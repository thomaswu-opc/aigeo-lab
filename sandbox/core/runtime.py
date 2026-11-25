# -*- coding: utf-8 -*-
"""runtime.py — 沙盒的三件套：账本、工具总线、引用解析。

设计要点（这三点是「可证明」的全部机制）：
  1. 账本 append-only + 哈希链：每一步调用、参数、结果摘要、耗时都留痕，改一行即断链。
  2. 总线是唯一入口：plan 只能通过注册过的工具接触几何，没有 eval、没有 open、没有 ifcopenshell 直连。
  3. 工具不知道「题目」：注册表里不存在任何任务编号，任务信息只存在于 card（字符串）与 plan（数据）里。
"""
import hashlib
import inspect
import json
import os
import time
import traceback


# ------------------------------------------------------------ 账本


class Ledger:
    def __init__(self, path):
        self.path = path
        self.prev = "0" * 16
        self.seq = 0
        self.records = []
        with open(path, "w", encoding="utf-8"):
            pass

    def append(self, kind, payload):
        rec = {"seq": self.seq, "t": round(time.time(), 4), "kind": kind,
               "payload": payload, "prev": self.prev}
        body = json.dumps(rec, sort_keys=True, ensure_ascii=False, default=_jsonable)
        rec["hash"] = hashlib.sha256((self.prev + body).encode()).hexdigest()[:16]
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=_jsonable) + "\n")
        self.prev = rec["hash"]
        self.seq += 1
        self.records.append(rec)
        return rec

    @staticmethod
    def verify(path):
        prev = "0" * 16
        bad = []
        for i, ln in enumerate(open(path, encoding="utf-8")):
            rec = json.loads(ln)
            body = json.dumps({k: rec[k] for k in ("seq", "t", "kind", "payload", "prev")},
                              sort_keys=True, ensure_ascii=False, default=_jsonable)
            if rec["prev"] != prev or hashlib.sha256((prev + body).encode()).hexdigest()[:16] != rec["hash"]:
                bad.append(rec["seq"])
            prev = rec["hash"]
        return {"ok": not bad, "n": i + 1 if "i" in dir() else 0, "broken": bad}


def _jsonable(o):
    try:
        import numpy as np
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    if hasattr(o, "__dict__"):
        return {k: v for k, v in vars(o).items() if not k.startswith("_")}
    return str(o)


# ------------------------------------------------------------ 工具总线


TOOLS = {}


def tool(name, desc, params, pre, degrade, unit="", returns=""):
    """注册一个工具。pre=前置条件，degrade=降级路径，两者都会进 tool schema 供 LLM 选择。"""
    def deco(fn):
        sig = inspect.signature(fn)
        TOOLS[name] = {"fn": fn, "desc": desc, "params": params, "pre": pre,
                       "degrade": degrade, "returns": returns, "unit": unit,
                       "sig": list(sig.parameters)}
        return fn
    return deco


def schema():
    return [{"name": k, "description": v["desc"], "parameters": v["params"],
             "preconditions": v["pre"], "degrade": v["degrade"],
             "returns": v["returns"], "unit": v["unit"]}
            for k, v in TOOLS.items()]


class Bus:
    def __init__(self, ledger, scene=None):
        self.ledger = ledger
        self.scene = scene or []
        # 场景代号：每次整体换场景/替换实体就 +1。指纹缓存把它算进键里 —— 只靠 id(S)
        # 不行：load 之后旧 Solid 被回收，新 Solid 可能**拿到同一个 id**，缓存于是返回
        # 上一个模型的指纹（实测：同进程里连读两个模型，第二个模型的椅子拿到了第一个的坐标）。
        self.gen = 0

    def touch(self):
        self.gen += 1

    # --- 引用解析：@step.field[3].k ---
    def resolve(self, obj, results):
        if isinstance(obj, str) and obj.startswith("@"):
            return self._deref(obj, results)
        if isinstance(obj, dict):
            return {k: self.resolve(v, results) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.resolve(v, results) for v in obj]
        return obj

    def _deref(self, ref, results):
        path = ref[1:].split(".")
        root = path[0]
        if root not in results:
            raise KeyError(f"未定义的引用 {ref}（已完成步骤：{list(results)}）")
        cur = results[root]
        for seg in path[1:]:
            for part in seg.split("["):
                part = part.strip("]")
                if not part:
                    continue
                if isinstance(cur, dict):
                    if part not in cur:
                        # 说清楚「这个步骤的返回里到底有哪些字段」。只报 KeyError: 'solids'
                        # 时模型会换个同义词再试一次，把轮次耗在猜字段名上。
                        keys = [k for k in cur if not str(k).startswith("_")]
                        raise KeyError("%s 在 %r 处不存在；该步骤返回的字段是 %s"
                                       % (ref, part, keys[:14]))
                    cur = cur[part]
                elif isinstance(cur, list):
                    cur = cur[int(part)]
                else:
                    raise KeyError(f"{ref} 在 {seg} 处无法下钻：{type(cur)}")
        return cur

    def _coerce(self, name, args, step_id):
        """把「本该是数组/对象、却被写成 JSON 字符串」的参数还原。

        实测：模型会传 items='["O004","O005"]'（一整串），于是 select 把字符串当清单逐字符
        遍历，报 KeyError；同一个错法重复三次就被判停滞，整轮作废。这不是它不懂几何，
        只是一个格式问题 —— 与 where 条件、evidence 列表同样的宽容处理：认得出就还原，
        并在账本里记一笔，不偷偷改。
        """
        spec = TOOLS[name]["params"]
        out = {}
        for k, v in args.items():
            want = (spec.get(k) or "").strip().split(" ")[0].split("\u3000")[0]
            if isinstance(v, str) and want in ("list", "dict") and v[:1] in "[{":
                try:
                    j = json.loads(v)
                except Exception:
                    j = None
                if isinstance(j, (list, dict)) and (want == "dict") == isinstance(j, dict):
                    self.ledger.append("arg_coerced", {"step": step_id, "tool": name, "arg": k})
                    v = j
            out[k] = v
        return out

    def call(self, name, args, step_id=""):
        if name not in TOOLS:
            raise KeyError(f"未注册的工具 {name}；可用：{sorted(TOOLS)}")
        args = self._coerce(name, args, step_id)
        t0 = time.perf_counter()
        try:
            out = TOOLS[name]["fn"](self, **args)
            ok = True
            err = None
        except Exception as e:
            out, ok, err = None, False, f"{type(e).__name__}: {e}"
            self.ledger.append("tool_error", {"step": step_id, "tool": name, "args": _short(args),
                                              "error": err, "trace": traceback.format_exc()[-1200:]})
        dt = (time.perf_counter() - t0) * 1000
        digest = hashlib.sha256(json.dumps(out, sort_keys=True, ensure_ascii=False,
                                           default=_jsonable).encode()).hexdigest()[:12] if ok else None
        self.ledger.append("tool_call", {"step": step_id, "tool": name, "args": _short(args),
                                         "ok": ok, "ms": round(dt, 1), "digest": digest,
                                         "result_excerpt": _short(out)})
        if not ok:
            raise RuntimeError(f"[{step_id}] {name} 失败：{err}")
        return out


def _short(o, n=1400):
    s = json.dumps(o, ensure_ascii=False, default=_jsonable)
    return s if len(s) <= n else s[:n] + f"…(+{len(s)-n})"


# ------------------------------------------------------------ 只读过滤（不执行字符串）


_OPS = (">=", "<=", "==", "!=", ">", "<")


_OPMAP = {"$gte": ">=", "$lte": "<=", "$gt": ">", "$lt": "<", "$ne": "!=",
          "$eq": "==", ">=": ">=", "<=": "<=", ">": ">", "<": "<", "==": "==", "!=": "!=",
          "$in": "in", "$nin": "not in"}


def _cmp(op, got, want):
    try:
        if op == "in":
            return got in want
        if op == "not in":
            return got not in want
        if op == ">=":
            return got >= want
        if op == "<=":
            return got <= want
        if op == ">":
            return got > want
        if op == "<":
            return got < want
        if op == "!=":
            return got != want
        return got == want
    except TypeError:
        return False


def match(item, where):
    """纯数据比较，绝不 eval —— 这是「不让 LLM 注入代码」的底线。

    接受两种等价写法（模型两种都会用，与其纠正它不如都接住）：
      {"min_thickness<=": 0.06}          ← 后缀紧凑式
      {"min_thickness": {"$lte": 0.06}}  ← 操作符式
    以及逻辑组合：
      {"$and": [c1, c2]} / {"$or": [...]} / {"$not": c}
    """
    if not isinstance(where, dict):
        return False
    for key, want in where.items():
        if key in ("$and", "$or"):
            conds = want if isinstance(want, list) else [want]
            ok = all(match(item, c) for c in conds) if key == "$and" \
                else any(match(item, c) for c in conds)
            if not ok:
                return False
            continue
        if key == "$not":
            if match(item, want):
                return False
            continue

        if isinstance(want, dict) and want and all(k in _OPMAP for k in want):
            for o, v in want.items():
                if key not in item or not _cmp(_OPMAP[o], item[key], v):
                    return False
            continue

        op, field = "==", key
        for o in _OPS:
            if key.endswith(o):
                field, op = key[: -len(o)], o
                break
        if field not in item:
            return False
        if not _cmp(op, item[field], want):
            return False
    return True


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_jsonable)
