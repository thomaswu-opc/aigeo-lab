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
        # 被拆掉的父实体 -> 它拆出来的子实体。零件拆解会从场景里移除父件，但后续
        # 引用仍会写旧编号（实测 T03 有 37 次 KeyError 全是这一个原因）。
        # 记在这里，报错时就能直接说「C017 已拆成 C017.1/C017.2，改用新编号」，
        # 而不是只丢一句「场景里没有 C017」让它自己猜。
        self.split_map = {}
        # 本次 resolve 期间发生的「自动别名解析」：引用写错但意图明确时按最近一步解析，
        # 并把这件事记下来告诉调用方（账本 + 观察里各一份），不能悄悄改成想要的样子。
        self.notes = []

    def touch(self):
        self.gen += 1

    def missing_msg(self, sid):
        """「场景里没有 X」的完整解释：现在有哪些、X 是否已被拆件换号。"""
        have = [s.sid for s in self.scene]
        msg = "场景里没有 %s；现有 %s…" % (sid, have[:12])
        kids = self.split_map.get(sid)
        if kids:
            msg += ("；%s 已被 parts 拆分为 %s —— 原编号不再存在，后续引用请改用新编号"
                    % (sid, "/".join(kids)))
        else:
            # 常见错法：写的是 C017 但实际编号带后缀，或反过来。
            near = [x for x in have if x.startswith(str(sid).rstrip("0123456789"))]
            if near:
                msg += "；相近编号：%s" % near[:6]
        return msg

    # --- 引用解析：@step.field[3].k ---
    # 字段别名：模型在 load 之后写 @all.items，实际字段叫 solids。这类错在账本里
    # 出现 22 次，每次都要重发一次调用才纠回来。名字对不上但意图唯一时，直接按别名
    # 解析，并把「正式字段名」写回观察里 —— 既不浪费一轮，也不让它以为随便写都行。
    FIELD_ALIAS = {"items": ("solids", "entities", "parts", "list"),
                   "solids": ("items", "entities"),
                   "value": ("items", "value"),
                   "values": ("value", "items", "solids")}

    def resolve(self, obj, results, toolmap=None):
        """toolmap: 步骤 id -> 产生它的工具名。用来把 @load 这种「按工具名引用」纠到实际步骤。"""
        if isinstance(obj, str) and obj.startswith("@"):
            return self._deref(obj, results, toolmap)
        if isinstance(obj, dict):
            return {k: self.resolve(v, results, toolmap) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.resolve(v, results, toolmap) for v in obj]
        return obj

    def _alias_label(self, root, results, toolmap):
        """把 @load / @Load / @load_2 这类引用纠到实际存在的步骤 id。

        实测 37 次失败都是同一个写法：模型把「步骤名」当成「工具名」。
        load 这一步实际叫 all（或它自己起的别名），于是 @load.solids 永远解析不到。
        纠错只在**意图唯一**时发生：同名工具只出现过一次、或取最近一次。
        """
        if not results:
            return None
        if toolmap:
            rl = root.lower()
            cands = [k for k, t in toolmap.items() if str(t).lower() == rl and k in results]
            if len(cands) == 1:
                return cands[0]
            if len(cands) > 1:
                return cands[-1]
            # @load_2 / @select3：去掉尾部序号再试一次
            base = root.rstrip("0123456789_")
            if base and base != root:
                cands = [k for k, t in toolmap.items() if t == base and k in results]
                if cands:
                    return cands[-1]
        low = {k.lower(): k for k in results}
        if root.lower() in low:
            return low[root.lower()]
        return None

    def _deref(self, ref, results, toolmap=None):
        path = ref[1:].split(".")
        root = path[0]
        if root not in results:
            lab = self._alias_label(root, results, toolmap)
            if lab is None:
                steps = ", ".join("%s(%s)" % (k, (toolmap or {}).get(k, "?")) for k in results)
                raise KeyError("未定义的引用 %s；已完成步骤是 %s。引用要写**步骤名**，"
                               "不是工具名 —— 步骤名见左侧总览表第一列。" % (ref, steps or "（无）"))
            self.notes.append({"ref": ref, "from": root, "to": lab,
                               "tool": (toolmap or {}).get(lab)})
            root = lab
        cur = results[root]
        for seg in path[1:]:
            for part in seg.split("["):
                part = part.strip("]")
                if not part:
                    continue
                if isinstance(cur, dict):
                    if part not in cur:
                        # 字段名对不上但意图唯一（items ↔ solids 之类）：按别名解析一次，
                        # 同时把正式字段名记下来。多数情况下这一步就等于「省掉一轮」。
                        alt = next((a for a in self.FIELD_ALIAS.get(part, ())
                                    if a in cur and not str(a).startswith("_")), None)
                        if alt:
                            self.notes.append({"ref": ref, "field": part, "to": alt,
                                               "step": root})
                            cur = cur[alt]
                            continue
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
