# -*- coding: utf-8 -*-
"""ops.py — 工具实现。共 8 个，全部与任务无关；任务差异只体现在 plan 的组合里。"""
import json
import os

import numpy as np

from core.runtime import tool, match
from tools import geom
from tools import brep as brepmod
from tools import world


def _get(bus, sid):
    for s in bus.scene:
        if s.sid == sid:
            return s
    raise KeyError(f"场景里没有 {sid}；现有 {[s.sid for s in bus.scene][:12]}…")


def _items(bus, items):
    """接受 describe 列表 / sid 列表 / 单个 sid 字符串 / 逗号或空白分隔的 sid 串。

    字符串形态必须支持：模型经常直接写 "C008" 或 "C008 C009"，早期版本会把它当可迭代
    对象逐字符处理，抛 KeyError('C')，然后被记录成工具失败 —— 一个措辞问题被误当成
    「证据不足」，白白触发自救。
    """
    if isinstance(items, str):
        # 模型经常把数组序列化成字符串交过来："[\"C011\", \"C014\"]"。
        # 只按分隔符切会把引号留在 token 上，查不到实体，然后被记成「场景里没有 C011」——
        # 一个引号导致整条测量链断掉。这里把括号引号一起去掉。
        items = [x.strip("\"'") for x in
                 items.replace("[", " ").replace("]", " ").replace(",", " ").split()]
        items = [x for x in items if x]
    elif isinstance(items, dict):
        items = items.get("items") or items.get("solids") or [items]
    out = []
    for it in items:
        sid = it["sid"] if isinstance(it, dict) else it
        out.append(_get(bus, sid))
    return out


@tool("load",
      "读入模型，返回全部实体的几何指纹。此后所有工具都用返回的 sid 指代实体。",
      {"path": "str  模型文件路径", "keep_metadata": "bool 是否保留文件里的名称/类别（默认 false；任务09 必须 false）"},
      pre="文件存在且为 ifc/obj/gltf/usd 之一",
      degrade="不支持 fbx/blend 时先离线转 obj，并在 meta 里标注",
      returns="{meta, solids[describe...]}",
      unit="m（必要时自动做单位裁决，依据写进 meta.scale_source）")
def t_load(bus, path, keep_metadata=False):
    solids, meta = world.load(path, keep_names=keep_metadata)
    bus.scene = solids
    if hasattr(bus, "touch"):
        bus.touch()
    items = []
    for s in solids:
        d = geom.describe(s)
        if not keep_metadata:
            d.pop("src_type", None)
        items.append(d)
    return {"meta": meta, "solids": items, "n": len(items)}


def _desc(bus, S):
    """实体的几何指纹，按 sid 缓存。

    select 的输出必须始终是「完整指纹」而不是「裸 sid」。早期版本原样回传收到的元素，
    于是 select（sid 列表）之后，字段全丢，下一次 select 的 where 在缺字段的 dict 上
    恒为假、静默返回 0 —— 表现为「判据失效」，实际是工具在丢信息。这是最危险的一类 bug：
    它不报错，只让智能体得出错误结论。
    """
    cache = getattr(bus, "_descache", None)
    if cache is None:
        cache = bus._descache = {}
    key = (S.sid, id(S), getattr(bus, "gen", 0))
    d = cache.get(key)
    if d is None:
        d = geom.describe(S)
        d.pop("src_type", None)          # 类别属 metadata，几何任务一律不往外带
        cache[key] = d
    return dict(d)


@tool("select",
      "按几何字段过滤实体清单，返回命中的完整几何指纹。where 支持 "
      "{字段: 值} / {字段<=: 值} / {字段: {$lte: 值}} / {$and: [...]} / {$or: [...]}。",
      {"items": "list  实体清单，或引用字符串（如 @load.solids、@clu.groups[0].items）",
       "where": "dict  过滤条件；空 {} 表示全选",
       "probe": "bool  判据体检：反例检验（每条条件能指出谁被它排除了）+ 余量检验（阈值还能挪多少而不改变结论）"},
      pre="items 能被解析为场景里存在的 sid",
      degrade="无匹配时返回空列表，不报错 —— 由调用方决定下一步；"
              "若 where 里的字段在该实体上不存在，该条件判假（不会静默忽略）",
      returns="{items[完整指纹], n, where, probe?}")
def t_select(bus, items, where=None, probe=False):
    """过滤，并保证输出元素携带完整指纹。

    关键细节：输入元素自带的量优先。principal_frame 输出的是「主轴系下的 dims」，
    若这里无条件用世界轴重算，就把参照系成果覆盖掉了 —— 表现为「切了参照系还是筛不到」。
    正确语义是「缺什么补什么」，不是「以场景为准重算」。

    probe=True 时额外做一次逐条件体检。这不是「验算答案」，而是让命题自己暴露毛病：
    哪些条件在真正排除候选、哪些条件是凑数的（去掉也不影响命中）、
    哪些条件单独看一个都命中不了（阈值写歪了，或者参照系用错了）。
    """
    resolved = []
    for it in items:
        sid = it["sid"] if isinstance(it, dict) else it
        S = _get(bus, sid)
        d = _desc(bus, S)
        if isinstance(it, dict):
            d.update(it)
        d.pop("src_type", None)
        resolved.append((S, d))
    rows = [d for S, d in resolved]
    where = where or {}
    keep = [d for d in rows if match(d, where)]
    out = {"items": keep, "n": len(keep), "where": where}
    if probe:
        out["probe"] = _probe(rows, where, len(keep))
    return out


def _atoms(where):
    """把 where 摊成「一条一条」的条件，用来单独看每条的边际作用。"""
    if not isinstance(where, dict):
        return []
    if isinstance(where.get("$and"), list):
        return [a for a in where["$and"] if isinstance(a, dict)]
    out = []
    for k, v in where.items():
        if k in ("$or", "$not"):
            out.append({k: v})
        else:
            out.append({k: v})
    return out


_OP_SUFFIX = (">=", "<=", "==", "!=", ">", "<")


def _atom_parts(atom):
    """把一条条件拆成 (字段, 运算符, 阈值)。拆不出数值比较就返回 (字段, None, None)。

    两种写法都要认：{"min_thickness<=": 0.06} 与 {"min_thickness": {"$lte": 0.06}}。
    """
    if not isinstance(atom, dict) or len(atom) != 1:
        return None
    k, v = next(iter(atom.items()))
    if isinstance(v, dict) and len(v) == 1:
        op, t = next(iter(v.items()))
        op = {"$lte": "<=", "$lt": "<", "$gte": ">=", "$gt": ">",
              "$eq": "==", "$ne": "!="}.get(op)
        if op is None:
            return (k, None, None)
        return (k, op, t)
    for s in _OP_SUFFIX:
        if k.endswith(s):
            return (k[: -len(s)], s, v)
    return (k, None, None)


def _slack(rows, atom):
    """阈值还能挪多少而不改变这条判据的结果 —— 也就是「唯一」是稳的，还是踩在边界上。

    <= t：抬高会放进最近的一个被拒候选（slack_up），压低会让最近的一个通过者掉出去
    （slack_down）；>= t 反过来。两个数都是「挪这么多，结论就不一样了」。
    """
    parts = _atom_parts(atom)
    if not parts:
        return None
    field, op, t = parts
    if op not in (">=", "<=", ">", "<") or not isinstance(t, (int, float)):
        return None
    vals = [d[field] for d in rows
            if isinstance(d.get(field), (int, float)) and not isinstance(d.get(field), bool)]
    if not vals:
        return None
    cmp = {"<=": (lambda v: v <= t), "<": (lambda v: v < t),
           ">=": (lambda v: v >= t), ">": (lambda v: v > t)}[op]
    inside = [v for v in vals if cmp(v)]
    outside = [v for v in vals if not cmp(v)]
    up = down = None
    if op in ("<=", "<"):
        up = (min(outside) - t) if outside else None        # 抬高多少才会漏进一个
        down = (t - max(inside)) if inside else None        # 压低多少才会踢出一个
    else:
        down = (t - max(outside)) if outside else None      # 降低多少才会漏进一个
        up = (min(inside) - t) if inside else None          # 抬高多少才会踢出一个
    near = min([x for x in (up, down) if x is not None], default=None)
    fragile = None if near is None else (near <= max(1e-9, abs(float(t)) * 0.05))
    return {"op": op, "t": round(float(t), 6),
            "slack_up": None if up is None else round(float(up), 6),
            "slack_down": None if down is None else round(float(down), 6),
            "fragile": fragile}


def _probe(rows, where, n_hit):
    """逐条判据做两件事：找反例、量余量。

    反例检验：一条判据要算数，必须能指出「谁被它排除了」—— 即存在一个候选，
    满足其余所有条件、偏偏不满足这一条。找不出反例的条件是凑数的，它不构成证据。
    余量检验：阈值还能挪多少而不改变结论。挪一点点结论就变，说明「唯一」是巧合。
    """
    atoms = _atoms(where)
    keep = {d["sid"] for d in rows if match(d, where)}
    if len(atoms) < 2:
        return {"total": len(rows), "n_hit": n_hit, "atoms": [],
                "note": "只有一个条件，谈不上哪条在起作用；要看稳健性请补一个正交条件"}
    items = []
    for a in atoms:
        alone = {d["sid"] for d in rows if match(d, a)}
        rest = [b for b in atoms if b is not a]
        without = {d["sid"] for d in rows if match(d, {"$and": rest})}
        # 反例 = 满足其余条件、但不满足这一条的候选。有反例，这条判据才真的在干活。
        wit = sorted(without - alone)
        if wit:
            shown = "、".join(wit[:3]) + ("…" if len(wit) > 3 else "")
            role = "有效：反例 %s 满足其余条件、不满足这一条" % shown
        elif n_hit == 0 and not alone:
            role = "过严：这条单独用也一个都命中不了（阈值写歪了，或参照系不对）"
        elif not alone:
            role = "空判：场景里没有任何实体满足它，写它等于什么都没说"
        else:
            role = "冗余：其余条件已经排干净了，它没有独立排除任何人（删掉结论不变）"
        row = {"cond": json.dumps(a, ensure_ascii=False)[:80],
               "alone": len(alone), "without": len(without),
               "witness": wit[:3], "role": role}
        sl = _slack(rows, a)
        if sl:
            row["slack"] = sl
            if sl["fragile"]:
                near = min([x for x in (sl["slack_up"], sl["slack_down"]) if x is not None])
                row["role"] += ("；阈值正好压在某候选的实测值上，收紧一点就会丢命中"
                                if near <= 1e-9 else
                                "；阈值余量只有 %.3g，挪这么点就会换命中集合" % near)
        items.append(row)
    notes = []
    empty = [i for i in items if i["role"].startswith(("冗余", "空判"))]
    fragile = [i for i in items if (i.get("slack") or {}).get("fragile")]
    if n_hit == 0:
        notes.append("命中 0 个：命题太严或参照系不对 —— 先把单独命中数也是 0 的那条拎出来改")
    elif n_hit > 1:
        notes.append("命中 %d 个，还不是唯一：加一个正交条件（一个能单独把某个候选排除掉的判据）" % n_hit)
    if empty:
        notes.append("%d 条判据找不到反例：其余条件已经把它们要排除的候选排掉了。"
                     "不算错，但这两条不构成独立证据 —— 想让判据站得住，换成能单独指出反例的条件"
                     % len(empty))
    if fragile:
        notes.append("%d 条判据的阈值余量不足 5%%：现在唯一，稍微一动就可能不再唯一"
                     % len(fragile))
    if not notes:
        notes.append("每条判据都有具体反例，阈值也留了余量：这个唯一命中是稳的")
    return {"total": len(rows), "n_hit": n_hit, "atoms": items, "note": "；".join(notes)}


def witness_check(rows, where, sid=None):
    """给一次「唯一命中」打分：m 条判据里，有几条能指出具体反例。

    编排层在登记角色时顺手调它（纯计算，不占轮次）。一个角色的判据如果全都指不出反例，
    那它只是「唯一」而已 —— 报告里会显示成 8/12 这种数字，评委一眼能看出判据的密度。
    """
    atoms = _atoms(where)
    if not isinstance(rows, (list, tuple)) or len(atoms) < 2:
        return None
    rows = [r for r in rows if isinstance(r, dict) and "sid" in r]
    if len(rows) < 2:
        return None
    out = []
    for a in atoms:
        alone = {d["sid"] for d in rows if match(d, a)}
        rest = [b for b in atoms if b is not a]
        without = {d["sid"] for d in rows if match(d, {"$and": rest})}
        w = sorted(without - alone)
        if w:
            out.append({"cond": json.dumps(a, ensure_ascii=False)[:60], "witness": w[0]})
    return {"sid": sid, "n_cond": len(atoms), "n_witnessed": len(out), "witness": out[:4]}


@tool("gaps",
      "算目标实体到一组实体的包围盒间隙并升序排列。仅作粗筛，不能当最终距离用。",
      {"sid": "str 目标实体", "items": "list 候选实体清单", "limit": "int 返回前几个（默认 8）"},
      pre="两者均已加载",
      degrade="包围盒对薄壁/斜置零件会假阳性，必须用 distance 复核",
      returns="[{sid, aabb_gap}] 升序",
      unit="m")
def t_gaps(bus, sid, items, limit=30):
    a = _get(bus, sid)
    rows = []
    for s in _items(bus, items):
        if s.sid == sid:
            continue
        rows.append({"sid": s.sid, "aabb_gap": round(geom.aabb_gap(a, s), 6)})
    rows.sort(key=lambda r: r["aabb_gap"])
    return {"sid": sid, "rows": rows[:limit], "n": min(len(rows), limit),
            "n_total": len(rows)}


@tool("contains",
      "判断一组实体的包围盒是否整体落在目标实体的包围盒内。plane 可指定只看某个投影面"
      "（例如「家具占地是否在地板边界内」是 xy 平面问题，比三维包含才对）。",
      {"outer": "str 容器实体", "items": "list 候选实体清单",
       "plane": "str  'none' 三维（默认）/ 'xy' / 'xz' / 'yz'"},
      pre="两者均已加载",
      degrade="包围盒包含不等于实体包含；只作为证据链的一环，需与 segment 的洞口尺寸互证",
      returns="{inside:[{sid}], outside:[{sid}], margins:[{sid, margin, axes}], plane}")
def t_contains(bus, outer, items, plane="none"):
    ax = {"none": [0, 1, 2], "xy": [0, 1], "xz": [0, 2], "yz": [1, 2]}[plane]
    O = _get(bus, outer)
    lo, hi = O.bbox()
    inb, out, marg = [], [], []
    for s in _items(bus, items):
        l, h = s.bbox()
        # 只看 plane 指定的轴。三维包含会把「坐在地板上的椅子」判成越界 ——
        # 它的 z 上限 0.9 当然超出地板 -0.12..0，但那不是题目问的事。
        ok = all(l[i] >= lo[i] - 1e-6 and h[i] <= hi[i] + 1e-6 for i in ax)
        (inb if ok else out).append({"sid": s.sid})
        m = min(min(l[i] - lo[i], hi[i] - h[i]) for i in ax)
        marg.append({"sid": s.sid, "margin": round(float(m), 6),
                     "axes": ["xyz"[i] for i in ax],
                     "lo_gap": [round(float(l[i] - lo[i]), 6) for i in ax],
                     "hi_gap": [round(float(hi[i] - h[i]), 6) for i in ax]})
    return {"outer": outer, "plane": plane,
            "outer_bbox": [[round(float(x), 6) for x in lo], [round(float(x), 6) for x in hi]],
            "inside": inb, "outside": out, "margins": marg,
            # 「全部包含」这个结论必须由工具给，不能让读的人自己看 inside 列表空不空 ——
            # 空列表在观察里可能被截断，也可能被误读成「没查」。
            "all_inside": not out, "n_inside": len(inb), "n_outside": len(out)}


def _uf(n):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    return find, union


def _gap_boxes(b1, b2):
    """两个轴对齐盒的间隙向量范数（同一坐标系下才有意义）。"""
    lo1, hi1 = b1
    lo2, hi2 = b2
    d = np.maximum(0.0, np.maximum(lo1 - hi2, lo2 - hi1))
    return float(np.linalg.norm(d))


def _cluster_pass(solids, boxes, tol):
    find, union = _uf(len(solids))
    for i in range(len(solids)):
        for j in range(i + 1, len(solids)):
            if _gap_boxes(boxes[i], boxes[j]) <= tol:
                union(i, j)
    groups = {}
    for i in range(len(solids)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


@tool("clusters",
      "把实体分成互不相连的簇（一个文件里常装着多个彼此独立的模型）。"
      "间隙在主参照系下计算，因此整体旋转过的模型也能正确分离。",
      {"items": "list  实体清单（通常直接给 load.solids）",
       "tol": "float  判定「相邻」的间隙阈值，米（默认 1.0）",
       "frame": "str  'auto' 用场景主参照系 / 'world' 用世界轴",
       "metric": "str  'bbox' 包围盒间隙（快）/ 'surface' 真实表面最短距离（准，仅对候选对精算）"},
      pre="items 非空；frame='auto' 时需要场景里存在显著平面朝向（info.coverage 会给出）",
      degrade="frame='auto' 但覆盖率不足时自动退回世界轴，并在 frame.used 里标注；"
              "metric='surface' 时若候选对过多则退回 bbox 并标注",
      returns="{groups[{cid,n,sids,items,bbox_world,span_frame,separation}], "
              "frame, consistency{world_n, frame_n, agree}, tol, metric}")
def t_clusters(bus, items, tol=1.0, frame="auto", metric="bbox"):
    solids = _items(bus, items)
    if not solids:
        return {"groups": [], "n_groups": 0, "tol": tol, "metric": metric,
                "note": "空清单"}
    n = len(solids)

    # --- 1) 选定参照系。为什么不用世界轴：轴对齐包围盒只是真实距离的下界，
    # 整体旋转 θ 时这个下界最多膨胀到 √2 倍，簇间间隙随之塌缩 —— 45° 时实测
    # 火箭与房间的 AABB 间隙从 2.8 m 掉到 0.71 m，三个模型被并成一簇。
    # 在主参照系里，包围盒重新变紧，判据恢复与位姿无关。
    axes, origin, finfo = geom.frame_axes(solids)
    if frame == "world" or finfo.get("coverage", 0.0) < 0.5:
        axes, origin = np.eye(3), np.zeros(3)
        used = "world"
        if frame == "auto":
            finfo = dict(finfo, fallback="覆盖率 %.3f 不足，退回世界轴" % finfo.get("coverage", 0.0))
    else:
        used = "frame"
    Q = [(S.P - origin) @ axes.T for S in solids]
    boxes_f = [(q.min(0), q.max(0)) for q in Q]
    boxes_w = [S.bbox() for S in solids]

    # --- 2) 聚类
    groups_idx = _cluster_pass(solids, boxes_f, tol)
    world_n = len(_cluster_pass(solids, boxes_w, tol))
    note = None

    if metric == "surface":
        # 先用（已收紧的）包围盒间隙粗筛候选对，再做精确表面距离，只连真实相邻者。
        find, union = _uf(n)
        cand = 0
        budget = 400
        cheap = [(i, j) for i in range(n) for j in range(i + 1, n)
                 if _gap_boxes(boxes_f[i], boxes_f[j]) <= max(tol * 2.0, tol + 0.5)]
        if len(cheap) > budget:
            note = "候选对 %d 超过预算 %d，metric 退回 bbox" % (len(cheap), budget)
            metric = "bbox"
            for i, j in cheap:
                if _gap_boxes(boxes_f[i], boxes_f[j]) <= tol:
                    union(i, j)
        else:
            for i, j in cheap:
                cand += 1
                d = geom.distance_min(solids[i], solids[j])["value"]
                if d <= tol:
                    union(i, j)
        g2 = {}
        for i in range(n):
            g2.setdefault(find(i), []).append(i)
        groups_idx = list(g2.values())

    # --- 3) 输出。bbox_world 给渲染与人工核对，span_frame 给判据使用。
    src = {it["sid"]: it for it in items if isinstance(it, dict)}
    groups_idx.sort(key=lambda g: (-len(g), min(g)))
    out = []
    for cid, g in enumerate(groups_idx):
        lo = np.min([boxes_w[i][0] for i in g], axis=0)
        hi = np.max([boxes_w[i][1] for i in g], axis=0)
        flo = np.min([boxes_f[i][0] for i in g], axis=0)
        fhi = np.max([boxes_f[i][1] for i in g], axis=0)
        # 与其他簇的最小间隙（主参照系下）：给 agent 判断「分得干不干净」的依据
        sep = None
        for cid2, g2 in enumerate(groups_idx):
            if cid2 == cid:
                continue
            sep = min([_gap_boxes(boxes_f[i], boxes_f[j]) for i in g for j in g2]
                      + ([sep] if sep is not None else []))
        out.append({"cid": cid, "n": len(g),
                    "sids": [solids[i].sid for i in g],
                    "items": [src.get(solids[i].sid, {"sid": solids[i].sid}) for i in g],
                    "bbox_world": [[round(float(x), 6) for x in lo],
                                   [round(float(x), 6) for x in hi]],
                    "span_frame": [round(float(x), 6) for x in (fhi - flo)],
                    "separation": None if sep is None else round(float(sep), 4)})

    return {"groups": out, "n_groups": len(out), "tol": tol, "metric": metric,
            "frame": {"used": used,
                      "axes": [[round(float(x), 9) for x in a] for a in axes],
                      "origin": [round(float(x), 9) for x in origin],
                      "coverage": finfo.get("coverage"),
                      "is_axis_aligned": finfo.get("is_axis_aligned")},
            "consistency": {"world_n_groups": world_n, "frame_n_groups": len(out),
                            "agree": world_n == len(out)},
            **({"note": note} if note else {})}


@tool("pick",
      "从清单里取出 key==value 的第一条。用于把「另一条支线算出来的清单」按实体对齐，"
      "例如把某实体的测量行对齐到最终报告的字段上。",
      {"items": "list 清单（rows / items 均可）", "key": "str 字段名", "value": "any 目标值"},
      pre="清单元素是 dict 且含该字段",
      degrade="找不到时返回 found=false 且 item=null，下游引用会显式失败，不会静默给错值",
      returns="{item, found}")
def t_pick(bus, items, key, value):
    seq = items.get("rows") or items.get("items") or items if isinstance(items, dict) else items
    for it in seq:
        if isinstance(it, dict) and it.get(key) == value:
            return {"item": it, "found": True, "n": 1, "key": key, "value": value}
    return {"item": None, "found": False, "n": 0, "key": key, "value": value}


@tool("merge",
      "把多份实体清单按顺序拼成一份（去重按 sid）。",
      {"lists": "list 多份实体清单"},
      pre="各清单来自 select 或 load",
      degrade="重复 sid 只保留首次出现",
      returns="{items, n}")
def t_merge(bus, lists):
    seen, out = set(), []
    for lst in lists:
        items = lst.get("items", lst) if isinstance(lst, dict) else lst
        for it in items:
            sid = it["sid"] if isinstance(it, dict) else it
            if sid in seen:
                continue
            seen.add(sid)
            out.append(it)
    return {"items": out, "n": len(out)}


@tool("distance",
      "两实体最短表面距离（点-面双向 + 边-边精修），返回见证点对。",
      {"a": "str 实体A", "b": "str 实体B"},
      pre="两网格闭合、世界坐标、可相交（相交返回 0）",
      degrade="网格过大时自动分块，仅在精修预算内做边-边；预算不够会在 method 里降级标注",
      returns="{value, unit, witness_a, witness_b, method, uncertainty}",
      unit="m")
def t_distance(bus, a, b):
    r = geom.distance_min(_get(bus, a), _get(bus, b))
    r["a"], r["b"] = a, b          # 让数值自带走「它是哪一对的距离」，复核时不必再猜
    return r


@tool("rank",
      "把目标实体到一组实体的最短表面距离算出来并升序排序（逐个调用 distance）。",
      {"a": "str 目标实体", "items": "list 候选实体清单"},
      pre="候选清单已通过上一级过滤缩小（建议 ≤ 6 个，否则很慢）",
      degrade="候选过多时先 gaps 粗筛",
      returns="[{sid, value, witness_a, witness_b}] 升序",
      unit="m")
def t_rank(bus, a, items):
    A = _get(bus, a)
    rows = []
    for s in _items(bus, items):
        if s.sid == a:
            continue
        r = geom.distance_min(A, s)
        rows.append({"sid": s.sid, "value": r["value"],
                     "witness_a": r["witness_a"], "witness_b": r["witness_b"],
                     "method": r["method"]})
    rows.sort(key=lambda r: r["value"])
    return {"a": a, "rows": rows, "n": len(rows)}


@tool("principal_frame",
      "估计场景自身的竖直参照系，并把实体在参照系里重新描述一次。用于模型整体不与坐标轴对齐时"
      "（旋转、倾斜、坐标原点任意）——此时「薄/长/方」这些基于 AABB 的描述量会失效。",
      {"items": "list 实体清单（通常是整个场景或一个簇）",
       "cos_tol": "float 法向聚类的平行阈值（默认 0.999，约 2.5°）"},
      pre="场景里有显著面积的平面（墙/板/箱），且竖直方向存在",
      degrade="曲面为主（无平面簇）时 coverage 会很低，此时返回原始世界轴并置 up_confident=false，"
              "调用方应退回世界轴判据而不是相信这个 frame",
      returns="{axes, origin, info{coverage,dev_from_world_deg,is_axis_aligned}, items_frame[...]}",
      unit="m")
def t_principal_frame(bus, items, cos_tol=0.999):
    solids = _items(bus, items)
    axes, origin, info = geom.frame_axes(solids, cos_tol=cos_tol)
    # 竖直基准沿用世界 Z：不把地面标高重定基。否则「落地」这个判据（z_min≈0）
    # 会被整体平移 0.12 m 而全部失效 —— 这是坐标系归一化最容易埋的坑。
    origin = np.array([float(origin[0]), float(origin[1]), 0.0])
    described = []
    for s in solids:
        d = geom.describe(geom.in_frame(s, axes, origin))
        d["sid"] = s.sid                       # sid 保持与原场景一致
        described.append(d)
    return {"axes": [[round(float(x), 9) for x in a] for a in axes],
            "origin": [round(float(x), 9) for x in origin],
            "info": info, "items_frame": described, "n": len(described),
            "transform": "p_frame = (p_world - origin) @ axes.T  (origin.z 恒为 0)"}


@tool("segment",
      "把实体还原成共面区域（B-Rep 面）：法向、面积、平面度、边界环、内环（开口）、缺口面积。",
      {"sid": "str 实体", "sharp_deg": "float 二面角阈值（默认20）", "max_faces": "int 返回前几大面（默认10）"},
      pre="实体为闭合网格",
      degrade="曲面片在低阈值下会被拆碎、高阈值下会被误合并；阈值敏感性要报告",
      returns="{faces:[{n, area, dev, planar, holes[], reflex, notch_area}]}")
def t_segment(bus, sid, sharp_deg=20.0, max_faces=10):
    return brepmod.faces(_get(bus, sid), sharp_deg=sharp_deg, max_faces=max_faces)


@tool("section",
      "水平（或沿指定轴）截面面积，用于「面积为 X 的截面在什么高度」这类反问题。",
      {"sid": "str 实体", "axis": "int 0/1/2 对应 x/y/z", "height": "float 截面位置"},
      pre="实体闭合",
      degrade="不闭合时截面线段可能不配对，n_segments 会暴露异常",
      returns="{value, n_segments, axis, height}",
      unit="m²")
def t_section(bus, sid, axis=2, height=0.0):
    return geom.section_area(_get(bus, sid), axis=axis, height=height)


@tool("answer",
      "提交结构化答案与证据引用。沙盒只接受带证据引用的答案。",
      {"values": "dict 结论数值（键自定，单位必须写在键名或单独字段）",
       "evidence": "list 证据步骤 id 列表，只写步骤 id 本身，如 ['s5','s6']（不要写 @s5.items 这种引用路径）",
       "confidence": "dict 每项置信度 high/medium/low",
       "unknowns": "list 几何无法确定的项及缺什么信息",
       "notes": "str 方法说明"},
      pre="所有数值都能追溯到某个工具步骤",
      degrade="证据不足时把项写进 unknowns，不要猜",
      returns="原样回显，作为 run 的最终答案")
def t_answer(bus, values, evidence=None, confidence=None, unknowns=None, notes=""):
    return {"values": values, "evidence": evidence or [], "confidence": confidence or {},
            "unknowns": unknowns or [], "notes": notes}


# 写层（edit）与只读层分开注册；放在文件末尾 import，避免与只读工具产生循环依赖。
import tools.edit  # noqa: E402,F401
import tools.checks  # noqa: E402,F401  （只读判据：占地、距边距离）
import tools.collide  # noqa: E402,F401  （运动层：接触/穿插判定、旋转与平移扫掠）
import tools.shape  # noqa: E402,F401  （形状分解：连通分量、轴向凹坑、2D 刚体配准）
import tools.measure   # noqa: E402,F401  反解与拟合
import tools.contact   # noqa: E402,F401  区域关系与接触面积
import tools.family    # noqa: E402,F401  洞口族：接触 + 连通（归属关系）
