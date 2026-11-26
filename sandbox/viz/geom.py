# -*- coding: utf-8 -*-
"""geom.py — 把场景导成「浏览器能直接画」的紧凑几何。

为什么不再只给 PNG：快照有三个改不掉的问题 —— 遮挡关系依赖画家算法（会错）、
视角固定（看不到侧面/顶部）、不能点（看不出这块是哪个实体）。
本机实测：20 个实体、9245 顶点、18426 三角面，扁平 JSON 约 500 KB，
gzip 后 ~150 KB，本机局域网传输是零感知的量级。所以「重不重」不是问题，
问题只在「要不要为此写一套前端」。

格式刻意压平（不嵌套 [[x,y,z],...]）：
  v  顶点扁平数组 x,y,z,x,y,z...
  f  索引扁平数组 a,b,c,a,b,c...
  g  实体分组 {sid, v0, vn, f0, fn, dim, c}
这样前端 `new Float32Array(g.v)` 直接进 BufferGeometry，不需要再 map 一遍。
"""
import json
import os

import numpy as np

# 与 viz/render.py 的 PALETTE 对齐，两张视图里同一个 sid 颜色一致
PALETTE = [
    (150, 162, 178), (176, 186, 200), (128, 142, 160),
    (214, 158, 106), (198, 132, 96), (226, 186, 128),
    (108, 168, 148), (140, 150, 200), (196, 150, 186),
]


def dump(solids, path, tag=None, step=None, turn=None, note=None, marks=None):
    """写一份 geom.json。原子替换：页面 800 ms 轮询，不能读到半个文件。"""
    if not solids:
        return None
    order = sorted(solids, key=lambda s: s.sid)
    v, f, groups = [], [], []
    color_of = {}
    for i, s in enumerate(order):
        color_of[s.sid] = PALETTE[i % len(PALETTE)]

    voff = foff = 0
    for s in order:
        P = np.asarray(s.P, float)
        F = np.asarray(s.F, int)
        # 4 位小数 = 0.1 mm，足够"看"，也让文件比全精度小一半
        v.extend(np.round(P, 4).reshape(-1).tolist())
        f.extend((F + voff).reshape(-1).tolist())
        dim = (P.max(0) - P.min(0)).tolist() if len(P) else [0, 0, 0]
        groups.append({"sid": s.sid, "v0": voff, "vn": len(P), "f0": foff, "fn": len(F),
                       "dim": [round(x, 4) for x in dim], "c": list(color_of[s.sid])})
        voff += len(P)
        foff += len(F)

    payload = {
        "tag": tag, "step": step, "turn": turn, "note": note,
        "n_solids": len(order), "n_tri": len(f) // 3,
        "v": v, "f": f, "g": groups,
    }
    if marks is not None:
        payload["marks"] = marks
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    os.replace(tmp, path)
    return {"bytes": os.path.getsize(path), "n_solids": len(order), "n_tri": len(f) // 3}


def live_geometry(bus, outdir, every=4, min_gap=1.5):
    """跑动中更新 runs/<tag>/geom.json。与 live_preview 分开节流：

    预览图是给人看「有没有在动」，几何是给人转着看「现在是什么样」，
    后者更有用，所以刷得比图勤一点，但同样不能每步都写。
    """
    import time
    path = os.path.join(outdir, "geom.json")
    state = {"n": -1, "t": 0.0}

    def _cb(loop):
        if not bus.scene:
            return
        n = len(loop.steps)
        now = time.time()
        if state["n"] >= 0 and (n - state["n"] < every and now - state["t"] < min_gap):
            return
        last = loop.steps[-1] if loop.steps else {}
        try:
            marks = step_marks(loop)
            dump(bus.scene, path, step=last.get("id"), turn=loop.turn,
                 note="%s · %s" % (loop.phase, (last.get("tool") or "—")),
                 marks=marks)
            state["n"], state["t"] = n, now
        except Exception:
            pass

    return _cb


# ================================================================ 每步三维注解（marks）
# 目的：三维画面不只显示「现在场景长什么样」，还要把「这一步工具做了什么」
# 直接画出来 —— 选中了谁、量了哪两点、量出的角度在哪、切面在哪。
# 数据源是 loop.results[sid]（完整原始返回值），不是被截断到 300 字的 obs。

def _sid_list(v):
    """把工具参数里的实体参数摊成 sid 列表。"""
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v else []
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            if isinstance(x, str):
                out.append(x)
            elif isinstance(x, dict) and isinstance(x.get("sid"), str):
                out.append(x["sid"])
        return out
    if isinstance(v, dict) and isinstance(v.get("sid"), str):
        return [v["sid"]]
    return []


def _pt(v, nd=4):
    """点坐标规整成小数组；非点返回 None。"""
    try:
        a = [round(float(x), nd) for x in v]
        return a if len(a) == 3 else None
    except (TypeError, ValueError):
        return None


def step_marks(loop):
    """遍历已成功执行的步骤，每步生成一条三维注解。

    返回 [{step, turn, tool, phase, focus, markers, lines, arcs, planes}]。
    focus = 高亮的实体 sid；其余是叠加的几何图元。只读，失败返回 None 的项被跳过。
    """
    out = []
    for st in (loop.steps or []):
        sid = st.get("id")
        tool = st.get("tool")
        if not sid or st.get("error") or st.get("rejected"):
            continue
        res = (loop.results or {}).get(sid)
        if not isinstance(res, dict):
            continue
        args = st.get("args")
        # args 是 JSON 字符串（截断过）；解析失败就当没有
        A = {}
        if isinstance(args, str):
            try:
                A = json.loads(args)
            except Exception:
                A = {}
        elif isinstance(args, dict):
            A = args

        m = {"step": sid, "turn": st.get("turn"), "tool": tool,
             "phase": st.get("phase"), "why": (st.get("why") or "")[:80],
             "focus": [], "markers": [], "lines": [], "arcs": [], "planes": []}

        # —— 选中/列举类：focus = 命中的实体
        if tool in ("select", "pick", "merge", "clusters", "segment", "principal_frame"):
            items = res.get("items")
            if tool == "clusters":
                groups = res.get("groups") or []
                sids = []
                for g in groups:
                    sids += _sid_list(g.get("items") or g.get("members"))
                m["focus"] = sids
            elif items:
                m["focus"] = _sid_list(items)
            elif isinstance(res.get("item"), dict):
                m["focus"] = _sid_list([res["item"]])
        elif tool == "parts":
            m["focus"] = _sid_list(res.get("parts")) + _sid_list(res.get("parent"))

        # —— 测量类：距离见证点 + 连线
        if tool == "distance":
            a, b = res.get("a"), res.get("b")
            m["focus"] = _sid_list([a, b])
            wa, wb = _pt(res.get("witness_a")), _pt(res.get("witness_b"))
            if wa and wb:
                m["lines"].append({"a": wa, "b": wb, "label": "%s m" % (res.get("value"),),
                                   "color": "#ffd166", "kind": "measure"})
                m["markers"].append({"pos": wa, "label": "A", "color": "#ffd166"})
                m["markers"].append({"pos": wb, "label": "B", "color": "#ffd166"})
        elif tool == "rank":
            m["focus"] = _sid_list([res.get("a")])
            rows = res.get("rows") or []
            for r in rows[:1]:
                wa, wb = _pt(r.get("witness_a")), _pt(r.get("witness_b"))
                if wa and wb:
                    m["lines"].append({"a": wa, "b": wb, "label": "%s m" % (r.get("value"),),
                                       "color": "#ffd166", "kind": "measure"})
                    m["markers"].append({"pos": wb, "label": r.get("sid"), "color": "#ffd166"})

        # —— 角度：两个法向 → 从交点出发的两根短线 + 弧
        if tool == "angle":
            na, nb = _pt(res.get("n_a")), _pt(res.get("n_b"))
            acute = res.get("acute_deg")
            # 法向起点取两个实体包围盒中心的中点附近；精确交点工具没给，用实体 bbox 中心
            sid_a, sid_b = A.get("a"), A.get("b")
            pa = _pt(_bbox_center(loop, sid_a)) or [0, 0, 0]
            if na and nb:
                L = 0.6
                enda = [pa[i] + na[i] * L for i in range(3)]
                endb = [pa[i] + nb[i] * L for i in range(3)]
                m["lines"].append({"a": pa, "b": enda, "label": "n_a",
                                   "color": "#7ee787", "kind": "normal"})
                m["lines"].append({"a": pa, "b": endb, "label": "n_b",
                                   "color": "#ff9bce", "kind": "normal"})
                m["arcs"].append({"center": pa, "n_a": na, "n_b": nb,
                                  "label": "%s°" % (acute,), "color": "#a5d6ff"})
            m["focus"] = _sid_list([sid_a, sid_b])

        # —— 间隙：两点连线
        if tool == "clearance":
            na, ne = _pt(res.get("nearest_on_footprint")), _pt(res.get("nearest_on_edge"))
            m["focus"] = _sid_list([A.get("sid")])
            if na and ne:
                m["lines"].append({"a": na, "b": ne, "label": "%s m" % (res.get("value"),),
                                   "color": "#ffd166", "kind": "measure"})
                m["markers"].append({"pos": ne, "label": "线", "color": "#ffd166"})

        # —— 洞口底边：一条参考线
        if tool == "opening_edge":
            edge = res.get("edge") or []
            if len(edge) == 2:
                p1, p2 = _pt(edge[0]), _pt(edge[1])
                if p1 and p2:
                    m["lines"].append({"a": p1, "b": p2, "label": "洞口底边",
                                       "color": "#a5d6ff", "kind": "ref"})

        # —— 碰撞/接触：deepest 点
        if tool == "collide":
            ua, ub = res.get("a"), res.get("b")
            m["focus"] = _sid_list(ua) + _sid_list(ub)
            dep = _pt(res.get("deepest"))
            if dep and (res.get("n_pairs") or 0) > 0:
                m["markers"].append({"pos": dep, "label": "最深 %s" % (res.get("depth"),),
                                     "color": "#ff6b6b"})

        # —— 切面：平面法向 + 位置 → 一个半透明四边形
        if tool in ("cut_plane", "section"):
            pl = res.get("plane")
            if isinstance(pl, dict):
                n = _pt(pl.get("normal"))
                d = pl.get("d")
                m["planes"].append({"normal": n, "d": d, "label": "切面",
                                    "color": "#c792ea"})
                m["focus"] = _sid_list([A.get("sid")])

        # —— 编辑类：focus = 被改的实体
        if tool in ("translate", "rotate_axis", "stretch_axis"):
            m["focus"] = _sid_list(A.get("sid") or A.get("sids"))

        # 只保留有内容的
        if m["focus"] or m["markers"] or m["lines"] or m["arcs"] or m["planes"]:
            out.append(m)

    return out


def _bbox_center(loop, sid):
    """取某实体当前包围盒中心（用于角度法向起点）。取不到返回 None。"""
    try:
        S = loop.bus.scene.get(sid)
    except Exception:
        S = None
    if S is None:
        return None
    try:
        P = np.asarray(S.P, float)
        return (P.min(0) + P.max(0)) / 2.0
    except Exception:
        return None
