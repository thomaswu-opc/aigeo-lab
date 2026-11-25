# -*- coding: utf-8 -*-
"""checks.py — 只读判据：占地、到边距离、包含关系。

为什么单独成文件：这些量是「验收判据」，会被 04/05/08 反复用，而且必须可被审计。
它们不改场景，只把一个模糊要求翻译成一个数。

翻译的依据来自赛方 FAQ，这里逐条固定下来，避免每次重新解释：
  * 椅子占地 = 整张椅子垂直投影到地板上的范围，重叠部分只算一次；
  * 占地中心 = 沿房间左右/前后方向，刚好包住该范围的最小矩形的中心；
  * 家具距门距离 = 家具占地**边缘**到「门洞底边」线段的**最短平面距离**，
    不量家具中心到门扇的距离。
"""
import numpy as np

from core.runtime import tool
from tools import geom


def _vertical_axis(axes):
    """参照系里哪一行是竖直轴：与 [0,0,1] 最平行的那一行。返回 (k_up, k_u, k_v)。"""
    k = int(np.argmax(np.abs(axes[:, 2])))
    rest = [i for i in range(3) if i != k]
    return k, rest[0], rest[1]


def _footprint(S, axes, origin):
    """占地：把顶点投影到水平面，取沿 u/v 的最小包围矩形。"""
    k, ku, kv = _vertical_axis(axes)
    Q = (S.P - origin) @ axes.T
    u, v, w = Q[:, ku], Q[:, kv], Q[:, k]
    return {"rect": [float(u.min()), float(v.min()), float(u.max()), float(v.max())],
            "center": [float((u.min() + u.max()) / 2), float((v.min() + v.max()) / 2)],
            "size": [float(u.max() - u.min()), float(v.max() - v.min())],
            "u": u, "v": v, "w": w, "k": (ku, kv)}


def _pt_in_tri(p, T):
    def cr(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2, d3 = cr(T[0], T[1], p), cr(T[1], T[2], p), cr(T[2], T[0], p)
    neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (neg and pos)


def _seg_pt(p, a, b):
    """点到线段距离。"""
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-18:
        return float(np.linalg.norm(p - a)), a
    t = max(0.0, min(1.0, float((p - a) @ ab) / L2))
    q = a + ab * t
    return float(np.linalg.norm(p - q)), q


def _seg_seg(p1, p2, q1, q2):
    """两条 2D 线段的最短距离与最近点。相交返回 0。"""
    r, s = p2 - p1, q2 - q1
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) > 1e-15:
        t = ((q1[0] - p1[0]) * s[1] - (q1[1] - p1[1]) * s[0]) / den
        u = ((q1[0] - p1[0]) * r[1] - (q1[1] - p1[1]) * r[0]) / den
        if -1e-12 <= t <= 1 + 1e-12 and -1e-12 <= u <= 1 + 1e-12:
            p = p1 + r * t
            return 0.0, p, p
    best = None
    for p, a, b, which in ((p1, q1, q2, "p1"), (p2, q1, q2, "p2"),
                           (q1, p1, p2, "q1"), (q2, p1, p2, "q2")):
        d, q = _seg_pt(p, a, b)
        if best is None or d < best[0]:
            best = (d, p, q)
    return best


@tool("footprint",
      "实体在水平面上的占地范围：沿场景左右/前后轴的最小包围矩形、矩形中心、尺寸，"
      "以及落地情况（z 范围）。这就是赛方定义的「椅子占地」与「占地中心」。",
      {"sids": "list 实体（或单个 sid）",
       "frame": "str 'auto' 用场景主参照系（可对抗整体旋转）/ 'world' 用世界轴"},
      pre="实体已加载；frame='auto' 时场景需有显著平面朝向",
      degrade="曲面为主时覆盖率不足，自动退回世界轴并在 frame.used 标注；"
              "占地是「投影的包围矩形」，若实体本身倾斜，矩形会大于真实投影",
      returns="{rows:[{sid, rect, center, size, z_min, z_max, grounded}], frame}",
      unit="m / m²")
def t_footprint(bus, sids, frame="auto"):
    from tools.edit import _sids
    sol = _sids(bus, sids)
    axes, origin, info = geom.frame_axes(sol)
    used = "frame"
    if frame == "world" or info.get("coverage", 0.0) < 0.5:
        axes, origin, used = np.eye(3), np.zeros(3), "world"
        if frame == "auto":
            info = dict(info, fallback="覆盖率 %.3f 不足，退回世界轴" % info.get("coverage", 0.0))
    # 竖直原点沿用世界 z=0，与 principal_frame 保持一致：地面就是 z=0 这件事
    # 不能在两个工具里有两种说法，否则「落地」判据在一边成立、在另一边不成立。
    origin = np.array([float(origin[0]), float(origin[1]), 0.0])
    rows = []
    for S in sol:
        fp = _footprint(S, axes, origin)
        k, ku, kv = _vertical_axis(axes)
        # 参照系 → 世界：矩形四角换回去，便于人核对（判据仍用参照系下的数）
        u0, v0, u1, v1 = fp["rect"]
        corners = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]], float)
        W = np.zeros((4, 3))
        W[:, ku], W[:, kv] = corners[:, 0], corners[:, 1]
        Ww = W @ axes + origin
        cu, cv = fp["center"]
        c3 = np.zeros(3)
        c3[ku], c3[kv] = cu, cv
        cw = c3 @ axes + origin
        zmin = float(S.P[:, 2].min())
        row = {
            "sid": S.sid,
            # 参照系下的量：严格、与整体旋转无关
            "rect": [round(x, 9) for x in fp["rect"]],
            "center": [round(x, 9) for x in fp["center"]],
            "size": [round(x, 9) for x in fp["size"]],
            # 世界坐标下的同一件事：给人算「距左墙多少」用，不必自己换基
            "center_world": [round(float(cw[0]), 9), round(float(cw[1]), 9), round(zmin, 9)],
            "rect_world_xy": [round(float(Ww[:, 0].min()), 9), round(float(Ww[:, 1].min()), 9),
                              round(float(Ww[:, 0].max()), 9), round(float(Ww[:, 1].max()), 9)],
            "z_min": round(zmin, 9),
            "z_max": round(float(S.P[:, 2].max()), 9),
            "grounded": bool(abs(zmin) < 1e-3),
        }
        if used != "world":
            # 参照系的 u/v 不一定对应世界 x/y。不说清楚，看的人会拿 u 当 x 去算距离。
            # ku/kv 是**参照系里的行号**，不是世界轴编号。早先直接把 "xyz"[ku] 当时世界轴
            # 输出，结果房间报成 "u=x"，而 u 实际沿 y —— 净宽净深因此整体对调。
            # 正确的映射是：u 方向 = 基向量 axes[ku]，它最贴近哪个世界轴就是哪个。
            row["axis_map"] = {
                "u": "xyz"[int(np.argmax(np.abs(axes[ku])))],
                "v": "xyz"[int(np.argmax(np.abs(axes[kv])))],
            }
            if not info.get("is_axis_aligned"):
                row["note"] = "模型整体未与坐标轴对齐：rect/center/size 在参照系内，center_world 已换回世界"
        rows.append(row)
    return {"rows": rows, "n": len(rows),
            "frame": {"used": used, "axes": [[round(float(x), 9) for x in a] for a in axes],
                      "origin": [round(float(x), 9) for x in origin],
                      "coverage": info.get("coverage"),
                      "rule": "rect/center/size 均在参照系 (u,v) 内；(u,v) = 除竖直轴外的两行"}}


@tool("clearance",
      "实体占地边缘到一条水平线段的**最短平面距离**。用于「家具距门洞是否 ≥1 m」："
      "线段取门洞底边（门洞开口与地板相接的那条线）。不量中心到线段的距离。",
      {"sid": "str 实体", "p1": "list 线段端点1 [x,y,z]", "p2": "list 线段端点2 [x,y,z]",
       "frame": "str 'auto' / 'world'"},
      pre="p1、p2 在同一水平面上；线段应与被量实体处于同一个房间",
      degrade="距离按占地（投影）算，若实体悬空则该投影不代表实际接触；"
              "线段与占地相交时返回 0",
      returns="{sid, value, nearest_on_footprint, nearest_on_edge, footprint_rect}",
      unit="m")
def t_clearance(bus, sid, p1, p2, frame="auto"):
    from tools.edit import _sids
    S = _sids(bus, sid)[0]
    axes, origin, info = geom.frame_axes(bus.scene)
    if frame == "world" or info.get("coverage", 0.0) < 0.5:
        axes, origin = np.eye(3), np.zeros(3)
    origin = np.array([float(origin[0]), float(origin[1]), 0.0])
    k, ku, kv = _vertical_axis(axes)
    fp = _footprint(S, axes, origin)
    A = ((np.asarray(p1, float) - origin) @ axes.T)[[ku, kv]]
    B = ((np.asarray(p2, float) - origin) @ axes.T)[[ku, kv]]
    pts = np.stack([fp["u"], fp["v"]], 1)
    tris = pts[S.F]                                  # (T,3,2) 投影后的三角形
    best = None
    for T in tris:
        d, p, q = _seg_seg(A, B, T[0], T[1])
        cand = [(d, p, q)]
        for e0, e1 in ((T[1], T[2]), (T[2], T[0])):
            dd, pp, qq = _seg_seg(A, B, e0, e1)
            cand.append((dd, pp, qq))
        for c in cand:
            if best is None or c[0] < best[0]:
                best = c
    d, on_fp, on_edge = best
    # 参照系 → 世界（只补回竖直分量，取线段所在高度）
    def back(pt2, z):
        v3 = np.zeros(3)
        v3[ku], v3[kv] = pt2[0], pt2[1]
        return (v3 @ axes + origin) + np.array([0, 0, z])
    z = float(np.asarray(p1, float)[2])
    return {"sid": S.sid, "value": round(float(d), 9),
            "nearest_on_footprint": [round(float(x), 6) for x in back(on_fp, z)],
            "nearest_on_edge": [round(float(x), 6) for x in back(on_edge, z)],
            "footprint_rect": [round(float(x), 9) for x in fp["rect"]],
            "n_tri_checked": int(len(tris))}


@tool("opening_edge",
      "由「墙 + 洞口实体 + 地板」推出洞口底边线段（位于室内墙面上、与地板相接的那条线）。"
      "这是「家具距门距离」的测量基准线：两端取墙体开口边缘，不扣除门框。",
      {"walls": "list 候选墙体（会自动挑出被该洞口贯穿的那一面）",
       "opening": "str 洞口实体（贯穿墙体的那个盒状几何）",
       "floor": "str 地板实体，用于判断墙的哪一侧朝向室内"},
      pre="洞口实体确实贯穿某面墙：在这面墙的薄方向上，洞口的两端都超出墙面",
      degrade="若没有独立的洞口实体（洞口只体现为墙自身的缺口），本工具不适用，"
              "此时应改用 segment 读墙面的缺口（notch_area / reflex）并自行取边缘",
      returns="{wall, opening, edge[[x,y,z],[x,y,z]], width, height, sill_z, inward_normal}",
      unit="m")
def t_opening_edge(bus, walls, opening, floor):
    from tools.edit import _sids
    O = _sids(bus, opening)[0]
    F = _sids(bus, floor)[0]
    olo, ohi = O.bbox()
    oc = (olo + ohi) / 2
    flo, fhi = F.bbox()
    fc = (flo + fhi) / 2

    # 1) 挑被贯穿的那面墙：洞口中心落在墙的包围盒内（留一点容差）
    pick = None
    for W in _sids(bus, walls):
        lo, hi = W.bbox()
        if np.all(oc >= lo - 1e-6) and np.all(oc <= hi + 1e-6):
            pick = W
            break
    if pick is None:
        raise ValueError("没有哪面墙的包围盒包含洞口中心；洞口可能与墙不共面")
    wlo, whi = pick.bbox()
    wd = whi - wlo

    # 2) 墙的薄方向就是洞口的贯穿方向
    k = int(np.argmin(wd))
    rest = [i for i in range(3) if i != k]
    # 洞口在墙平面内的水平方向（水平轴），另一个是竖直方向
    horiz = [i for i in rest if i != 2]
    if not horiz:
        raise ValueError("墙体没有水平延伸方向，无法定义开口宽度")
    u = horiz[0]

    # 3) 室内侧：墙面朝向地板中心的那一侧
    inner = wlo[k] if abs(fc[k] - wlo[k]) < abs(fc[k] - whi[k]) else whi[k]
    inward = np.zeros(3)
    inward[k] = 1.0 if inner == whi[k] else -1.0

    # 4) 底边：与地板相接的标高（洞口下沿），两端取洞口在 u 方向的边界
    sill = float(olo[2]) if abs(olo[2]) < 1e-6 else float(olo[2])
    p1 = np.zeros(3)
    p2 = np.zeros(3)
    p1[k] = p2[k] = inner
    p1[u], p2[u] = float(olo[u]), float(ohi[u])
    p1[2] = p2[2] = sill
    return {"wall": pick.sid, "opening": O.sid, "floor": F.sid,
            "edge": [[round(float(x), 9) for x in p1], [round(float(x), 9) for x in p2]],
            "width": round(float(ohi[u] - olo[u]), 9),
            "height": round(float(ohi[2] - olo[2]), 9),
            "sill_z": round(sill, 9),
            "axis_u": "xyz"[u], "axis_thin": "xyz"[k],
            "inward_normal": [round(float(x), 6) for x in inward],
            "rule": "底边在室内墙面(沿薄方向朝地板中心的一侧)上，u 范围取洞口两端，标高取洞口下沿"}
