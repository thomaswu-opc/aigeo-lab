# -*- coding: utf-8 -*-
"""edit.py — 写层：改几何、体检、落盘。

与 tools/geom.py 的分工写在最前面，因为这是整个 EDIT 层唯一需要守住的纪律：

  * geom 只读，输入 Solid 输出数字；
  * edit 改变 bus.scene，并且**每次改动都返回 before/after 的可核对量**。

工具绝不返回「已完成」「成功」这类自证结论。一次编辑是否成立，只能由读回落盘文件、
用同一套只读工具重新量出来的数字来判断 —— 否则「编辑」就退化成模型自己说自己对。

坐标一律世界坐标、单位米。编辑原语不认识任何任务：translate / stretch_axis / cut_plane
可以拼出「移家具」「加宽房间」「拉高方块并切球」，但代码里没有出现这些词。
"""
import json
import math
import os

import numpy as np

from core.runtime import tool
from tools import geom
from tools.world import Solid

WELD_TOL = 1e-7          # 顶点焊接容差（米）。房间尺度 5 m，1e-7 相当于 0.02 µm


# ================================================================ 网格原语


def _welded(P, F, tol=WELD_TOL):
    """合并重合顶点，并丢弃未被任何面引用的顶点。

    编辑链上焊接是必须的：拉伸后相邻实体的共享边上会出现只差 1e-16 的「两个顶点」，
    它们会让后续切割/封口把一条边认成两条，切口环因此断开。
    压缩同样必须：裁剪会保留原顶点表，被切掉那半边的顶点没人引用却仍在数组里，
    体积不受影响，但 V-E+F 会算成几千、落盘文件也白白变大。
    """
    P = np.asarray(P, float).reshape(-1, 3)
    F = np.asarray(F, int).reshape(-1, 3)
    if len(P) == 0 or len(F) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    key = np.round(P / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    P2 = P[first]
    F2 = inv[F]
    ok = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])
    F2 = F2[ok]
    used = np.unique(F2)
    remap = -np.ones(len(P2), int)
    remap[used] = np.arange(len(used))
    return P2[used], remap[F2]


def _plane(axis=None, value=0.0, normal=None):
    """构造平面 n·p = d。axis 写法给轴对齐平面，normal 写法给任意朝面。"""
    if normal is not None:
        n = np.asarray(normal, float)
        ln = np.linalg.norm(n)
        if ln < 1e-12:
            raise ValueError("normal 不能为零向量")
        n = n / ln
        return n, float(value)
    n = np.zeros(3)
    n[int(axis)] = 1.0
    return n, float(value)


def _clip_halfspace(P, F, n, d, side=1, eps=1e-9):
    """把三角网格裁到 n·p*side >= d*side 的半空间里（苏瑟兰-霍奇曼，逐三角形）。

    返回 (P2, F2, segs, stats)：
      segs  每条形如 ((ka, kb), pa, pb) —— 切割边，端点是平面上的插值点，
            ka/kb 是按容差量化出来的顶点键，用来把相邻三角形的切口接成环。
      stats 报三个计数：整留、整丢、被切。三者之和必须等于原三角形数，
            这是裁剪没有静默丢面的自检。
    """
    P = np.asarray(P, float).reshape(-1, 3)
    F = np.asarray(F, int).reshape(-1, 3)
    f = (P @ n - d) * side                    # >0 = 保留侧
    fs = f[F]
    keep_all = np.all(fs >= -eps, axis=1)
    drop_all = np.all(fs <= eps, axis=1)
    mix = ~(keep_all | drop_all)

    outP = [P]
    nv = len(P)
    outF = [F[keep_all]]
    segs = []
    q = 1.0 / max(eps * 1e3, WELD_TOL)

    def _key(p):
        return (int(round(p[0] * q)), int(round(p[1] * q)), int(round(p[2] * q)))

    cache = {}

    def _pt_on_edge(i, j):
        """边 i→j 与平面的交点；同一对顶点只算一次，键也一并对齐。"""
        a, b = (i, j) if i < j else (j, i)
        hit = cache.get((a, b))
        if hit is None:
            t = fs_flat[a] / (fs_flat[a] - fs_flat[b])
            p = P[a] + (P[b] - P[a]) * t
            hit = (_key(p), p)
            cache[(a, b)] = hit
        return hit

    fs_flat = f
    for t in np.nonzero(mix)[0]:
        ids = [int(F[t, 0]), int(F[t, 1]), int(F[t, 2])]
        poly = []
        cut = []
        for k in range(3):
            i, j = ids[k], ids[(k + 1) % 3]
            fi, fj = f[i], f[j]
            if fi >= -eps:
                poly.append(P[i])
                cut.append(None if abs(fi) > eps else _key(P[i]))
            if (fi > eps and fj < -eps) or (fi < -eps and fj > eps):
                kk, pp = _pt_on_edge(i, j)
                poly.append(pp)
                cut.append(kk)
        if len(poly) < 3:
            continue
        base = nv
        outP.append(np.array(poly))
        nv += len(poly)
        for m in range(1, len(poly) - 1):
            outF.append(np.array([[base, base + m, base + m + 1]]))
        # 切口线段：多边形里相邻两点都落在平面上
        m = len(cut)
        for k in range(m):
            a, b = cut[k], cut[(k + 1) % m]
            if a is not None and b is not None and a != b:
                segs.append(((a, b), poly[k], poly[(k + 1) % m]))

    P2 = np.vstack(outP) if outP else np.zeros((0, 3))
    F2 = np.vstack(outF) if outF else np.zeros((0, 3), int)
    stats = {"kept_whole": int(keep_all.sum()), "dropped_whole": int(drop_all.sum()),
             "split": int(mix.sum()), "n_in": int(len(F))}
    return P2, F2, segs, stats


def _rings(segs, eps=1e-9):
    """接环逻辑只有一份，在 geom 里；这里保留旧名，避免 cut_plane 的调用点散开。"""
    return geom.rings_from_segs(segs)


def _basis(n):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = a - n * a.dot(n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def _ear_clip(pts2):
    """二维多边形耳切。输入须逆时针（或凸）；退化时退回扇形，并如实上报。"""
    idx = list(range(len(pts2)))
    tri, guard = [], 0

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def inside(p, a, b, c):
        d1, d2, d3 = cross(a, b, p), cross(b, c, p), cross(c, a, p)
        return (d1 > 1e-14 and d2 > 1e-14 and d3 > 1e-14)

    while len(idx) > 3 and guard < 4 * len(pts2) + 16:
        guard += 1
        cut = False
        for m in range(len(idx)):
            i0, i1, i2 = idx[m - 1], idx[m], idx[(m + 1) % len(idx)]
            a, b, c = pts2[i0], pts2[i1], pts2[i2]
            if cross(a, b, c) <= 1e-14:
                continue
            if any(inside(pts2[k], a, b, c) for k in idx if k not in (i0, i1, i2)):
                continue
            tri.append((i0, i1, i2))
            idx.pop(m)
            cut = True
            break
        if not cut:
            break
    if len(idx) == 3:
        tri.append((idx[0], idx[1], idx[2]))
        return tri, "ear"
    if len(idx) > 3:
        tri = [(idx[0], idx[m], idx[m + 1]) for m in range(1, len(idx) - 1)]
        return tri, "fan"
    return tri, "degenerate"


def _cap(rings, n, d, want_flip):
    """把切口环封成平面面片。法向朝 +n（want_flip=True 时反向，保证壳朝外）。"""
    u, v = _basis(n)
    tris, notes = [], []
    for ring in rings:
        c = ring.mean(0)
        p2 = np.stack([(ring - c) @ u, (ring - c) @ v], 1)
        area2 = 0.5 * np.sum(p2[:, 0] * np.roll(p2[:, 1], -1) - np.roll(p2[:, 0], -1) * p2[:, 1])
        if area2 < 0:
            ring, p2 = ring[::-1], p2[::-1]
        tri, how = _ear_clip([tuple(x) for x in p2])
        notes.append({"n": len(ring), "area": round(abs(float(area2)), 9), "method": how})
        for a, b, cc in tri:
            tris.append((ring[a], ring[b], ring[cc]) if not want_flip else (ring[a], ring[cc], ring[b]))
    return tris, notes


def mesh_report(P, F):
    """网格体检：封闭性、边界环、重复面、退化面、朝向一致性、欧拉特征。

    这就是 Task 07 第 4 问要的那份报告，但它不针对球 —— 对任何网格都成立。
    """
    P = np.asarray(P, float).reshape(-1, 3)
    F = np.asarray(F, int).reshape(-1, 3)
    out = {"n_vert": int(len(P)), "n_tri": int(len(F))}
    if len(F) == 0:
        return dict(out, closed=False, n_boundary_edges=0, n_boundary_loops=0,
                    n_duplicate_faces=0, n_degenerate=0, euler=None,
                    signed_volume=0.0, surface_area=0.0, watertight=False,
                    bbox=None, dims=None)
    lo, hi = P.min(0), P.max(0)
    out["bbox"] = [_f(lo), _f(hi)]
    out["dims"] = [round(float(x), 9) for x in (hi - lo)]

    A, B, C = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    cr = np.cross(B - A, C - A)
    ar = np.linalg.norm(cr, axis=1) / 2.0
    vol = float(np.einsum("ij,ij->i", A, np.cross(B, C)).sum() / 6.0)
    # 包围盒体积与填充率。墙上有洞时填充率必然小于 1，且「少了多少」正好是洞口体积：
    # 这是判断「这面墙上有没有开口」唯一不依赖洞口实体的办法，也是 T-09 里
    # 区分「窗（墙上有贯通洞）」与「画框（墙是整的）」的那条分水岭。
    bv = float(np.prod(hi - lo))
    out["bbox_volume"] = round(bv, 9)
    out["fill"] = round(vol / bv, 9) if bv > 1e-15 else None

    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    ukey = np.sort(E, axis=1)
    uniq, cnt = np.unique(ukey, axis=0, return_counts=True)
    boundary = uniq[cnt == 1]
    nonman = uniq[cnt > 2]

    # 同向重复的有向边 = 相邻两面朝向相反（壳体没统一朝向）
    ue, ce = np.unique(E, axis=0, return_counts=True)
    flip_edges = int((ce > 1).sum())

    fkey = np.sort(F, axis=1)
    _, cf = np.unique(fkey, axis=0, return_counts=True)
    dup_faces = int((cf > 1).sum())

    # 边界环：把边界边连成环，数量就是「破洞个数」
    loops = 0
    if len(boundary):
        adj = {}
        for a, b in boundary:
            adj.setdefault(int(a), []).append(int(b))
            adj.setdefault(int(b), []).append(int(a))
        seen = set()
        for s in adj:
            if s in seen:
                continue
            stack, comp = [s], []
            seen.add(s)
            while stack:
                x = stack.pop()
                comp.append(x)
                for y in adj[x]:
                    if y not in seen:
                        seen.add(y)
                        stack.append(y)
            loops += 1

    out.update({
        "closed": len(boundary) == 0,
        "watertight": len(boundary) == 0 and len(nonman) == 0 and flip_edges == 0,
        "n_boundary_edges": int(len(boundary)),
        "n_boundary_loops": loops,
        "n_nonmanifold_edges": int(len(nonman)),
        "n_duplicate_faces": dup_faces,
        "n_degenerate": int((ar < 1e-12).sum()),
        "n_flipped_edges": flip_edges,
        "euler": int(len(P) - len(uniq) + len(F)),
        "signed_volume": round(vol, 9),
        "volume": round(abs(vol), 9),
        "surface_area": round(float(ar.sum()), 9),
        "n_used_vert": int(len(np.unique(F))),
    })
    if out["n_used_vert"] != out["n_vert"]:
        out["note_unused_vert"] = "有未引用的顶点（焊接或裁剪残留）"
    return out


# ================================================================ 场景改造


def _replace(bus, S, P, F, tag=""):
    """用新几何替换场景里的同一个 sid，并清掉指纹缓存。

    为什么必须换对象实例而不是原地改 S.P：_desc 的缓存键是 (sid, id(S))，
    原地改数组 id 不变，缓存会返回改动前的指纹 —— 后续 select 就在旧数据上过滤，
    表现为「改了但判据不认」。换实例让缓存自然失效。
    """
    P, F = _welded(P, F)
    if hasattr(bus, "touch"):
        bus.touch()          # 场景代号 +1：指纹缓存按 (sid, id(S), gen) 取，避免 id 复用命中旧缓存
    new = Solid(S.sid, S.name, P, F, (S.src or "?") + ("+" + tag if tag else "+edit"))
    i = bus.scene.index(S)
    bus.scene[i] = new
    cache = getattr(bus, "_descache", None)
    if cache is not None:
        cache.clear()
    return new


def _get(bus, sid):
    for s in bus.scene:
        if s.sid == sid:
            return s
    raise KeyError("场景里没有 %s；现有 %s…" % (sid, [x.sid for x in bus.scene][:12]))


def _flatten_sids(sids):
    """把各种形态的「实体清单」压成一维 sid 字符串列表。

    `["@leaf.items", "@chair.items"]` 这种写法解析出来是嵌套的
    （`[[{sid:"C011"}], [{sid:"C017"}]]`），逐项当 sid 处理会抛
    「场景里没有 [{'sid': ...}]」，然后被记成工具失败 —— 一个写法问题
    被误判成证据不足。这里递归展平，字符串、dict、list 三种输入都接住。
    """
    if sids is None:
        return []
    if isinstance(sids, str):
        return [x for x in sids.replace("[", " ").replace("]", " ")
                .replace(",", " ").replace("\"", " ").replace("'", " ").split() if x]
    if isinstance(sids, dict):
        if "sid" in sids:
            return [sids["sid"]]
        return _flatten_sids(sids.get("sids") or sids.get("items") or [])
    out = []
    for it in sids:
        out.extend(_flatten_sids(it))
    return out


def _sids(bus, sids):
    return [_get(bus, x) for x in _flatten_sids(sids)]


def _f(v):
    return [round(float(x), 9) for x in np.asarray(v).reshape(-1)]


# ================================================================ 工具


@tool("translate",
      "把一组实体整体平移（同一个位移量、刚性、不缩放不转动）。多个实体一起给即整体移动，"
      "例如「火箭各部件作为一个整体移动」。",
      {"sids": "list 要平移的实体 sid 列表（或单个 sid）",
       "delta": "list [dx,dy,dz] 位移，米"},
      pre="实体已加载；delta 是世界坐标下的位移向量",
      degrade="若只想移动单个实体，传单元素列表；不提供绕轴转动",
      returns="{moved:[{sid,bbox_before,bbox_after}], delta, n}",
      unit="m")
def t_translate(bus, sids, delta):
    d = np.asarray(delta, float).reshape(3)
    rows = []
    for S in _sids(bus, sids):
        lo0, hi0 = S.bbox()
        S = _replace(bus, S, S.P + d, S.F, "translate")
        lo1, hi1 = S.bbox()
        rows.append({"sid": S.sid, "bbox_before": _f([lo0, hi0]), "bbox_after": _f([lo1, hi1])})
    return {"moved": rows, "n": len(rows), "delta": _f(d)}


@tool("rotate_axis",
      "把一组实体绕一条固定轴整体旋转一个角度（右手定则）。刚性变换，不缩放不剪切。"
      "转轴用「轴上一点 + 方向」给出，因此转轴不必过原点。",
      {"sids": "list 要旋转的实体 sid 列表（或单个 sid）",
       "pivot": "list [px,py,pz] 轴上的任一点，米",
       "axis": "str|list 轴向，'x'/'y'/'z' 或三分量向量",
       "angle_deg": "float 旋转角（度），右手定则：拇指指 axis 正向，四指方向为正"},
      pre="实体已加载；pivot 与 axis 共同确定一条无限长直线",
      degrade="只做刚体旋转；要绕自身中心转就把 pivot 取成该实体的包围盒中心",
      returns="{rotated:[{sid,bbox_before,bbox_after,max_shift}], pivot, axis, angle_deg}",
      unit="m / deg")
def t_rotate_axis(bus, sids, pivot, axis, angle_deg):
    piv = np.asarray(pivot, float).reshape(3)
    if isinstance(axis, str):
        m = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}
        if axis.lower() not in m:
            raise ValueError(f"轴只能取 x/y/z 或三分量向量，收到 {axis!r}")
        k = np.array(m[axis.lower()], float)
    else:
        k = np.asarray(axis, float).reshape(3)
    nn = float(np.linalg.norm(k))
    if nn < 1e-12:
        raise ValueError("轴向量长度为 0")
    k = k / nn
    th = math.radians(float(angle_deg))
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)
    rows = []
    for S in _sids(bus, sids):
        lo0, hi0 = S.bbox()
        P0 = np.asarray(S.P, float)
        P1 = (P0 - piv) @ R.T + piv
        shift = float(np.linalg.norm(P1 - P0, axis=1).max()) if len(P0) else 0.0
        S = _replace(bus, S, P1, S.F, "rotate")
        lo1, hi1 = S.bbox()
        rows.append({"sid": S.sid, "bbox_before": _f([lo0, hi0]), "bbox_after": _f([lo1, hi1]),
                     "max_shift": round(shift, 9)})
    return {"rotated": rows, "n": len(rows), "pivot": _f(piv),
            "axis": _f(k), "angle_deg": float(angle_deg)}


@tool("stretch_axis",
      "沿某一坐标轴做分段线性重映射：给若干锚点对 [旧坐标, 新坐标]，锚点之间线性插值、"
      "两端按最外段斜率外推。这是「参数联动」的通用表达 —— 换锚点即换参数，几何随之重建。",
      {"sids": "list 受影响的实体 sid 列表",
       "axis": "int 0/1/2 对应 x/y/z",
       "anchors": "list [[from,to],...] 至少 2 对，按 from 升序，如 [[4.6,4.6],[9.4,11.4],[9.6,11.6]]"},
      pre="anchors 严格递增且 at 两两不等；实体顶点按该轴分量落段",
      degrade="锚点少于 2 对时直接报错，不做默认值猜测；段外按端点斜率外推，会拉伸到无穷远，"
              "故跨过整个实体的锚点应显式给出",
      returns="{stretched:[{sid,n_vert,max_shift,bbox_before,bbox_after}], anchors, axis}",
      unit="m")
def t_stretch_axis(bus, sids, axis, anchors):
    ax = int(axis)
    pts = sorted(([float(a), float(b)] for a, b in anchors), key=lambda p: p[0])
    if len(pts) < 2:
        raise ValueError("anchors 至少需要 2 对 [from,to]")
    fr = [p[0] for p in pts]
    to = [p[1] for p in pts]
    for i in range(1, len(fr)):
        if not (fr[i] > fr[i - 1] + 1e-12):
            raise ValueError("anchors 的 from 必须严格递增：%s" % fr)

    rows = []
    for S in _sids(bus, sids):
        lo0, hi0 = S.bbox()
        x = S.P[:, ax].copy()
        y = np.interp(x, fr, to)          # 段外 np.interp 用端点值(平台)，不是外推
        # np.interp 在段外是常数，会把握手/贴边的顶点挤成一堆。手动补外推。
        left = x < fr[0]
        right = x > fr[-1]
        if left.any():
            k = (to[1] - to[0]) / (fr[1] - fr[0])
            y[left] = to[0] + (x[left] - fr[0]) * k
        if right.any():
            k = (to[-1] - to[-2]) / (fr[-1] - fr[-2])
            y[right] = to[-1] + (x[right] - fr[-1]) * k
        P2 = S.P.copy()
        P2[:, ax] = y
        shift = float(np.abs(y - x).max()) if len(x) else 0.0
        S = _replace(bus, S, P2, S.F, "stretch")
        lo1, hi1 = S.bbox()
        rows.append({"sid": S.sid, "n_vert": int(len(S.P)), "max_shift": round(shift, 9),
                     "bbox_before": _f([lo0, hi0]), "bbox_after": _f([lo1, hi1])})
    return {"stretched": rows, "n": len(rows), "axis": "xyz"[ax],
            "anchors": [[round(a, 9), round(b, 9)] for a, b in pts],
            "rule": "x' = interp(x, from→to)；段外按最外段斜率外推"}


@tool("cut_plane",
      "用一个平面切开实体，保一侧，并把切口用平面面片封上（不封就是破洞）。"
      "返回切口环数量、面积、被切三角形数，以及封口前后的封闭性，便于自查。",
      {"sid": "str 被切实体",
       "axis": "int 0/1/2 平面法向沿哪根轴（与 normal 二选一）",
       "value": "float 平面位置（n·p = value 的 value），米",
       "normal": "list 可选，任意方向的平面法向 [nx,ny,nz]",
       "keep": "str 'above' 保留 n·p>=value 一侧 / 'below' 保留另一侧",
       "cap": "bool 是否封口（默认 true）"},
      pre="实体是闭合网格；平面与该实体确有相交",
      degrade="切口环不是闭合环时（有悬边）会在 dangling 里报出来，此时封口会漏，"
              "应换切割位置或先检查网格；环退化时封口退回扇形三角化并在 method 里标注",
      returns="{sid, kept_side, stats{kept_whole,dropped_whole,split,n_in}, cut{rings,area,loops},"
              " health_before, health_after, n_tri}",
      unit="m")
def t_cut_plane(bus, sid, axis=2, value=0.0, normal=None, keep="above", cap=True):
    S = _get(bus, sid)
    n, d = _plane(axis, value, normal)
    side = 1 if str(keep).lower() in ("above", ">=", "high", "upper", "1") else -1
    h0 = mesh_report(S.P, S.F)
    P2, F2, segs, stats = _clip_halfspace(S.P, S.F, n, d, side)
    rings, dangling = _rings(segs)
    cut_area = 0.0
    u, v = _basis(n)
    for r in rings:
        c = r.mean(0)
        p2 = np.stack([(r - c) @ u, (r - c) @ v], 1)
        cut_area += abs(0.5 * float(np.sum(p2[:, 0] * np.roll(p2[:, 1], -1)
                                          - np.roll(p2[:, 0], -1) * p2[:, 1])))
    cap_info = None
    if cap and len(rings):
        # 封口面要朝外：保留 above 时，远端是 -n 方向，盖子朝 -n
        tris, notes = _cap(rings, n, d, want_flip=(side > 0))
        if tris:
            base = len(P2)
            capP = np.vstack(tris)                                  # (n_cap_tri, 3, 3)
            F2 = np.vstack([F2, np.arange(base, base + len(capP)).reshape(-1, 3)])
            P2 = np.vstack([P2, capP.reshape(-1, 3)])
            P2, F2 = _welded(P2, F2)
        cap_info = {"loops": notes, "n_cap_tri": int(len(tris))}
    S = _replace(bus, S, P2, F2, "cut")
    h1 = mesh_report(S.P, S.F)
    return {"sid": sid, "kept_side": "above" if side > 0 else "below",
            "plane": {"normal": _f(n), "d": round(d, 9)},
            "stats": stats,
            "cut": {"n_loops": len(rings), "area": round(cut_area, 9),
                    "dangling": len(dangling), "cap": cap_info},
            "health_before": h0, "health_after": h1, "n_tri": int(len(S.F))}


@tool("mesh_health",
      "网格体检：是否封闭、破洞个数与边数、重复面、退化面、朝向一致性、欧拉特征、体积与表面积。"
      "用于检查编辑后的结果，也可对原始模型做基线。",
      {"sids": "list 要检查的实体（或单个 sid）"},
      pre="实体已加载",
      degrade="只统计三角形网格；多面片 n-gon 已在读入时三角化",
      returns="{rows:[{sid, ...}]}",
      unit="m / m² / m³")
def t_mesh_health(bus, sids):
    rows = []
    for S in _sids(bus, sids):
        r = mesh_report(S.P, S.F)
        r["sid"] = S.sid
        rows.append(r)
    return {"rows": rows, "n": len(rows)}


@tool("emit",
      "把当前场景写到磁盘，形成可再次读入的模型文件。这是编辑闭环的关键一步："
      "编辑只有落盘、再用同一个 load 读回、用只读工具重新量出的数字才算成立。",
      {"path": "str 输出路径（.gltf 或 .obj）",
       "sids": "list 可选，只写这些实体；省略则写整个场景",
       "fmt": "str 'gltf' 或 'obj'（默认按扩展名）"},
      pre="路径目录可写",
      degrade="gltf 单文件内嵌 base64 buffer，不产生外部 .bin；obj 不携带单位，"
              "读回时会走对角线先验，故默认用 gltf",
      returns="{path, format, n_solids, order[原 sid 顺序], bytes, unit}",
      unit="m")
def t_emit(bus, path, sids=None, fmt=None):
    solids = _sids(bus, sids) if sids else list(bus.scene)
    ext = os.path.splitext(path)[1].lower()
    fmt = (fmt or ("gltf" if ext in (".gltf", ".glb") else "obj")).lower()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if fmt == "obj":
        _write_obj(path, solids)
        unit = None
    else:
        _write_gltf(path, solids, unit="metre")
        unit = "metre"
    return {"path": os.path.abspath(path), "format": fmt, "n_solids": len(solids),
            "order": [S.sid for S in solids], "bytes": os.path.getsize(path),
            "unit": unit,
            "note": "读回后 sid 按 order 顺序重编，前缀随格式：gltf→G000, obj→O000；"
                    "第 k 个对应 order[k]"}


# ================================================================ 落盘


def _write_obj(path, solids):
    lines, off = [], 0
    for S in solids:
        lines.append("o %s" % S.sid)
        lines.extend("v %.9f %.9f %.9f" % tuple(p) for p in S.P)
        lines.extend("f %d %d %d" % (a + 1 + off, b + 1 + off, c + 1 + off) for a, b, c in S.F)
        off += len(S.P)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# AIGeoLab sandbox · 世界坐标三角网格 · 单位 m\n")
        f.write("\n".join(lines) + "\n")


def _write_gltf(path, solids, unit="metre"):
    """写 glTF 2.0 单文件（buffer 用 base64 data URI）。

    选 glTF 而不是只写 OBJ 的原因：glTF 的规范单位是米，且允许在 asset.extras 里
    显式声明。读回时据此跳过尺度推断 —— 否则一次「加宽 1.2 m」的编辑可能因为
    对角线先验被读成毫米，闭环就假了。
    """
    import base64
    import struct

    blob = bytearray()
    views, accessors, meshes, nodes = [], [], [], []

    def _add(data, atype, comp_type, target=None, minmax=False):
        nonlocal blob
        while len(blob) % 4:
            blob.append(0)
        off = len(blob)
        raw = data.tobytes()
        blob.extend(raw)
        bv = {"buffer": 0, "byteOffset": off, "byteLength": len(raw)}
        if target:
            bv["target"] = target
        views.append(bv)
        acc = {"bufferView": len(views) - 1, "componentType": comp_type,
               "count": int(data.shape[0]), "type": atype}
        if minmax:
            acc["min"] = [float(x) for x in data.min(0)]
            acc["max"] = [float(x) for x in data.max(0)]
        accessors.append(acc)
        return len(accessors) - 1

    for S in solids:
        P = np.asarray(S.P, np.float32)
        # 索引必须先展平再写：accessor.count 对 SCALAR 是「元素个数」，
        # 直接传 (n_tri,3) 会让 count 写成 n_tri，只落进 1/3 的索引 ——
        # 读回来是 1/3 个壳，几何不报错、体积正好是三分之一。踩过这个坑。
        F = np.asarray(S.F, np.uint32).reshape(-1)
        a_pos = _add(P, "VEC3", 5126, 34962, minmax=True)
        a_idx = _add(F, "SCALAR", 5125, 34963)
        meshes.append({"name": S.sid, "primitives": [
            {"attributes": {"POSITION": a_pos}, "indices": a_idx, "mode": 4, "material": 0}]})
        nodes.append({"name": S.sid, "mesh": len(meshes) - 1})

    g = {
        "asset": {"version": "2.0", "generator": "AIGeoLab sandbox",
                  "extras": {"unit": unit, "axis": "Z-up", "frame": "world"}},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes, "meshes": meshes,
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [0.72, 0.76, 0.80, 1.0],
                                                "metallicFactor": 0.0, "roughnessFactor": 0.85}}],
        "accessors": accessors, "bufferViews": views,
        "buffers": [{"byteLength": len(blob),
                     "uri": "data:application/octet-stream;base64," + base64.b64encode(bytes(blob)).decode()}],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(g, f)


# ================================================================ 开口重建


def _ring_angles(seg, w, h, cu=0.0, cv=0.0):
    """环带采样的角度序列：按矩形四角把一周分成四段，段内均匀、端点恰为角点。

    纯等角度会把四个角斜切掉（角点不在序列里），而切掉的角是实打实的体积 ——
    实测 6.4×3 的板因此少算 0.041 m³。角点角度必须**从圆心看**，不能用板中心：
    圆心偏移 0.1 m 时两者不同，用错就切错角。
    """
    if seg % 4:
        seg = max(4, (seg // 4) * 4)
    per = max(2, seg // 4)
    cw = np.array([[w, h], [-w, h], [-w, -h], [w, -h]], float)
    corners = np.sort(np.mod(np.arctan2(cw[:, 1] - cv, cw[:, 0] - cu), 2 * np.pi))
    angs = []
    for i in range(4):
        a0 = float(corners[i])
        a1 = float(corners[(i + 1) % 4]) + (2 * np.pi if i == 3 else 0.0)
        for m in range(per):
            angs.append(a0 + (a1 - a0) * m / per)
    return np.array(angs), per * 4


def _unit_poly_area(angs):
    """单位角度序列围成的多边形面积（r=1）。用来把目标面积精确换回半径。"""
    p = np.stack([np.cos(angs), np.sin(angs)], 1)
    q = np.roll(p, -1, axis=0)
    return 0.5 * abs(float(np.sum(p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1])))


def _ray_to_rect(cu, cv, dx, dy, w, h):
    """从圆心 (cu,cv) 沿方向 (dx,dy) 射到矩形边界（±w, ±h）的距离。

    这里必须从**圆心**出发求交点。用 w/|dx| 这种「从板中心出发」的写法，
    圆心一偏移，外点就会整体平移——实测整块墙被搬走 0.1 m，z 从 0..3 变成 0.1..3.1。
    """
    ts = []
    if abs(dx) > 1e-12:
        for bx in (w, -w):
            t = (bx - cu) / dx
            if t > 0:
                ts.append(t)
    if abs(dy) > 1e-12:
        for by in (h, -h):
            t = (by - cv) / dy
            if t > 0:
                ts.append(t)
    return min(ts) if ts else 0.0


def _disc_ring_tris(angs, r, w, h, cu=0.0, cv=0.0):
    """把「矩形外框 − 同心圆」这一个环带三角化：相邻两条射线的内外点围成四边形。"""
    inner, outer = [], []
    for a in angs:
        d = np.array([np.cos(a), np.sin(a)])
        t = _ray_to_rect(cu, cv, d[0], d[1], w, h)
        inner.append(np.array([cu, cv]) + d * r)
        outer.append(np.array([cu, cv]) + d * t)
    n = len(inner)
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris.append((inner[i], outer[i], outer[j]))
        tris.append((inner[i], outer[j], inner[j]))
    return np.array(inner), np.array(outer), tris


def _tri_area_2d(a, b, c):
    return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


@tool("reopen_round",
      "把板上的洞口改成圆形（同面积），并把板、孔壁整体重建为闭合网格。"
      "用于「方窗改圆窗」：不挪动几何、不缩放板，只重建开口。",
      {"sid": "str 带洞口的板（轴对齐，单一厚度方向）",
       "center": "list [x,y,z] 新洞口的中心（只取板平面内的两个分量）",
       "area": "float 目标开口面积 m²（圆半径由它反算）",
       "segments": "int 圆的边数（默认 48，越大越接近理想圆）"},
      pre="板是轴对齐的六面体，且只有一个洞口；圆必须完全落在板面内",
      degrade="圆超出板面时直接报错，不做裁剪 —— 静默裁剪会得到一个看起来对、"
              "但开口面积不对的结果；非轴对齐的板不适用，应先用 stretch/布尔路径",
      returns="{sid, opening{area, diameter, center}, health_before, health_after, n_tri}",
      unit="m / m²")
def t_reopen_round(bus, sid, center, area, segments=48):
    S = _get(bus, sid)
    lo, hi = S.bbox()
    d = hi - lo
    k = int(np.argmin(d))
    rest = [i for i in range(3) if i != k]
    ku, kv = rest
    c = np.asarray(center, float).reshape(3)
    half_w = float(d[ku]) / 2.0
    half_h = float(d[kv]) / 2.0
    cu = float(c[ku] - (lo[ku] + hi[ku]) / 2.0)     # 圆心相对板中心的偏移
    cv = float(c[kv] - (lo[kv] + hi[kv]) / 2.0)
    # 半径由「实际采样点的单位多边形面积」反算，而不是 πr² 或正 n 边形公式。
    # 题目约束的是开口面积；用 r=sqrt(A/π) 时内接多边形只有 A 的 99.7%，
    # 而这 0.3% 恰好是本题要报的数，不能靠「近似」糊过去。
    angs, seg0 = _ring_angles(max(8, int(segments)), half_w, half_h, cu, cv)
    r = math.sqrt(float(area) / _unit_poly_area(angs))

    # 圆必须整个落在板面内，且离板边留出余量，否则环带三角化会自交
    if abs(cu) + r > half_w - 1e-6 or abs(cv) + r > half_h - 1e-6:
        raise ValueError("圆超出板面（或贴边）：圆心偏移 (%.4f, %.4f)，半径 %.4f，板半宽 %.4f/%.4f"
                         % (cu, cv, r, half_w, half_h))

    seg = seg0
    inner, outer, ring = _disc_ring_tris(angs, r, half_w, half_h, cu, cv)
    ring = [t for t in ring if _tri_area_2d(*t) > 1e-12]

    k0, k1 = float(lo[k]), float(hi[k])
    P, F = [], []
    # 两层顶点：先把内圆与环带外点各放两组（k0 / k1），再补矩形四角
    def add(pt2, kk):
        # inner/outer 已经是「以板中心为原点、含圆心偏移」的坐标，这里只补板中心
        p = np.zeros(3)
        p[ku] = pt2[0] + (lo[ku] + hi[ku]) / 2
        p[kv] = pt2[1] + (lo[kv] + hi[kv]) / 2
        p[k] = kk
        P.append(p)
        return len(P) - 1

    # 环带外点已落在矩形边界上，无需再补角点
    idx = {}
    for kk, tag in ((k0, "a"), (k1, "b")):
        for i in range(seg):
            idx[(tag, "in", i)] = add(inner[i], kk)
            idx[(tag, "out", i)] = add(outer[i], kk)

    def tri(i, j, l):
        F.append((i, j, l))

    # 两个大面：k1 侧朝 +k，k0 侧朝 -k
    for i in range(seg):
        j = (i + 1) % seg
        tri(idx[("b", "in", i)], idx[("b", "out", i)], idx[("b", "out", j)])
        tri(idx[("b", "in", i)], idx[("b", "out", j)], idx[("b", "in", j)])
        tri(idx[("a", "in", i)], idx[("a", "out", j)], idx[("a", "out", i)])
        tri(idx[("a", "in", i)], idx[("a", "in", j)], idx[("a", "out", j)])
    # 孔壁：朝圆心
    for i in range(seg):
        j = (i + 1) % seg
        tri(idx[("a", "in", i)], idx[("b", "in", i)], idx[("b", "in", j)])
        tri(idx[("a", "in", i)], idx[("b", "in", j)], idx[("a", "in", j)])
    # 侧面：沿矩形四条边，把相邻外点按 k 方向连成带（共线三角形会被面积过滤掉）
    for i in range(seg):
        j = (i + 1) % seg
        quad = (idx[("a", "out", i)], idx[("a", "out", j)],
                idx[("b", "out", j)], idx[("b", "out", i)])
        tri(quad[0], quad[1], quad[2])
        tri(quad[0], quad[2], quad[3])

    P = np.array(P, float)
    F = np.array(F, int)
    h0 = mesh_report(S.P, S.F)
    S = _replace(bus, S, P, F, "reopen")
    # 侧面会产生大量共线退化三角形（直边上多点共线），保留会让「退化面」计数失真。
    keep = []
    A, B, C = S.P[S.F[:, 0]], S.P[S.F[:, 1]], S.P[S.F[:, 2]]
    ar = np.linalg.norm(np.cross(B - A, C - A), axis=1) / 2.0
    keep = S.F[ar > 1e-12]
    if len(keep) != len(S.F):
        S = _replace(bus, S, S.P, keep, "reopen_clean")
    h1 = mesh_report(S.P, S.F)
    return {"sid": sid,
            "opening": {"area": round(float(area), 9),
                        "diameter": round(2 * r * math.cos(math.pi / seg), 9),
                        "circumscribed_diameter": round(2 * r, 9),
                        "center": [round(float(c[0]), 9), round(float(c[1]), 9), round(float(c[2]), 9)],
                        "radius": round(r, 9), "segments": seg,
                        "axis_in_plane": ["xyz"[ku], "xyz"[kv]], "axis_thin": "xyz"[k]},
            "plate": {"bbox": [_f(lo), _f(hi)], "size": [round(float(x), 9) for x in d]},
            "health_before": h0, "health_after": h1, "n_tri": int(len(S.F))}
