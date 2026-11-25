# -*- coding: utf-8 -*-
"""shape.py — 形状分解与配准。

这一层解决的是「一个实体里到底有几件东西」和「这两组东西怎么对上去」：

  * `parts`  —— 按网格连通性拆分量。一个 IFC 实体里塞了箭体和三只插脚时，
                 连通性拆不出来（它们焊在一起），但「箭体 + 三个独立凸台」这种情况能拆。
  * `pockets` —— 沿某个轴找「陷进去的平底坑」：法向朝外、高度低于最高面、有封闭轮廓。
                 它不认识「凹槽」这个词，只认识「朝 +z 的、不在顶面的平面补丁」。
  * `regid2d` —— 给若干对对应点，解最优 2D 刚体变换（绕轴转角 + 平移）。
                 闭式解，不用迭代；残差一起返回，残差大就说明配错了对。

三个工具都不含任务知识：`pockets` 不知道自己在找火箭发射台的凹槽，
`regid2d` 也不知道自己在对接插脚 —— 调用方给什么点，它算什么变换。
"""
import math

import numpy as np

from core.runtime import tool
from tools import brep
from tools import geom
from tools.world import Solid

TOL = 1e-6


def _solid_of(bus, sid):
    for s in bus.scene:
        if s.sid == sid:
            return s
    raise KeyError(f"场景里没有 {sid}")


def _axis_index(axis):
    if isinstance(axis, str):
        m = {"x": 0, "y": 1, "z": 2}
        if axis.lower() not in m:
            raise ValueError(f"轴只能取 x/y/z，收到 {axis!r}")
        return m[axis.lower()]
    v = np.asarray(axis, float).reshape(3)
    return int(np.argmax(np.abs(v)))


# ================================================================ 连通分量


def _components(P, F, tol=1e-6):
    """按共享顶点把三角形分成连通块。返回 [[tri indices], ...]。

    用并查集而不是 BFS：一个 1278 面的实体上，BFS 要为每个顶点建邻接表，
    并查集只要一遍扫描全部顶点键。
    """
    F = np.asarray(F, int)
    key = np.round(np.asarray(P, float) / tol).astype(np.int64)
    parent = list(range(len(F)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    first = {}
    for i, tri in enumerate(F):
        for v in tri:
            k = tuple(key[v])
            if k in first:
                union(i, first[k])
            else:
                first[k] = i
    groups = {}
    for i in range(len(F)):
        groups.setdefault(find(i), []).append(i)
    # 大的排前面：通常「主体」是最大那块，后面的小分量才是附件
    return sorted(groups.values(), key=len, reverse=True)


@tool("parts",
      "按网格连通性把一个实体拆成若干独立实体，每个分量成为场景里的新实体并返回几何指纹。"
      "用于「一个大实体里含多个部件」的情形（例如箭体与三只插脚）。",
      {"sid": "str  要拆的实体", "min_tri": "int  小于这个面数的分量丢弃（默认 2，滤掉碎片）",
       "keep_original": "bool  是否保留原实体（默认 false，拆完移除，避免重复计数）"},
      pre="实体存在",
      degrade="焊成一体的部件（共享顶点）拆不出来；此时只能靠几何特征区分，不能靠连通性",
      returns="{parent, parts:[describe...], n}",
      unit="m")
def t_parts(bus, sid, min_tri=2, keep_original=False):
    S = _solid_of(bus, sid)
    comps = _components(S.P, S.F)
    out = []
    k = 0
    for idx in comps:
        if len(idx) < min_tri:
            continue
        idx = np.asarray(idx, int)
        verts = np.unique(S.F[idx])
        remap = {int(v): i for i, v in enumerate(verts)}
        P = np.asarray(S.P, float)[verts]
        F = np.array([[remap[int(v)] for v in tri] for tri in S.F[idx]], int)
        k += 1
        name = f"{getattr(S, 'name', sid)}#{k}"
        sub = Solid(f"{sid}.{k}", name, P, F, S.src or "?")
        bus.scene.append(sub)
        d = geom.describe(sub)
        d["parent"] = sid
        d["share"] = round(len(idx) / max(len(S.F), 1), 6)
        out.append(d)
    if not keep_original:
        bus.scene = [s for s in bus.scene if s.sid != sid]
    out.sort(key=lambda d: -d["ntri"])
    return {"parent": sid, "parts": out, "n": len(out), "kept_original": bool(keep_original)}


# ================================================================ 轴向凹坑


def _hull2(pts):
    """Andrew monotone chain。返回逆时针凸包顶点（索引）。"""
    pts = np.asarray(pts, float)
    n = len(pts)
    if n <= 2:
        return list(range(n))
    order = sorted(range(n), key=lambda i: (pts[i, 0], pts[i, 1]))

    def cross(o, a, b):
        return (pts[a, 0] - pts[o, 0]) * (pts[b, 1] - pts[o, 1]) - \
               (pts[a, 1] - pts[o, 1]) * (pts[b, 0] - pts[o, 0])

    lower = []
    for i in order:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], i) <= 1e-12:
            lower.pop()
        lower.append(i)
    upper = []
    for i in reversed(order):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], i) <= 1e-12:
            upper.pop()
        upper.append(i)
    return lower[:-1] + upper[:-1]


def _poly_area2(Q):
    if len(Q) < 3:
        return 0.0
    x, y = Q[:, 0], Q[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _poly_centroid(Q):
    """多边形面积质心（不是顶点平均）。

    这个区别在本题上很实在：切出来的轮廓经常在直边中间多出一个共线点
    （网格三角化的副产物），顶点平均会被它拽偏 —— 平行四边形的 8 点环顶点平均
    偏离真中心 2 cm，而面积质心对共线点完全不敏感。
    """
    Q = np.asarray(Q, float)
    if len(Q) < 3:
        return Q.mean(0) if len(Q) else np.zeros(2)
    x, y = Q[:, 0], Q[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cr = x * yn - xn * y
    A = cr.sum() / 2.0
    if abs(A) < 1e-15:
        return Q.mean(0)
    return np.array([((x + xn) * cr).sum() / (6 * A), ((y + yn) * cr).sum() / (6 * A)])


@tool("pockets",
      "沿指定轴找实体上的「平底凹陷」：法向朝 +轴、高度低于最高面、有封闭轮廓的平面补丁。"
      "返回每个凹坑的轮廓多边形、尺寸、面积、深度与中心。用于识别平台上的凹槽这类特征。",
      {"sid": "str  实体", "axis": "str  轴（默认 'z'，即找顶面往下的坑）",
       "min_area": "float  小于此面积忽略（默认 1e-4 m²）"},
      pre="实体闭合且闭合面朝外",
      degrade="斜面坑、锥形坑不是平面补丁，此工具只认平底；遇到斜底会返回空并说明",
      returns="{axis, top, pockets:[{center, depth, area, size, polygon, n_tri}], n}",
      unit="m")
def t_pockets(bus, sid, axis="z", min_area=1e-4):
    S = _solid_of(bus, sid)
    ai = _axis_index(axis)
    bi, ci = [i for i in (0, 1, 2) if i != ai]
    P = np.asarray(S.P, float)
    F = np.asarray(S.F, int)
    V = P[F]
    # 三角形法向（未归一）与面积
    cr = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0])
    mag = np.linalg.norm(cr, axis=1)
    up = mag > 1e-14
    nz = np.zeros_like(cr)
    nz[up] = cr[up] / mag[up][:, None]
    h = V[:, :, ai].mean(1)
    top = float(P[:, ai].max())

    # 朝 +轴 且水平的三角形
    sel = up & (np.abs(nz[:, ai] - 1.0) < 1e-6)
    if not sel.any():
        return {"axis": axis, "top": round(top, 6), "pockets": [], "n": 0,
                "note": "没有朝 +%s 的平面补丁" % axis}
    # 按高度分组（同一坑底的三角形高度相同）
    hs = np.round(h[sel], 6)
    groups = {}
    for t, hh in zip(np.nonzero(sel)[0], hs):
        groups.setdefault(float(hh), []).append(int(t))

    out = []
    for hh, tris in sorted(groups.items()):
        if hh > top - 1e-5:
            continue                      # 顶面本身，不是坑
        # 同一高度可能有好几个互不相连的坑。这里按「三角形是否共享顶点」分簇，
        # 而不是按距离阈值：坑底的 4 个角彼此相距 0.2~0.3 m，用距离聚类会把角点各自
        # 分成一簇（坑的尺寸和坑间距是同一量级，不存在放之四海皆准的阈值）。
        for grp in _by_vertex(tris, F):
            vs = np.unique(F[grp])
            Q = np.unique(np.round(P[vs][:, [bi, ci]], 9), axis=0)
            if len(Q) < 3:
                continue
            hull = Q[_hull2(Q)]
            A = abs(_poly_area2(hull))
            if A < min_area:
                continue
            size = [round(float(hull[:, 0].max() - hull[:, 0].min()), 6),
                    round(float(hull[:, 1].max() - hull[:, 1].min()), 6)]
            # 边长比 AABB 更有用：多边形坑的「长×宽」要靠边长，AABB 会把斜放的矩形撑大
            sides = [round(float(np.linalg.norm(hull[(i + 1) % len(hull)] - hull[i])), 6)
                     for i in range(len(hull))]
            ctr = _poly_centroid(hull)
            c = [0.0, 0.0, 0.0]
            c[ai] = round(hh, 6)
            c[bi] = round(float(ctr[0]), 6)
            c[ci] = round(float(ctr[1]), 6)
            out.append({
                "center": c,
                "depth": round(top - hh, 6),
                "area": round(A, 6),
                "size": size,
                "sides": sides,
                "polygon": [[round(float(x), 6), round(float(y), 6)] for x, y in hull],
                "n_vert": int(len(Q)),
                "n_tri": len(grp),
                "bottom": round(hh, 6),
            })
    out.sort(key=lambda d: -d["area"])
    return {"axis": axis, "top": round(top, 6), "pockets": out, "n": len(out),
            "size_axes": ["xyz"[bi], "xyz"[ci]]}


def _by_vertex(tris, F):
    """把一组三角形按「共享顶点」聚成连通簇（共享即同簇）。

    比距离阈值可靠：相邻三角形一定共享顶点，不相邻的一定不共享（除非模型本身就焊在一起）。
    """
    tris = [int(t) for t in tris]
    par = {t: t for t in tris}

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    own = {}
    for t in tris:
        for v in F[t]:
            v = int(v)
            if v in own:
                ra, rb = find(own[v]), find(t)
                if ra != rb:
                    par[rb] = ra
            else:
                own[v] = t
    g = {}
    for t in tris:
        g.setdefault(find(t), []).append(t)
    return sorted(g.values(), key=len, reverse=True)


# ================================================================ 轴向截面轮廓


def _qkey(v, tol=1e-7):
    return tuple(np.round(np.asarray(v, float) / tol).astype(np.int64).tolist())


def _ring_area_2d(Q):
    return abs(_poly_area2(Q))


def _simplify(Q, tol=1e-6):
    """去掉直边上的多余拐点。tol 是相邻两段夹角的**正弦**上限。

    为什么要做这一步：截面环是被三角化过的，一条直边常常被切成两段，
    于是「4 条边的平行四边形」交出来是 8 条边。不化简的话，拼接关系就只能靠
    边的多重集去猜，而 0.05 这种被切出来的半截边会污染匹配。
    """
    Q = np.asarray(Q, float)
    n = len(Q)
    if n <= 3:
        return Q
    keep = []
    for i in range(n):
        v1 = Q[i] - Q[i - 1]
        v2 = Q[(i + 1) % n] - Q[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-12 or n2 < 1e-12:
            continue
        cr = abs(v1[0] * v2[1] - v1[1] * v2[0]) / (n1 * n2)
        if cr > tol:
            keep.append(i)
    if len(keep) < 3:
        return Q
    return Q[keep]


def _thin_ends(Sq, h):
    """轮廓最短的两条边的中点（补上高度分量）。用于取薄板构件的两个端面。"""
    n = len(Sq)
    if n < 3:
        return []
    L = [(float(np.linalg.norm(Sq[(i + 1) % n] - Sq[i])), i) for i in range(n)]
    L.sort()
    out = []
    for ln, i in L[:2]:
        m = (Sq[i] + Sq[(i + 1) % n]) / 2.0
        out.append({"edge_len": round(ln, 6),
                    "mid": [round(float(m[0]), 6), round(float(m[1]), 6), round(float(h), 6)]})
    return out


@tool("levels",
      "列出实体沿某轴的「顶点高度层」：每层给高度与该层的顶点数。用于找台阶、底面、顶面、"
      "部件分界这些特征平面 —— 顶点数突然变化的地方就是几何在这里换了截面。",
      {"sid": "str  实体", "axis": "str  轴（默认 'z'）", "min_vert": "int  少于这个顶点数的层忽略（默认 1）"},
      pre="实体存在",
      degrade="只统计顶点，不保证该高度是一个完整平面；要确认平面请配合 profile 看截面",
      returns="{axis, levels:[{h, n_vert, share}], n}",
      unit="m")
def t_levels(bus, sid, axis="z", min_vert=1):
    S = _solid_of(bus, sid)
    ai = _axis_index(axis)
    v = np.asarray(S.P, float)[:, ai]
    key = np.round(v / TOL).astype(np.int64)
    uk, cnt = np.unique(key, return_counts=True)
    out = []
    for k, c in zip(uk, cnt):
        if c < min_vert:
            continue
        out.append({"h": round(float(k) * TOL, 6), "n_vert": int(c),
                    "share": round(float(c) / max(len(v), 1), 6)})
    return {"axis": axis, "levels": out, "n": len(out)}


@tool("profile",
      "沿指定轴在某高度切一个实体，返回截面的**全部闭合轮廓环**：面积、周长、中心、边长、"
      "多边形顶点，并标出哪些环被别的环包住（内环＝孔洞）。",
      {"sid": "str  实体", "axis": "str  轴（默认 'z'）", "height": "float  切面位置",
       "min_area": "float  小于此面积忽略（默认 1e-6 m²）"},
      pre="实体闭合；切面与该实体有交",
      degrade="切面恰好穿过顶点或与某个面共面时，该处轮廓会断开，dangling 字段会暴露",
      returns="{axis, height, rings:[{area, perimeter, center, sides, polygon, inner}], n, dangling}",
      unit="m")
def t_profile(bus, sid, axis="z", height=0.0, min_area=1e-6):
    from tools import geom                    # 接环与质心只有一份实现，在 geom 里
    S = _solid_of(bus, sid)
    ai = _axis_index(axis)
    bi, ci = [i for i in (0, 1, 2) if i != ai]
    rings, dangling, probe_h = geom.section_rings(S, ai, height, min_area=min_area)
    out = []
    for r in rings:
        Q = r["polygon"]
        A = r["area"]
        per_len = float(np.linalg.norm(np.diff(np.vstack([Q, Q[:1]]), axis=0), axis=1).sum())
        ctr = geom.poly_centroid_2d(Q)
        c = [0.0, 0.0, 0.0]
        c[ai] = round(float(height), 6)
        c[bi] = round(float(ctr[0]), 6)
        c[ci] = round(float(ctr[1]), 6)
        Sq = _simplify(Q)
        out.append({
            "area": round(A, 6),
            "perimeter": round(per_len, 6),
            "center": c,
            "sides": [round(float(np.linalg.norm(Q[(i + 1) % len(Q)] - Q[i])), 6)
                      for i in range(len(Q))],
            "polygon": [[round(float(x), 6), round(float(y), 6)] for x, y in Q],
            "n_vert": int(len(Q)),
            "inner": bool(r["inner"]),
            "depth": int(r["depth"]),
            # 化简后的轮廓：直边上的三角化副产物已去掉，边数与边长反映真实形状
            "simple": {
                "n_vert": int(len(Sq)),
                "sides": [round(float(np.linalg.norm(Sq[(i + 1) % len(Sq)] - Sq[i])), 6)
                          for i in range(len(Sq))],
                "polygon": [[round(float(x), 6), round(float(y), 6)] for x, y in Sq],
                # 轮廓最短的两条边的中点。薄板类构件（门扇、玻璃、画板）的两个「端面」
                # 就是这两条边 —— 旋转轴、贴合边这类量都从这里取，不用再猜。
                "thin_ends": _thin_ends(Sq, c[ai]),
            },
        })
    out.sort(key=lambda r: -r["area"])
    return {"axis": axis, "height": round(float(height), 6), "probe_height": round(probe_h, 9),
            "rings": out, "n": len(out), "n_inner": sum(1 for r in out if r["inner"]),
            "dangling": len(dangling),
            "size_axes": ["xyz"[bi], "xyz"[ci]]}


def _pt_in_poly(p, Q):
    """射线法：从 p 向 +u 打一条水平线，与多边形边界的交点数奇偶。"""
    x, y = p
    inside = False
    n = len(Q)
    for i in range(n):
        x1, y1 = Q[i]
        x2, y2 = Q[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if xi > x:
                inside = not inside
    return inside


# ================================================================ 2D 刚体配准


@tool("regid2d",
      "给若干对对应点，解最优 2D 刚体变换（绕指定轴旋转 + 平移）。返回转角、旋转中心、"
      "平移量、每对点的残差。残差大即说明对应关系配错了。",
      {"pairs": "数组  每项 {from:[x,y,z], to:[x,y,z]}，至少 2 对（3 对及以上才唯一确定）",
       "axis": "str  绕哪根轴转（默认 'z'）"},
      pre="点对已建立；只用垂直于 axis 的两个分量解转角",
      degrade="只有 1 对点时无法定转角，返回 angle=0 并说明；共线点对同样无解",
      returns="{angle_deg, turn, pivot, translate, translate_plane, direction_unit, shift_xy_length, residual, rms, max, rotation_axis}",
      unit="m / deg")
def t_regid2d(bus, pairs, axis="z"):
    if isinstance(pairs, dict):
        pairs = pairs.get("pairs") or []
    ai = _axis_index(axis)
    bi, ci = [i for i in (0, 1, 2) if i != ai]
    src, dst = [], []
    for pr in pairs:
        a = pr.get("from") if isinstance(pr, dict) else pr[0]
        b = pr.get("to") if isinstance(pr, dict) else pr[1]
        src.append(np.asarray(a, float).reshape(3))
        dst.append(np.asarray(b, float).reshape(3))
    if len(src) < 2:
        raise ValueError("至少需要 2 对点；1 对点只能定平移，定不了转角")
    S = np.array(src)
    D = np.array(dst)
    cs = S.mean(0)
    cd = D.mean(0)
    p = S[:, [bi, ci]] - cs[[bi, ci]]
    q = D[:, [bi, ci]] - cd[[bi, ci]]
    # 闭式最小二乘：theta = atan2(sum(p × q), sum(p · q))
    cross = float((p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]).sum())
    dot = float((p * q).sum())
    if abs(cross) < 1e-14 and abs(dot) < 1e-14:
        raise ValueError("点对退化（重合或全同），解不出转角")
    th = math.atan2(cross, dot)
    c, s = math.cos(th), math.sin(th)
    # 验算：把 src 绕质心转 th，再平移到 dst 质心
    pr = np.stack([p[:, 0] * c - p[:, 1] * s, p[:, 0] * s + p[:, 1] * c], 1)
    res = np.linalg.norm(pr - q, axis=1)
    tr = cd - cs
    # 平面内那一份平移。三维的 translate 里混着沿轴的分量（质心的轴向高差），
    # 那个分量不是「水平移动量」—— 直接拿去做水平平移会把物体整体压下去。
    tr_plane = tr.copy()
    tr_plane[ai] = 0.0
    return {
        "angle_deg": round(math.degrees(th), 6),
        "rotation_axis": list(np.eye(3)[ai]),
        "pivot": [round(float(x), 6) for x in cs],
        "translate": [round(float(x), 6) for x in tr],
        # 「水平移动」只该用这一份：沿轴分量恒为 0，正是绕竖直轴调整朝向之后
        # 在水平面内那一步位移。
        "translate_plane": [round(float(x), 6) for x in tr_plane],
        # 水平分量长度：只算垂直于旋转轴的平面内的平移长度。
        # 「水平移动距离」这个量必须由工具给，不能让人自己从向量里开平方 ——
        # 那样报出来的数就没有可追溯的出处了。
        "shift_xy_length": round(float(np.linalg.norm(tr[[bi, ci]])), 6),
        "shift_axis_len": round(float(tr[ai]), 6),
        # 水平移动的**单位向量**。题面要「方向」和「距离」两个量，方向给单位向量才能
        # 与距离分开写；让调用方自己拿向量除长度，除法就脱离了工具出处。
        "direction_unit": [round(float(x), 6) for x in
                           (tr_plane / np.linalg.norm(tr_plane)
                            if np.linalg.norm(tr_plane) > 1e-12 else np.zeros(3))],
        # 转向按右手定则说清楚：正角 = 绕轴的 +方向逆时针（从轴的正端看）。
        "turn": ("逆时针（右手定则，绕 rotation_axis 的正向）" if th > 1e-12 else
                 "顺时针（右手定则，绕 rotation_axis 的负向）" if th < -1e-12 else "不转"),
        "residual": [round(float(x), 6) for x in res],
        "rms": round(float(np.sqrt((res ** 2).mean())), 6),
        "max": round(float(res.max()), 6),
        "n_pairs": len(src),
        "note": "先绕 pivot 转 angle_deg，再平移 translate（两者都是世界坐标）。"
                "只做平面内的移动时用 translate_plane，它的沿轴分量恒为 0。",
    }
