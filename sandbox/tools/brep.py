# -*- coding: utf-8 -*-
"""brep.py — 从三角汤还原「可判读的面结构」：共面区域 + 边界环 + 内环（开口）。

只输入顶点与三角面；不读名称、类别、属性。
局限（必须写进报告）：锐边阈值是全局单一值，曲面片会被过度合并；
内环只覆盖「完全落在面内部」的开口，贴地面的门洞表现为边界缺口（notch），
需要配合 loop 顶点数与反折角判断，或改用 compare_with_wall 的贯通性证据。
"""
import math

import numpy as np

from core.runtime import tool


def _weld(P, F, tol=1e-5):
    key, idx = {}, []
    for q in P:
        k = tuple(np.round(q / tol).astype(np.int64))
        if k not in key:
            key[k] = len(key)
        idx.append(key[k])
    nP = np.zeros((len(key), 3))
    for k, i in key.items():
        nP[i] = np.array(k) * tol
    keep = [(a, b, c) for (a, b, c) in [(idx[a], idx[b], idx[c]) for a, b, c in F] if len({a, b, c}) == 3]
    return nP, keep


def _frames(P, F):
    A = P[[f[0] for f in F]]
    B = P[[f[1] for f in F]]
    C = P[[f[2] for f in F]]
    N = np.cross(B - A, C - A)
    ar = np.linalg.norm(N, axis=1) / 2
    ok = ar > 1e-12
    N[ok] /= (2 * ar[ok])[:, None]
    return N, ar


def _adj(F):
    em = {}
    for ti, (a, b, c) in enumerate(F):
        for u, v in ((a, b), (b, c), (c, a)):
            em.setdefault((min(u, v), max(u, v)), []).append(ti)
    return em


def _grow(F, N, em, cos_thr):
    n = len(F)
    lab = -np.ones(n, int)
    cur = 0
    for seed in range(n):
        if lab[seed] >= 0:
            continue
        lab[seed] = cur
        st = [seed]
        while st:
            ti = st.pop()
            a, b, c = F[ti]
            for u, v in ((a, b), (b, c), (c, a)):
                for tj in em[(min(u, v), max(u, v))]:
                    if lab[tj] < 0 and N[ti].dot(N[tj]) > cos_thr:
                        lab[tj] = cur
                        st.append(tj)
        cur += 1
    return lab, cur


def _plane(P, F, tris):
    A = P[[F[t][0] for t in tris]]
    B = P[[F[t][1] for t in tris]]
    C = P[[F[t][2] for t in tris]]
    nr = np.cross(B - A, C - A)
    ar = np.linalg.norm(nr, axis=1) / 2
    n = nr.sum(0)
    ln = np.linalg.norm(n)
    n = n / ln if ln > 1e-12 else np.array([0, 0, 1.0])
    c = ((A + B + C) / 3 * ar[:, None]).sum(0) / max(ar.sum(), 1e-18)
    dev = np.abs((np.vstack([A, B, C]) - c) @ n)
    return n, c, float(dev.max()), float(ar.sum())


def _loops(F, tris, P):
    di = {}
    for ti in tris:
        a, b, c = F[ti]
        for u, v in ((a, b), (b, c), (c, a)):
            di[(u, v)] = di.get((u, v), 0) + 1
    nxt = {}
    for (u, v), k in di.items():
        if di.get((v, u), 0) == 0:
            nxt.setdefault(u, []).append(v)
    loops, seen = [], set()
    for u0 in list(nxt):
        while nxt[u0]:
            v0 = nxt[u0].pop()
            if (u0, v0) in seen:
                continue
            loop, u, v = [u0], u0, v0
            seen.add((u, v))
            guard = 0
            while v != u0 and guard < 20000:
                loop.append(v)
                cand = [w for w in nxt.get(v, []) if (v, w) not in seen]
                if not cand:
                    break
                w = cand[0]
                seen.add((v, w))
                u, v = v, w
                guard += 1
            if v == u0 and len(loop) >= 3:
                loops.append(loop)
    loops.sort(key=lambda L: -_poly_area(P[L]))
    return loops


def _poly_area(Q):
    c = Q.mean(0)
    n = np.zeros(3)
    for i in range(len(Q)):
        n += np.cross(Q[i] - c, Q[(i + 1) % len(Q)] - c)
    return float(np.linalg.norm(n) / 2)


def _basis(n):
    a = np.array([1.0, 0, 0])
    if abs(n @ a) > 0.9:
        a = np.array([0, 1.0, 0])
    u = a - n * (a @ n)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def _reflex_count(Q, n):
    """外环上的反折顶点数：>0 说明边界带凹口（典型：贴地门洞）。"""
    u, v = _basis(n)
    pts = np.stack([Q @ u, Q @ v], 1)
    m = len(pts)
    k = 0
    for i in range(m):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % m]
        # 手写二维叉积：numpy 2.x 已移除对二维向量的 np.cross，
        # 而这里一旦抛异常，调用方看到的是「工具不可用」而不是「这个面没有凹口」。
        cr = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if cr < -1e-9:
            k += 1
    return k


def faces(S, sharp_deg=20.0, max_faces=12):
    P, F = _weld(S.P, S.F)
    if not F:
        return {"sid": S.sid, "faces": [], "note": "weld 后为空"}
    N, ar = _frames(P, F)
    em = _adj(F)
    lab, k = _grow(F, N, em, math.cos(math.radians(sharp_deg)))
    out = []
    for r in range(k):
        tris = np.nonzero(lab == r)[0].tolist()
        if len(tris) < 2 and k > 1:
            continue
        n, c, dev, area = _plane(P, F, tris)
        loops = _loops(F, tris, P)
        u, v = _basis(n)
        holes = []
        for L in loops[1:]:
            Q = P[L]
            uu, vv = Q @ u, Q @ v
            if abs(_poly_area(Q)) < 1e-6:
                continue
            holes.append({"area": round(abs(_poly_area(Q)), 6),
                          "w": round(float(uu.max() - uu.min()), 6),
                          "h": round(float(vv.max() - vv.min()), 6),
                          "center": [round(float(x), 6) for x in Q.mean(0)]})
        outer_area = abs(_poly_area(P[loops[0]])) if loops else 0.0
        out.append({
            "n": [round(float(x), 6) for x in n],
            "centroid": [round(float(x), 6) for x in c],
            "area": round(area, 6),
            "tri": len(tris),
            "dev": round(dev, 8),
            "planar": bool(dev < 1e-3),
            "n_loops": len(loops),
            "outer_loop_area": round(outer_area, 6),
            "outer_vertices": len(loops[0]) if loops else 0,
            "reflex": _reflex_count(P[loops[0]], n) if loops else 0,
            "holes": holes,
            # 缺口式开口：外环面积明显大于面本身，说明边界被切掉一块
            "notch_area": round(max(0.0, outer_area - area), 6),
        })
    out.sort(key=lambda f: -f["area"])
    return {"sid": S.sid, "sharp_deg": sharp_deg, "n_regions": k,
            "faces": out[:max_faces], "n_faces": len(out)}


# ------------------------------------------------------------ 洞口：面内的缺料

def _cells_covered(us, vs, tris_uv):
    """把「面内缺料」算出来，而不是靠文件里有没有洞口实体。

    做法：取实体顶点在面内的**所有不同坐标**当作网格线 —— 矩形洞口的边一定落在
    这些线上，所以这个网格是精确的，不是近似栅格。每个格子取中心点判「是否落在任一个
    三角面内」，落不进的就是缺料。再把相连的空格连成一块，就是一个个洞口。

    为什么不用包围盒相减：一个洞口贴着地板（门洞）时，墙面多边形是「带缺口的外环」，
    内环是空的（holes=[]），包围盒差也含别的角料。只有按面覆盖率算，门洞和窗洞才是同一套算法。
    """
    nu, nv = len(us) - 1, len(vs) - 1
    if nu <= 0 or nv <= 0:
        return np.zeros((0, 0), bool)
    cu = (us[:-1] + us[1:]) / 2.0
    cv = (vs[:-1] + vs[1:]) / 2.0
    U, V = np.meshgrid(cu, cv, indexing="ij")
    P = np.stack([U.ravel(), V.ravel()], 1)
    cov = np.zeros(len(P), bool)
    for a, b, c in tris_uv:
        v0, v1 = b - a, c - a
        den = v0[0] * v1[1] - v0[1] * v1[0]
        if abs(den) < 1e-15:
            continue
        w = P - a
        s = (w[:, 0] * v1[1] - w[:, 1] * v1[0]) / den
        t = (v0[0] * w[:, 1] - v0[1] * w[:, 0]) / den
        cov |= (s >= -1e-9) & (t >= -1e-9) & (s + t <= 1 + 1e-9)
    return cov.reshape(nu, nv)


def _components(mask):
    """四邻接连通块。返回每块的 (i0,i1,j0,j1) 闭区间。"""
    nu, nv = mask.shape
    seen = np.zeros_like(mask, bool)
    out = []
    for i in range(nu):
        for j in range(nv):
            if not mask[i, j] or seen[i, j]:
                continue
            stack = [(i, j)]
            seen[i, j] = True
            cells = []
            while stack:
                a, b = stack.pop()
                cells.append((a, b))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    x, y = a + da, b + db
                    if 0 <= x < nu and 0 <= y < nv and mask[x, y] and not seen[x, y]:
                        seen[x, y] = True
                        stack.append((x, y))
            ii = [c[0] for c in cells]
            jj = [c[1] for c in cells]
            out.append((min(ii), max(ii), min(jj), max(jj)))
    return out


@tool("openings",
      "找出一块板状构件（墙、板、盖）**面内的缺料**：每个洞口给出中心、宽、高、下沿标高、"
      "面积，以及下沿是否贴到构件底边（贴底＝门洞式的通底开口）。不依赖文件里有没有洞口实体，"
      "通底缺口与内环孔洞用同一套算法。",
      {"sid": "str  构件（通常是一面墙）",
       "min_area": "float  小于这个面积的缺料忽略（默认 1e-3 m²）"},
      pre="构件是板状（最短边明显小于另两边），且至少有一个平面主导面",
      degrade="曲面构件（壳、圆柱面）没有定义良好的面内缺料，会直接报错；"
              "构件被拆成多块实体时，先 merge 再查",
      returns="{sid, thickness, axes, wall, openings:[{center, width, height, area, sill_z, "
              "grounded, touches_top, rect, corners, prism}], n, resolution}",
      unit="m / m²")
def t_openings(bus, sid, min_area=1e-3):
    from tools.shape import _solid_of
    S = _solid_of(bus, sid)
    lo, hi = S.bbox()
    dims = hi - lo
    k = int(np.argmin(dims))                       # 厚度方向
    if dims[k] > 0.5 * min(dims[(k + 1) % 3], dims[(k + 2) % 3]):
        raise ValueError("%s 不是板状构件（三边 %s），面内缺料无从定义"
                         % (sid, [round(float(d), 3) for d in dims]))
    rest = [i for i in range(3) if i != k]
    v = 2 if 2 in rest else rest[int(np.argmax(dims[rest]))]   # 竖直方向
    u = [i for i in rest if i != v][0]                          # 面内水平方向

    P, F = _weld(S.P, S.F)
    us = np.unique(np.round(P[:, u], 6))
    vs = np.unique(np.round(P[:, v], 6))
    if len(us) * len(vs) > 200000:
        raise ValueError("顶点坐标组合过多（%d×%d），这个构件不是规整板状，改用 profile 逐高度切"
                         % (len(us), len(vs)))
    tris_uv = []
    for f in F:
        Q = P[[int(t) for t in f]][:, [u, v]]
        tris_uv.append((Q[0], Q[1], Q[2]))
    cov = _cells_covered(us, vs, tris_uv)

    outs = []
    for i0, i1, j0, j1 in _components(~cov):
        u0, u1 = float(us[i0]), float(us[i1 + 1])
        v0, v1 = float(vs[j0]), float(vs[j1 + 1])
        w, h = u1 - u0, v1 - v0
        if w * h < min_area:
            continue
        c = np.zeros(3)
        c[u], c[v] = (u0 + u1) / 2.0, (v0 + v1) / 2.0
        c[k] = float((lo[k] + hi[k]) / 2.0)        # 厚度方向取中面
        # 洞口矩形的几何：中心、宽高之外，还要给**矩形本身**（四个角点 + 两向单位矢量）。
        # 只给 center/width/height 时，任何"构件与洞口是什么关系"的判断都得让模型自己做
        # 加减乘除 —— 那既违反"派生量走 calc"，也会在旋转/换单位后悄悄错位。
        corners = []
        for cu, cv in ((u0, v0), (u1, v0), (u1, v1), (u0, v1)):
            q = np.zeros(3)
            q[u], q[v], q[k] = cu, cv, c[k]
            corners.append([round(float(t), 6) for t in q])
        prism_lo = np.array(lo, float).copy()
        prism_hi = np.array(hi, float).copy()
        prism_lo[u], prism_hi[u] = u0, u1
        prism_lo[v], prism_hi[v] = v0, v1
        outs.append({
            "center": [round(float(x), 6) for x in c],
            "width": round(w, 6),
            "height": round(h, 6),
            "area": round(w * h, 6),
            "sill_z": round(v0 if v == 2 else float(lo[2]), 6),
            "grounded": bool(abs(v0 - float(lo[v])) <= 1e-6),   # 贴着构件底边
            "touches_top": bool(abs(v1 - float(hi[v])) <= 1e-6),
            "rect": {
                "u_dir": [1.0 if i == u else 0.0 for i in range(3)],
                "v_dir": [1.0 if i == v else 0.0 for i in range(3)],
                "n_dir": [1.0 if i == k else 0.0 for i in range(3)],
                "u_len": round(w, 6), "v_len": round(h, 6),
                "depth": [round(float(lo[k]), 6), round(float(hi[k]), 6)],
                "u_range": [round(u0, 6), round(u1, 6)],
                "v_range": [round(v0, 6), round(v1, 6)],
            },
            "corners": corners,
            # 洞口沿墙厚方向拉伸出的棱柱（世界轴 AABB）：判"构件在不在洞里"直接用它
            "prism": {"lo": [round(float(x), 6) for x in prism_lo],
                      "hi": [round(float(x), 6) for x in prism_hi]},
        })
    outs.sort(key=lambda o: -o["area"])
    return {"sid": sid, "thickness": round(float(dims[k]), 6),
            "axes": {"u": "xyz"[u], "v": "xyz"[v], "n": "xyz"[k]},
            "wall": {"lo": [round(float(x), 6) for x in lo],
                     "hi": [round(float(x), 6) for x in hi]},
            "openings": outs, "n": len(outs),
            "resolution": "顶点坐标线网格（精确，非栅格）"}
