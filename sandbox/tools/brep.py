# -*- coding: utf-8 -*-
"""brep.py — 从三角汤还原「可判读的面结构」：共面区域 + 边界环 + 内环（开口）。

只输入顶点与三角面；不读名称、类别、属性。
局限（必须写进报告）：锐边阈值是全局单一值，曲面片会被过度合并；
内环只覆盖「完全落在面内部」的开口，贴地面的门洞表现为边界缺口（notch），
需要配合 loop 顶点数与反折角判断，或改用 compare_with_wall 的贯通性证据。
"""
import math

import numpy as np


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
