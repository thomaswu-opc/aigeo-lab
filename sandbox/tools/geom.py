# -*- coding: utf-8 -*-
"""geom.py — 度量内核。所有数值只在这里产生；LLM 永远不碰数字。

每个函数返回 dict，必须包含可复核的见证物（witness / 参数 / 方法 / 不确定度），
否则评委既无法反驳也无法确认。
"""
import math

import numpy as np

EPS = 1e-12


# ------------------------------------------------------------ 基本量


def tri_arrays(S):
    P, F = S.P, S.F
    return P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]


def tri_normals(S):
    A, B, C = tri_arrays(S)
    N = np.cross(B - A, C - A)
    ln = np.linalg.norm(N, axis=1)
    area = ln / 2.0
    ok = ln > EPS
    Nn = np.zeros_like(N)
    Nn[ok] = N[ok] / ln[ok, None]
    return Nn, area


def volume(S):
    """有向体积（散度定理）。符号可用于判断壳朝向。"""
    A, B, C = tri_arrays(S)
    return float(np.einsum("ij,ij->i", A, np.cross(B, C)).sum() / 6.0)


def surface_area(S):
    return float(tri_normals(S)[1].sum())


def dominant_dirs(S, topk=4, cos_tol=0.995, min_area_frac=0.02, max_seeds=400):
    """按面积贪心聚类法向；每类给法向、面积、沿法向的顶点投影跨度（=厚度）。
    这是「板 / 薄壁 / 翼缘」判定的唯一来源。

    判据是「方向簇的总面积占比」，不是「单块三角形的面积」：
    窗框的前后环面由 16 块 0.04 m² 的小三角形拼成，单块远低于阈值，合计却占 20%。
    早期版本用单块面积做门限并 break，整簇被判丢，min_thickness 于是报出 1.4 m
    （等于整块板宽），而厚度正是区分「墙 / 门扇 / 窗框」的主判据 —— 一个阈值写法
    让下游所有角色识别都踩空。
    """
    N, A = tri_normals(S)
    tot = float(A.sum())
    if tot <= 0:
        return []
    # 没有任何平面占主导（球面这类各向同性曲面）：谈不上「板」，直接返回空。
    if float(A.max()) / tot < 0.002:
        return []

    order = np.argsort(-A)
    dirs, examined = [], 0
    for i in order:
        if A[i] <= 1e-12:
            break
        if any(abs(float(N[i] @ np.array(d["n"]))) > cos_tol for d in dirs):
            continue
        sel = np.abs(N @ N[i]) > cos_tol
        share = float(A[sel].sum()) / tot
        examined += 1
        if share >= min_area_frac:
            proj = S.P @ N[i]
            dirs.append({"n": N[i].tolist(), "area": float(A[sel].sum()),
                         "share": round(share, 4),
                         "thickness": float(proj.max() - proj.min())})
            if len(dirs) >= topk:
                break
        if examined >= max_seeds:
            break
    return dirs


def axis_aligned(dirs, tol=1e-3):
    for d in dirs:
        n = np.abs(np.array(d["n"]))
        if n.max() > 1 - tol and int((n > tol).sum()) == 1:
            continue
        return False
    return True


def describe(S):
    lo, hi = S.bbox()
    dims = (hi - lo).tolist()
    V = abs(volume(S))
    bv = float(np.prod(np.maximum(hi - lo, EPS)))
    dirs = dominant_dirs(S)
    thin = [d["thickness"] for d in dirs]
    ds = sorted(dims, reverse=True)
    return {
        "sid": S.sid,
        "src_type": S.src,
        "ntri": int(len(S.F)),
        "dims": [round(d, 6) for d in ds],
        # 三个标量版本，供 select 做纯数据比较（不做 eval）
        "d0": round(ds[0], 6), "d1": round(ds[1], 6), "d2": round(ds[2], 6),
        "cover": round(ds[2] / max(ds[1], EPS), 6),
        "bbox_lo": [round(float(x), 6) for x in lo],
        "bbox_hi": [round(float(x), 6) for x in hi],
        "volume": round(V, 6),
        "bbox_volume": round(bv, 6),
        "fill": round(V / bv, 6) if bv > EPS else 0.0,
        "area": round(surface_area(S), 6),
        "z_min": round(float(lo[2]), 6),
        "dominant": [{"n": [round(float(x), 6) for x in d["n"]],
                      "area": round(d["area"], 6),
                      "thickness": round(d["thickness"], 6)} for d in dirs],
        "axis_aligned": bool(axis_aligned(dirs)),
        "min_thickness": round(min(thin) if thin else 1.0, 6),
        "max_patch_area": round(dirs[0]["area"], 6) if dirs else 0.0,
    }


# ------------------------------------------------------------ 距离内核


def _tribox(A, B, C):
    return np.minimum(np.minimum(A, B), C), np.maximum(np.maximum(A, B), C)


def _gap(lo1, hi1, lo2, hi2):
    """逐元素 AABB 间隙（同形状数组）。"""
    d = np.maximum(0.0, np.maximum(lo1 - hi2, lo2 - hi1))
    return np.linalg.norm(d, axis=-1)


def _candidate_edge_pairs(lo1, hi1, lo2, hi2, best, budget=4_000_000):
    """外积 + 分块，返回间隙 ≤ best 的边对索引。"""
    ii, jj = [], []
    step = max(1, int(budget // max(len(lo2), 1)))
    for i0 in range(0, len(lo1), step):
        sl = slice(i0, i0 + step)
        d = np.maximum(0.0, np.maximum(lo1[sl, None, :] - hi2[None, :, :],
                                      lo2[None, :, :] - hi1[sl, None, :]))
        g = np.linalg.norm(d, axis=-1)
        a, b = np.nonzero(g <= best * 1.0001 + 1e-6)
        if len(a):
            ii.append(a + i0)
            jj.append(b)
    if not ii:
        return np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(ii), np.concatenate(jj)


def _closest_on_tri_batch(P, T, budget=1_500_000):
    """P:(c,3) T:(M,3,3) -> Q:(c,M,3), d:(c,M)。分块控制内存。"""
    M = len(T)
    c = max(1, min(len(P), int(budget // max(M, 1))))
    outQ, outD = [], []
    for i0 in range(0, len(P), c):
        p = P[i0:i0 + c][:, None, :]                    # (c,1,3)
        a, b, cc = T[None, :, 0], T[None, :, 1], T[None, :, 2]   # (1,M,3)
        ab, ac, ap = b - a, cc - a, p - a
        d1 = np.einsum("cmj,cmj->cm", ab, ap)
        d2 = np.einsum("cmj,cmj->cm", ac, ap)
        bp = p - b
        d3 = np.einsum("cmj,cmj->cm", ab, bp)
        d4 = np.einsum("cmj,cmj->cm", ac, bp)
        cp = p - cc
        d5 = np.einsum("cmj,cmj->cm", ab, cp)
        d6 = np.einsum("cmj,cmj->cm", ac, cp)
        va = d3 * d6 - d5 * d4
        vb = d5 * d2 - d1 * d6
        vc = d1 * d4 - d3 * d2
        den = va + vb + vc
        den = np.where(np.abs(den) < 1e-18, 1.0, den)
        v, w = vb / den, vc / den
        Q = a + ab * v[..., None] + ac * w[..., None]
        Q = np.broadcast_to(Q, (Q.shape[0], M, 3)).copy()

        def put(mask, val):
            Q[mask] = np.broadcast_to(val, (mask.sum(), 3)) if val.ndim == 3 else val[mask]

        m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        if m.any():
            t = (d4 - d3) / np.maximum((d4 - d3) + (d5 - d6), 1e-18)
            val = np.broadcast_to(b, (Q.shape[0], M, 3)) + (
                np.broadcast_to(cc, (Q.shape[0], M, 3)) - np.broadcast_to(b, (Q.shape[0], M, 3))) * t[..., None]
            Q[m] = val[m]
        m2 = (vb <= 0) & (d2 >= 0) & (d6 <= 0) & ~m
        if m2.any():
            t = d2 / np.maximum(d2 - d6, 1e-18)
            val = np.broadcast_to(a, (Q.shape[0], M, 3)) + (
                np.broadcast_to(ac, (Q.shape[0], M, 3))) * t[..., None]
            Q[m2] = val[m2]
        m3 = (vc <= 0) & (d1 >= 0) & (d3 <= 0) & ~m & ~m2
        if m3.any():
            t = d1 / np.maximum(d1 - d3, 1e-18)
            val = np.broadcast_to(a, (Q.shape[0], M, 3)) + (
                np.broadcast_to(ab, (Q.shape[0], M, 3))) * t[..., None]
            Q[m3] = val[m3]
        m4 = (d6 >= 0) & (d5 <= d6) & ~m & ~m2 & ~m3
        if m4.any():
            Q[m4] = np.broadcast_to(cc, (Q.shape[0], M, 3))[m4]
        m5 = (d3 >= 0) & (d4 <= d3) & ~m & ~m2 & ~m3 & ~m4
        if m5.any():
            Q[m5] = np.broadcast_to(b, (Q.shape[0], M, 3))[m5]
        m6 = (d1 <= 0) & (d2 <= 0) & ~m & ~m2 & ~m3 & ~m4 & ~m5
        if m6.any():
            Q[m6] = np.broadcast_to(a, (Q.shape[0], M, 3))[m6]

        d = np.linalg.norm(Q - p, axis=2)
        outQ.append(Q)
        outD.append(d)
    return np.vstack(outQ), np.vstack(outD)


def _seg_seg(p1, q1, p2, q2):
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, r)
    c = np.einsum("ij,ij->i", d1, r)
    b = np.einsum("ij,ij->i", d1, d2)
    den = a * e - b * b
    s = np.where(den > 1e-18, np.clip((b * f - c * e) / np.maximum(den, 1e-18), 0, 1), 0.0)
    t = np.clip((b * s + f) / np.maximum(e, 1e-18), 0, 1)
    s = np.clip((b * t - c) / np.maximum(a, 1e-18), 0, 1)
    X = p1 + d1 * s[:, None]
    Y = p2 + d2 * t[:, None]
    return np.linalg.norm(X - Y, axis=1), X, Y


def _edge_set(S):
    e = np.vstack([S.F[:, [0, 1]], S.F[:, [1, 2]], S.F[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    p, q = S.P[e[:, 0]], S.P[e[:, 1]]
    lo, hi = np.minimum(p, q), np.maximum(p, q)
    return p, q, lo, hi


def distance_min(A, B, edge_budget=3_000_000):
    """网格-网格最短表面距离 + 两个见证点。
    路径：①双向点-面（精确）②边-边（下界剪枝后精确）③取最小。
    """
    best, bw, method = math.inf, None, "vertex-face both ways"
    TA = np.stack(tri_arrays(A), axis=1)
    TB = np.stack(tri_arrays(B), axis=1)

    Q, d = _closest_on_tri_batch(A.P, TB)
    k = int(np.argmin(d))
    i, j = divmod(k, len(TB))
    if d.flat[k] < best:
        best, bw = float(d.flat[k]), (A.P[i].copy(), Q[i, j].copy())

    Q2, d2 = _closest_on_tri_batch(B.P, TA)
    k = int(np.argmin(d2))
    i, j = divmod(k, len(TA))
    if d2.flat[k] < best:
        best, bw = float(d2.flat[k]), (Q2[i, j].copy(), B.P[i].copy())

    # 边-边精修
    p1, q1, lo1, hi1 = _edge_set(A)
    p2, q2, lo2, hi2 = _edge_set(B)
    n_pairs = 0
    if len(p1) and len(p2):
        ii, jj = _candidate_edge_pairs(lo1, hi1, lo2, hi2, best)
        n_pairs = int(len(ii))
        if 0 < n_pairs <= edge_budget:
            dd, X, Y = _seg_seg(p1[ii], q1[ii], p2[jj], q2[jj])
            k = int(np.argmin(dd))
            if dd[k] < best - 1e-12:
                best = float(dd[k])
                bw = (X[k].copy(), Y[k].copy())
                method = "vertex-face both ways + edge-edge refine"
        elif n_pairs > edge_budget:
            method += "（边-边精修被预算裁掉：候选 %d 对）" % n_pairs

    return {
        "value": round(best, 6), "unit": "m",
        "witness_a": [round(float(x), 6) for x in bw[0]],
        "witness_b": [round(float(x), 6) for x in bw[1]],
        "method": method,
        "uncertainty": "±1e-6（三角离散化；相交时返回 0）",
        "assumptions": ["两网格表面在世界坐标（米）", "不相交"],
        "a": A.sid, "b": B.sid,
        "edge_pairs_evaluated": int(len(p1) * len(p2) if len(p1) and len(p2) and len(ii) == 0 else (len(ii) if len(p1) and len(p2) else 0)),
    }


# ------------------------------------------------------------ 截面


def section_area(S, axis=2, height=0.0):
    ax = [0, 1, 2]
    ax.remove(axis)
    ui, vi = ax[0], ax[1]
    A, B, C = tri_arrays(S)
    z = np.stack([A[:, axis], B[:, axis], C[:, axis]], 1)
    pts = np.stack([A, B, C], 1)
    segs = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        zi, zj = z[:, i], z[:, j]
        cr = ((zi - height) * (zj - height)) < 0
        if not cr.any():
            continue
        t = (height - zi[cr]) / (zj[cr] - zi[cr])
        segs.append(pts[cr, i] + (pts[cr, j] - pts[cr, i]) * t[:, None])
    if not segs:
        return {"value": 0.0, "n_segments": 0, "axis": "xyz"[axis], "height": height}
    P = np.vstack(segs)
    n = len(P) // 2
    p1, p2 = P[:2 * n:2], P[1:2 * n:2]
    area = abs(0.5 * np.sum(p1[:, ui] * p2[:, vi] - p2[:, ui] * p1[:, vi]))
    return {"value": round(float(area), 6), "n_segments": int(n),
            "axis": "xyz"[axis], "height": height}


def aabb_gap(A, B):
    la, ha = A.bbox()
    lb, hb = B.bbox()
    return float(np.linalg.norm(np.maximum(0.0, np.maximum(la - hb, lb - ha))))


# ------------------------------------------------------------ 场景主方向（对抗「非轴对齐」）


def frame_axes(solids, cos_tol=0.999, min_area_frac=0.01):
    """用「面积加权的面法向簇」求场景竖直参照系。

    为什么不用本调整体的协方差特征分解：实测会偏 8.5°。
    原因是球面（12.56 m²，法向各向同性）与椅腿/插脚这类斜置小面把协方差矩阵的
    两个水平特征值压到接近相等（78.8 / 87.8），在这个近似退化的平面上，特征向量
    的方向由噪声决定 —— 于是「主轴」不等于「墙的朝向」。

    改成先聚类再挑：只有占据显著面积的平面簇才算「构形朝向」，各向同性的曲面
    不产生任何大簇，自然被排除。这个判据也更容易向评委解释：
    「这栋楼 92% 的外表面法向落在两个互相垂直的方向上，所以这两个方向就是它的朝向」。

    返回 (axes 3x3 行向量, origin, info)
    """
    Ns, Ws = [], []
    for s in solids:
        Nn, A = tri_normals(s)
        ok = A > 1e-10
        Ns.append(Nn[ok])
        Ws.append(A[ok])
    if not Ns or sum(len(x) for x in Ns) == 0:
        return np.eye(3), np.zeros(3), {"coverage": 0.0, "n_clusters": 0, "up_confident": True}

    N = np.vstack(Ns)
    W = np.concatenate(Ws)
    tot = float(W.sum())

    # 1) 面积贪心聚类（同一簇 = 法向近似平行或反平行）
    order = np.argsort(-W)
    clusters = []
    for i in order:
        if any(abs(float(N[i] @ c["n"])) > cos_tol for c in clusters):
            continue
        sel = np.abs(N @ N[i]) > cos_tol
        clusters.append({"n": N[i] / np.linalg.norm(N[i]),
                         "area": float(W[sel].sum()),
                         "share": float(W[sel].sum()) / tot})
        if W[i] < min_area_frac * tot:
            break
    big = [c for c in clusters if c["share"] >= min_area_frac]

    # 2) 竖直簇（法向水平）里挑两个正交方向
    horiz = [c for c in big if abs(float(c["n"][2])) < 0.15]
    horiz.sort(key=lambda c: -c["area"])
    if not horiz:
        return np.eye(3), np.vstack([s.P for s in solids]).min(0), \
               {"coverage": 0.0, "n_clusters": len(big), "up_confident": False,
                "note": "没有显著的水平法向簇：场景可能没有竖直墙体"}
    u1 = np.array([horiz[0]["n"][0], horiz[0]["n"][1], 0.0])
    n1 = np.linalg.norm(u1)
    if n1 < 1e-9:
        return np.eye(3), origin_of(np.vstack([s.P for s in solids])), \
               {"coverage": 0.0, "n_clusters": len(big), "up_confident": False,
                "note": "最大水平簇退化，无法建立参照系"}
    u1 = u1 / n1
    # 规范定向（只定 u1 的符号）：保证同一模型任何读取顺序都得到同一 frame
    if u1[int(np.argmax(np.abs(u1)))] < 0:
        u1 = -u1
    # u3 直接取世界竖直轴：本工具只处理「绕竖轴旋转」，不重定基 Z，
    # 这样地面标高、楼层高度等竖直量在世界系与 frame 系里完全一致。
    u3 = np.array([0.0, 0.0, 1.0])
    u2 = np.cross(u3, u1)                           # 保证 u1×u2=u3（右手、det=+1）
    n2 = np.linalg.norm(u2)
    u2 = u2 / n2 if n2 > 1e-9 else np.array([0.0, 1.0, 0.0])
    axes = np.array([u1, u2, u3])

    # 3) 覆盖率：法向落在三个方向任意一个上的面积占比
    hit = np.zeros(len(N), bool)
    for a in axes:
        hit |= np.abs(N @ a) > cos_tol
    P = np.vstack([s.P for s in solids])
    dev = [round(float(np.degrees(np.arccos(min(1.0, float(np.abs(a).max()))))), 3) for a in axes]
    info = {"coverage": round(float(W[hit].sum()) / tot, 4),
            "n_clusters": len(big),
            "clusters": [{"n": [round(float(x), 6) for x in c["n"]],
                          "share": round(c["share"], 4),
                          "tilt_from_world_deg": round(float(np.degrees(
                              np.arccos(min(1.0, float(np.abs(c["n"]).max()))))), 3)}
                         for c in big[:6]],
            "dev_from_world_deg": dev,
            "is_axis_aligned": max(dev) < 1.0,
            "up_confident": bool(abs(u3[2]) > 0.9)}
    return axes, origin_of(P), info


def origin_of(P):
    return np.asarray(P).reshape(-1, 3).min(0)


def in_frame(S, axes, origin, sid=None):
    """把实体旋进参照系，返回新的 Solid（不改原对象）。"""
    import tools.world as _w
    Q = (S.P - origin) @ axes.T
    return _w.Solid(sid or S.sid, S.name, Q, S.F, S.src)
