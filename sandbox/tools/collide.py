# -*- coding: utf-8 -*-
"""collide.py — 碰撞判定与扫掠。

读层的工具回答「多远」，写层的工具回答「改成什么」，这一层回答第三类问题：
**在两个姿态之间，有没有一刻撞上了。**

三件事必须分清楚，否则结论会混：

  * 分离 —— 任何时刻都不接触；
  * 接触 —— 某个姿态下表面相切，相交三角形对存在但顶点没有互相进入；
  * 重叠 —— 顶点已经进到对方内部，是实打实的相互穿入（`depth > 0`）。

题面在 08 里同时要「首次碰到的角度」和「实体重叠的角度区间」，就是要求分开报这两件事。
所以 `_penetration` 不返回布尔，返回「谁进到谁里边、进了多深」。

关于精度：`sweep_*` 只做「粗扫 + 边界二分」。粗扫步长会写进返回值（`scan_step_deg`），
二分精度也写进去（`refine_deg`）。报告里必须同时给出这两个数 —— 「检查到 0.1°」这种说法
不说明方法就只是一句空话。窄于粗扫步长的孤立接触可能漏检，连续重叠段不会漏；这个限制
写在 `limits` 里返回，而不是藏在文档里。

坐标一律世界坐标、单位米。工具不认识任务：`sweep_rotate` 不知道自己在测门，
`collide` 不知道自己在测插脚 —— 它们只知道两组三角形和一个运动。
"""
import math

import numpy as np

from core.runtime import tool
from tools import geom
from tools.world import Solid

# 判「刚好接触」的尺度。1 µm：远小于任何建模留缝（本套模型的最小间隙是 7.5 mm），
# 又远大于浮点噪声（5 m 量级上双精度噪声 ~1e-15）。
CONTACT_TOL = 1e-6


# ================================================================ 三角形对相交


def _tri_tri(A, B):
    """两个三角形是否相交（分离轴，11 轴）。

    分离轴只有 11 根：两面法向 + 3×3 条边的叉积。轴上投影区间只要有空隙就是分离。

    有两个地方容易写错：
      * 退化轴（两边平行的叉积近似零向量）必须跳过，否则投影全为 0，会被误判成相交；
      * 判分离用严格小于。刚好相切（投影区间端点相等）必须算作相交 —— 门扇最后一刻擦过
        椅面，物理上是碰上了，不能因为差 1e-16 就报「没碰」。
    """
    ea = (A[1] - A[0], A[2] - A[1], A[0] - A[2])
    eb = (B[1] - B[0], B[2] - B[1], B[0] - B[2])
    axes = [np.cross(ea[0], ea[1]), np.cross(eb[0], eb[1])]
    for u in ea:
        for v in eb:
            axes.append(np.cross(u, v))
    for ax in axes:
        if float(ax @ ax) < 1e-18:
            continue
        pa = A @ ax
        pb = B @ ax
        if pa.max() < pb.min() or pb.max() < pa.min():
            return False
    return True


def _tri_tri_batch(A, B):
    """批量版：A、B 都是 (n,3,3)，返回 (n,) 布尔。

    逐对调 `_tri_tri` 在扫掠里会成为瓶颈（一次扫掠上千个角度 × 上百对）。
    这里把 11 根轴摊成 11 次 einsum —— 每次是对 (n,3,3) 的整体投影，
    没有 Python 循环，n=100 时一批只要几十微秒。
    """
    if len(A) == 0:
        return np.zeros(0, bool)
    ea0, ea1, ea2 = A[:, 1] - A[:, 0], A[:, 2] - A[:, 1], A[:, 0] - A[:, 2]
    eb0, eb1, eb2 = B[:, 1] - B[:, 0], B[:, 2] - B[:, 1], B[:, 0] - B[:, 2]
    axes = [np.cross(ea0, ea1), np.cross(eb0, eb1)]
    for u in (ea0, ea1, ea2):
        for v in (eb0, eb1, eb2):
            axes.append(np.cross(u, v))
    live = np.ones(len(A), bool)
    for ax in axes:
        n2 = np.einsum("ni,ni->n", ax, ax)
        ok = n2 > 1e-18
        if not ok.any():
            continue
        pa = np.einsum("nij,nj->ni", A, ax)
        pb = np.einsum("nij,nj->ni", B, ax)
        sep = (pa.max(1) < pb.min(1)) | (pb.max(1) < pa.min(1))
        live &= ~(ok & sep)
        if not live.any():
            return live
    return live


def _tri_aabb(P, F):
    V = P[np.asarray(F, int)]
    return V.min(1), V.max(1)


def _aabb_pairs(la, ha, lb, hb):
    """两批三角形之间的 AABB 重叠对。返回 (ia, ib)。

    这是整个扫掠的主要开销，所以写成 6 条链式比较（每个轴两条），不用 all(axis=2)
    —— 后者会先堆一个 (na,nb,3) 的布尔中间量，na·nb=9.5万时多占几 MB。
    """
    ov = ((la[:, None, 0] <= hb[None, :, 0]) & (lb[None, :, 0] <= ha[:, None, 0]) &
          (la[:, None, 1] <= hb[None, :, 1]) & (lb[None, :, 1] <= ha[:, None, 1]) &
          (la[:, None, 2] <= hb[None, :, 2]) & (lb[None, :, 2] <= ha[:, None, 2]))
    return np.nonzero(ov)


def _hit_pairs(Pa, Fa, Pb, Fb):
    """返回真正相交的三角形对索引 (ia, ib)。"""
    la, ha = _tri_aabb(Pa, Fa)
    lb, hb = _tri_aabb(Pb, Fb)
    ia, ib = _aabb_pairs(la, ha, lb, hb)
    if len(ia) == 0:
        return ia, ib
    A = Pa[np.asarray(Fa, int)][ia]
    B = Pb[np.asarray(Fb, int)][ib]
    ok = _tri_tri_batch(A, B)
    return ia[ok], ib[ok]


def _hits(Pa, Fa, Pb, Fb, la=None, ha=None):
    """候选对 → 真正相交的三角对数目。"""
    ia, _ = _hit_pairs(Pa, Fa, Pb, Fb)
    return int(len(ia))


def _edge_pierce_tri(T1, T2, tol=CONTACT_TOL):
    """T1 的三条边里，有没有一条**严格穿过** T2 的内部。

    「严格」三件事缺一不可：两端点在 T2 平面两侧且各自离面超过 tol；
    交点参数在开区间内；交点在三角形内部（重心坐标都为正且和小于 1，留 tol 余量）。
    少任何一条都会把「贴着边擦过」算成穿透。

    为什么需要这个判据：门扇只有 8 个顶点。板穿过椅腿时，腿的 8 个顶点和板的 8 个顶点
    可能全都在对方外面 —— 顶点包含法在这里是瞎的。但腿的水平边（顶面/座面轮廓）必然
    横跨板面，所以「边穿面」一定成立。两个判据互补，缺一个就会漏报。
    """
    e0 = T2[1] - T2[0]
    e1 = T2[2] - T2[0]
    n2 = np.cross(e0, e1)
    L = float(np.linalg.norm(n2))
    if L < 1e-14:
        return False
    n2 = n2 / L
    for i in range(3):
        p, q = T1[i], T1[(i + 1) % 3]
        dp = float((p - T2[0]) @ n2)
        dq = float((q - T2[0]) @ n2)
        if abs(dp) <= tol or abs(dq) <= tol or dp * dq > 0:
            continue
        t = dp / (dp - dq)
        x = p + (q - p) * t
        v2 = x - T2[0]
        d00 = float(e0 @ e0)
        d01 = float(e0 @ e1)
        d11 = float(e1 @ e1)
        d20 = float(v2 @ e0)
        d21 = float(v2 @ e1)
        den = d00 * d11 - d01 * d01
        if abs(den) < 1e-18:
            continue
        u = (d11 * d20 - d01 * d21) / den
        v = (d00 * d21 - d01 * d20) / den
        m = 1e-9
        if not (u > m and v > m and u + v < 1 - m):
            continue
        # 交点还必须离 T2 的三条边都有实质距离，否则是「擦着边过去」。
        # 这一条是实测补上的：箭体底面（水平）压在凹槽壁上（竖直），交线正好落在
        # 凹槽壁的顶边 —— 只判重心坐标时会被算成 104 对「穿刺」，而实际只是表面贴合。
        # 差 2.4e-8 m 的浮点错位，足够让重心坐标落进内部 1e-9 的缝里。
        edge_clear = True
        for k in range(3):
            a2 = T2[k]
            e2 = T2[(k + 1) % 3] - a2
            L2 = float(e2 @ e2)
            if L2 < 1e-18:
                edge_clear = False
                break
            t2 = min(1.0, max(0.0, float((x - a2) @ e2) / L2))
            if float(np.linalg.norm(x - (a2 + e2 * t2))) <= tol:
                edge_clear = False
                break
        if edge_clear:
            return True
    return False


def _pierce_count(Pa, Fa, Pb, Fb, ia, ib):
    """相交三角形对里，有多少对存在「边严格穿过对方三角面」。"""
    n = 0
    Fa = np.asarray(Fa, int)
    Fb = np.asarray(Fb, int)
    for a, b in zip(ia, ib):
        Ta = Pa[Fa[a]]
        Tb = Pb[Fb[b]]
        if _edge_pierce_tri(Ta, Tb) or _edge_pierce_tri(Tb, Ta):
            n += 1
    return n


# ================================================================ 内外判定


_RAY = np.array([0.3713907, 0.5570860, 0.7427814])   # 单位向量，故意非轴对齐


def _inside(pts, P, F):
    """点是否在闭合网格内部：沿 _RAY 打射线，奇数个交点 = 在内。

    一次把 (npts × ntri) 的射线-三角形求交摊开算完。逐点循环在扫掠里是灾难：
    门扇加把手 540 个顶点、椅子 176 个三角形，逐点要 540 次 numpy 调用；
    摊开只要一次，中间量 540×176×3 也就 2 MB。

    对开网格（例如只有一层面片的投影）这个判定没有意义，所以调用方要先确认网格闭合。
    后果是可预期的：把开网格当闭合会低估穿透深度，不会凭空造出「撞上了」。
    """
    pts = np.asarray(pts, float).reshape(-1, 3)
    F = np.asarray(F, int)
    if len(pts) == 0 or len(F) == 0:
        return np.zeros(len(pts), bool)
    V0 = P[F[:, 0]]
    E1 = P[F[:, 1]] - V0
    E2 = P[F[:, 2]] - V0
    pvec = np.cross(_RAY, E2)                        # (ntri,3)
    det = np.einsum("ij,ij->i", E1, pvec)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tv = pts[:, None, :] - V0[None, :, :]            # (npts,ntri,3)
    u = np.sum(tv * pvec[None, :, :], axis=2) * inv
    # Möller–Trumbore 里 qvec = cross(tvec, E1) —— 是**边 E1**，不是射线方向。
    # 写成 cross(tvec, dir) 不会报错、也能算出数，只是那个数不是 v，奇偶计数随即乱掉：
    # 表现为「明明在盒子外面 3.5 m 的顶点被判成在盒子里」。
    qvec = np.cross(tv, E1[None, :, :])              # (npts,ntri,3)
    v = (qvec @ _RAY) * inv
    t = np.sum(qvec * E2[None, :, :], axis=2) * inv
    good = ok[None, :] & (u >= -1e-12) & (v >= -1e-12) & (u + v <= 1 + 1e-12) & (t > 1e-9)
    return (good.sum(1) % 2) == 1


def _pt_tri_dist(pts, P, F):
    """每个点到网格的最近距离。用 geom 里现成的最近点批处理。

    `_closest_on_tri_batch(P, T)` 返回 (Q, d)，Q 是 (c,M,3)、d 是 (c,M)；
    注意返回顺序是「最近点在前、距离在后」，早期版本按名字顺序取会拿到坐标当距离。
    """
    T = P[np.asarray(F, int)]
    if len(pts) == 0 or len(T) == 0:
        return np.zeros(len(pts))
    Q, D = geom._closest_on_tri_batch(np.asarray(pts, float).reshape(-1, 3), T)
    return D.min(1)


def _penetration(Pa, Fa, Pb, Fb):
    """谁进了谁、进多深。

    返回：
      n_pairs   相交三角形对数（>0 即「碰上了」，含相切）
      n_pierce  存在「边严格穿过对方三角面」的相交对数
      depth     最大顶点侵入深度（在对方内部的顶点到对方表面的最近距离）
      n_in_a    被 b 包住的 a 顶点数；n_in_b 反之
      deepest   取得最大深度的顶点坐标

    两个判据缺一不可：
      * **顶点侵入**能抓住「实体块戳进实心件」（插脚进凹槽、箭体压平台）；
      * **边穿面**能抓住「薄板穿过细杆」（门扇穿过椅腿）—— 这时两侧的顶点可能全在
        对方外面，顶点法完全失效。
    只用一个，会稳定漏掉另一类；用错一个，会把「刚好贴住」报成「撞穿」。
    """
    ia, ib = _hit_pairs(Pa, Fa, Pb, Fb)
    n_pairs = int(len(ia))
    n_pierce = _pierce_count(Pa, Fa, Pb, Fb, ia, ib) if n_pairs else 0
    ua = np.unique(np.round(Pa, 9), axis=0)
    ub = np.unique(np.round(Pb, 9), axis=0)
    in_b = _inside(ua, Pb, Fb)
    in_a = _inside(ub, Pa, Fa)
    depth, deepest = 0.0, None
    if in_b.any():
        d = _pt_tri_dist(ua[in_b], Pb, Fb)
        j = int(np.argmax(d))
        if d[j] > depth:
            depth, deepest = float(d[j]), ua[in_b][j].tolist()
    if in_a.any():
        d = _pt_tri_dist(ub[in_a], Pa, Fa)
        j = int(np.argmax(d))
        if d[j] > depth:
            depth, deepest = float(d[j]), ub[in_a][j].tolist()
    return {"n_pairs": n_pairs, "n_pierce": n_pierce, "depth": round(depth, 9),
            "n_in_a": int(in_a.sum()), "n_in_b": int(in_b.sum()),
            "deepest": None if deepest is None else [round(x, 6) for x in deepest]}


def _is_overlap(r, tol=CONTACT_TOL):
    """相交是否已经属于「实体重叠」而不是「表面相切」。"""
    return bool(r["n_pierce"] > 0 or r["depth"] > tol)


# ================================================================ 刚体运动


def _axis_vec(axis):
    if isinstance(axis, str):
        m = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}
        if axis.lower() not in m:
            raise ValueError(f"轴只能取 x/y/z 或三分量向量，收到 {axis!r}")
        v = np.array(m[axis.lower()], float)
    else:
        v = np.asarray(axis, float).reshape(3)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("轴向量长度为 0")
    return v / n


def _rot_mat(axis, ang):
    """Rodrigues：绕单位轴转 ang 弧度。"""
    k = _axis_vec(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


def _rot_pts(P, pivot, axis, ang):
    R = _rot_mat(axis, ang)
    piv = np.asarray(pivot, float).reshape(3)
    return (P - piv) @ R.T + piv


def _solid_of(bus, sid):
    for s in bus.scene:
        if s.sid == sid:
            return s
    raise KeyError(bus.missing_msg(sid))


def _gather(bus, sids):
    """把若干实体合成一个 (P, F) —— 运动是整体发生的，不能逐件处理，
    否则「门扇已经碰到椅子、把手还没碰到」会被报成两组互相矛盾的结论。
    """
    from tools.edit import _flatten_sids       # 复用同一套清单展平，避免两套写法
    Ps, Fs, off, used = [], [], 0, []
    for sid in _flatten_sids(sids):
        s = _solid_of(bus, sid)
        Ps.append(np.asarray(s.P, float))
        Fs.append(np.asarray(s.F, int) + off)
        off += len(s.P)
        used.append(s.sid)
    if not Ps:
        raise ValueError("moving/fixed 至少各要一个实体")
    return np.vstack(Ps), np.vstack(Fs), used


# ================================================================ 工具：单次判定


def _closed(P, F):
    """网格是否闭合：每条无向边恰好被两个三角形共用。

    这里不复用 edit.mesh_report —— 它要焊接、算欧拉特征、算有符号体积，为一次布尔判定
    付那个代价不值当。侵入深度只在闭合网格上有意义，所以这个判断必须廉价且先做。
    """
    F = np.asarray(F, int)
    if len(F) == 0:
        return False
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.sort(e, axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return bool((cnt == 2).all())


@tool("collide",
      "判定两个实体集合在**当前姿态**下的接触与穿插：相交三角形对数、最大侵入深度、"
      "互相进入的顶点数。二者都不碰时 n_pairs=0。",
      {"a": "str[]  实体 sid 列表（可逗号分隔）", "b": "str[]  另一组 sid"},
      pre="两组都已在场景里",
      degrade="若网格不闭合，侵入深度会低估；closed 字段会标出参与判定的实体是否闭合",
      returns="{n_pairs, depth, contact, overlap, n_in_a, n_in_b, deepest, closed_a, closed_b}",
      unit="m")
def t_collide(bus, a, b):
    Pa, Fa, ua = _gather(bus, a)
    Pb, Fb, ub = _gather(bus, b)
    r = _penetration(Pa, Fa, Pb, Fb)
    ov = _is_overlap(r)
    # contact：表面相切（有相交对但没人进去）；overlap：真穿入
    return {
        "n_pairs": r["n_pairs"],
        "n_pierce": r["n_pierce"],
        "depth": r["depth"],
        "contact": bool(r["n_pairs"] > 0 and not ov),
        "overlap": ov,
        "n_in_a": r["n_in_a"], "n_in_b": r["n_in_b"],
        "deepest": r["deepest"],
        "closed_a": _closed(Pa, Fa), "closed_b": _closed(Pb, Fb),
        "a": ua, "b": ub,
        "tol": CONTACT_TOL,
    }


# ================================================================ 工具：旋转扫掠


@tool("sweep_rotate",
      "让一组实体绕固定轴从 a0 转到 a1（度），逐角度检查与另一组实体是否相交。"
      "返回首次接触角、全部重叠区间（度，已二分细化）与检查方法。",
      {"moving": "str[]  一起转的实体（刚体，整体运动）",
       "fixed": "str[]  不动的实体",
       "pivot": "float[3]  轴上的任一点",
       "axis": "str|float[3]  转轴方向，'x'/'y'/'z' 或向量",
       "a0": "float  起始角（度）", "a1": "float  终止角（度）",
       "scan_step": "float  粗扫步长（度，默认 0.5）",
       "refine": "float  边界二分精度（度，默认 0.005）"},
      pre="moving 与 fixed 都已在场景里；角度按右手定则绕 axis 正向增加",
      degrade="窄于 scan_step 的孤立接触可能漏检；连续重叠段不会漏。limits 字段会写明",
      returns="{first_contact_deg, first_n_pairs, intervals[[a,b]...], n_probes, method, limits}",
      unit="deg")
def t_sweep_rotate(bus, moving, fixed, pivot, axis, a0, a1, scan_step=0.5, refine=0.005):
    Pm, Fm, um = _gather(bus, moving)
    Pf, Ff, uf = _gather(bus, fixed)
    piv = np.asarray(pivot, float).reshape(3)

    def hit_at(deg):
        P2 = _rot_pts(Pm, piv, axis, math.radians(deg))
        return _hits(P2, Fm, Pf, Ff)

    def ov_at(deg):
        P2 = _rot_pts(Pm, piv, axis, math.radians(deg))
        return _is_overlap(_penetration(P2, Fm, Pf, Ff))

    n = max(2, int(round(abs(a1 - a0) / abs(scan_step))) + 1)
    grid = np.linspace(a0, a1, n)
    vals = [hit_at(float(g)) for g in grid]

    # 连续相交段
    segs, cur = [], None
    for g, v in zip(grid, vals):
        if v > 0 and cur is None:
            cur = [float(g), float(g)]
        elif v > 0:
            cur[1] = float(g)
        elif cur is not None:
            segs.append(cur)
            cur = None
    if cur is not None:
        segs.append(cur)

    def bisect(lo, hi, want_hit):
        """在 [lo,hi] 里找碰撞起点，返回精度 refine 的边界。"""
        while abs(hi - lo) > refine:
            mid = (lo + hi) / 2
            if (hit_at(mid) > 0) == want_hit:
                hi = mid
            else:
                lo = mid
        return hi

    def bisect_depth(lo, hi):
        """[lo,hi] 里 lo 不重叠、hi 重叠，二分找开始重叠的角度。

        这一步是题面要求的另一半：只说「首次碰到」还不够，还得说清从哪一刻起
        是**实体重叠**而不是表面相切。两者相差通常只有零点几度，但性质完全不同 ——
        相切在装配里叫「刚好贴合」，重叠叫「撞坏了」。
        """
        if not ov_at(hi):
            return None
        while abs(hi - lo) > refine:
            mid = (lo + hi) / 2
            if ov_at(mid):
                hi = mid
            else:
                lo = mid
        return hi

    def bisect_depth_out(lo, hi):
        """[lo,hi] 里 lo 重叠、hi 不重叠，找重叠结束的角度。"""
        while abs(hi - lo) > refine:
            mid = (lo + hi) / 2
            if ov_at(mid):
                lo = mid
            else:
                hi = mid
        return lo

    out_segs = []
    for s in segs:
        i = int(np.searchsorted(grid, s[0]))
        lo = s[0]
        if i > 0 and vals[i - 1] == 0:
            lo = bisect(float(grid[i - 1]), s[0], True)
        j = int(np.searchsorted(grid, s[1]))
        hi = s[1]
        if j + 1 < len(grid) and vals[j + 1] == 0:
            hi = bisect(s[1], float(grid[j + 1]), False)
        out_segs.append([round(float(lo), 4), round(float(hi), 4)])

    # 重叠区间：在每个相交段内先按小步长采样，再对重叠子区间的两个边界各二分一次。
    #
    # 不能假设「段首相切、段尾重叠」—— 常见的情形恰恰是**两端都相切、中间才穿进去**
    # （门板扫过椅子：进入时擦一下、离开时又擦一下）。按段首段尾判会把这种段整体判成
    # 「只是相切」，报出「无重叠」，而实际上中间穿进去好几毫米。
    ov_segs = []
    for s in segs:
        lo, hi = s[0], s[1]
        K = 9
        ts = [lo + (hi - lo) * k / (K - 1) for k in range(K)]
        vs = [ov_at(t) for t in ts]
        idx = [k for k, v in enumerate(vs) if v]
        if not idx:
            ov_segs.append(None)
            continue
        i0, i1 = idx[0], idx[-1]
        start = ts[i0] if i0 == 0 else bisect_depth(ts[i0 - 1], ts[i0])
        end = ts[i1] if i1 == K - 1 else bisect_depth_out(ts[i1], ts[i1 + 1])
        ov_segs.append([round(float(start), 4), round(float(max(start, end)), 4)])

    first = out_segs[0][0] if out_segs else None
    first_ov = next((s[0] for s in ov_segs if s), None)
    return {
        "moving": um, "fixed": uf,
        "pivot": [round(float(x), 6) for x in piv],
        "axis": list(_axis_vec(axis).round(6)),
        "a0": a0, "a1": a1,
        "first_contact_deg": None if first is None else round(first, 4),
        "first_contact_n_pairs": (hit_at(first) if first is not None else 0),
        "first_overlap_deg": first_ov,
        "intervals": out_segs,
        "overlap_intervals": [s for s in ov_segs if s],
        "n_probes": int(n + 2 * len(out_segs) * 8 + len(ov_segs) * 14),
        "method": ("粗扫 %.3g° + 边界二分到 %.3g°；相交判据为三角形对分离轴测试（相切算相交），"
                   "重叠判据为「顶点已进入对方内部」，容差 %.0e m"
                   % (scan_step, refine, CONTACT_TOL)),
        "limits": ("窄于 %.3g° 的孤立接触可能漏检；重叠段边界已细化，不会漏段" % scan_step),
        "scan_step_deg": scan_step, "refine_deg": refine,
    }


# ================================================================ 工具：平移扫掠


@tool("sweep_translate",
      "让一组实体沿向量平移（t 从 0 到 1），逐 t 检查与另一组实体是否相交。"
      "返回首次接触的 t、对应位移距离与全部相交区间。",
      {"moving": "str[]  一起平移的实体", "fixed": "str[]  不动的实体",
       "vector": "float[3]  总平移向量（米）",
       "scan_step": "float  粗扫步数占比（默认按 200 段）",
       "refine": "float  边界二分精度（0..1 的比例，默认 1e-4）"},
      pre="两组都已在场景里",
      degrade="与 sweep_rotate 相同：窄接触可能漏检，limits 里写明",
      returns="{first_contact_t, first_contact_m, intervals_t, intervals_m, method, limits}",
      unit="m")
def t_sweep_translate(bus, moving, fixed, vector, scan_step=None, refine=1e-4):
    Pm, Fm, um = _gather(bus, moving)
    Pf, Ff, uf = _gather(bus, fixed)
    vec = np.asarray(vector, float).reshape(3)
    L = float(np.linalg.norm(vec))
    n = int(scan_step) if scan_step else 201

    def hit_at(t):
        P2 = Pm + vec * t
        return _hits(P2, Fm, Pf, Ff, *_tri_aabb(P2, Fm))

    grid = np.linspace(0.0, 1.0, n)
    vals = [hit_at(float(g)) for g in grid]
    segs, cur = [], None
    for g, v in zip(grid, vals):
        if v > 0 and cur is None:
            cur = [float(g), float(g)]
        elif v > 0:
            cur[1] = float(g)
        elif cur is not None:
            segs.append(cur)
            cur = None
    if cur is not None:
        segs.append(cur)

    def bisect(lo, hi, want_hit):
        while abs(hi - lo) > refine:
            mid = (lo + hi) / 2
            if (hit_at(mid) > 0) == want_hit:
                hi = mid
            else:
                lo = mid
        return hi

    out_segs = []
    for s in segs:
        i = int(np.searchsorted(grid, s[0]))
        lo = s[0]
        if i > 0 and vals[i - 1] == 0:
            lo = bisect(float(grid[i - 1]), s[0], True)
        j = int(np.searchsorted(grid, s[1]))
        hi = s[1]
        if j + 1 < len(grid) and vals[j + 1] == 0:
            hi = bisect(s[1], float(grid[j + 1]), False)
        out_segs.append([round(lo, 6), round(hi, 6)])

    first = out_segs[0][0] if out_segs else None
    return {
        "moving": um, "fixed": uf,
        "vector": [round(float(x), 6) for x in vec], "length": round(L, 6),
        "first_contact_t": None if first is None else round(first, 6),
        "first_contact_m": None if first is None else round(first * L, 6),
        "intervals_t": out_segs,
        "intervals_m": [[round(a * L, 6), round(b * L, 6)] for a, b in out_segs],
        "n_probes": int(n + 2 * len(out_segs) * 14),
        "method": "粗扫 %d 段 + 边界二分到 %.1e（按向量比例 t）" % (n, refine),
        "limits": "窄于向量长度 1/%d 的孤立接触可能漏检；重叠段不会漏" % (n - 1),
    }
