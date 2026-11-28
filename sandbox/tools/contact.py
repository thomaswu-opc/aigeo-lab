# -*- coding: utf-8 -*-
"""contact.py — 区域关系与接触面积。

两个工具，回答两类「不是距离」的问题：

  * `sphere_region` —— 一个球与一个长方体区域的关系：球心到区域的有符号距离、
    球在区域内的体积、以及关系分类（完全在内 / 内切 / 相交 / 外切 / 完全在外）。
    只报最短距离是不够的：球心在区域正中央时距离 1.25 m，球心在面外 0.01 m 时
    距离也是 0.01 m，两者听起来差不多，实际一个是「整颗球在里面」、
    一个是「基本在外面」。体积才分得开。

  * `contact_area` —— 两个实体的**贴合面积**。面接触有一块面积，边接触、角接触
    的面积是 0。题面说「仅有边、角接触不算面接触」，判据就落在这个数上。
    注意它和 `collide` 不是一回事：`collide` 判「有没有穿进去」，这里判
    「贴在一起的地方有多大」。

两个工具都不含任务知识。
"""
import math

import numpy as np

from core.runtime import tool
from tools import geom
from tools.shape import _solid_of


def _int_sqrt(a, b, r):
    """∫_a^b √(r²−u²) du，|u|≤r。解析式，不用数值积分。"""
    def F(u):
        u = max(-r, min(r, u))
        return 0.5 * (u * math.sqrt(max(0.0, r * r - u * u)) + r * r * math.asin(u / r))
    return F(b) - F(a)


def _circle_rect_area(cx, cy, r, x0, y0, x1, y1):
    """圆与轴对齐矩形的交面积。

    沿 x 分段解析积分：把 x 轴按「矩形左右边」与「圆与 y0/y1 的交点」切成若干段，
    每一段内被积函数（上边界减下边界）的形态是固定的（不是直线就是圆弧），
    所以每段都能用 √(r²−u²) 的原函数精确积出来。
    不用网格采样：采样会在圆弧边缘留下 O(1/N) 的偏差，而这个数要拿去和体积对账。
    """
    if r <= 0:
        return 0.0
    xa, xb = max(x0, cx - r), min(x1, cx + r)
    if xb <= xa:
        return 0.0
    ks = [xa, xb]
    for yy in (y0, y1):
        dy = yy - cy
        if abs(dy) < r:
            dx = math.sqrt(r * r - dy * dy)
            for x in (cx - dx, cx + dx):
                if xa < x < xb:
                    ks.append(x)
    ks = sorted(set(ks))
    tot = 0.0
    for a, b in zip(ks[:-1], ks[1:]):
        if b - a < 1e-15:
            continue
        m = 0.5 * (a + b)
        s = math.sqrt(max(0.0, r * r - (m - cx) ** 2))
        # 段内「圆的上边界是否越过矩形上边」不会变 —— 变的地方正是关键点
        top = y1 * (b - a) if (cy + s) > y1 else cy * (b - a) + _int_sqrt(a - cx, b - cx, r)
        bot = y0 * (b - a) if (cy - s) < y0 else cy * (b - a) - _int_sqrt(a - cx, b - cx, r)
        tot += max(0.0, top - bot)
    return tot


def _sphere_box_volume(c, r, lo, hi):
    """球与轴对齐盒的交体积。

    沿 z 积分「圆与矩形的交面积」。z 用 z = cz + r·sinθ 换元：被积函数在球顶球底
    有 √ 型奇性，直接对 z 积分用辛普森收敛很慢；换成 θ 之后被积式变光滑，
    256 段就能到 1e-10。
    """
    z0, z1 = max(lo[2], c[2] - r), min(hi[2], c[2] + r)
    if z1 <= z0 or r <= 0:
        return 0.0
    t0 = math.asin(max(-1.0, min(1.0, (z0 - c[2]) / r)))
    t1 = math.asin(max(-1.0, min(1.0, (z1 - c[2]) / r)))
    n = 256
    h = (t1 - t0) / n
    s = 0.0
    for i in range(n + 1):
        th = t0 + i * h
        z = c[2] + r * math.sin(th)
        rho = r * math.cos(th)
        A = _circle_rect_area(c[0], c[1], rho, lo[0], lo[1], hi[0], hi[1])
        w = 1 if i in (0, n) else (4 if i % 2 else 2)
        s += w * A * r * math.cos(th)
    return s * h / 3.0


@tool("sphere_region",
      "球与长方体区域的关系：球心到区域的有符号距离、球在区域内的体积，"
      "以及分类（完全在内 / 内切 / 相交 / 外切 / 完全在外）。",
      {"center": "list 球心 [x,y,z]", "radius": "float 半径",
       "box": "str 作为区域的实体（取其轴对齐包围盒）"},
      pre="半径 > 0；区域按实体的轴对齐包围盒理解",
      degrade="区域不是长方体时，这里算的是其包围盒的关系，不是实体本身的关系",
      returns="{relation, signed_gap, center_gap, inside_volume, outside_volume, ratio, cap_height}"
      "  —— signed_gap 是**球面**到区域边界的距离：0 相切、正有间隙、负凸出",
      unit="m / m³")
def t_sphere_region(bus, center, radius, box):
    # 全项目只有这一个工具把这个参数叫 box，其余都叫 sid：模型按习惯写 {"sid": ...}
    # 会 TypeError 整步作废（实测 T02）。别名由 Bus.call 的 _ARG_ALIAS 统一接住 ——
    # 在这里加 sid=None 默认值会把 box 从 required 里摘掉，模型干脆两个都不传。
    S = _solid_of(bus, box)
    P = np.asarray(S.P, float)
    lo, hi = P.min(0), P.max(0)
    c = np.asarray(center, float)
    r = float(radius)
    # 有符号距离统一按「**球面**到区域边界」量：相切为 0，分离/内含为正，凸出为负。
    # 早先按球心量，外切时球心距面正好等于 r，报告里却写成 1.0 m ——
    # 「相切」在数上完全看不出来，读的人只能靠关系名去猜。
    d_out = np.maximum(0.0, np.maximum(lo - c, c - hi))
    center_gap = float(np.linalg.norm(d_out))
    inside = bool(np.all(c >= lo) and np.all(c <= hi))
    if inside:
        d_face = float(np.min(np.minimum(c - lo, hi - c)))
        signed = d_face - r
    else:
        signed = center_gap - r
    vol = _sphere_box_volume(c, r, lo, hi)
    v_all = 4.0 / 3.0 * math.pi * r ** 3
    # 分类：相切用 1e-9 的容差判，否则「正好贴住」会随机落到两边
    tol = 1e-9
    if inside:
        if signed > tol:
            rel = "完全在区域内"
        elif abs(signed) <= tol:
            rel = "完全在区域内且与边界内切"
        else:
            rel = "与区域相交（球心在区域内，部分凸出）"
    else:
        if abs(signed) <= tol:
            rel = "完全在区域外且与边界外切"
        elif signed > tol:
            rel = "完全在区域外"
        else:
            rel = "与区域相交（球心在区域外，部分进入）"
    # 球冠高度 = 凸出量。相交时用于和解析式对账（球冠体积 = πh²(r−h/3)）
    cap = max(0.0, -signed) if abs(signed) < r else 0.0
    return {
        "relation": rel,
        # 球面到区域边界的有符号距离：0 = 相切，正 = 有间隙，负 = 凸出/重叠
        "signed_gap": round(signed, 9),
        "center_gap": round(center_gap if not inside else -abs(signed + r), 9),
        "inside_volume": round(vol, 9),
        "outside_volume": round(max(0.0, v_all - vol), 9),
        "sphere_volume": round(v_all, 9),
        "ratio": round(vol / v_all, 9) if v_all > 0 else 0.0,
        "cap_height": round(cap, 9),
        "sphere_center": [round(float(x), 6) for x in c],
        "radius": round(r, 6),
        "box_lo": [round(float(x), 6) for x in lo],
        "box_hi": [round(float(x), 6) for x in hi],
        "method": "有符号距离按包围盒取；体积沿 z 积分圆∩矩形面积（解析），"
                  "z 用 sinθ 换元后辛普森 256 段",
        "limits": "区域按包围盒理解；球必须是球（先看 fit_sphere 的 roundness）",
    }


def _tri_overlap_area(a2, b2):
    """两个 2D 三角形的重叠面积。Sutherland–Hodgman：用 b 的三条边裁 a。"""
    poly = [tuple(p) for p in a2]
    for i in range(3):
        if not poly:
            return 0.0
        p1, p2 = b2[i], b2[(i + 1) % 3]
        ex, ey = p2[0] - p1[0], p2[1] - p1[1]

        def side(p):
            return ex * (p[1] - p1[1]) - ey * (p[0] - p1[0])

        out = []
        for k in range(len(poly)):
            cur, nxt = poly[k], poly[(k + 1) % len(poly)]
            sc, sn = side(cur), side(nxt)
            if sc <= 1e-15:
                out.append(cur)
            if (sc > 1e-15) != (sn > 1e-15):
                t = sc / (sc - sn)
                out.append((cur[0] + t * (nxt[0] - cur[0]), cur[1] + t * (nxt[1] - cur[1])))
        poly = out
    if len(poly) < 3:
        return 0.0
    s = 0.0
    for k in range(len(poly)):
        x1, y1 = poly[k]
        x2, y2 = poly[(k + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return abs(0.5 * s)


@tool("contact_area",
      "两个实体贴合在一起的总面积，按共面三角形对逐块求重叠面积后求和。"
      "面接触得到一块正面积；边接触、角接触得到 0。",
      {"a": "str  实体", "b": "str  另一个实体", "tol": "float  共面与贴合的容差（默认 1e-6 m）"},
      pre="两实体在当前位置上贴合或相交（先用 collide 看是不是分离）",
      degrade="曲面与曲面贴合时，三角形之间不严格共面，面积会被低估；此时 n_pairs 会很小",
      returns="{area, n_pairs, max_patch, normal, tol}",
      unit="m²")
def t_contact_area(bus, a, b, tol=1e-6):
    A, B = _solid_of(bus, a), _solid_of(bus, b)
    d = geom.distance_min(A, B)
    if d["value"] > tol:
        return {"area": 0.0, "n_pairs": 0, "max_patch": 0.0, "normal": None,
                "min_distance": round(float(d["value"]), 9), "tol": tol,
                "note": "两实体未接触：最短表面距离大于容差，贴合面积为 0"}
    TA, TB = A.P[A.F], B.P[B.F]
    nA = np.cross(TA[:, 1] - TA[:, 0], TA[:, 2] - TA[:, 0])
    nB = np.cross(TB[:, 1] - TB[:, 0], TB[:, 2] - TB[:, 0])
    lA = np.linalg.norm(nA, axis=1)
    lB = np.linalg.norm(nB, axis=1)
    nA = np.divide(nA, lA[:, None], out=np.zeros_like(nA), where=lA[:, None] > 1e-14)
    nB = np.divide(nB, lB[:, None], out=np.zeros_like(nB), where=lB[:, None] > 1e-14)
    # 粗筛：法向近似平行（同向或反向），且两个面到彼此平面的距离都在容差内
    dot = np.abs(nA @ nB.T)
    cand = np.nonzero(dot > 1 - 1e-9)
    total, npair, best, best_n = 0.0, 0, 0.0, None
    for i, j in zip(*cand):
        a3, b3 = TA[i], TB[j]
        n = nA[i]
        if abs(float((b3[0] - a3[0]) @ n)) > tol:
            continue
        u, v = geom._basis(n) if hasattr(geom, "_basis") else (None, None)
        if u is None:
            u = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
            u = u - n * float(u @ n)
            u /= np.linalg.norm(u)
            v = np.cross(n, u)
        a2 = np.stack([a3 @ u, a3 @ v], 1)
        b2 = np.stack([b3 @ u, b3 @ v], 1)
        ar = _tri_overlap_area(a2, b2)
        if ar > 1e-12:
            total += ar
            npair += 1
            if ar > best:
                best, best_n = ar, [round(float(x), 6) for x in n]
    return {"area": round(total, 9), "n_pairs": int(npair), "max_patch": round(best, 9),
            "normal": best_n, "min_distance": round(float(d["value"]), 9), "tol": tol,
            "note": "逐对共面三角形求 2D 重叠面积后求和；面积 > 0 即面接触，"
                    "边接触与角接触的面积为 0"}
