# -*- coding: utf-8 -*-
"""measure.py — 反解与拟合：给定结果求参数，给定网格求理想量。

这一层是「正向度量」的逆运算。放进来的三个工具有一个共同点：
**答案不是量出来的，是解出来的**，所以每个都必须同时交代「解在哪儿」「有解吗」
「残差多大」—— 只给一个数，读的人无法判断它是不是二分跑飞之后的产物。

  * `fit_sphere`          —— 网格顶点 → 最小二乘球（中心、半径、残差）。
                             用来回答「按理想球体计算是多少」：半径不是抄题面的 1 m，
                             是从这一份网格里拟合出来的，残差说明它到底有多像球。
  * `proj_area`           —— 一组实体沿某轴投影的**凸包面积**。
                             比包围矩形的面积小、比逐个部件相加小：
                             前者把空缺算成实体，后者把重叠算两次。
  * `solve_section_height` —— 给定目标面积，求截面高度（二分）。
                             先扫一遍看单调性，不单调就报出来，不闷头二分。

三个工具都不含任务知识：`solve_section_height` 不知道自己要找的是球缺，
`fit_sphere` 也不知道自己在量的是赛题里的球。
"""
import math

import numpy as np

from core.runtime import tool
from tools.geom import describe as geom_describe
from tools.shape import _hull2, _solid_of


def _uniq_pts(P, tol=1e-9):
    """焊接重复顶点。三角化会把同一个角点重复存进多个三角形，不去重会把
    顶点密度当成权重 —— 拟合出来的球心会被「三角形多的地方」拉偏。"""
    q = np.round(np.asarray(P, float) / tol).astype(np.int64)
    _, idx = np.unique(q, axis=0, return_index=True)
    return np.asarray(P, float)[np.sort(idx)]


@tool("fit_sphere",
      "把实体顶点最小二乘拟合成一个球：返回球心、半径、残差（rms / max），"
      "以及由该半径算出的理想体积与理想表面积。用于「按理想球体计算」这类对照。",
      {"sid": "str  实体",
       "tol": "float  顶点焊接容差（默认 1e-9 m）"},
      pre="实体开口或闭合均可；顶点足以约束一个球",
      degrade="实体不是球（被切平、被拉伸）时残差会显著变大；残差大即拟合不成立，不是球不好",
      returns="{center, radius, ideal_volume, ideal_area, rms, max, n_pts, roundness}",
      unit="m")
def t_fit_sphere(bus, sid, tol=1e-9):
    S = _solid_of(bus, sid)
    P = _uniq_pts(S.P, tol)
    if len(P) < 4:
        return {"ok": False, "error": "顶点少于 4 个，无法约束一个球", "n_pts": int(len(P))}
    # |p|² = 2 c·p + (r² − |c|²)  →  对 (c, k) 是线性的，闭式最小二乘
    A = np.c_[2.0 * P, np.ones(len(P))]
    b = (P ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    k = float(sol[3])
    r2 = k + float(c @ c)
    if r2 <= 0:
        return {"ok": False, "error": "拟合半径的平方为负：这些点不构成球", "n_pts": int(len(P))}
    r = math.sqrt(r2)
    d = np.linalg.norm(P - c, axis=1)
    res = np.abs(d - r)
    # 对外报告的半径先取整，理想量必须用**同一个取整后的半径**算。
    # 否则调用方拿 radius=1.0 去算球体积，本工具用 0.9999997 算理想体积，
    # 两者差 3e-6 m³ —— 看起来像积分不准，其实是自己和自己不一致。
    r_rep = round(float(r), 9)
    return {
        "ok": True,
        "center": [round(float(x), 6) for x in c],
        "radius": r_rep,
        # 理想球的两个量都由**拟合半径**算出，不抄题面 —— 这样「网格偏小多少」
        # 才是两个独立来源的比较，而不是同一个数自己减自己。
        "ideal_volume": round(float(4.0 / 3.0 * math.pi * r_rep ** 3), 9),
        "ideal_area": round(float(4.0 * math.pi * r_rep ** 2), 9),
        "rms": round(float(np.sqrt((res ** 2).mean())), 9),
        "max": round(float(res.max()), 9),
        "n_pts": int(len(P)),
        # 残差相对半径的比例：这个数 >1e-3 就说明网格根本不是球，
        # 下面的 ideal_volume 也就不该被当作对照基准。
        "roundness": round(float(res.max() / r), 9),
        "method": "代数最小二乘（|p|² 对 c 与 r²−|c|² 线性），残差为顶点到球面的距离",
        "limits": "只对整球成立；被削平或拉长的实体先看 roundness，再决定要不要用",
    }


@tool("proj_area",
      "一组实体沿某轴投影的**凸包面积**，同时返回包围矩形面积作对照。"
      "用于「占地」这类平面量：凸包不算空隙、也不重复算重叠。",
      {"sids": "list  实体（可含多个部件）",
       "axis": "str  投影方向（默认 'z'，即水平占地）"},
      pre="实体顶点已焊接（内部自动去重）",
      degrade="凸包会把凹形占地的凹处填上；凹形明显的实体要另看轮廓",
      returns="{area, hull_n, bbox_area, bbox_size, fill_ratio}",
      unit="m²")
def t_proj_area(bus, sids, axis="z"):
    ai = {"x": 0, "y": 1, "z": 2}[axis.lower()] if isinstance(axis, str) else int(axis)
    bi, ci = [i for i in (0, 1, 2) if i != ai]
    from tools.edit import _flatten_sids
    # 实体清单在观察里是 [{"sid": ...}, ...]，直接调用时也可能只给一个 sid 字符串。
    # 两种都要认，否则「传了但没生效」会表现成「这一步没有记录」而不是报错。
    Ps = []
    for sid in _flatten_sids(sids):
        Ps.append(np.asarray(_solid_of(bus, sid).P, float))
    if not Ps:
        return {"ok": False, "error": "没有实体", "area": 0.0}
    P = _uniq_pts(np.vstack(Ps))
    Q = P[:, [bi, ci]]
    idx = _hull2(Q)
    H = Q[idx]
    x, y = H[:, 0], H[:, 1]
    area = abs(float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))
    lo, hi = Q.min(0), Q.max(0)
    size = hi - lo
    bbox_area = float(size[0] * size[1])
    return {
        "ok": True,
        "area": round(area, 6),
        "hull_n": int(len(H)),
        "hull": [[round(float(a), 6), round(float(b), 6)] for a, b in H],
        "bbox_area": round(bbox_area, 6),
        "bbox_size": [round(float(v), 6) for v in size],
        # 凸包 / 包围矩形：这个比值把「包围盒高估了多少」变成一个可核对的数。
        "fill_ratio": round(float(area / bbox_area) if bbox_area > 1e-15 else 0.0, 6),
        "axes": ["xyz"[bi], "xyz"[ci]],
        "n_pts": int(len(P)),
    }


@tool("solve_section_height",
      "给定目标截面面积，求该面积出现在哪个高度（沿轴二分）。先扫一遍判单调性，"
      "不单调时把跨越点全部列出，而不是二分出一个看似合理的数。",
      {"sid": "str  实体", "target": "float  目标面积", "axis": "str  轴（默认 'z'）",
       "lo": "float  搜索下界（默认实体最小坐标）", "hi": "float  搜索上界（默认最大坐标）",
       "samples": "int  单调性预扫点数（默认 33）", "tol": "float  面积收敛容差（默认 1e-6）"},
      pre="截面面积在搜索区间内单调（预扫会核对）",
      degrade="非单调时返回 monotonic=false 与全部跨越点，height 取最靠近 lo 的那一段的解",
      returns="{height, area, n_iter, monotonic, crossings, bracket, samples}",
      unit="m")
def t_solve_section_height(bus, sid, target, axis="z", lo=None, hi=None,
                           samples=33, tol=1e-6):
    from tools import geom
    S = _solid_of(bus, sid)
    ai = {"x": 0, "y": 1, "z": 2}[axis.lower()] if isinstance(axis, str) else int(axis)
    P = np.asarray(S.P, float)
    a_lo = float(P[:, ai].min()) if lo is None else float(lo)
    a_hi = float(P[:, ai].max()) if hi is None else float(hi)
    tgt = float(target)

    def A(h):
        return float(geom.section_area(S, ai, h)["value"])

    xs = np.linspace(a_lo, a_hi, int(samples))
    ys = [A(x) for x in xs]
    # 单调性：扫一遍看曲线有没有回头。回头就说明「面积=目标」不止一处，
    # 单一二分给的那个数只是其中一处，不说明是哪一处。
    dif = np.diff(ys)
    rising = bool((dif >= -1e-9).all())
    falling = bool((dif <= 1e-9).all())
    crossings = []
    for i in range(len(xs) - 1):
        if (ys[i] - tgt) * (ys[i + 1] - tgt) <= 0 and abs(ys[i + 1] - ys[i]) > 1e-12:
            crossings.append([round(float(xs[i]), 6), round(float(xs[i + 1]), 6)])

    # 只取第一个跨越段做二分 —— 后面的段会在 crossings 里如实列出
    bracket = crossings[0] if crossings else [a_lo, a_hi]
    l, h = bracket
    fl, fh = A(l), A(h)
    n_iter = 0
    if (fl - tgt) * (fh - tgt) > 0:
        return {"ok": False, "height": None, "area": None, "n_iter": 0,
                "monotonic": bool(rising or falling), "crossings": crossings,
                "bracket": [round(l, 6), round(h, 6)],
                "error": "搜索区间内没有达到目标面积的截面",
                "samples": [[round(float(x), 6), round(float(y), 6)] for x, y in zip(xs, ys)]}
    for n_iter in range(1, 81):
        m = 0.5 * (l + h)
        fm = A(m)
        if abs(fm - tgt) <= tol or (h - l) <= 1e-9:
            break
        if (fl - tgt) * (fm - tgt) <= 0:
            h, fh = m, fm
        else:
            l, fl = m, fm
    m = 0.5 * (l + h)
    return {
        "ok": True,
        "height": round(float(m), 9),
        # 报「相对最低点的偏移」这件事交给调用方：这里只说绝对高度，
        # 因为「最低点」是另一个工具的事（fit/bbox），不该在这一层混进来。
        "area": round(float(A(m)), 9),
        "n_iter": int(n_iter),
        "monotonic": bool(rising or falling),
        "direction": "递增" if rising else ("递减" if falling else "非单调"),
        "crossings": crossings,
        "n_solutions_brackets": len(crossings),
        "bracket": [round(float(bracket[0]), 6), round(float(bracket[1]), 6)],
        "search": [round(a_lo, 6), round(a_hi, 6)],
        "samples": [[round(float(x), 6), round(float(y), 6)] for x, y in zip(xs, ys)],
        "method": "预扫 %d 点判单调 → 首个跨越段内二分（并列解全部列出）" % samples,
        "limits": "非单调时只解第一处；其余跨越段在 crossings 里，需逐段再解",
    }


@tool("calc",
      "两个数的四则运算。派生量（面积相减、体积相对差、高度偏移）必须由它算出来，"
      "这样结果才有工具出处，而不是报告里凭空出现的一个数。",
      {"a": "float  左操作数", "op": "str  + - * /", "b": "float  右操作数",
       "unit": "str  结果单位（可选，仅记录）"},
      pre="op ∈ {+,-,*,/}；除法时 b≠0",
      degrade="除数为 0 时返回 ok=false，不给 inf",
      returns="{value, op, a, b, unit}",
      unit="同输入")
def t_calc(bus, a, op, b, unit=""):
    a, b = float(a), float(b)
    if op == "+":
        v = a + b
    elif op == "-":
        v = a - b
    elif op == "*":
        v = a * b
    elif op == "/":
        if abs(b) < 1e-15:
            return {"ok": False, "error": "除数为 0", "a": a, "b": b, "op": op}
        v = a / b
    else:
        return {"ok": False, "error": "不支持的运算 %r（只接受 + - * /）" % op}
    return {"ok": True, "value": round(float(v), 9), "op": op, "a": a, "b": b,
            "unit": unit, "note": "派生量：value = a %s b" % op}


@tool("angle",
      "两个平面（或两根轴）之间的夹角，单位度。给实体时取它的主平面法向 —— 即最大共面区域的"
      "朝外法向（开门后的门扇就是这样拿到「门扇所在平面」的）。也可直接给两个 [x,y,z] 向量。"
      "返回值同时给 dot 与两向夹角、补角：平面之间用 acute_deg，别自己再取一次补角。",
      {"a": "list  实体 sid（字符串），或 [x,y,z] 方向向量",
       "b": "list  同上",
       "kind": "str  'plane' 平面法向（默认） / 'axis' 轴向；两者都返回法向夹角与补角"},
      pre="a / b 是场景里已加载的实体，或长度 3 的非零向量",
      degrade="曲面为主（dominant_share 小）时主法向不可靠，share 会一起返回；"
              "近乎平行（dot≈±1）或垂直（dot≈0）时夹角对数值噪声敏感，用 dot 复核",
      returns="{angle_deg, acute_deg, dot, n_a, n_b, share_a, share_b, kind, note}",
      unit="deg")
def t_angle(bus, a, b, kind="plane"):
    def pick(x):
        """实体 → 主平面法向；向量 → 原样归一化。返回 (n, share, src)。"""
        if isinstance(x, str) and x.strip().startswith("["):
            import json as _json
            x = _json.loads(x)
        if not isinstance(x, str) and hasattr(x, "__len__"):
            v = np.asarray([float(t) for t in x], float)
            nv = np.linalg.norm(v)
            if nv < 1e-12:
                raise ValueError("向量为零")
            return v / nv, None, "vector"
        S = _solid_of(bus, str(x))
        d = geom_describe(S)
        dom = d.get("dominant") or []
        if not dom:
            raise ValueError("%s 上没有可用的主平面" % x)
        n = np.asarray(dom[0]["n"], float)
        nv = np.linalg.norm(n)
        if nv < 1e-12:
            raise ValueError("%s 的主法向为零" % x)
        share = float(dom[0]["area"]) / max(float(d.get("area") or 0.0), 1e-12)
        return n / nv, share, "dominant0(%s)" % x

    n_a, s_a, src_a = pick(a)
    n_b, s_b, src_b = pick(b)
    dot = float(np.clip(float(n_a @ n_b), -1.0, 1.0))
    ang = math.degrees(math.acos(dot))
    acute = min(ang, 180.0 - ang)
    return {
        "angle_deg": round(ang, 6),
        "acute_deg": round(acute, 6),
        "dot": round(dot, 9),
        "n_a": [round(float(v), 6) for v in n_a],
        "n_b": [round(float(v), 6) for v in n_b],
        "share_a": None if s_a is None else round(s_a, 4),
        "share_b": None if s_b is None else round(s_b, 4),
        "kind": kind,
        "note": ("法向取自：%s / %s。两个平面之间的较小夹角 = acute_deg；"
                 "angle_deg 是两根法向的有向夹角（可能 >90°）" % (src_a, src_b)),
    }
