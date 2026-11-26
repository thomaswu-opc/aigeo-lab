# -*- coding: utf-8 -*-
"""family.py — 洞口与「挂在它上面的构件」的归属关系。

为什么不是「重合度」：
    重合是**位置**关系，位置会随开合变。开着的门扇与门洞的重合度接近 0，可它当然是门的一部分。
    用位置描述归属，就必须给「开着」打特例补丁；每来一种新开合状态就再补一个。

这里用的是一条**拓扑**关系：接触 + 连通。
    * 面接触（fill）：构件落在洞口沿墙厚拉伸的棱柱里 —— 玻璃、关着的扇；
    * 线接触（edge）：构件的点或面贴到洞口**边界环**（四条边线段）—— 框、门套，
      以及**开着的扇**（铰链边始终贴边界，开多大都一样）；
    * 连通（link）：与族内成员接触即并入（玻璃经框、把手经扇、扇经门套）。
开合不是特例，它退化成一件可测的事：扇平面与洞口平面的夹角（0 度关、65 度开）。

判定要点：线接触必须双向量 —— 只量「构件顶点到边界线」会漏掉「构件的面穿过边界线」
（门套就是这样：顶点离边界线 0.13 m，但它的面正压在边界线上）。
所以同时算「顶点到线段」与「边界采样点到构件三角面」，取最小。

所有阈值都取**无量纲**形式（线接触按洞口短边比例、连通按毫米），
换单位、换朝向、换模型尺度都不影响判断。
"""
import numpy as np

from core.runtime import tool, TOOLS
from tools.geom import distance_min, _closest_on_tri_batch, tri_arrays
from tools import geom
import tools.ops  # noqa: F401  确保 openings 已注册


def _as_sids(xs):
    """接受 sid 字符串、dict（select/describe 的返回值）、或引用解析后的混合列表。

    实测：模型会把 select 的整份指纹列表直接当 walls 传进来，于是 `sorted(walls)`
    在 dict 之间比大小，抛 TypeError，一步失败 → 它再试三遍 → 四轮白烧。
    这里统一成 sid：不猜内容，只取 sid 字段。
    """
    if xs is None:
        return []
    if isinstance(xs, str):
        xs = [xs]
    out = []
    for x in xs:
        if isinstance(x, dict):
            if "sid" in x:
                out.append(x["sid"])
        elif x is not None:
            out.append(str(x))
    return out


def _solid(bus, sid):
    for S in bus.scene:
        if S.sid == sid:
            return S
    raise KeyError(bus.missing_msg(sid))


def _seg_gap(P, segs):
    """点集到一组线段的最短距离（顶点方向）。"""
    best = np.inf
    for a, b in segs:
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-18:
            d = np.linalg.norm(P - a, axis=1)
        else:
            t = np.clip(((P - a) @ ab) / L2, 0.0, 1.0)
            d = np.linalg.norm(P - (a + t[:, None] * ab), axis=1)
        best = min(best, float(d.min()))
    return best


def _ring_points(corners, n=21):
    """洞口边界环上的采样点（每条边 n 个）。"""
    cs = [np.asarray(c, float) for c in corners]
    out = []
    for i in range(4):
        a, b = cs[i], cs[(i + 1) % 4]
        for t in np.linspace(0.0, 1.0, n):
            out.append(a + t * (b - a))
    return np.array(out)


def _ring_to_mesh_gap(S, ring):
    """边界采样点到构件三角面的最短距离（面方向）。"""
    T = np.stack(tri_arrays(S), axis=1)
    if len(T) == 0:
        return np.inf
    Q, d = _closest_on_tri_batch(ring, T)
    return float(np.min(d))


def _strip_cover(S, ui, vi, u0, u1, v0, h=0.05, nu=24, nv=2):
    """洞口下沿那条窄带被这件构件覆盖的比例（把三角面投影到洞口平面上判）。

    这是"有没有下框"的几何定义：落地窗的下框横跨整个洞宽（覆盖≈1），
    门套只有两根竖梃在两端（覆盖≈0.1），上槛根本不在下沿（覆盖 0）。
    """
    P = np.asarray(S.P, float)
    us = np.linspace(u0, u1, nu)
    vs = np.linspace(v0 + 0.005, v0 + h, nv)
    Q = np.array([[a, b] for a in us for b in vs])
    cov = np.zeros(len(Q), bool)
    for f in S.F:
        Tri = P[[int(i) for i in f]][:, [ui, vi]]
        a0, e1, e2 = Tri[0], Tri[1] - Tri[0], Tri[2] - Tri[0]
        den = e1[0] * e2[1] - e1[1] * e2[0]
        if abs(den) < 1e-15:
            continue
        w = Q - a0
        sa = (w[:, 0] * e2[1] - w[:, 1] * e2[0]) / den
        sb = (e1[0] * w[:, 1] - e1[1] * w[:, 0]) / den
        cov |= (sa >= -1e-9) & (sb >= -1e-9) & (sa + sb <= 1 + 1e-9)
    return float(cov.mean())


def _plate_angle(S, n_dir):
    """构件的最薄方向与洞口法向的夹角（度）。薄板且夹角>10° = 一片"转开了的扇"。

    返回 (angle, is_plate)。夹角 0 = 与洞口平面平行（关着）；65 = 开 65 度。
    """
    P = np.asarray(S.P, float)
    if len(P) < 4:
        return 0.0, False
    Pc = P - P.mean(0)
    try:
        _, sv, vt = np.linalg.svd(Pc, full_matrices=False)
    except Exception:
        return 0.0, False
    nrm = vt[2] / max(np.linalg.norm(vt[2]), 1e-12)
    c = abs(float(nrm @ np.asarray(n_dir, float)))
    ang = float(np.degrees(np.arccos(min(1.0, c))))
    is_plate = bool(sv[2] <= 0.25 * max(sv[0], 1e-9))
    return ang, is_plate


def _families(bus, walls, items, tol_rel, link_mm, vol_ratio, plate_ratio):
    """plate_ratio：洞口所在的板必须比洞口**大得多**，否则它是个框，不是墙。

    窗框、画框本身也是带洞的环 —— 把整个场景当 walls 传进来时，`openings` 会在它们身上
    查出"洞"（窗框中间就是玻璃那块），于是同一个窗被拆成两层族。用面积比把它们挡掉：
    墙的板面积是洞口的 5~8 倍，框只有 1.2 倍。这是比值，换单位换尺度都成立。
    """
    ops, skipped = [], []
    for w in walls:
        try:
            r = TOOLS["openings"]["fn"](bus, w)
        except Exception:
            continue
        plate = 1.0
        try:
            span = np.sort(np.array(r["wall"]["hi"]) - np.array(r["wall"]["lo"]))[[1, 2]]
            plate = float(span[1] * span[0])            # 板的两个面内尺寸之积
        except Exception:
            pass
        for o in r["openings"]:
            if plate < plate_ratio * o["area"]:
                skipped.append({"wall": w, "opening_area": o["area"],
                                "plate_area": round(plate, 3),
                                "why": "板面积不足洞口的 %.1f 倍，判定为「框」而不是「墙」"
                                       % plate_ratio})
                continue
            ops.append({"wall": w, "op": o, "wall_t": r["thickness"]})
    if not ops:
        return [], list(items), skipped

    rows = []
    for sid in items:
        S = _solid(bus, sid)
        lo, hi = S.bbox()
        bv = float(np.prod(np.maximum(hi - lo, 1e-9)))
        try:
            vfill = float(abs(geom.volume(S))) / bv if bv > 0 else 0.0
        except Exception:
            vfill = 0.0
        # 体积门槛用**网格体积**而非包围盒体积：开着的扇包围盒被撑大（0.67 m³），
        # 实际只有 0.07 m³。用包围盒会把门扇挡在族外，而地板（2.3 m³）反而要靠它挡。
        rows.append({"sid": sid, "S": S, "lo": lo, "hi": hi,
                     "vol": bv, "vfill": vfill, "mvol": vfill * bv})

    fams = []
    used = set()
    for e in ops:
        o = e["op"]
        rect = o["rect"]
        u0, u1 = rect["u_range"]
        v0, v1 = rect["v_range"]
        n0, n1 = rect["depth"]
        ui = int(np.argmax(np.abs(rect["u_dir"])))
        vi = int(np.argmax(np.abs(rect["v_dir"])))
        ni = int(np.argmax(np.abs(rect["n_dir"])))
        short = min(u1 - u0, v1 - v0)
        edge_tol = max(tol_rel * short, 1e-3)
        plo = np.array(o["prism"]["lo"], float)
        phi = np.array(o["prism"]["hi"], float)
        cs = [np.asarray(c, float) for c in o["corners"]]
        segs = [(cs[i], cs[(i + 1) % 4]) for i in range(4)]
        ring = _ring_points(o["corners"])
        prism_vol = float(np.prod(np.maximum(phi - plo, 1e-9)))

        members = {}
        for r in rows:
            if r["sid"] == e["wall"]:
                continue
            # 预筛：洞口平面内有交集；墙厚方向放宽 0.4 m（洞口体积盒比墙厚得多）
            if not (r["hi"][ui] >= u0 - edge_tol and r["lo"][ui] <= u1 + edge_tol
                    and r["hi"][vi] >= v0 - edge_tol and r["lo"][vi] <= v1 + edge_tol):
                continue
            if not (r["hi"][ni] >= n0 - 0.4 and r["lo"][ni] <= n1 + 0.4):
                continue
            P = np.asarray(r["S"].P, float)
            inside = float(np.all((P >= plo - 1e-9) & (P <= phi + 1e-9), axis=1).mean())
            in_uv = float(((P[:, ui] >= u0 - edge_tol) & (P[:, ui] <= u1 + edge_tol)
                           & (P[:, vi] >= v0 - edge_tol) & (P[:, vi] <= v1 + edge_tol)).mean())
            gap = min(_seg_gap(P, segs), _ring_to_mesh_gap(r["S"], ring))
            spans_n = bool(r["lo"][ni] <= n0 + edge_tol and r["hi"][ni] >= n1 - edge_tol)
            covers = bool((r["hi"][ui] - r["lo"][ui]) >= 0.95 * (u1 - u0)
                          and (r["hi"][vi] - r["lo"][vi]) >= 0.95 * (v1 - v0))
            # 洞口体积盒：占满洞口 + 贯穿墙厚 + 自身实心。它是洞口的另一种写法，
            # 不参与门窗语义竞争（用「自身填充率」区分它和环形窗框：0.16 对 1.00）
            if covers and spans_n and r["vfill"] >= 0.9:
                rel = "void"
            elif inside >= 0.95:
                rel = "fill"
            elif gap <= edge_tol:
                rel = "edge"
            else:
                continue
            members[r["sid"]] = {"sid": r["sid"], "rel": rel,
                                 "in_prism": round(inside, 4),
                                 "cover_uv": round(in_uv, 4),
                                 "edge_gap": round(float(gap), 4),
                                 "spans_wall": spans_n,
                                 "solid_fill": round(r["vfill"], 3), "via": None}

        # 连通：从直接成员出发，把与族内成员接触的构件并进来（限体积，别把地板串进来）
        fam_lo, fam_hi = plo.copy(), phi.copy()
        for sid in members:
            r = next(x for x in rows if x["sid"] == sid)
            fam_lo = np.minimum(fam_lo, r["lo"])
            fam_hi = np.maximum(fam_hi, r["hi"])
        link = link_mm / 1000.0
        changed = True
        while changed:
            changed = False
            for r in rows:
                if r["sid"] in members or r["sid"] == e["wall"]:
                    continue
                if r["mvol"] > prism_vol * vol_ratio:
                    continue
                if not (np.all(r["hi"] >= fam_lo - link) and np.all(r["lo"] <= fam_hi + link)):
                    continue
                for sid in sorted(members):
                    m = next(x for x in rows if x["sid"] == sid)
                    d = distance_min(r["S"], m["S"])["value"]
                    if d <= link:
                        members[r["sid"]] = {"sid": r["sid"], "rel": "linked",
                                             "in_prism": 0.0, "cover_uv": 0.0,
                                             "edge_gap": round(float(d), 4),
                                             "spans_wall": False,
                                             "solid_fill": round(r["vfill"], 3),
                                             "via": sid}
                        fam_lo = np.minimum(fam_lo, r["lo"])
                        fam_hi = np.maximum(fam_hi, r["hi"])
                        changed = True
                        break

        # 语义范围的收窄就靠这几个几何量：模型不需要再翻遍全场景找"哪个像门"
        rail, rail_sid = 0.0, None
        sash, sash_ang, sash_sid = False, None, None
        op_area = max((u1 - u0) * (v1 - v0), 1e-9)
        for m in members.values():
            r = next(x for x in rows if x["sid"] == m["sid"])
            if m["rel"] == "void":
                # 洞口体积盒把整个洞填满，下沿当然是"全覆盖" —— 它是洞口本身，不是框料
                m["bottom_cover"] = 0.0
                continue
            # 下框：横跨洞宽、且落在洞口下沿那条窄带上（落地窗有，门套只有两端两根竖梃）
            span_u = float(r["hi"][ui] - r["lo"][ui])
            cov = _strip_cover(r["S"], ui, vi, u0, u1, v0) if span_u >= 0.9 * (u1 - u0) else 0.0
            m["bottom_cover"] = round(cov, 3)
            if cov > rail:
                rail, rail_sid = cov, m["sid"]
            # 扇：薄板 + 平面与洞口面夹角≥10° + 投影占洞口面积≥25%
            # （门套也薄、也垂直，但投影只占 5%；窗框/玻璃与洞口共面，夹角 0）
            ang, is_plate = _plate_angle(r["S"], rect["n_dir"])
            m["plane_angle_deg"] = round(ang, 1)
            proj = float(r["hi"][ui] - r["lo"][ui]) * float(r["hi"][vi] - r["lo"][vi])
            m["proj_ratio"] = round(proj / op_area, 3)
            if is_plate and ang >= 10.0 and proj / op_area >= 0.25 and not sash:
                sash, sash_ang, sash_sid = True, round(ang, 1), m["sid"]
        used |= set(members)
        fams.append({"wall": e["wall"],
                     "facts": {"grounded": bool(o["grounded"]),
                               "has_bottom_rail": bool(rail >= 0.8),
                               "bottom_cover": round(rail, 3),
                               "bottom_rail_member": rail_sid if rail >= 0.8 else None,
                               "has_sash": sash, "sash_angle_deg": sash_ang,
                               "sash_member": sash_sid,
                               "has_void_box": any(m["rel"] == "void" for m in members.values()),
                               "n_members": len(members)},
                     "opening": {k: o[k] for k in ("center", "width", "height", "area",
                                                   "sill_z", "grounded", "touches_top",
                                                   "corners", "prism")},
                     "members": sorted(members.values(),
                                       key=lambda m: (-m["in_prism"], m["edge_gap"])),
                     "n": len(members),
                     "tol_rel": tol_rel, "edge_tol": round(edge_tol, 4)})
    loose = [r["sid"] for r in rows if r["sid"] not in used]
    return fams, loose, skipped


@tool("assembly",
      "把洞口和「挂在它上面的构件」认成族。归属用接触+连通判，不用重合判："
      "面接触（构件落在洞口棱柱内，in_prism）、线接触（构件点或面贴到洞口边界环，"
      "edge_gap —— 开着的扇靠铰链边也在这里）、连通（与族内成员接触即并入，via）。"
      "占满洞口且贯穿墙厚的实心盒体标 void（洞口的另一种写法），环形框不会被误标。",
      {"walls": "list 板状构件（墙/板），从它们身上找洞口",
       "items": "list 可选，候选构件清单（默认整个场景）",
       "tol_rel": "float 线接触阈值，按洞口短边比例（默认 0.02 = 2%）",
       "link_mm": "float 连通阈值（mm，默认 15）：装配件之间的功能缝（铰链、门套与墙）"
                  "在 10 mm 量级；5 mm 会把开着的门扇挡在族外（实测 8.8 mm）",
       "vol_ratio": "float 连通时允许的最大**网格体积**比（相对洞口棱柱，默认 1.0）",
       "plate_ratio": "float 洞口所在的板面积至少要大于洞口面积的几倍（默认 2.0）。"
                      "窗框/画框本身也是带洞的环，不挡掉会把一个窗拆成两层族"},
      pre="walls 里至少有一面能查出洞口（先用 openings 确认）",
      degrade="查不到洞口时返回空族；曲面构件上的开口无定义，会被跳过",
      returns="{families:[{wall, opening{center,width,height,sill_z,grounded,corners,prism}, "
              "members:[{sid, rel(fill|edge|void|linked), in_prism, cover_uv, edge_gap, "
              "spans_wall, solid_fill, via}], n}], loose:[不属于任何洞口的构件], "
              "skipped:[被面积比挡掉的「框上的洞」], n}",
      unit="m / m2")
def t_assembly(bus, walls, items=None, tol_rel=0.02, link_mm=15.0, vol_ratio=1.0,
               plate_ratio=2.0):
    """带记忆：同一组墙 + 同一组参数算过一次就复用。

    `items` 只影响 loose 清单，不影响族（族由洞口和接触决定）。实测模型会拿不同的
    items 子集反复调同一个族查询（一次运行 10 次），每次重算一遍接触图纯属浪费。
    """
    walls = _as_sids(walls)
    items = ([S.sid for S in bus.scene] if items is None else _as_sids(items))
    memo = getattr(bus, "_famemo", None)
    if memo is None:
        memo = bus._famemo = {}
    key = (tuple(sorted(walls)), round(float(tol_rel), 6), round(float(link_mm), 6),
           round(float(vol_ratio), 6), round(float(plate_ratio), 6),
           getattr(bus, "gen", 0))
    hit = memo.get(key)
    if hit is not None:
        fams, loose_all, skipped = hit
        keep = set(items)
        loose = [x for x in loose_all if x in keep]
        return {"families": fams, "loose": loose, "n": len(fams), "skipped": skipped,
                "memo": "同样的墙与参数已经算过，直接复用（items 只影响 loose 清单）",
                "params": {"tol_rel": tol_rel, "link_mm": link_mm,
                           "vol_ratio": vol_ratio, "plate_ratio": plate_ratio}}
    fams, loose, skipped = _families(bus, walls, list(items), float(tol_rel),
                                     float(link_mm), float(vol_ratio), float(plate_ratio))
    memo[key] = (fams, loose, skipped)
    return {"families": fams, "loose": loose, "n": len(fams), "skipped": skipped,
            "params": {"tol_rel": tol_rel, "link_mm": link_mm,
                       "vol_ratio": vol_ratio, "plate_ratio": plate_ratio}}
