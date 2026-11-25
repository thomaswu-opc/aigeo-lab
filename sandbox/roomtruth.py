# -*- coding: utf-8 -*-
"""roomtruth.py — 给「非实题」用例算期望值：不跑 agent，也不看 agent 的答案。

题目形状固定（第 01 题那句问法），但**问谁**、**模型改成什么数值**可以随便换。
换完之后期望值不能靠人记，得当场从这个模型本身算出来 —— 于是有了这个文件。

它与 `verify_edits.py` 里的 T01 那段是同源但独立的实现：
  * 门洞用「墙面顶点网格 + 三角形覆盖判定」算缺料，不用 `openings` 工具；
  * 房间方位不靠世界坐标正负，而是靠几何事实：
      右向 = 从门洞所在那面侧墙指向它对面那面侧墙；
      后向 = 从开口侧（地板上没有墙贴着的那条边）指向有墙的一侧。
    这两条与朝向无关，房间转多少度都成立。
  * 目标中心取顶点包围盒在水平面内的中点（不依赖 footprint）。
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from core.runtime import Bus, Ledger, TOOLS      # noqa: E402
import tools.ops                                 # noqa: E402

_cache_bus = []


def bus():
    if not _cache_bus:
        import tempfile
        _cache_bus.append(Bus(Ledger(os.path.join(tempfile.mkdtemp(), "ledger.jsonl"))))
    return _cache_bus[0]


def rows_of(path):
    """读入模型，返回 sid -> 几何指纹。"""
    b = bus()
    out = TOOLS["load"]["fn"](b, path)
    return {s["sid"]: TOOLS["select"]["fn"](b, [s["sid"]], {})["items"][0]
            for s in out["solids"]}


def room(rows):
    """地板 + 净宽轴 + 净深轴 + 地板包围盒。"""
    floor = None
    best = 0.0
    for sid, r in rows.items():
        if r["dims"][2] <= 0.3 and r["z_min"] < 0 and r["dims"][0] >= 3 and r["dims"][0] > best:
            floor, best = sid, r["dims"][0]
    if floor is None:
        raise RuntimeError("找不到地板（z_min<0 的水平大板）")
    lo = np.array(rows[floor]["bbox_lo"])
    hi = np.array(rows[floor]["bbox_hi"])
    span = hi - lo
    w = int(np.argmax(span[:2]))          # 净宽轴 = 长水平边
    d = 1 - w
    return floor, w, d, lo, hi


def _walls(rows):
    """墙 = 高、薄、且长。只写「高 + 薄」会把门扇、画框也算进来。"""
    return [sid for sid, r in rows.items()
            if r["sz"] >= 1.5 and r["z_min"] <= 0.1
            and r["min_thickness"] <= 0.25 and r["d0"] >= 3.0]


def _plane_dist(wall_row, pt):
    """点到墙面（不是墙的包围盒）的距离：沿墙最薄的那根轴量。"""
    span = np.array(wall_row["bbox_hi"])[:2] - np.array(wall_row["bbox_lo"])[:2]
    ax = int(np.argmin(span))
    plane = (wall_row["bbox_lo"][ax] + wall_row["bbox_hi"][ax]) / 2
    return abs(pt[ax] - plane)


def _edge_cover(rows, lo, hi, axis, side, tol=0.18):
    """地板某条边「被墙贴住」的比例（沿边走 9 个采样点，看有没有墙的平面包围盒盖住）。"""
    walk = 1 - axis
    a = lo[axis] if side == 0 else hi[axis]
    pts = np.linspace(lo[walk], hi[walk], 9)[1:-1]
    covered = 0
    for t in pts:
        p = np.zeros(2)
        p[axis] = a
        p[walk] = t
        for sid in _walls(rows):
            r = rows[sid]
            blo = np.array(r["bbox_lo"])[:2]
            bhi = np.array(r["bbox_hi"])[:2]
            if blo[0] - tol <= p[0] <= bhi[0] + tol and blo[1] - tol <= p[1] <= bhi[1] + tol:
                covered += 1
                break
    return covered / max(1, len(pts))


def door(rows, w, d, lo, hi):
    """门洞 = 某面墙上「通底」的那块缺料（与 tools.openings 不同的实现）。"""
    found = []
    for W in _walls(rows):
        S = next(x for x in bus().scene if x.sid == W)
        blo, bhi = S.bbox()
        t = int(np.argmin(bhi - blo))
        if t == 2:
            continue
        u, v = [i for i in range(3) if i != t]
        P = S.P
        us = np.unique(np.round(P[:, u], 6))
        vs = np.unique(np.round(P[:, v], 6))
        if len(us) < 2 or len(vs) < 2:
            continue
        cu = (us[:-1] + us[1:]) / 2.0
        cv = (vs[:-1] + vs[1:]) / 2.0
        U, V = np.meshgrid(cu, cv, indexing="ij")
        Q = np.stack([U.ravel(), V.ravel()], 1)
        cov = np.zeros(len(Q), bool)
        for f in S.F:
            Tri = P[[int(i) for i in f]][:, [u, v]]
            a0, v0, v1 = Tri[0], Tri[1] - Tri[0], Tri[2] - Tri[0]
            den = v0[0] * v1[1] - v0[1] * v1[0]
            if abs(den) < 1e-15:
                continue
            ww = Q - a0
            sa = (ww[:, 0] * v1[1] - ww[:, 1] * v1[0]) / den
            sb = (v0[0] * ww[:, 1] - v0[1] * ww[:, 0]) / den
            cov |= (sa >= -1e-9) & (sb >= -1e-9) & (sa + sb <= 1 + 1e-9)
        grid = cov.reshape(len(us) - 1, len(vs) - 1)
        seen = np.zeros_like(grid, bool)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] or seen[i, j]:
                    continue
                stack, cells = [(i, j)], []
                seen[i, j] = True
                while stack:
                    a_, b_ = stack.pop()
                    cells.append((a_, b_))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        x, y = a_ + da, b_ + db
                        if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] \
                                and not grid[x, y] and not seen[x, y]:
                            seen[x, y] = True
                            stack.append((x, y))
                ii = [c[0] for c in cells]
                jj = [c[1] for c in cells]
                u0, u1 = float(us[min(ii)]), float(us[max(ii) + 1])
                v0_, v1_ = float(vs[min(jj)]), float(vs[max(jj) + 1])
                found.append({"wall": W, "u": u, "v": v, "urange": (u0, u1),
                              "vrange": (v0_, v1_), "area": (u1 - u0) * (v1_ - v0_),
                              "grounded": abs(v0_ - float(blo[v])) <= 1e-6})
    doors = [f for f in found if f["grounded"] and (f["vrange"][1] - f["vrange"][0]) > 1.5
             and f["area"] > 1.0]
    if len(doors) != 1:
        return None
    h = doors[0]
    rng = {h["u"]: h["urange"], h["v"]: h["vrange"]}
    blo = np.array(rows[h["wall"]]["bbox_lo"])
    bhi = np.array(rows[h["wall"]]["bbox_hi"])

    def ctr(ax):
        return sum(rng[ax]) / 2.0 if ax in rng else float((blo[ax] + bhi[ax]) / 2)

    c3 = np.zeros(3)
    c3[w], c3[d] = ctr(w), ctr(d)
    return {"wall": h["wall"], "center": (float(c3[0]), float(c3[1])),
            "center_wd": (ctr(w), ctr(d))}


def frame(rows):
    """房间自身的两根有向轴：右向、后向（各给一个 +1/-1 的符号，作用在 w / d 上）。"""
    floor, w, d, lo, hi = room(rows)
    dr = door(rows, w, d, lo, hi)
    if dr is None:
        raise RuntimeError("找不到唯一通底洞口")
    wall_c = rows[dr["wall"]]["bbox_lo"][w], rows[dr["wall"]]["bbox_hi"][w]
    floor_c = (lo[w] + hi[w]) / 2
    # 右向 = 从门洞墙指向它对面那面侧墙
    right = +1 if (sum(wall_c) / 2) < floor_c else -1
    # 后向 = 从开口侧指向有墙的那一侧
    cov_lo = _edge_cover(rows, lo, hi, d, 0)
    cov_hi = _edge_cover(rows, lo, hi, d, 1)
    back = +1 if cov_lo <= cov_hi else -1
    return {"floor": floor, "w": w, "d": d, "lo": lo, "hi": hi, "door": dr,
            "right": right, "back": back, "cover": (cov_lo, cov_hi),
            "door_wall": dr["wall"]}


def pick(rows, kind):
    """按几何挑目标。挑不出来就返回 None —— 期望值宁可不给，也不猜。"""
    fr = frame(rows)
    w, d = fr["w"], fr["d"]
    if kind == "chair":
        c = [sid for sid, r in rows.items()
             if r["z_min"] == 0 and r["d0"] <= 1.1 and r["d2"] >= 0.3]
        return c[0] if len(c) == 1 else None
    if kind in ("window_frame", "picture_frame"):
        # 挂在墙上的薄片：板厚薄、不落地、水平尺寸在 0.8~1.6 m
        plates = [sid for sid, r in rows.items()
                  if r["min_thickness"] <= 0.12 and r["z_min"] > 0.3
                  and 0.8 <= r["d0"] <= 1.6]
        cand = []
        for sid in plates:
            r = rows[sid]
            pt = np.array([r["cx"], r["cy"]])
            best = None
            for W in _walls(rows):
                dist = _plane_dist(rows[W], pt)
                if best is None or dist < best[0]:
                    best = (dist, W)
            if best and best[0] <= 0.35:
                cand.append((sid, best[1], rows[best[1]]["fill"]))
        if kind == "window_frame":
            # 窗：贴在一面「有洞、但不是门洞」的墙上；框比玻璃更空（fill 小）
            ok = [c for c in cand if 0.0 < c[2] < 0.95 and c[1] != fr["door_wall"]]
        else:
            # 画：贴在一面整墙上
            ok = [c for c in cand if c[2] >= 0.999]
        if not ok:
            return None
        ok.sort(key=lambda c: rows[c[0]]["fill"])
        return ok[0][0]
    raise KeyError(kind)


def expect(rows, target_sid):
    """期望答案：左右/前后 + 两个方向的距离（沿房间自身的轴）。"""
    fr = frame(rows)
    w, d = fr["w"], fr["d"]
    t = rows[target_sid]
    dc = fr["door"]["center"]                 # 世界 (x, y)
    dw = (t["cx"] - dc[w]) * fr["right"] if w == 0 else (t["cy"] - dc[w]) * fr["right"]
    dd = (t["cx"] - dc[d]) * fr["back"] if d == 0 else (t["cy"] - dc[d]) * fr["back"]
    return {
        "left_right": "右" if dw > 0 else "左",
        "distance_width_m": round(abs(dw), 3),
        "front_back": "后" if dd > 0 else "前",
        "distance_depth_m": round(abs(dd), 3),
        "door_center_xy_m": [round(dc[0], 3), round(dc[1], 3)],
        "target_center_xy_m": [round(t["cx"], 3), round(t["cy"], 3)],
        "axes": {"width": "xy"[w], "right_sign": fr["right"],
                 "depth": "xy"[d], "back_sign": fr["back"]},
    }


if __name__ == "__main__":
    import json
    for p in sys.argv[1:]:
        rws = rows_of(p)
        fr = frame(rws)
        print("==", os.path.basename(p),
              "地板", fr["floor"], "净宽轴", "xy"[fr["w"]], "净深轴", "xy"[fr["d"]],
              "右向", fr["right"], "后向", fr["back"], "边覆盖", fr["cover"],
              "门洞中心", [round(x, 3) for x in fr["door"]["center"]])
        for k in ("chair", "window_frame", "picture_frame"):
            sid = pick(rws, k)
            print("   ", k, "=", sid, json.dumps(expect(rws, sid), ensure_ascii=False) if sid else "（挑不出唯一）")
