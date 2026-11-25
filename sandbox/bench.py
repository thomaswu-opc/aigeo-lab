# -*- coding: utf-8 -*-
"""bench.py — 宏观性能：同一批任务重复跑 N 次，看成功率、轮次、成本、失败模式。

为什么要重复跑：单次运行不可作证据。实测同一张卡、同一模型、temperature=0，
T01-3 连跑四次得到 33/24/23/24 轮、4 次里错 1 次（法向取反算出 25°）。
一次成功说明「这条路走得通」，说明不了「稳定走得通」。

判定不看 agent 自己的报告，用独立算法重算：
  * T01-1 / case_C —— roomtruth.expect（墙面覆盖法找门洞、有向轴判方位）
  * T01-3 —— 门扇顶点 SVD 最小奇异向量 vs 墙面厚度轴（与 tools.angle 不同实现）
  * T03 —— 净宽/净深由地板包围盒重算；椅子占地 0.46×0.50
  * T04 —— 椅子占地中心由 roomtruth 给（顶点包围盒中心）
  * T05 / T08 —— 用题面给的相对位移（(0.90,0) / (0.50,0.78) / (-0.15,-0.67)），
    不需要知道墙在哪，只要求内部自洽
  * T02 / T06 / T07 —— 闭式（2.5、1.0、6.0、5.4、18.75、去球冠的球体积）
  * T09 —— 答案里 16 个实体是否逐一给了角色（只核覆盖，不核每个标签对错）
  * T10 —— 只核结构（角度在 (0,180]、有水平位移、拟合残差 ≤5mm）；期望角度要完整复算，
    那一步在 verify_edits 里，所以这一格的强度低于其它题

每题只核「该题最核心、且能独立算出的那一两个量」。完整的 90 项产物验收由
verify_edits.py 负责 —— bench 回答的是「现场重跑还给不给这个结论」，不是「产物每一处都对」。

用法：
  python bench.py                 # 全部任务，默认 3 次
  python bench.py --reps 5
  python bench.py --only T01_1,T01_3
  python bench.py --report        # 只汇总已有 runs/bench_*，不跑
"""
import json
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np                                     # noqa: E402
import roomtruth as RT                                 # noqa: E402

IFC = "../pkg/AI-Geometry-Models/IFC/AI_Geometry_Input.ifc"


def _rows(path):
    return RT.rows_of(os.path.join(HERE, path))


def truth_t01_1():
    r = _rows(IFC)
    e = RT.expect(r, RT.pick(r, "chair"))
    return {"left_right": e["left_right"], "distance_width_m": e["distance_width_m"],
            "front_back": e["front_back"], "distance_depth_m": e["distance_depth_m"]}


def truth_case_c():
    r = _rows(IFC)
    e = RT.expect(r, RT.pick(r, "picture_frame"))
    return {"left_right": e["left_right"], "distance_width_m": e["distance_width_m"],
            "front_back": e["front_back"], "distance_depth_m": e["distance_depth_m"]}


def truth_t01_3():
    """门扇平面与墙面夹角：顶点 SVD 的最小奇异向量 vs 墙面最薄方向。"""
    from core.runtime import TOOLS
    b = RT.bus()
    TOOLS["load"]["fn"](b, os.path.join(HERE, IFC))
    leaf = wall = None
    for S in b.scene:
        r = TOOLS["select"]["fn"](b, [S.sid], {})["items"][0]
        if (not r["axis_aligned"]) and r["min_thickness"] <= 0.06 and r["z_min"] <= 0.05 \
                and r["dims"][0] > 1.5:
            leaf = S
        if r["sz"] >= 1.5 and r["min_thickness"] <= 0.25 and r["d0"] >= 3.0:
            lo, hi = S.bbox()
            if int(np.argmin(hi - lo)) == 0:                 # 法向 ±x 的那面墙
                wall = S
    P = np.asarray(leaf.P, float)
    _, _, vt = np.linalg.svd(P - P.mean(0), full_matrices=False)
    n_leaf = vt[2] / np.linalg.norm(vt[2])
    lo, hi = wall.bbox()
    n_wall = np.zeros(3)
    n_wall[int(np.argmin(hi - lo))] = 1.0
    cos = abs(float(n_leaf @ n_wall))
    return {"angle_deg": round(math.degrees(math.acos(min(1.0, cos))), 3)}


def check_t01_1(v):
    t = truth_t01_1()
    bad = []
    for k in ("left_right", "front_back"):
        if v.get(k) != t[k]:
            bad.append("%s=%s≠%s" % (k, v.get(k), t[k]))
    for k in ("distance_width_m", "distance_depth_m"):
        try:
            if abs(float(v.get(k)) - t[k]) > 5e-3:
                bad.append("%s=%s≠%s" % (k, v.get(k), t[k]))
        except (TypeError, ValueError):
            bad.append("%s 缺失" % k)
    return bad


def check_case_c(v):
    t = truth_case_c()
    bad = []
    for k in ("left_right", "front_back"):
        if v.get(k) != t[k]:
            bad.append("%s=%s≠%s" % (k, v.get(k), t[k]))
    for k in ("distance_width_m", "distance_depth_m"):
        try:
            if abs(float(v.get(k)) - t[k]) > 0.02:
                bad.append("%s=%s≠%s" % (k, v.get(k), t[k]))
        except (TypeError, ValueError):
            bad.append("%s 缺失" % k)
    return bad


def check_t01_3(v):
    t = truth_t01_3()["angle_deg"]
    try:
        got = float(v.get("angle_deg"))
    except (TypeError, ValueError):
        return ["angle_deg 缺失"]
    return [] if abs(got - t) <= 0.05 else ["angle_deg=%.2f≠%.2f" % (got, t)]


def check_t09(v):
    """只核覆盖：16 个 sid 是否都在答案里出现且各带一个说法。

    这张卡没固定字段名（自由中文键），机器只能核"有没有都说到"，
    核不到"每个标签对不对" —— 那是验收层的活（verify_edits 用墙填充率独立复核）。
    """
    blob = json.dumps(v, ensure_ascii=False)
    sids = set(re.findall(r"[CO]\d{3}", blob))
    if len(sids) < 16:
        return ["只提到 %d/16 个实体" % len(sids)]
    return []


# ---------------------------------------------------------------- 其余六题
# 判据取向：每题只核「该题最核心、且能独立算出的那一两个量」。完整的 90 项产物验收
# 由 verify_edits.py 负责；bench 要回答的是「现场重跑还给不给这个结论」，不是「产物每一处都对」。
# 强度分两档：strict = 期望值闭式可算或由 roomtruth 独立重算；loose = 只能核结构/合理范围。

def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def check_t02(v):
    """盒子边长 2.5、球半径 1.0 —— 题面给定参考量，闭式。"""
    bad = []
    for key, want in (("box_edge_m", 2.5), ("sphere_radius_m", 1.0)):
        got = _f(v.get(key))
        if got is None or abs(got - want) > 2e-3:
            bad.append("%s=%s≠%s" % (key, v.get(key), want))
    return bad


def check_t03(v):
    """净宽/净深由地板包围盒独立重算；椅子占地 = 0.46×0.50。"""
    r = _rows(IFC)
    fr = RT.frame(r)
    w, d = fr["w"], fr["d"]
    bad = []
    for key, want in (("net_width_m", fr["hi"][w] - fr["lo"][w]),
                      ("net_depth_m", fr["hi"][d] - fr["lo"][d])):
        got = _f(v.get(key))
        if got is None or abs(got - want) > 2e-3:
            bad.append("%s=%s≠%.3f" % (key, v.get(key), want))
    area = _f(v.get("chair_footprint_area_m2"))
    if area is None or abs(area - 0.46 * 0.50) > 2e-3:
        bad.append("chair_footprint_area_m2=%s≠0.230" % v.get("chair_footprint_area_m2"))
    return bad


def check_t04(v):
    """椅子占地中心由 roomtruth 独立给（顶点包围盒中心）。"""
    r = _rows(IFC)
    c = r[RT.pick(r, "chair")]
    want = (c["cx"], c["cy"])
    got = v.get("footprint_center_world") or v.get("footprint_center_xy_m") or []
    try:
        dx, dy = abs(float(got[0]) - want[0]), abs(float(got[1]) - want[1])
    except (TypeError, ValueError, IndexError):
        return ["footprint_center 缺失"]
    return [] if max(dx, dy) <= 0.01 else [
        "footprint_center=(%.3f, %.3f)≠(%.3f, %.3f)" % (got[0], got[1], *want)]


def check_t05(v):
    """三个位置的相对位移是题面给定的：(0.90, 0) 与 (0.50, 0.78)。

    用「相对原始中心的位移」而不是绝对坐标：不需要知道墙在哪，只要求三处一致。
    """
    base = v.get("chair_center_original_xy_m")
    try:
        bx, by = float(base[0]), float(base[1])
    except (TypeError, ValueError, IndexError):
        return ["chair_center_original_xy_m 缺失"]
    bad = []
    for key, want in (("p2_center_xy_m", (0.90, 0.0)), ("p3_center_xy_m", (0.50, 0.78))):
        pt = v.get(key)
        try:
            dx, dy = float(pt[0]) - bx, float(pt[1]) - by
        except (TypeError, ValueError, IndexError):
            bad.append("%s 缺失" % key)
            continue
        if abs(dx - want[0]) > 3e-3 or abs(dy - want[1]) > 3e-3:
            bad.append("%s 位移=(%.3f, %.3f)≠(%.2f, %.2f)" % (key, dx, dy, *want))
    return bad


def check_t06(v):
    """扩宽到 6.0 / 5.4；窗洞 1.4×1.4 = 1.96 m²（题面给定）。"""
    bad = []
    for key, want in (("net_width_6_m", 6.0), ("net_width_54_m", 5.4)):
        got = _f(v.get(key))
        if got is None or abs(got - want) > 3e-3:
            bad.append("%s=%s≠%s" % (key, v.get(key), want))
    ww, wh = _f(v.get("window_width_m")), _f(v.get("window_height_m"))
    if ww is None or wh is None or abs(ww * wh - 1.96) > 5e-3:
        bad.append("窗洞 %s×%s≠1.4×1.4" % (v.get("window_width_m"), v.get("window_height_m")))
    return bad


def check_t07(v):
    """闭式体积：长方体 2.5×2.5×3.0；球去掉 0.5 高的球冠。"""
    import math as _m
    v_ball = 4 * _m.pi / 3 - _m.pi * 0.5 ** 2 * (3 * 1.0 - 0.5) / 3
    bad = []
    got = _f(v.get("cube_volume_after_m3"))
    if got is None or abs(got - 18.75) > 0.01 * 18.75:
        bad.append("cube_volume_after=%s≠18.75" % v.get("cube_volume_after_m3"))
    got = _f(v.get("ball_volume_after_m3"))
    if got is None or abs(got - v_ball) > 0.03 * v_ball:
        bad.append("ball_volume_after=%s≠%.4f" % (v.get("ball_volume_after_m3"), v_ball))
    return bad


def check_t08(v):
    """椅子从 (0.70, 1.62) 移到 (0.55, 0.95) ⇒ 位移 (-0.15, -0.67)，题面闭式。"""
    sh = v.get("chair_shift_m")
    try:
        dx, dy = abs(float(sh[0]) + 0.15), abs(float(sh[1]) + 0.67)
    except (TypeError, ValueError, IndexError):
        return ["chair_shift_m 缺失"]
    return [] if max(dx, dy) <= 3e-3 else [
        "chair_shift=(%.3f, %.3f)≠(-0.15, -0.67)" % (sh[0], sh[1])]


def check_t10(v):
    """对接的角度/位移要给出且自洽：角度落在 (0,180]、有水平位移、拟合残差小。

    期望角度依赖三个插脚与凹槽的配对，需完整复算（verify_edits 里有）；这里只核结构合理性，
    强度低于上面几题，如实标注。
    """
    bad = []
    ang = _f(v.get("rotation_angle_deg"))
    if ang is None or not (0 < abs(ang) <= 180.001):
        bad.append("rotation_angle_deg=%s 不在 (0,180]" % v.get("rotation_angle_deg"))
    dist = _f(v.get("horizontal_distance_m"))
    if dist is None or dist <= 1e-6:
        bad.append("horizontal_distance_m=%s 缺失或为 0" % v.get("horizontal_distance_m"))
    res = _f(v.get("fit_residual_max_m"))
    if res is not None and res > 5e-3:
        bad.append("fit_residual_max_m=%.4f>5mm" % res)
    return bad


CASES = [
    # (编号, 名称, 卡片, 模型, 判定函数, 说明)
    ("T01_1", "位置关系（椅子 vs 门洞）", "tasks/T01_1.md", IFC, check_t01_1, "读层 . 度量"),
    ("T01_3", "门扇与墙面夹角", "tasks/T01_3.md", IFC, check_t01_3, "读层 . 夹角"),
    ("T09", "16 个实体逐一说角色", "tasks/T09.md", IFC, check_t09, "读层 . 识别"),
    ("CASE_C", "换对象（问画框）", "tasks/gen/case_picture_frame.md", IFC, check_case_c, "泛化 . 换对象"),
    # 其余赛题：T01 已用两个小问代表，T02-T10 各一题
    ("T02", "球与盒子：五个位置", "tasks/T02.md", IFC, check_t02, "读层 . 相交体积"),
    ("T03", "面积与体积清单", "tasks/T03.md", IFC, check_t03, "读层 . 面积体积"),
    ("T04", "家具占地与边距", "tasks/T04.md", IFC, check_t04, "读层 . 占地"),
    ("T05", "椅子三种摆放", "tasks/T05.md", IFC, check_t05, "写层 . 平移"),
    ("T06", "后墙开洞加宽", "tasks/T06.md", IFC, check_t06, "写层 . 开洞"),
    ("T07", "长方体抬高 / 球切顶", "tasks/T07.md", IFC, check_t07, "写层 . 布尔"),
    ("T08", "开门碰椅子", "tasks/T08.md", IFC, check_t08, "写层 . 碰撞"),
    ("T10", "火箭对接", "tasks/T10.md", IFC, check_t10, "写层 . 对接"),
]


def run_one(cid, card, model, i):
    tag = "bench_%s_%d" % (cid, i)
    stale = os.path.join(HERE, "runs", tag, "result.json")
    if os.path.exists(stale):
        os.remove(stale)
    cmd = [sys.executable, "run_agent.py", "--card", card, "--model", model,
           "--llm", "auto", "--tag", tag, "--max-turns", "100",
           "--wall-budget", "420", "--no-video"]
    subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    try:
        return tag, json.load(open(stale, encoding="utf-8"))
    except FileNotFoundError:
        return tag, None


def summarize(rows):
    print("\n| 任务 | 类型 | n | 成功 | 轮次（min/中位/max） | tokens 中位 | 耗时中位 | 失败模式 |")
    print("|---|---|---|---|---|---|---|---|")
    for cid, name, kind, recs in rows:
        if not recs:
            # --report 模式下新加的任务还没有历史运行：写一行「未跑」，
            # 比让它把整张表崩掉有用（第一次扩到 10 题时就是这里 IndexError）。
            print("| %s %s | %s | 0 | **未跑** | — | — | — | 还没有 bench 运行 |" % (
                cid, name, kind))
            continue
        ok = [r for r in recs if r["ok"]]
        turns = sorted(r["turns"] for r in recs)
        toks = sorted(r["tokens"] for r in recs if r["tokens"])
        secs = sorted(r["sec"] for r in recs)
        med = lambda a: a[len(a) // 2] if a else 0
        fails = {}
        for r in recs:
            if not r["ok"]:
                fails[r["why"]] = fails.get(r["why"], 0) + 1
        print("| %s %s | %s | %d | **%d/%d** | %d / %d / %d | %s | %.0fs | %s |" % (
            cid, name, kind, len(recs), len(ok), len(recs),
            turns[0], med(turns), turns[-1],
            ("%.0fk" % (med(toks) / 1000)) if toks else "—", med(secs),
            "；".join("%s×%d" % (k, v) for k, v in fails.items()) or "—"))


def main():
    args = sys.argv[1:]
    reps = 3
    if "--reps" in args:
        reps = int(args[args.index("--reps") + 1])
    only = []
    if "--only" in args:
        only = [x.upper() for x in args[args.index("--only") + 1:] if x]
    report_only = "--report" in args

    rows = []
    for cid, name, card, model, checker, kind in CASES:
        if only and cid not in only:
            continue
        recs = []
        for i in range(1, reps + 1):
            tag = "bench_%s_%d" % (cid, i)
            path = os.path.join(HERE, "runs", tag, "result.json")
            if report_only:
                if not os.path.exists(path):
                    continue
                res = json.load(open(path, encoding="utf-8"))
            else:
                tag, res = run_one(cid, card, model, i)
                if res is None:
                    recs.append({"ok": False, "why": "无产物", "turns": 0, "tokens": 0, "sec": 0})
                    continue
            v = (res.get("answer") or {}).get("values") or {}
            bad = checker(v) if v else ["没有答案"]
            rec = {"ok": not bad and res.get("status") == "OK",
                   "why": ("；".join(bad) if bad else res.get("status")),
                   "turns": res.get("turns") or 0,
                   "tokens": (res.get("llm_usage") or {}).get("total_tokens") or 0,
                   "sec": res.get("elapsed_s") or 0}
            recs.append(rec)
            print("[%s #%d] %s 轮=%s tok=%sk %.0fs" % (
                cid, i, "OK" if rec["ok"] else "FAIL:" + rec["why"][:40],
                rec["turns"], rec["tokens"] // 1000, rec["sec"]), flush=True)
        rows.append((cid, name, kind, recs))

    summarize(rows)


if __name__ == "__main__":
    main()
