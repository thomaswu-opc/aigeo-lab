# -*- coding: utf-8 -*-
"""bench.py — 宏观性能：同一批任务重复跑 N 次，看成功率、轮次、成本、失败模式。

为什么要重复跑：单次运行不可作证据。实测同一张卡、同一模型、temperature=0，
T01-3 连跑四次得到 33/24/23/24 轮、4 次里错 1 次（法向取反算出 25°）。
一次成功说明「这条路走得通」，说明不了「稳定走得通」。

判定不看 agent 自己的报告，用独立算法重算：
  * T01-1 / case_C —— roomtruth.expect（墙面覆盖法找门洞、有向轴判方位）
  * T01-3 —— 门扇顶点 SVD 最小奇异向量 vs 墙面厚度轴（与 tools.angle 不同实现）
  * T09 —— 答案里 16 个实体是否逐一给了角色（只核覆盖，不核每个标签对错）

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


CASES = [
    # (编号, 名称, 卡片, 模型, 判定函数, 说明)
    ("T01_1", "位置关系（椅子 vs 门洞）", "tasks/T01_1.md", IFC, check_t01_1, "读层 . 度量"),
    ("T01_3", "门扇与墙面夹角", "tasks/T01_3.md", IFC, check_t01_3, "读层 . 夹角"),
    ("T09", "16 个实体逐一说角色", "tasks/T09.md", IFC, check_t09, "读层 . 识别"),
    ("CASE_C", "换对象（问画框）", "tasks/gen/case_picture_frame.md", IFC, check_case_c, "泛化 . 换对象"),
]


def run_one(cid, card, model, i):
    tag = "bench_%s_%d" % (cid, i)
    stale = os.path.join(HERE, "runs", tag, "result.json")
    if os.path.exists(stale):
        os.remove(stale)
    cmd = [sys.executable, "run_agent.py", "--card", card, "--model", model,
           "--llm", "auto", "--tag", tag, "--max-turns", "90",
           "--wall-budget", "320", "--no-video"]
    subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    try:
        return tag, json.load(open(stale, encoding="utf-8"))
    except FileNotFoundError:
        return tag, None


def summarize(rows):
    print("\n| 任务 | 类型 | n | 成功 | 轮次（min/中位/max） | tokens 中位 | 耗时中位 | 失败模式 |")
    print("|---|---|---|---|---|---|---|---|")
    for cid, name, kind, recs in rows:
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
