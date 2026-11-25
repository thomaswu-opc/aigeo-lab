# -*- coding: utf-8 -*-
"""try_cases.py — 非实题用例：换「问谁」、换模型里的数值，看同一套流程还成不成立。

这里刻意不用赛题答案：每一例的期望值都由 `roomtruth.py` 从**当前这个模型**重算，
算法与 agent 无关（门洞用墙面覆盖法、方位用「右向=门洞墙→对面墙」「后向=开口→有墙」）。

用例形状完全一样（第 01 题那句问法），变的是：
  A 基准          —— 原件，问椅子（对照，跑不过就没必要看别的）
  B 换对象        —— 问窗框
  C 换对象        —— 问画框
  D 换数值        —— 把椅子挪到另一处（现造变体），期望值随之改变
  E 换单位        —— 同一模型 ×1000（mm），答案必须还是米
  F 换朝向        —— 绕地板前左角转 90°，四个结论都该一字不变

用法：
  python try_cases.py                 # 跑全部（需要 LLM key）
  python try_cases.py --only A C      # 只跑两例
  python try_cases.py --truth-only     # 只算期望值，不跑 agent
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import roomtruth as RT                            # noqa: E402

BASE = "../pkg/AI-Geometry-Models/IFC/AI_Geometry_Input.ifc"
CARD_SRC = os.path.join(HERE, "tasks", "T01_1.md")
GEN = os.path.join(HERE, "tasks", "gen")

TARGETS = {
    "chair": ("椅子", "椅子占地中心：椅子在**水平面上的最小包围矩形**中心（用 `footprint` 取；"
                     "不要用三维包围盒的中心 —— 椅子有靠背或扶手时，这两个点不是同一个点）。"),
    "window_frame": ("窗框", "窗框中心：窗户的框（不是玻璃）在水平面内的包围盒中心。"),
    # C 例实测：模型把「门洞上方的横档」当成了画框（都是不落地的小薄片）。
    # 这是对象指代的歧义，不是几何判据的问题 —— 题面得说清指的是哪一幅画。
    "picture_frame": ("画框", "画框中心：挂在房间墙上的那一幅画（框与画心都算，取两者合并的包围盒）在水平面内的中心。\n"
                            "  这幅画大约 1 m 见方（水平尺寸 1.0~1.4 m），离地 1 m 以上；\n"
                            "  门、门上的横档、门把手、窗、窗玻璃都不是画框 —— 那些在门洞或窗洞那一带上。"),
}


def build_cards():
    """从任务卡模板生成用例卡。每一处替换都断言命中，避免模板改了而生成物悄悄走偏。"""
    src = open(CARD_SRC, encoding="utf-8").read()
    os.makedirs(GEN, exist_ok=True)
    paths = {}
    for kind, (label, bullet) in TARGETS.items():
        s = src
        reps = [
            ("# 任务卡 · T-01-1", "# 任务卡 · 泛化用例（问「%s」）" % label),
            ("模型：房间（三面墙、地板、门洞和打开的门、窗户、画框，以及一张椅子）。",
             "模型：房间（三面墙、地板、门洞和打开的门、窗户、画框，以及各种家具）。"),
            ("报告椅子占地中心在其左侧还是右侧", "报告%s在门洞中心的左侧还是右侧" % label),
            ("不是给椅子贴一个房间方位标签", "不是给%s贴一个房间方位标签" % label),
            ("    椅子沿右向的坐标比门洞中心大 ⇒ 椅子在门洞的**右侧**",
             "    %s沿右向的坐标比门洞中心大 ⇒ 它在门洞的**右侧**" % label),
            ("    椅子沿后向的坐标比门洞中心大 ⇒ 椅子在门洞的**后方**",
             "    %s沿后向的坐标比门洞中心大 ⇒ 它在门洞的**后方**" % label),
            ("- 椅子占地中心：椅子在**水平面上的最小包围矩形**中心（用 `footprint` 取；\n"
             "  不要用三维包围盒的中心 —— 椅子有靠背或扶手时，这两个点不是同一个点）。",
             "- " + bullet),
            ("说明门洞与椅子分别是如何", "说明门洞与%s分别是如何" % label),
            # 实测 A 例：门洞的 x 取了洞口、y 却取了墙面中心，于是两个距离里有一个是错的。
            # 「两个分量同来源」是口径问题，不是知识泄露，写进卡里。
            ("  竖直方向的高度不参与本题的两个距离。",
             "  竖直方向的高度不参与本题的两个距离。\n"
             "  `x`、`y` 两个分量必须取自**同一次**返回里的同一个点（例如同一条通底洞口的 center）；\n"
             "  不要把其中一个分量换成墙的包围盒中心 —— 墙中心与洞口中心通常不是同一个点。"),
            ("| `left_right` | `\"左\"` 或 `\"右\"` |",
             "| `left_right` | `\"左\"` 或 `\"右\"` |\n"
             "| `offset_along_right_m` | 数，**带符号**：正=在右侧，负=在左侧 |"),
            ("| `front_back` | `\"前\"` 或 `\"后\"` |",
             "| `front_back` | `\"前\"` 或 `\"后\"` |\n"
             "| `offset_along_back_m` | 数，**带符号**：正=在后方，负=在前方 |"),
            ("| `chair_center_floor_xy_m` | `[x, y]`，椅子占地中心 |",
             "| `target_center_floor_xy_m` | `[x, y]`，%s中心 |" % label),
        ]
        for a, b in reps:
            assert a in s, "模板里找不到：%r" % a[:40]
            s = s.replace(a, b)
        p = os.path.join(GEN, "case_%s.md" % kind)
        open(p, "w", encoding="utf-8").write(s)
        paths[kind] = p
    return paths


def make_chair_variant():
    """把椅子平移 (0.6, 1.2, 0)，落成 gltf：这是一处真实的数值改动，不是换个问法。"""
    out = os.path.join(HERE, "variants", "room_chair_moved.gltf")
    if os.path.exists(out):
        return out
    from core.runtime import TOOLS
    import tools.ops, tools.edit                       # noqa: F401
    b = RT.bus()
    TOOLS["load"]["fn"](b, BASE)
    rows = {S.sid: TOOLS["select"]["fn"](b, [S.sid], {})["items"][0] for S in b.scene}
    chair = [sid for sid, r in rows.items()
             if r["z_min"] == 0 and r["d0"] <= 1.1 and r["d2"] >= 0.3]
    assert len(chair) == 1, "椅子候选不唯一：%s" % chair
    TOOLS["translate"]["fn"](b, chair, [0.6, 1.2, 0.0])
    TOOLS["emit"]["fn"](b, out, None, "gltf")
    return out


CASES = [
    # (编号, 用例名, 模型, 目标类型, 说明)
    ("A", "基准 · 问椅子", BASE, "chair", "对照：跑不过这例，后面的都不用看"),
    ("B", "换对象 · 问窗框", BASE, "window_frame", "同一把问题，换问另一件东西"),
    ("C", "换对象 · 问画框", BASE, "picture_frame", "画框贴在整墙上，位置更容易混淆"),
    ("D", "换数值 · 椅子挪位", None, "chair", "椅子平移 (0.6, 1.2)，期望值随之改变"),
    ("E", "换单位 · mm", "variants/obj_mm.obj", "chair", "坐标 ×1000，答案仍是米"),
    ("F", "换朝向 · 转 90°", "variants/room_rot90_corner.obj", "chair", "刚体旋转，四个结论都不该变"),
]

def tag_of(cid):
    return "case_%s" % cid


TOL = 0.02          # 目标中心取的是包围盒中心：框、玻璃、画心之间差 1 cm 量级


def run_case(cid, name, model, kind, why, card):
    truth = RT.expect(RT.rows_of(os.path.join(HERE, model)), RT.pick(RT.rows_of(os.path.join(HERE, model)), kind))
    tag = tag_of(cid)
    stale = os.path.join(HERE, "runs", tag, "result.json")
    if os.path.exists(stale):
        os.remove(stale)
    # 用例本身比单题更长（要认一件新东西），把轮次与墙钟放宽，免得测到的是时钟而不是推理
    cmd = [sys.executable, "run_agent.py", "--card", card, "--model", model,
           "--llm", "auto", "--tag", tag, "--max-turns", "120",
           "--wall-budget", "300", "--no-video"]
    subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    try:
        res = json.load(open(stale, encoding="utf-8"))
    except FileNotFoundError:
        return truth, None, "无产物"
    return judge(truth, res)


def judge(truth, res):
    """只比对，不跑。判定分三层：数值、标签、标签与符号是否自洽。"""
    v = (res.get("answer") or {}).get("values") or {}
    got = {"left_right": v.get("left_right"),
           "distance_width_m": v.get("distance_width_m"),
           "front_back": v.get("front_back"),
           "distance_depth_m": v.get("distance_depth_m"),
           "offset_along_right_m": v.get("offset_along_right_m"),
           "offset_along_back_m": v.get("offset_along_back_m")}
    bad = []
    if got["left_right"] != truth["left_right"]:
        bad.append("左右 %s≠%s" % (got["left_right"], truth["left_right"]))
    if got["front_back"] != truth["front_back"]:
        bad.append("前后 %s≠%s" % (got["front_back"], truth["front_back"]))
    for k in ("distance_width_m", "distance_depth_m"):
        if got[k] is None or abs(float(got[k]) - truth[k]) > TOL:
            bad.append("%s %s≠%s" % (k.replace("distance_", "").replace("_m", ""),
                                     got[k], truth[k]))
    # 标签必须与符号一致：这一条把「算对了但标签写反」单独拎出来，不让它混进「量错了」里
    for key, lab, neg in (("offset_along_right_m", "left_right", "左"),
                          ("offset_along_back_m", "front_back", "前")):
        val = got.get(key)
        if val is None:
            continue
        want = neg if float(val) < 0 else ("右" if key.endswith("right_m") else "后")
        if got.get(lab) != want:
            bad.append("%s 与符号矛盾（%s=%s，却写 %s）" % (lab, key, val, got.get(lab)))
    return truth, got, ("；".join(bad) if bad else "一致")


def main():
    args = sys.argv[1:]
    only = []
    if "--only" in args:
        only = [x.upper() for x in args[args.index("--only") + 1:] if len(x) == 1]
    truth_only = "--truth-only" in args
    report_only = "--report-only" in args
    cards = build_cards()
    moved = make_chair_variant()

    rows = []
    for cid, name, model, kind, why in CASES:
        if only and cid not in only:
            continue
        m = model or moved
        if truth_only:
            t = RT.expect(RT.rows_of(os.path.join(HERE, m)),
                          RT.pick(RT.rows_of(os.path.join(HERE, m)), kind))
            rows.append((cid, name, t, None, why))
            continue
        if report_only:
            p = os.path.join(HERE, "runs", tag_of(cid), "result.json")
            if not os.path.exists(p):
                t = RT.expect(RT.rows_of(os.path.join(HERE, m)),
                              RT.pick(RT.rows_of(os.path.join(HERE, m)), kind))
                rows.append((cid, name, t, None, "未跑"))
                continue
            t, got, verdict = judge(RT.expect(RT.rows_of(os.path.join(HERE, m)),
                                              RT.pick(RT.rows_of(os.path.join(HERE, m)), kind)),
                                    json.load(open(p, encoding="utf-8")))
        else:
            t, got, verdict = run_case(cid, name, m, kind, why, cards[kind])
        rows.append((cid, name, t, got, verdict))
        print("[%s] %-14s %s" % (cid, name, verdict), flush=True)

    print("\n| 用例 | 期望（左右/净宽、前后/净深） | 实测 | 判定 |")
    print("|---|---|---|---|")
    for cid, name, t, got, verdict in rows:
        exp = "%s / %.3f，%s / %.3f" % (t["left_right"], t["distance_width_m"],
                                       t["front_back"], t["distance_depth_m"])
        if got is None:
            got_s = "（未跑）" if verdict != "无产物" else "无产物"
        else:
            got_s = "%s / %s，%s / %s" % (got["left_right"], got["distance_width_m"],
                                          got["front_back"], got["distance_depth_m"])
        print("| %s %s | %s | %s | %s |" % (cid, name, exp, got_s, verdict))


if __name__ == "__main__":
    main()
