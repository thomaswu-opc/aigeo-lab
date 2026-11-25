# -*- coding: utf-8 -*-
"""brief.py — 把一句自然语言编译成「本轮任务规格 + 注入 system 的说明」。

这里刻意不做成表单。用户唯一要给的只有两样：一段话、一个模型文件。
把「这段话到底要什么」读出来，是这条管线自己的活，读出来的结构也只活在本轮里：

    原话 ──compile_brief──> 结构化 brief（中间产物，可查、可存进 run 目录）
                              └──render_prompt──> 注入 LLM 的 system 说明

为什么不让人填结构化字段：
  1. 结构化字段会替 agent 完成「读懂题目」这一步，而那正是要考的能力；
  2. 一旦允许填表，题目的含义就混进了人的先验，答对了也说不清是谁答的。

所以这里的抽取是**规则**，不是第二次 LLM 调用：输入什么、抽出什么，可以逐条复现，
也不会在题目之外偷偷补一句「注意这是门」。

抽取失败不许编。认不出交付类型就退回「按原话执行」，认不出量就不列量 ——
给一个错的规格，比不给规格更坏。
"""
import re

# ---------------------------------------------------------------- 词表

# 改几何的动词。这些词一旦出现，交付物必然包含一个改完的模型文件。
_EDIT_VERBS = [
    "移动", "平移", "移到", "搬", "摆放", "放到", "挪",
    "加宽", "加长", "加高", "抬高", "抬到", "升高", "降低", "降到", "下降", "缩短", "延长", "拉伸", "拉长",
    "切平", "切掉", "切割", "剖开", "挖空", "挖掉", "挖",
    "开洞", "开一个洞", "开一扇门", "开门", "重建", "补面", "封口",
    "改成", "改成圆", "换成", "替换", "改为", "改圆", "改方",
    "对接", "装配", "拼合", "插入", "拧", "旋进",
    "旋转", "复制", "拷贝", "复制出", "阵列",
]

# 只有这些词时才把「旋转 / 转」当写操作（否则「给出旋转角」是读数）。
_EDIT_STRONG = [
    "移动", "平移", "搬到", "摆放", "加宽", "加长", "加高", "抬高", "抬到", "降低", "降到",
    "切平", "挖", "开洞", "开门", "重建", "替换", "对接", "装配", "插入", "阵列",
]

# 量词 -> (建议字段名, 单位, 口径)。字段名只是命名建议，含义以原话为准。
_QTY = [
    ("净宽", "clear_width_m", "m", "两个相对内表面之间的净距"),
    ("净深", "clear_depth_m", "m", "两个相对内表面之间的净距"),
    ("净高", "clear_height_m", "m", "地面到顶面的净距"),
    ("墙高", "wall_height_m", "m", "墙体内表面从地面到顶面的高度"),
    ("层高", "storey_height_m", "m", "地面到顶面"),
    ("厚度", "thickness_m", "m", "板厚 = 最小主法向跨度"),
    ("半径", "radius_m", "m", "拟合或解析半径"),
    ("直径", "diameter_m", "m", "两倍半径"),
    ("高度", "height_m", "m", "沿指定方向的跨度"),
    ("长度", "length_m", "m", "沿指定方向的跨度"),
    ("最短距离", "min_distance_m", "m", "两网格之间的最短表面距离"),
    ("距离", "distance_m", "m", "两网格之间的最短表面距离"),
    ("间隙", "gap_m", "m", "带符号：正=有间隙，负=穿插"),
    ("坐标", "center_m", "m", "几何中心的三维坐标"),
    ("中心", "center_m", "m", "几何中心的三维坐标"),
    ("面积", "area_m2", "m2", "投影 / 截面 / 表面积，按原话指定"),
    ("体积", "volume_m3", "m3", "封闭网格的体积"),
    ("角度", "angle_deg", "deg", "绕指定轴的旋转角"),
    ("占比", "ratio", "1", "部分 / 整体"),
    ("比例", "ratio", "1", "部分 / 整体"),
    ("填充率", "fill_ratio", "1", "实体体积 / 包围盒体积"),
    ("数量", "count", "1", "命中个数"),
    ("个数", "count", "1", "命中个数"),
    ("有几个", "count", "1", "命中个数"),
]

# 原话里可能点名的实体。只作为「要辨认什么」的提示，不告诉 agent 它们长什么样。
_SUBJECTS = [
    "椅子", "书桌", "桌子", "桌", "门扇", "门洞", "门", "窗框", "窗台", "窗",
    "画框", "画", "把手", "墙", "地板", "地面", "天花", "吊顶",
    "房间", "卧室", "客厅", "走廊",
    "球", "正方体", "立方体", "长方体", "圆柱", "圆锥", "盒子",
    "火箭", "发射台", "插脚", "凹槽", "槽", "销",
    "家具", "实体", "构件", "开口", "洞口",
]

_UNIT_HINT = {"m": "米", "m2": "平方米", "m3": "立方米", "deg": "度", "1": "无量纲"}


def _hits(text, words):
    out = []
    for w in words:
        i = text.find(w)
        if i >= 0:
            out.append((i, w))
    return [w for _, w in sorted(out)]


def compile_brief(text):
    """原话 -> brief。纯函数，同一句话永远得到同一个 brief。"""
    raw = (text or "").strip()
    low = raw.lower()

    # 1) 交付类型：改模型 / 只读量。两者都提到时按「改」算 —— 改完必然要量。
    edits = _hits(raw, _EDIT_VERBS)
    strong = [v for v in _EDIT_VERBS if v in _EDIT_STRONG and v in raw]
    kind = "edit" if (strong or (edits and ("模型" in raw or "文件" in raw or "输出" in raw))) else "measure"

    # 2) 要交的量。同一个量只留一次，保留原话里的先后顺序。
    seen, qty = set(), []
    for word, name, unit, note in _QTY:
        if word in raw and name not in seen:
            seen.add(name)
            qty.append({"from": word, "name": name, "unit": unit, "note": note})

    # 3) 原话点名要辨认的实体（去重、保持出现顺序）。
    subs, used = [], set()
    for w in _hits(raw, _SUBJECTS):
        if any(w != u and w in u for u in used):
            continue                      # 「门」已经被「门扇」覆盖
        if w in used:
            continue
        used.add(w)
        subs.append(w)

    # 4) 原话自己规定的口径（容差、小数位、单位、禁止项）。
    rules = []
    if "容差" in raw or "1e-6" in low:
        rules.append("原话提到容差：答案按原话给的容差取整/对齐，不要四舍五入到更粗的位。")
    if re.search(r"保留\s*\d+\s*位", raw):
        rules.append("原话要求保留小数位：" + re.search(r"保留\s*\d+\s*位", raw).group(0) + "，照办。")
    if "毫米" in raw or "mm" in low:
        rules.append("原话提到毫米：输入可能是毫米单位，量之前先确认比例来源（mesh_health.meta）。")
    if "不要" in raw or "不得" in raw or "不能" in raw:
        rules.append("原话含禁止项：把禁止项当硬约束，宁可 declare_insufficient 也不要违反。")
    if "排序" in raw or "从大到小" in raw or "从小到大" in raw:
        rules.append("原话要求排序：用 rank 出顺序，order 与数值一起交。")

    brief = {
        "raw": raw,
        "kind": kind,
        "kind_label": "改动模型并交出文件" if kind == "edit" else "只读测量并给出数值",
        "edit_verbs": strong[:6],
        "quantities": qty,
        "subjects": subs,
        "rules": rules,
        "chars": len(raw),
    }
    brief["prompt"] = render_prompt(brief)
    return brief


def render_prompt(brief):
    """brief -> 注入 LLM 的说明。用 markdown，但只写「从原话能站得住」的句子。"""
    L = ["## 本轮任务", "",
         "下面这段是用户给的**全部**任务描述。系统里没有内置题目，也没有预置的解题步骤：",
         "要做什么、做成什么样，只能从这里读出来。", "",
         "```text", brief["raw"], "```", "",
         "## 管线从原话编译出的执行要点",
         "",
         "这些要点是把上面那段话读一遍得到的，不是表单、不是模板。若与原文冲突，以原文为准；",
         "若原文没提，就按这里的默认口径做，并在报告里说明用了哪一条。", "",
         "- 交付类型：**%s**" % brief["kind_label"]]
    if brief["edit_verbs"]:
        L.append("- 原话里的写操作：%s" % "、".join(brief["edit_verbs"]))
    if brief["quantities"]:
        L.append("- 要交的量（字段名只是命名建议，含义以原话为准）：")
        for q in brief["quantities"]:
            L.append("    - `%s`（%s，%s）— 原话出现「%s」"
                     % (q["name"], _UNIT_HINT.get(q["unit"], q["unit"]), q["note"], q["from"]))
    else:
        L.append("- 要交的量：原话没有点名具体的量，按它问的那件事自己定；"
                 "每个量单独一项，单位写进字段名后缀。")
    if brief["subjects"]:
        L.append("- 原话点名要辨认的实体：%s" % "、".join(brief["subjects"]))
        L.append("  （这是**题面用词**，不是模型里的名字。模型里的名称/类别/Pset 一律不可用，"
                 "仍然只能用厚度、法向、落地标高、内环这些几何量把它们压到唯一命中。）")
    L += ["", "## 默认口径（原话没有另行规定时用这套）", "",
          "- 长度米、面积平方米、体积立方米、角度度；数值不要带单位字符串，单位体现在字段名里。",
          "- 距离 = 两网格之间的最短表面距离；带符号时正=有间隙、负=穿插、0=相切。",
          "- 面积按“投影/截面/表面”三者之一，选哪个必须在 why 里说清；体积只对封闭网格给，破洞先修或明说。",
          "- 每个数字必须来自某次工具返回；派生量（差、和、除）走 calc，不要在报告里手写算式。",
          "- 接触/贴合面积只算共面重叠；仅边接触、角接触的面积是 0，不要当贴合。",
          "- 认不出、量不准就 declare_insufficient 并列出缺什么。猜一个数比说不知道更糟。"]
    if brief["rules"]:
        L += ["", "## 原话自带的约束", ""] + ["- " + r for r in brief["rules"]]
    L += ["", "## 交卷前自检", "",
          "1. 原话问的每一个量，values 里都有对应的一项，且能找到 evidence 里的出处；",
          "2. 该交文件的任务，emit 之后必须读回**自己写出的那个文件**再量，不能只信编辑函数的返回；",
          "3. 结论里的定性判断（是否通过、是否越界、是门还是窗）都有判据，不是感觉。"]
    return "\n".join(L)


if __name__ == "__main__":  # 便于命令行核对：python -m agent.brief "一段话"
    import sys
    print(compile_brief(" ".join(sys.argv[1:]))["prompt"])
