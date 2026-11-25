# -*- coding: utf-8 -*-
"""rawcards.py — 生成「原题纯净版」任务卡。

为什么要这份对照：
    tasks/T*.md 是**增强版**卡 —— 赛题原文之后，还跟了「输入 / 判据定义 / 输出要求 /
    答案字段名 / 约束」。字段名表只给键不给值，但它确实在帮模型组织交付。
    于是「现场 10/24」这个数字里，混杂了「模型能不能算」和「题面帮了多少」两件事。

    纯净版只留赛题原文 + 一条运行时信息（模型路径、格式与编号不可信）。
    两版并排跑，差值就是脚手架的价值。

用法：
    python rawcards.py            # 生成 sandbox/tasks/raw/*.md
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "tasks")
OUT = os.path.join(SRC, "raw")

# 只保留原文，不需要"输入/判据/输出/字段名/约束"这些后加的段
DROP = ("## 输入", "## 判据定义", "## 关键定义", "## 输出要求", "## 答案字段名", "## 约束")

HEAD = """> 本卡是智能体在本次运行中收到的**全部**任务信息。卡内不含任何结论、不含对象编号、不含解题步骤。

## 任务原文（赛题 Task {no}）

"""

TAIL = """
## 输入

- 模型文件：`$MODEL`
- 格式未知、单位未知、朝向未知、对象编号与名称不可信；读取后请自行确认。
"""


def extract(text):
    """取「题目原文」段的正文（遇到下一个二级标题就停）。"""
    # 兼容「Task 02」与「Task 01 第 1 问」两种标题
    m = re.search(r"## 题目原文（赛题 Task (\d+)[^）]*）\s*\n(.*?)(?=\n## |\Z)", text, re.S)
    if not m:
        return None, None
    return m.group(1), m.group(2).strip()


def main():
    os.makedirs(OUT, exist_ok=True)
    made = []
    for name in sorted(os.listdir(SRC)):
        if not name.endswith(".md"):
            continue
        src = os.path.join(SRC, name)
        text = open(src, encoding="utf-8").read()
        no, body = extract(text)
        if no is None:
            print("跳过（没有原文段）：", name)
            continue
        card = HEAD.format(no=no) + body + "\n" + TAIL
        dst = os.path.join(OUT, name)
        open(dst, "w", encoding="utf-8").write(card)
        made.append((name, len(card)))
    print("生成 %d 张纯净版卡到 tasks/raw/：" % len(made))
    for n, sz in made:
        print("  %-12s %5d 字符" % (n, sz))


if __name__ == "__main__":
    main()
