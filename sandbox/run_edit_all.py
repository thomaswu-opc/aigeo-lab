# -*- coding: utf-8 -*-
"""run_edit_all.py — 一条命令跑完写层（EDIT）三道题，然后独立验收。

    python run_edit_all.py            # 全跑 + 验收
    python run_edit_all.py --only 07  # 只跑一道
    python run_edit_all.py --skip-run # 只重跑验收

跑的顺序有讲究：先 T07（单实体、两刀），再 T05（多副本、判据），最后 T06（多实体联动、
参数化）。前一题暴露的写层 bug 会在后一题放大，所以这个顺序下一旦挂，故障点很清楚。
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "..", "pkg", "AI-Geometry-Models", "IFC", "AI_Geometry_Input.ifc")

# tag -> (任务卡, 决策记录, 轮数上限)
JOBS = {
    "07": ("tasks/T07.md", "episodes/T07.edit.json", 40),
    "05": ("tasks/T05.md", "episodes/T05.edit.json", 60),
    "06": ("tasks/T06.md", "episodes/T06.edit.json", 80),
}


def sh(cmd):
    print("  $ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        raise SystemExit("[!] 失败：%s" % " ".join(cmd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="只跑某一道：05 / 06 / 07")
    ap.add_argument("--skip-run", action="store_true", help="跳过运行，只做验收")
    a = ap.parse_args()

    if not os.path.exists(MODEL):
        raise SystemExit("[!] 找不到模型包：%s\n"
                         "    赛方模型不随仓库分发，见 README 的下载说明。" % MODEL)

    order = [a.only] if a.only else ["07", "05", "06"]
    for t in order:
        if t not in JOBS:
            raise SystemExit("[!] 未知任务 %s，可选 %s" % (t, "/".join(JOBS)))

    if not a.skip_run:
        if "06" in order:
            # T06 的决策记录是生成出来的：加宽/改圆窗都是「一张锚点表换一组参数」，
            # 手抄 44 条 turn 迟早抄错一个坐标，所以交给生成器。
            print("== 生成 T06 决策记录 ==", flush=True)
            sh([sys.executable, "make_T06_episode.py"])

        for t in order:
            card, ep, turns = JOBS[t]
            print("== T-%s 写层：%s ==" % (t, card), flush=True)
            sh([sys.executable, "run_agent.py", "--card", card, "--model", MODEL,
                "--episode", ep, "--tag", "edit" + t,
                "--max-turns", str(turns), "--orbits", "2", "--no-video"])

    print("== 独立验收（只读磁盘产物，不读智能体的报告）==", flush=True)
    r = subprocess.run([sys.executable, "verify_edits.py"], cwd=HERE)
    raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
