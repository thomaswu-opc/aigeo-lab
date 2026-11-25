# -*- coding: utf-8 -*-
"""run_edit_all.py — 一条命令跑完九道题（读层四题 + 写层五题），然后独立验收。

    python run_edit_all.py            # 全跑 + 验收
    python run_edit_all.py --only 07  # 只跑一道
    python run_edit_all.py --skip-run # 只重跑验收

跑的顺序有讲究。**先读层、再写层**：读层四题只出数字、不出文件，跑完能立刻看出测量侧
（截面、体积、接触、包含）是否可信；一旦测量侧有问题，写层的失败会伪装成「编辑没做对」，
故障点就混了。

读层内部按依赖排：03（截面/体积/面积，最基本的量）→ 02（球与盒的体积交、面接触）
→ 04（平面包含与距离，用的就是 02 的接触判据）→ 09（分类 + 关系图，风险最大，放最后）。

写层内部同理：07（单实体、两刀）→ 05（多副本、判据）→ 06（多实体联动、参数化）
→ 08（同一实体换位置的两种状态 + 扫掠）→ 10（多实体装配 + 配准）。
前一题暴露的 bug 会在后一题放大：T07 考「切得准不准」，T05 考「判据能不能复用」，
T06 考「多实体改完还联不联动」，T08 考「同一套动作在两个状态上都必须成立」，
T10 考「动作由对应点解出来、不是照着坐标抄」。
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "..", "pkg", "AI-Geometry-Models", "IFC", "AI_Geometry_Input.ifc")

# tag -> (任务卡, 决策记录, 轮数上限)
JOBS = {
    "03": ("tasks/T03.md", "episodes/T03.edit.json", 45),
    "02": ("tasks/T02.md", "episodes/T02.edit.json", 45),
    "04": ("tasks/T04.md", "episodes/T04.edit.json", 30),
    "09": ("tasks/T09.md", "episodes/T09.edit.json", 50),
    "07": ("tasks/T07.md", "episodes/T07.edit.json", 40),
    "05": ("tasks/T05.md", "episodes/T05.edit.json", 60),
    "06": ("tasks/T06.md", "episodes/T06.edit.json", 80),
    "08": ("tasks/T08.md", "episodes/T08.edit.json", 60),
    "10": ("tasks/T10.md", "episodes/T10.edit.json", 60),
}


def sh(cmd):
    print("  $ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        raise SystemExit("[!] 失败：%s" % " ".join(cmd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="只跑某一道：02 / 03 / 04 / 05 / 06 / 07 / 08 / 09 / 10")
    ap.add_argument("--skip-run", action="store_true", help="跳过运行，只做验收")
    a = ap.parse_args()

    if not os.path.exists(MODEL):
        raise SystemExit("[!] 找不到模型包：%s\n"
                         "    赛方模型不随仓库分发，见 README 的下载说明。" % MODEL)

    order = [a.only] if a.only else ["03", "02", "04", "09", "07", "05", "06", "08", "10"]
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
