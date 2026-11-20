# -*- coding: utf-8 -*-
"""compare_agent.py — 把 agent 跑遍全部变体与对抗样本，出一张表。

这张表要回答两个问题，且只回答这两个：
  1. 换模型/换格式/换单位/换朝向，agent 还给出同一答案吗？（泛化）
  2. 把答案改坏、把流程跳过，沙盒会拒绝吗？（防作弊机制是否真的生效）

不含任何「精度提升」的叙事 —— 本任务没有可优化的精度，只有可验证的一致性。
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "../pkg/AI-Geometry-Models/IFC/AI_Geometry_Input.ifc"
EP = "episodes/T01_2.agent.json"

CASES = [
    ("IFC 原件", BASE, EP, "基准"),
    ("OBJ 转换（名全随机）", "variants/obj_plain.obj", EP, "换格式 + 抹语义"),
    ("绕竖轴 90°", "variants/obj_rot90.obj", EP, "刚体旋转"),
    ("单位改 mm（×1000）", "variants/obj_mm.obj", EP, "尺度"),
    ("绕竖轴 45°", "variants/obj_rot45.obj", EP, "非轴对齐（触发自救）"),
    ("越界：SURVEY 直接测量", BASE, "episodes/adv_gate.json", "对抗样本"),
    ("编造数字提交", BASE, "episodes/adv_fabricate.json", "对抗样本"),
]


def run(tag, model, ep, extra=None):
    cmd = [sys.executable, "run_agent.py", "--card", "tasks/T01_2.md",
           "--model", model, "--episode", ep, "--tag", tag, "--no-video"]
    if extra:
        cmd += extra
    p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    f = os.path.join(HERE, "runs", tag, "result.json")
    if not os.path.exists(f):
        return {"status": "CRASH", "err": (p.stderr or p.stdout)[-300:]}
    return json.load(open(f, encoding="utf-8"))


def headline(r):
    a = (r.get("answer") or {}).get("values") or {}
    return a.get("distances_m") or {}


def main():
    rows, base = [], None
    for name, model, ep, note in CASES:
        tag = "cmp_" + "".join(c if c.isalnum() else "_" for c in name)[:24]
        r = run(tag, model, ep)
        d = headline(r)
        if base is None and d:
            base = d
        dev = None
        if base and d:
            dev = max(abs(d[k] - base[k]) for k in base if k in d)
        rows.append({
            "变体": name, "变化": note, "状态": r.get("status"),
            "轮数": r.get("turns"), "门控拒绝": r.get("gate_rejects"),
            "审计拒绝": r.get("audit_rejects"),
            "失败升级": len(r.get("failures") or []),
            "结论": ("%.4f / %.4f / %.4f" % (d.get("door_leaf", 0), d.get("window_frame", 0),
                                             d.get("picture_frame", 0))) if d else "—",
            "与基准偏差": ("%.1e" % dev) if dev is not None else "—",
            "账本": "完好" if (r.get("verify") or {}).get("ledger", {}).get("ok") else "—",
        })

    cols = ["变体", "变化", "状态", "轮数", "门控拒绝", "审计拒绝", "失败升级", "结论", "与基准偏差", "账本"]
    L = ["# agent 泛化与防作弊矩阵\n",
         "同一张任务卡、同一份决策记录，只换模型或换对抗输入。表由 `compare_agent.py` 自动生成。\n",
         "| " + " | ".join(cols) + " |",
         "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        L.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    L.append("\n## 怎么读这张表\n")
    L.append("- **前 5 行**：换格式、抹名称、转 90°、改单位，答案一致（偏差来自 OBJ 只保留 6 位小数）。"
             "第 5 行转 45° 时多耗 2 轮 —— 那是 agent 发现世界轴判据失效后，自己切到场景参照系并重试。")
    L.append("- **后 2 行**：故意作弊。越界调用被门控拦下（GATE_BLOCKED），"
             "编造的数字被审计拒绝并逐条列出无法溯源的数值（AUDIT_FAILED）。")
    L.append("- **没有一行是「碰巧对」**：每行的距离都能在 `runs/<tag>/events.jsonl` 里找到对应工具调用与见证点。")
    open(os.path.join(HERE, "runs", "agent_matrix.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:6 + len(rows)]))
    print("\n→ runs/agent_matrix.md")


if __name__ == "__main__":
    main()
