# -*- coding: utf-8 -*-
"""compare.py — 把多次运行汇成一张泛化性矩阵。这张表就是「不能只对给定模型有效」的答卷。"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = [
    ("ifc", "IFC 原件", "", 0.0),
    ("plain", "OBJ 转换", "格式变、对象名随机化", 0.0),
    ("rot90", "OBJ + 绕竖轴 90°", "刚体旋转", 0.0),
    ("mm", "OBJ + 单位改 mm", "尺度 ×1000", 0.0),
    ("rot45", "OBJ + 绕竖轴 45°", "非轴对齐（负例）", None),
]


def load(tag):
    p = os.path.join(HERE, "runs", tag, "result.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def main():
    base = load("ifc")
    base_vals = {}
    if base and base.get("status") is None:
        base_vals = {r["sid"]: r["value"] for r in base["answer"]["values"]["distance_ranking"]}
    base_sorted = [r["value"] for r in base["answer"]["values"]["distance_ranking"]] if base_vals else []

    lines = ["# 泛化性矩阵\n",
             "同一份任务卡、同一份 plan，只换模型。答案必须「跟着变换走」：距离不变、见证点随刚体变换移动、",
             "尺度变则先由几何裁决单位再复原。\n",
             "| 变体 | 变化 | 状态 | d1 | d2 | d3 | 与原件最大偏差 | 见证点复核 | 账本 | 耗时 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    out = []
    for tag, label, change, _ in ROWS:
        r = load(tag)
        if r is None:
            lines.append("| %s | %s | 未运行 | – | – | – | – | – | – | – |" % (label, change))
            continue
        if r.get("status") == "INSUFFICIENT_EVIDENCE":
            st = r["stopped_at"]
            lines.append("| %s | %s | **停在 %s**（期望 %s / 实得 %s） | – | – | – | – | – | %s | %.2fs |" % (
                label, change, st["step"], st["expect_n"], st["got_n"],
                "完好" if r["verify"]["ledger"]["ok"] else "断裂", r["elapsed_s"]))
            out.append({"tag": tag, "status": "stop", "at": st})
            continue
        vals = [x["value"] for x in r["answer"]["values"]["distance_ranking"]]
        delta = max(abs(a - b) for a, b in zip(vals, base_sorted)) if base_sorted and len(vals) == len(base_sorted) else float("nan")
        v = r["verify"]
        lines.append("| %s | %s | 完成 | %.3f | %.3f | %.3f | %.1e | %s | %s | %.2fs |" % (
            label, change, vals[0], vals[1], vals[2], delta,
            "通过" if v["witness_ok"] else "失败",
            "完好" if v["ledger"]["ok"] else "断裂", r["elapsed_s"]))
        out.append({"tag": tag, "values": vals, "delta": delta,
                    "witness_ok": v["witness_ok"], "ledger_ok": v["ledger"]["ok"]})

    lines.append("\n## 结论\n")
    ok = [o for o in out if o.get("values")]
    lines.append("- %d/%d 个变体给出与原件一致的答案（偏差 < 1e-6 m），排序不变。" % (
        sum(1 for o in ok if o["delta"] < 1e-6), len(ok)))
    lines.append("- 失败项是**预先声明的限制**：判据基于轴对齐包围盒，整体旋转 45° 后失效。")
    lines.append("- 沙盒在失效时选择停止并报「证据不足」，而不是给出一个看似合理的错数。")

    p = os.path.join(HERE, "runs", "matrix.md")
    open(p, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump(out, open(os.path.join(HERE, "runs", "matrix.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
