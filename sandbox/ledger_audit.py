# -*- coding: utf-8 -*-
"""账本离线审计：不花一分模型钱，先把「哪些轮次是白烧的」数清楚。

为什么要它：
  现场 bench 每题 100 轮、每轮一次 LLM 调用，一轮 A/B 就是几十万 token。
  在模型资源有限的前提下，靠「多跑几次看分数」来选优化点是不可行的 ——
  样本少则结论不可靠，样本多则跑不起。
  但历史账本里已经记录了每次调用的参数、返回、错误。这些是**确定性数据**：
  同一条规则如果能在 174 次运行里拦下 300 步，它上线后省下的轮次是可预期的，
  不需要再问模型一次。

三个口径（都只依赖账本，可复算）：
  1. 参数重复（现状）：sha256(工具+args) 完全相同 —— 现有 _sig_count 用的就是这个
  2. 产物重复（新增）：只读工具的输出哈希相同 —— 抓住「换措辞重问同一个东西」。
     这是现有检测的盲区：args 从 sid 列表换成指纹列表、label 换个名，签名就变了。
  3. 报错重复：同一工具连续报同类错 —— 说明错误提示没被吸收，是提示设计问题。

用法：
  python ledger_audit.py                 # 汇总所有 bench_*/raw_* 运行
  python ledger_audit.py bench_T02_1     # 单题明细
"""
import glob
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
EDIT_TOOLS = {"move", "rotate", "cut", "widen", "open_hole", "translate", "union",
              "boolean", "place", "put", "rename", "delete", "edit"}
READONLY = {"load", "clusters", "principal_frame", "select", "pick", "contains", "merge",
            "gaps", "segment", "parts", "profile", "pockets", "levels", "openings",
            "assembly", "distance", "rank", "section", "mesh_health", "footprint",
            "clearance", "opening_edge", "collide", "sweep_rotate", "sweep_translate",
            "regid2d", "proj_area", "fit_sphere", "solve_section_height", "calc",
            "angle", "sphere_region", "contact_area"}


def out_key(tool, obs):
    """产物指纹：只看返回内容，不看它是怎么问出来的。"""
    if not obs:
        return None
    # 去掉编排层自己追加的提示字段，否则每条都会因为 _repeat/_next 不同而「不重复」
    txt = re.sub(r'"(?:_repeat|_next|_reload|_same_hits|_phase_forced|_label)"\s*:\s*'
                 r'("(?:[^"\\]|\\.)*"|\[[^\]]*\]|\{[^{}]*\})', '"~"', obs)
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()[:16]


def audit(run_dir):
    p = os.path.join(run_dir, "result.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    steps = d.get("steps") or []
    arg_seen = {}
    out_seen = {}
    exact_dup = out_dup = 0
    dup_extra_after_first = 0
    err_by_tool = Counter()
    err_key = Counter()
    phases = Counter()
    for s in steps:
        tool = s.get("tool") or "?"
        ph = s.get("phase") or "?"
        phases[ph] += 1
        if s.get("error"):
            err_by_tool[tool] += 1
            # 归类：取错误首句的主干
            e = re.sub(r"[0-9]+(\.[0-9]+)?", "#", s["error"])
            err_key[(tool, e[:70])] += 1
            continue
        if tool not in READONLY:
            continue
        a = s.get("args") or ""
        if a in arg_seen:
            exact_dup += 1
        else:
            arg_seen[a] = s.get("id")
        k = out_key(tool, s.get("obs"))
        if k and k in out_seen:
            out_dup += 1
            dup_extra_after_first += 1
        elif k:
            out_seen[k] = s.get("id")
    return {
        "tag": os.path.basename(run_dir),
        "status": d.get("status"),
        "turns": d.get("turns") or 0,
        "tok": (d.get("llm_usage") or {}).get("total_tokens") or 0,
        "steps": len(steps),
        "exact_dup": exact_dup,
        "out_dup": out_dup,
        "missed": max(0, out_dup - exact_dup),   # 现有检测抓不到的部分
        "err": sum(err_by_tool.values()),
        "err_by_tool": err_by_tool,
        "err_key": err_key,
        "phases": phases,
    }


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    dirs = [d for d in glob.glob(os.path.join(HERE, "runs", "*"))
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "result.json"))]
    if only:
        rows = [audit(d) for d in dirs if os.path.basename(d) == only]
        rows = [r for r in rows if r]
        for r in rows:
            print(json.dumps({k: v for k, v in r.items() if k != "err_key"},
                             ensure_ascii=False, indent=1))
            print("报错归类：")
            for (tool, e), n in r["err_key"].most_common(8):
                print("  %2dx %-10s %s" % (n, tool, e))
        return

    rows = [r for r in (audit(d) for d in dirs) if r and r["steps"] >= 10]
    print("共审计 %d 次运行（>=10 步）\n" % len(rows))
    print("%-20s %-16s 轮%-4s 步%-4s 参数重复%-4s 产物重复%-4s 漏检%-4s 报错%-4s"
          % ("运行", "状态", "", "", "", "", "", ""))
    tot = Counter()
    for r in sorted(rows, key=lambda x: -x["missed"])[:22]:
        print("%-20s %-16s %-5d %-5d %-9d %-9d %-7d %-5d"
              % (r["tag"], r["status"], r["turns"], r["steps"],
                 r["exact_dup"], r["out_dup"], r["missed"], r["err"]))
        for k in ("turns", "steps", "exact_dup", "out_dup", "missed", "err"):
            tot[k] += r[k]
    print("\n合计：轮 %d｜步 %d｜参数重复 %d｜产物重复 %d｜现有检测漏掉 %d｜报错 %d"
          % (tot["turns"], tot["steps"], tot["exact_dup"], tot["out_dup"], tot["missed"], tot["err"]))

    ek = Counter()
    for r in rows:
        ek.update(r["err_key"])
    print("\n跨运行的高频报错（同类 ≥5 次，说明提示没被吸收）：")
    for (tool, e), n in ek.most_common(10):
        if n >= 5:
            print("  %3dx %-12s %s" % (n, tool, e))


if __name__ == "__main__":
    main()
