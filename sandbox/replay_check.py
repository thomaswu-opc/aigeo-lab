# -*- coding: utf-8 -*-
"""replay_check.py — 反事实检查：把历史失败的决定，用**新的编排层**重新判一遍。

为什么不用再跑一遍模型：
  现场每一轮都是一次 LLM 调用。要判断「新规则能不能救回这一步」，不必再问模型一次 ——
  账本里已经记了当时它想干什么（工具 + 参数）、当时场景里有什么（之前的成功步骤），
  以及它得到了什么（报错原文）。这些足够确定性地回答一个问题：

      同一个决定，在新规则下是**仍然失败**，还是**能被接住**？

  能接住的比例，就是这套改动在「机械层面」的上限收益。它不预测模型会不会因此变聪明，
  只回答「有多少轮本来就不该失败」。

三类判定：
  A 引用名猜错   @load / @Load_2 —— 新规则按「最近一次同名工具步骤」解析
  B 字段名猜错   @all.items      —— 新规则按字段别名（items ↔ solids）解析
  C 编号失效     C017 被 parts 拆掉 —— 新规则在报错里点名拆到哪去了（仍需改参数，但不用猜）

用法：python replay_check.py [--detail]
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FIELD_ALIAS = {"items": ("solids", "entities", "parts", "list"),
               "solids": ("items", "entities"),
               "value": ("items", "value"),
               "values": ("value", "items", "solids")}


def alias_label(root, labels, toolmap):
    """与 core.runtime.Bus._alias_label 同一套规则（这里只做判定，不用真值）。"""
    if not labels:
        return None
    rl = root.lower()
    cands = [k for k in labels if str(toolmap.get(k, "")).lower() == rl]
    if cands:
        return cands[-1]
    base = root.rstrip("0123456789_")
    if base and base != root:
        cands = [k for k in labels if str(toolmap.get(k, "")).lower() == base.lower()]
        if cands:
            return cands[-1]
    low = {k.lower(): k for k in labels}
    return low.get(rl)


def scan(run_dir, detail=False):
    p = os.path.join(run_dir, "result.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    steps = d.get("steps") or []
    labels, toolmap = [], {}
    split_children = {}          # 父 sid -> [子 sid]
    rec = {"tag": os.path.basename(run_dir), "status": d["status"], "turns": d.get("turns") or 0,
           "err": 0, "A_fix": 0, "A_left": 0, "B_fix": 0, "C_explained": 0, "C_other": 0}
    detail_rows = []
    for s in steps:
        err = s.get("error") or ""
        sid = s.get("id")
        if err:
            rec["err"] += 1
            m = re.search(r"未定义的引用 (@[A-Za-z_][\w\-]*)", err)
            f = re.search(r"(@[\w.]+) 在 '([\w]+)' 处不存在", err)
            miss = re.search(r"场景里没有 ([A-Za-z]?\d+[A-Za-z0-9_.]*)", err)
            if m:
                ref = m.group(1)
                root = ref[1:].split(".")[0]
                got = alias_label(root, labels, toolmap)
                if got:
                    rec["A_fix"] += 1
                    detail_rows.append((sid, "A", "%s → @%s" % (ref, got)))
                else:
                    rec["A_left"] += 1
                    detail_rows.append((sid, "A", "%s 仍无解" % ref))
            elif f:
                ref, field = f.group(1), f.group(2)
                root = ref[1:].split(".")[0]
                lab = root if root in labels else alias_label(root, labels, toolmap)
                # 只有「字段别名能命中」才算修好：需要知道那一步返回里有哪些字段，
                # 账本里没有原始返回，所以按已知的 load 返回结构判定（meta/solids/n）。
                keys = ("meta", "solids", "n") if toolmap.get(lab) == "load" else ()
                alt = next((a for a in FIELD_ALIAS.get(field, ()) if a in keys), None)
                if lab and alt:
                    rec["B_fix"] += 1
                    detail_rows.append((sid, "B", "%s.%s → .%s" % (ref, field, alt)))
                else:
                    detail_rows.append((sid, "B", "%s.%s 仍无解" % (ref, field)))
            elif miss:
                sid_m = miss.group(1)
                if sid_m in split_children:
                    rec["C_explained"] += 1
                    detail_rows.append((sid, "C", "%s 已拆为 %s" % (sid_m, split_children[sid_m])))
                else:
                    rec["C_other"] += 1
            continue
        if s.get("rejected"):
            continue
        if sid:
            labels.append(sid)
            toolmap[sid] = s.get("tool")
        if s.get("tool") == "parts":
            # 从返回摘要里抓子件编号（obs 里每个 part 都带 sid）
            obs = s.get("obs") or ""
            kids = sorted(set(re.findall(r'"sid"\s*:\s*"([^"]+\.\d+)"', obs)))
            par = re.findall(r'"parent"\s*:\s*"([^"]+)"', obs)
            if par and kids:
                split_children[par[0]] = kids
    rec["detail"] = detail_rows
    return rec


def main():
    detail = "--detail" in sys.argv
    dirs = [d for d in glob.glob(os.path.join(HERE, "runs", "*"))
            if os.path.isdir(d) and os.path.basename(d).startswith(("bench_", "raw_"))]
    rows = [r for r in (scan(d, detail) for d in dirs) if r]
    print("现场运行 %d 次｜总轮次 %d｜报错 %d\n"
          % (len(rows), sum(r["turns"] for r in rows), sum(r["err"] for r in rows)))
    print("| 运行 | 状态 | 轮 | 报错 | A 引用名→已解析 | B 字段名→已别名 | C 拆件→已说明 |")
    print("|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: -(x["A_fix"] + x["B_fix"] + x["C_explained"])):
        v = r["A_fix"] + r["B_fix"] + r["C_explained"]
        if v == 0 and r["err"] == 0:
            continue
        print("| %s | %s | %d | %d | %d | %d | %d |"
              % (r["tag"], r["status"], r["turns"], r["err"],
                 r["A_fix"], r["B_fix"], r["C_explained"]))
        if detail:
            for sid, kind, txt in r["detail"]:
                print("|   %s %s | %s | | | | |" % (kind, sid, txt))
    a = sum(r["A_fix"] for r in rows)
    b = sum(r["B_fix"] for r in rows)
    c = sum(r["C_explained"] for r in rows)
    e = sum(r["err"] for r in rows)
    print("\n合计：报错 %d｜引用名可自动解析 %d｜字段名可自动别名 %d｜拆件失效可说明 %d｜"
          "合计能接住 %d（%.0f%%）"
          % (e, a, b, c, a + b + c, 100.0 * (a + b + c) / max(1, e)))
    print("仍无解：A %d｜C(非拆件) %d" % (sum(r["A_left"] for r in rows),
                                        sum(r["C_other"] for r in rows)))


if __name__ == "__main__":
    main()
