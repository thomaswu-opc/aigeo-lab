#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""逐题跑 10 个赛题，POST /api/solve，轮询完成，记录每题状态与关键答案。"""
import json
import os
import re
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8770"
IFC = "../pkg/AI-Geometry-Models/IFC/AI_Geometry_Input.ifc"

# 题卡 → wall_budget（秒）
TASKS = [
    ("T01_1", "tasks/T01_1.md", 120),
    ("T01_2", "tasks/T01_2.md", 150),
    ("T01_3", "tasks/T01_3.md", 120),
    ("T02", "tasks/T02.md", 240),
    ("T03", "tasks/T03.md", 180),
    ("T04", "tasks/T04.md", 150),
    ("T05", "tasks/T05.md", 240),
    ("T06", "tasks/T06.md", 240),
    ("T07", "tasks/T07.md", 180),
    ("T08", "tasks/T08.md", 240),
    ("T09", "tasks/T09.md", 180),
    ("T10", "tasks/T10.md", 240),
]


def extract_question(path):
    txt = open(path, encoding="utf-8").read()
    m = re.search(r"## 题目原文[^\n]*\n+(.*?)(?=\n## |\Z)", txt, re.S)
    if m:
        return m.group(1).strip()
    return txt[:400]


def post_solve(text, title, max_turns, wall):
    payload = {
        "text": text,
        "model": IFC,
        "title": title,
        "max_turns": max_turns,
        "wall": wall,
    }
    req = urllib.request.Request(
        BASE + "/api/solve",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def get_live(tag):
    try:
        with urllib.request.urlopen(BASE + "/api/live?tag=" + tag, timeout=20) as resp:
            return json.load(resp)
    except Exception:
        return {}


def run_one(cid, path, wall):
    text = extract_question(path)
    title = "%s rerun" % cid
    try:
        r = post_solve(text, title, max_turns=40, wall=wall)
        tag = r.get("tag")
        print("  started %s → %s" % (cid, tag), flush=True)
    except Exception as e:
        print("  %s POST失败: %s" % (cid, str(e)[:120]), flush=True)
        return None
    t0 = time.time()
    while time.time() - t0 < wall + 60:
        st = get_live(tag)
        if not st.get("running"):
            return {"cid": cid, "tag": tag, "status": st.get("status"),
                    "n": st.get("n"), "elapsed": round(time.time() - t0)}
        time.sleep(10)
    return {"cid": cid, "tag": tag, "status": "TIMEOUT", "n": None,
            "elapsed": round(time.time() - t0)}


def main():
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    results = []
    for cid, path, wall in TASKS:
        if only and cid not in only:
            continue
        print("=== 跑 %s ===" % cid, flush=True)
        r = run_one(cid, path, wall)
        if r:
            results.append(r)
            print("  %s → %s (%d 步, %ds)" % (cid, r["status"], r["n"], r["elapsed"]),
                  flush=True)
    print("\n=== 汇总 ===")
    for r in results:
        print("  %-8s %-20s %d 步" % (r["cid"], r["status"], r["n"] or 0))


if __name__ == "__main__":
    main()
