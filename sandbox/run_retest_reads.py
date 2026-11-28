#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""顺序重跑三个读层任务（T02/T03/T05），验证第五道审计能否逼 agent 把已量到的数整理进 values。"""
import subprocess, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
IFC = "../pkg/AI-Geometry-Models/IFC/AI_Geometry_Input.ifc"
for t in ("T02", "T03", "T05"):
    tag = "retest_%s_2" % t
    out = os.path.join(HERE, "runs_retest_%s.log" % t)
    cmd = [sys.executable, "run_agent.py", "--card", "tasks/%s.md" % t,
           "--model", IFC, "--llm", "auto", "--tag", tag,
           "--max-turns", "100", "--wall-budget", "420", "--no-video"]
    print("=== %s -> %s ===" % (t, tag), flush=True)
    with open(out, "w", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    print("=== %s done ===" % t, flush=True)
