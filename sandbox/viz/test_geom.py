# -*- coding: utf-8 -*-
"""几何导出的契约测试：前端依赖这几条，一破就整块画面错位。"""
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools.ops, tools.shape, tools.brep, tools.world  # noqa: E402,F401
from core.runtime import Bus  # noqa: E402
from viz import geom  # noqa: E402

ok = fail = 0


def check(name, got, want):
    global ok, fail
    if got == want:
        ok += 1
        print("OK   %-38s → %s" % (name, got))
    else:
        fail += 1
        print("FAIL %-38s → %s（期望 %s）" % (name, got, want))


class L(list):
    def append(self, k, p=None):
        list.append(self, {"kind": k, "payload": p})


bus = Bus(L())
bus.call("load", {"path": "../pkg/AI-Geometry-Models/IFC/AI_Geometry_Input.ifc"}, "all")
out = os.path.join("runs", "_geomtest", "geom.json")
r = geom.dump(bus.scene, out, tag="_geomtest", step="s1", turn=3, note="测试")
d = json.load(open(out, encoding="utf-8"))

check("实体数", d["n_solids"], 20)
check("三角面数", d["n_tri"], sum(len(s.F) for s in bus.scene))
check("顶点扁平数组长度 = 3×顶点数", len(d["v"]), sum(len(s.P) for s in bus.scene) * 3)
check("索引扁平数组长度 = 3×面数", len(d["f"]), d["n_tri"] * 3)

# 分组不重不漏：每个 sid 一次，且 f0/fn 与 v0/vn 首尾相接
sids = [g["sid"] for g in d["g"]]
check("sid 无重复", len(sids), len(set(sids)))
check("分组按 v 连续", [g["v0"] for g in d["g"]][0], 0)
check("分组按 f 连续", [g["f0"] for g in d["g"]][0], 0)
check("最后一组收尾对上 v", d["g"][-1]["v0"] + d["g"][-1]["vn"], len(d["v"]) // 3)
check("最后一组收尾对上 f", d["g"][-1]["f0"] + d["g"][-1]["fn"], len(d["f"]) // 3)

# 索引必须落在本分组的局部顶点范围内（前端会减 v0，越界就是花屏）
bad = [g["sid"] for g in d["g"]
       if any(not (g["v0"] <= d["f"][g["f0"] * 3 + i] < g["v0"] + g["vn"])
              for i in range(g["fn"] * 3))]
check("索引都在本组范围内", bad, [])

# 坐标精度：4 位小数，最大误差不超过 0.1 mm
worst = 0.0
for g, s in zip(d["g"], sorted(bus.scene, key=lambda x: x.sid)):
    flat = np.array(d["v"][g["v0"] * 3:(g["v0"] + g["vn"]) * 3]).reshape(-1, 3)
    worst = max(worst, float(np.abs(flat - s.P).max()))
check("坐标误差 < 1e-4 m", worst < 1e-4, True)

# 体积：导出体积必须能压到 200 KB 以内（浏览器一次拉取的量级）
gz = len(gzip.compress(open(out, "rb").read(), 6))
check("gzip 后 < 200 KB", gz < 200 << 10, True)
print("     （原始 %.0f KB / gzip %.0f KB）" % (os.path.getsize(out) / 1024, gz / 1024))

# 原子写：不能留 .tmp
check("没有残留 .tmp", os.path.exists(out + ".tmp"), False)

# 空场景：不能抛异常（load 失败时 tick 仍会被调用）
check("空场景返回 None", geom.dump([], os.path.join("runs", "_geomtest", "empty.json")), None)

print("\n%d 通过 / %d 失败" % (ok, fail))
sys.exit(1 if fail else 0)
