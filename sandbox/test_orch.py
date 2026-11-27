# -*- coding: utf-8 -*-
"""test_orch.py — 编排层改动的离线单测（不调用任何模型）。

这里验的四条，都是「账本里出现过几十次」的具体错法，改完后必须能在不花 token
的情况下证明它确实被接住了：

  1. @load 这种「按工具名引用」→ 解析到实际步骤（账本 37 次）
  2. @all.items 这种「字段名猜错」→ 按别名解析并回报正式字段（账本 22 次）
  3. parts 拆完父件消失 → 返回 removed，后续「场景里没有」能说出拆到哪去了（账本 87 次）
  4. 收口窗口 → 预算将尽时自动进 VERIFY 并给出交卷提示（T06 类「算对没交」）

跑法：python test_orch.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.runtime import Bus
from tools.world import Solid
from tools.shape import t_parts
from agent.loop import Loop

ok = fail = 0


def check(name, got, want):
    global ok, fail
    if got == want:
        ok += 1
        print("OK   %s" % name)
    else:
        fail += 1
        print("FAIL %s\n     got  %r\n     want %r" % (name, got, want))


def ledger():
    class L(list):
        def append(self, kind, payload=None):
            list.append(self, {"kind": kind, "payload": payload})
    return L()


# ---------------------------------------------------------------- 1. 引用别名
bus = Bus(ledger())
results = {"all": {"meta": {"path": "x"}, "solids": [{"sid": "C001"}], "n": 1}}
toolmap = {"all": "load"}
check("@load.solids → 解析到 all",
      bus.resolve("@load.solids", results, toolmap), [{"sid": "C001"}])
check("别名记了一笔", bus.notes and bus.notes[0]["to"], "all")

bus.notes = []
check("@Load 大小写也认",
      bus.resolve("@Load.solids", results, toolmap), [{"sid": "C001"}])

# 意图不唯一时不乱猜：两处 load 才出现过，取最近一次（后写覆盖）
results2 = {"all": {"solids": [1]}, "all_2": {"solids": [2]}}
check("@load 取最近一次", bus.resolve("@load.solids", results2, {"all": "load", "all_2": "load"}), [2])

# 完全无解的引用：报错里要带「已完成步骤 + 工具名」，而不是只丢一句 KeyError
try:
    bus.resolve("@nope.solids", results, toolmap)
    print("FAIL 无解引用应当报错")
    fail += 1
except KeyError as e:
    check("无解引用报错含步骤与工具名", ("all(load)" in str(e)), True)

# ---------------------------------------------------------------- 2. 字段别名
bus.notes = []
check("@all.items → 用 solids 解析", bus.resolve("@all.items", results, toolmap), [{"sid": "C001"}])
check("字段别名记了正式名", bus.notes[0]["to"], "solids")

# ---------------------------------------------------------------- 3. parts 拆件
def tetra(offset):
    P = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float) + offset
    F = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    return P, F


P1, F1 = tetra(np.zeros(3))
P2, F2 = tetra(np.array([5.0, 0, 0]))
bus3 = Bus(ledger())
bus3.scene = [Solid("C017", "chair", np.vstack([P1, P2]), np.vstack([F1, F2 + 4]))]
out = t_parts(bus3, "C017")
check("parts 返回 removed 映射", out.get("removed"), {"C017": ["C017.1", "C017.2"]})
check("parts 后父件不在场景", [s.sid for s in bus3.scene], ["C017.1", "C017.2"])
check("原 sid 重新走工具时报错指出拆到哪",
      "拆分为 C017.1/C017.2" in bus3.missing_msg("C017"), True)
check("keep_original 时不动场景", t_parts(bus3, "C017.1", keep_original=True).get("removed"), None)
# 只有一个连通分量：不该换号（换号会把后续引用全打断，而场景一点没变）
bus4a = Bus(ledger())
bus4a.scene = [Solid("C018", "box", *tetra(np.zeros(3)))]
one = t_parts(bus4a, "C018")
check("单件不换号", [x.sid for x in bus4a.scene], ["C018"])
check("单件返回 n=1 且不报 removed", (one["n"], one.get("removed")), (1, None))

# ---------------------------------------------------------------- 4. 收口窗口
class FakePolicy:
    kind = "fake"

    def decide(self, ctx):
        return None


bus4 = Bus(ledger())
loop = Loop(bus4, ledger(), FakePolicy(), "card", "card.md", "m.ifc",
            max_turns=100, wall_budget=600.0)
loop.turn = 10
check("早期不触发收口", loop._maybe_close(), False)
loop.turn = 92   # 剩余 8 轮
check("还没量过就不催交卷", loop._maybe_close(), False)
loop.measurements = ["w1"]
check("剩余 8 轮触发收口", loop._maybe_close(), True)
# 非写层任务（没进过 EDIT）收口先进 EDIT，逼它改几何再 emit，而不是直接进 VERIFY 交卷
check("收口后阶段进 EDIT（写层任务）", loop.phase, "EDIT")
check("收口提示写明剩余轮次", "只剩 8 轮" in (loop.closing_note or ""), True)
check("只触发一次", loop._maybe_close(), False)

# 草稿板：把已量到的数摆出来（「算对没交」的直接对策）
bus6 = Bus(ledger())
loop6 = Loop(bus6, ledger(), FakePolicy(), "card", "card.md", "m.ifc", max_turns=100)
loop6.steps = [{"id": "w1", "phase": "MEASURE", "tool": "distance",
                "args": '{"a": "C011", "b": "C002"}', "obs": '{"distance": 4.8, "m": 4.8}'},
               {"id": "w2", "phase": "MEASURE", "tool": "footprint",
                "args": '{"sid": "C017"}', "obs": '{"area": 0.23, "value": 0.23}'}]
loop6.measurements = ["w1", "w2"]
bd = loop6._draft_board() or ""
check("草稿板列出步骤与数值", ("w1" in bd and "4.8" in bd and "0.23" in bd), True)
check("草稿板不去重成一条", bd.count("→"), 2)

# 录制回放不受收口影响（否则验收链会漂移）
class FakeRec:
    kind = "recorded"
    def decide(self, ctx): return None


bus7 = Bus(ledger())
loop7 = Loop(bus7, ledger(), FakeRec(), "card", "card.md", "m.ifc", max_turns=40)
loop7.turn = 38
loop7.measurements = ["w1"]
check("回放模式不触发收口", loop7._maybe_close(), False)

# 已交卷就不该再催
bus5 = Bus(ledger())
loop5 = Loop(bus5, ledger(), FakePolicy(), "card", "card.md", "m.ifc", max_turns=100)
loop5.turn = 95
loop5.answer = {"values": {"x": 1}}
check("已交卷不触发收口", loop5._maybe_close(), False)

# ---------------------------------------------------------------- 5. 量到了却填 null
bus8 = Bus(ledger())
loop8 = Loop(bus8, ledger(), FakePolicy(), "card", "card.md", "m.ifc", max_turns=70)
loop8.steps = [{"id": "d1", "phase": "MEASURE", "tool": "distance",
                "args": '{"a": "C017", "b": "C011"}',
                "obs": '{"value": 0.6965, "unit": "m", "edge_pairs_evaluated": 45}'}]
loop8.measurements = ["d1"]
loop8.results["d1"] = {"value": 0.6965}
ans = {"values": {"chair": "C017", "min_distance_chair_to_door_leaf_m": None},
        "unknowns": ["距离：distance 工具调用未返回可用数值"],
        "evidence": ["d1"]}
check("账本里有数 → 有 _has_numbers", loop8._has_numbers(), True)
loop8._audit("a1", ans)
check("半空答案被拒一次", loop8.audit_rejects, 1)
rej = [x for x in loop8.steps if x.get("id") == "audit"][-1]
check("驳回理由里带着实测值", "0.6965" in rej["obs"], True)
# 第二次放行（有些项确实量不出来，不能死锁）
loop8._audit("a2", ans)
check("第二次不再拒同一条", loop8.audit_rejects, 1)

# 常驻测量板：VERIFY/REPORT 才有
loop8.phase = "MEASURE"
check("测量阶段不贴板", loop8.context()["measured"], None)
loop8.phase = "VERIFY"
check("复核阶段贴板", "0.6965" in (loop8.context()["measured"] or ""), True)

print("\n%d 通过 / %d 失败" % (ok, fail))
sys.exit(1 if fail else 0)
