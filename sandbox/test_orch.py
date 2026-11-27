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
import tools.ops  # noqa: F401 — 注册全部工具（含 EDIT_TOOLS：stretch_axis/translate 等）
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

# ---------------------------------------------------------------- 6. 预算分层 + 容错区
# 声明预算（max_turns）驱动收口提示；终止预算（hard_turns）驱动 while 硬停。
# 容错区 = [max_turns, hard_turns)：超声明但未到硬停，不直接停，打 overrun 标记供评估减分。

def mk_loop(max_turns=40, wall=180.0, grace=None, gwall=None, **kw):
    b = Bus(ledger())
    return Loop(b, ledger(), FakePolicy(), "card", "card.md", "m.ifc",
                max_turns=max_turns, wall_budget=wall,
                grace_turns=grace, grace_wall=gwall, **kw)

# 6a. 默认容错：声明 40 → 硬停 1.5×（至少 +10）
lp = mk_loop(40, 180.0)
check("默认硬停 = max(1.5×, +10)", lp.hard_turns, max(int(40 * 1.5), 40 + 10))
check("默认硬停墙钟 = max(1.5×, +60)", lp.hard_wall, max(180.0 * 1.5, 180.0 + 60.0))

# 6b. 显式容错：声明 40 + grace 8 → 硬停 48
lp2 = mk_loop(40, 180.0, grace=8, gwall=30.0)
check("显式容错轮数", lp2.hard_turns, 48)
check("显式容错墙钟", lp2.hard_wall, 210.0)

# 6c. 收口提示仍用声明预算（左=声明-当前），不随硬停漂移
lp3 = mk_loop(100, 600.0)
lp3.turn = 92
lp3.measurements = ["w1"]
lp3._maybe_close()
check("收口窗口相对声明预算", "只剩 8 轮" in (lp3.closing_note or ""), True)

# 6d. 主循环硬停判据：turn 到达 hard_turns 边界才算 BUDGET_EXHAUSTED
#     （用 _in_grace 判断是否落在容错区）
lp4 = mk_loop(40, 180.0, grace=8)
lp4.turn = 45   # 已超声明 40，未到硬停 48
check("turn=45 落在容错区", lp4._in_grace(), True)
check("容错区未交卷 → BUDGET_OVERRUN",
      lp4._end_status("BUDGET_EXHAUSTED"), "BUDGET_OVERRUN")

lp4.turn = 48   # 到达硬停边界
check("turn=48 仍算容错区（含边界）", lp4._in_grace(), True)
check("边界未交卷 → BUDGET_OVERRUN",
      lp4._end_status("BUDGET_EXHAUSTED"), "BUDGET_OVERRUN")

# 6e. 已交卷即使超预算也改判 OK（不因超时判失败）
lp5 = mk_loop(40, 180.0, grace=8)
lp5.turn = 46
lp5.answer = {"values": {"x": 1}}
check("超预算但已交卷 → OK", lp5._end_status("BUDGET_EXHAUSTED"), "OK")
check("report 里 overrun 标记为真", lp5.report()["overrun"], True)

# 6f. 未超声明预算 → 不在容错区，正常 BUDGET_EXHAUSTED
lp6 = mk_loop(40, 180.0, grace=8)
lp6.turn = 40   # 恰好等于声明（未超）
check("turn=40 未超声明", lp6._in_grace(), False)
check("未超声明耗尽 → BUDGET_EXHAUSTED",
      lp6._end_status("BUDGET_EXHAUSTED"), "BUDGET_EXHAUSTED")

# 6g. report 暴露预算字段
rp = lp2.report()
check("report 带 declared_turns", rp["declared_turns"], 40)
check("report 带 hard_turns", rp["hard_turns"], 48)

# ---------------------------------------------------------------- 7. EDIT_PLAN 规划 + checklist
# 写层任务先 plan_edit 出 checklist 再进 EDIT 逐项执行，最后一项 emit；
# EDIT 内同一处改动重复 ≥2 次判空转。这是防「左挪 1mm 右挪 1mm」死循环的核心。

def mk_editloop():
    """构造一个已进 EDIT_PLAN 的写层 loop（非 read_only）。"""
    b = Bus(ledger())
    lp_ = Loop(b, ledger(), FakePolicy(), "card", "card.md", "m.ifc",
               max_turns=60, wall_budget=300.0, read_only=False)
    lp_.phase = "EDIT_PLAN"
    return lp_

# 7a. EDIT_PLAN 阶段在白名单里，且只允许 plan_edit + advance_phase
from agent.loop import PHASE_TOOLS
check("PHASES 含 EDIT_PLAN（MEASURE 后）", PHASE_TOOLS.get("EDIT_PLAN"),
      {"plan_edit", "advance_phase"})

# 7b. EDIT_PLAN 门控：没规划时被拒
lp_e = mk_editloop()
check("EDIT_PLAN 未规划门控被拒", lp_e._gate_pass(), False)

# 7c. plan_edit 合法 checklist → 存下并过门控
lp_e._dispatch({"tool": "plan_edit", "checklist": [
    {"tool": "stretch_axis", "sids": ["C004", "C006"], "target": "净宽 4.8→6.0"},
    {"tool": "translate", "sids": ["C007"], "target": "右墙随动 +1.2"},
    {"tool": "reopen_round", "sids": ["C006"], "target": "方窗改同面积圆窗"},
    {"tool": "emit", "sids": [], "target": "落盘"},
]})
check("plan_edit 存下 checklist（4 项）", len(lp_e.edit_checklist), 4)
check("EDIT_PLAN 规划后门控放行", lp_e._gate_pass(), True)
check("checklist 首项 tool", lp_e.edit_checklist[0]["tool"], "stretch_axis")

# 7d. plan_edit 拒绝空 checklist
lp_e2 = mk_editloop()
lp_e2._dispatch({"tool": "plan_edit", "checklist": []})
check("空 checklist 被拒（不存）", lp_e2.edit_checklist, [])

# 7e. plan_edit 拒绝最后一项非 emit
lp_e3 = mk_editloop()
lp_e3._dispatch({"tool": "plan_edit", "checklist": [
    {"tool": "stretch_axis", "sids": ["C004"], "target": "x"},
]})
check("最后一项非 emit 被拒", lp_e3.edit_checklist, [])

# 7f. plan_edit 拒绝非法工具（answer 不在 EDIT 白名单）
lp_e4 = mk_editloop()
lp_e4._dispatch({"tool": "plan_edit", "checklist": [
    {"tool": "answer", "sids": [], "target": "x"},   # answer 不在 EDIT 白名单
    {"tool": "emit", "sids": [], "target": "落盘"},
]})
check("非 EDIT 工具被拒", lp_e4.edit_checklist, [])

# 7g. EDIT_PLAN → EDIT 推进后，context() 带 checklist 进度
lp_e._advance("规划完成")
check("规划后推进到 EDIT", lp_e.phase, "EDIT")
cl = lp_e.context()["checklist"]
check("EDIT 阶段 context 带 checklist", (cl is not None and "☐" in cl and "emit" in cl), True)

# 7h. EDIT 内 checklist 打勾：edit 动作匹配 tool+sids 即标记完成
#     用真实几何场景（stretch_axis 要真正跑到成功返回，checklist 逻辑才在 out 分支里被触发）
from tools.world import Solid as _Solid
_box = _Solid("C004", "floor",
              np.array([[0,0,0],[4.8,0,0],[4.8,4,0],[0,4,0],
                        [0,0,.12],[4.8,0,.12],[4.8,4,.12],[0,4,.12]], float),
              np.array([[0,1,2],[0,2,3],[4,6,5],[4,7,6],
                        [0,4,5],[0,5,1],[1,5,6],[1,6,2],
                        [2,6,7],[2,7,3],[3,7,4],[3,4,0]], int))
lp_e.bus.scene = [_box]
lp_e._dispatch({"tool": "stretch_axis", "args": {"sids": ["C004"], "axis": 0,
                                                 "anchor_side": "lo", "delta": 1.2},
                "why": "加宽房间"})
check("stretch_axis 命中 checklist [0]", 0 in lp_e._edit_done, True)

# 7i. EDIT 内同一处改动重复 ≥2 次判空转（左挪右挪死循环）
lp_f = mk_editloop()
lp_f.phase = "EDIT"
lp_f._edit_before_snap = {}
lp_f.edit_checklist = [
    {"tool": "translate", "sids": ["C007"], "target": "x", "done": False},
    {"tool": "emit", "sids": [], "target": "落盘", "done": False},
]
_box2 = _Solid("C007", "wall",
               np.array([[0,0,0],[0.2,0,0],[0.2,4,0],[0,4,0],
                         [0,0,3],[0.2,0,3],[0.2,4,3],[0,4,3]], float),
               np.array([[0,1,2],[0,2,3],[4,6,5],[4,7,6],
                         [0,4,5],[0,5,1],[1,5,6],[1,6,2],
                         [2,6,7],[2,7,3],[3,7,4],[3,4,0]], int))
lp_f.bus.scene = [_box2]
for _ in range(2):
    lp_f._dispatch({"tool": "translate", "args": {"sids": ["C007"], "delta": [0.001, 0, 0]},
                    "why": "微调右墙"})
warns = [e for e in lp_f.L if e.get("kind") == "edit_loop_warn"]
check("同一处改动重复 2 次触发空转警告", len(warns) >= 1, True)

print("\n%d 通过 / %d 失败" % (ok, fail))
sys.exit(1 if fail else 0)
