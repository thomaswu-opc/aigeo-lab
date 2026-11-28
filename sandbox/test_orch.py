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

# 7o. 第五道审计：有数没整理进 values —— claims 为空（一个可核对数值都没写）但量出过数
#     实测 T02 交 {"probe":"check"}（只有字符串探针、n_claims=0），四道检查全放行、漏网。
lp_m = Loop(Bus(ledger()), ledger(), FakePolicy(), "card", "card.md", "m.ifc", max_turns=70)
lp_m.steps = [{"id": "sr1", "phase": "MEASURE", "tool": "sphere_region",
               "args": '{"items": "C001"}',
               "obs": '{"relation": "完全在区域内", "radius": 1.0, "sphere_center": [-5.25, 0.0, 1.25]}'}]
lp_m.measurements = ["sr1"]
lp_m.results["sr1"] = {"radius": 1.0}
check("账本有数但答案无数值 → _has_numbers", lp_m._has_numbers(), True)
# answer 只有探针字符串、没有任何数值（claims 为空）→ 第五道拦下、贴 draft board
ans_probe = {"values": {"probe": "check"}, "unknowns": [], "evidence": ["sr1"]}
lp_m._audit("a1", ans_probe)
check("有数没整理进 values 被拒一次", lp_m.audit_rejects, 1)
rej = [x for x in lp_m.steps if x.get("id") == "audit"][-1]
check("驳回理由指出缺可核对数值", "一个可核对的数值都没写" in rej["obs"], True)
check("驳回理由贴了已量到的数", "radius" in rej["obs"], True)
# 补上数值后放行
lp_m._audit("a2", {"values": {"sphere_radius_m": 1.0}, "unknowns": [], "evidence": ["sr1"]})
check("整理进数值后放行", lp_m.audit_rejects, 1)
# 纯识别任务（无数值结论、_has_numbers 为空）不触发这条
lp_i = Loop(Bus(ledger()), ledger(), FakePolicy(), "card", "card.md", "m.ifc", max_turns=70)
lp_i.steps = [{"id": "sel1", "phase": "IDENTIFY", "tool": "select",
               "args": '{"items": "C004"}', "obs": '{"n": 1}'}]
lp_i.results["sel1"] = {"n": 1}   # 证据要存在，否则先撞「证据不存在」而非第五道
lp_i._audit("a1", {"values": {"door": "C011"}, "unknowns": [], "evidence": ["sel1"]})
check("纯识别任务无数值不触发第五道", lp_i.audit_rejects, 0)

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

# 7j. plan_edit 清单没变时触发收紧提示（plan_unchanged）+ plan_note 进上下文
lp_g = mk_editloop()
cl7j = [
    {"tool": "stretch_axis", "sids": ["C004", "C006"], "target": "净宽"},
    {"tool": "emit", "sids": [], "target": "落盘"},
]
lp_g._dispatch({"tool": "plan_edit", "checklist": cl7j})
check("首次 plan_edit 无收紧提示", lp_g.plan_note, None)
first_plan_events = [e for e in lp_g.L if e.get("kind") == "plan_edit"]
check("首次 plan_edit 写事件", len(first_plan_events), 1)
# 重发同一份清单
lp_g._dispatch({"tool": "plan_edit", "checklist": cl7j})
check("重发同清单触发收紧提示", lp_g.plan_note is not None
      and "advance_phase" in lp_g.plan_note, True)
unchanged = [e for e in lp_g.L if e.get("kind") == "plan_unchanged"]
check("重发同清单记 plan_unchanged 事件", len(unchanged), 1)
# 清单没变时不重复写 plan_edit 事件
check("重发同清单不重复写 plan_edit 事件",
      len([e for e in lp_g.L if e.get("kind") == "plan_edit"]), 1)
# 重发后 plan_note 要进 context（EDIT_PLAN 阶段）
check("收紧提示进 context.plan_note", "advance_phase" in (lp_g.context().get("plan_note") or ""),
      True)
# 清单变了 → 不触发收紧提示，正常重规划
lp_g._dispatch({"tool": "plan_edit", "checklist": [
    {"tool": "stretch_axis", "sids": ["C004"], "target": "净宽改法"},
    {"tool": "emit", "sids": [], "target": "落盘"},
]})
check("清单变了清除收紧提示", lp_g.plan_note, None)
check("清单变了正常写 plan_edit 事件",
      len([e for e in lp_g.L if e.get("kind") == "plan_edit"]), 2)

# 7k. emit 时清单未完成 → 软提示（不硬拦，但要 agent 看到"还差哪几项"）
#     实测 T06：做完 8/13 项就 emit，跳过 5.4 那一步 → 答案 net_width_54_m=null。
lp_h = mk_editloop()
lp_h.phase = "EDIT"
# 3 项清单：两项 edit + 一项 emit。只做第一项 edit，直接 emit → 应提示"还差 2 项"
lp_h.edit_checklist = [
    {"tool": "translate", "sids": ["C007"], "target": "右墙随动", "done": False},
    {"tool": "reopen_round", "sids": ["C006"], "target": "方窗改圆窗", "done": False},
    {"tool": "emit", "sids": [], "target": "落盘", "done": False},
]
_box3 = _Solid("C007", "wall",
               np.array([[0,0,0],[0.2,0,0],[0.2,4,0],[0,4,0],
                         [0,0,3],[0.2,0,3],[0.2,4,3],[0,4,3]], float),
               np.array([[0,1,2],[0,2,3],[4,6,5],[4,7,6],
                         [0,4,5],[0,5,1],[1,5,6],[1,6,2],
                         [2,6,7],[2,7,3],[3,7,4],[3,4,0]], int))
lp_h.bus.scene = [_box3]
# 做完第一项 edit（translate C007）
lp_h._dispatch({"tool": "translate",
                "args": {"sids": ["C007"], "delta": [1.2, 0, 0]}, "why": "右墙随动"})
check("emit 前只完成 1/3 项", len(lp_h._edit_done), 1)
# 直接 emit（跳过第 2 项 reopen_round）→ 触发"清单未完成"提示
_out = lp_h._call("emit", {"path": os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "runs", "_tmp_7k_emit.gltf")},
                  "落盘", label="emit_test")
pend_ev = [e for e in lp_h.L if e.get("kind") == "checklist_pending_at_emit"]
check("emit 未完成清单记 checklist_pending_at_emit 事件", len(pend_ev), 1)
check("提示里点明还差 2 项",
      pend_ev and pend_ev[0]["payload"].get("done") == 1
      and pend_ev[0]["payload"].get("total") == 3, True)
# hint 里要写明缺哪一项（reopen_round）
_h = [s for s in lp_h.steps if s.get("hint", "").startswith("清单还差")]
check("emit 步骤 hint 指出缺 reopen_round",
      _h and "reopen_round" in _h[-1]["hint"], True)
# 清掉临时文件
try:
    os.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "runs", "_tmp_7k_emit.gltf"))
except OSError:
    pass

# 7l. emit 硬门控：清单未完成占比高时拒 emit（_dispatch 路径），最多拒 2 次后放行
lp_i = mk_editloop()
lp_i.phase = "EDIT"
lp_i.edit_checklist = [
    {"tool": "translate", "sids": ["C007"], "target": "第一步", "done": False},
    {"tool": "reopen_round", "sids": ["C006"], "target": "第二步", "done": False},
    {"tool": "stretch_axis", "sids": ["C004"], "target": "第三步", "done": False},
    {"tool": "emit", "sids": [], "target": "落盘", "done": False},
]
# 3 项工作，0 项完成 → 占比 0% < 50%，第 1 次 emit 应被拒
_emit_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "_tmp_7l_emit.gltf")
lp_i._dispatch({"tool": "emit", "args": {"path": _emit_path}, "why": "落盘"})
rej1 = [e for e in lp_i.L if e.get("kind") == "emit_gate_reject"]
check("第 1 次 emit 未完成占比高被拒", len(rej1), 1)
check("拒绝时不写盘", len(lp_i.emitted), 0)
check("拒绝计数 = 1", lp_i._emit_gate_rejects, 1)
# 第 2 次仍被拒
lp_i._dispatch({"tool": "emit", "args": {"path": _emit_path}, "why": "落盘"})
check("第 2 次 emit 仍被拒", len([e for e in lp_i.L if e.get("kind") == "emit_gate_reject"]), 2)
# 第 3 次超过上限 → 放行（防死锁）
lp_i._dispatch({"tool": "emit", "args": {"path": _emit_path}, "why": "落盘"})
check("第 3 次 emit 超上限放行", len(lp_i.emitted) >= 1, True)
check("放行后仍只有 2 次拒绝", len([e for e in lp_i.L if e.get("kind") == "emit_gate_reject"]), 2)
for p_ in list(lp_i.emitted):
    try:
        os.remove(p_)
    except OSError:
        pass

# 7m. emit 门控：清单完成过半时直接放行，不拒
lp_j = mk_editloop()
lp_j.phase = "EDIT"
lp_j.edit_checklist = [
    {"tool": "translate", "sids": ["C007"], "target": "第一步", "done": False},
    {"tool": "reopen_round", "sids": ["C006"], "target": "第二步", "done": False},
    {"tool": "emit", "sids": [], "target": "落盘", "done": False},
]
lp_j._edit_done = {0}   # 2 项工作完成 1 项 = 50%，不 < 50% → 放行
_emit_path2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "_tmp_7m_emit.gltf")
lp_j._dispatch({"tool": "emit", "args": {"path": _emit_path2}, "why": "落盘"})
check("清单过半完成时 emit 不拒", len([e for e in lp_j.L if e.get("kind") == "emit_gate_reject"]), 0)
for p_ in list(lp_j.emitted):
    try:
        os.remove(p_)
    except OSError:
        pass

# 7n. 方案 2：select 回传旧指纹时给提醒，纯 sid 不误报
# 真实场景加载 + 对 C004 做一次 stretch_axis，再故意回传带旧 bbox_hi 的 dict
lp_k = mk_editloop()
lp_k.phase = "EDIT"
lp_k._call("load", {"path": "../pkg/AI-Geometry-Models/IFC/AI_Geometry_Input.ifc"},
           "载入场景")
lp_k._call("stretch_axis", {"sids": ["C004"], "axis": 0, "anchor_side": "lo", "delta": 1.2},
           "加宽到 6.0（几何已改）")
# 回传旧 bbox_hi（原始 4.8 宽），应触发 stale_fingerprint
# select 不在 EDIT 白名单，_dispatch 会拦；用 _call 直接走到检测块
lp_k._call("select",
           {"items": [{"sid": "C004", "bbox_lo": [4.6, -2.0, -0.12],
                       "bbox_hi": [9.4, 2.0, 0.0], "sx": 4.8}], "where": {}},
           "复核右边界（旧指纹）")
stale_ev = [e for e in lp_k.L if e.get("kind") == "stale_fingerprint"]
check("select 回传旧 bbox_hi 触发 stale_fingerprint", len(stale_ev), 1)
_h = [s for s in lp_k.steps if s.get("hint", "").startswith("select 回传旧指纹")]
check("stale 步骤 hint 提示改用纯 sid 重查", len(_h) >= 1, True)
# 回传纯 sid（不带旧几何字段），不应触发 stale
lp_k._call("select", {"items": [{"sid": "C004"}], "where": {}},
           "用纯 sid 重查（正确做法）")
check("纯 sid 回查不再触发 stale_fingerprint", len(stale_ev), 1)

print("\n%d 通过 / %d 失败" % (ok, fail))
sys.exit(1 if fail else 0)
