# -*- coding: utf-8 -*-
"""selftest.py — 不需要 LLM 的判据单测。

跑：`python selftest.py`（几秒，无网络、无模型）。

这里只测**纯逻辑**：哪些条件该判真、哪些该判假。它们是最容易"松一点就误判、
紧一点就卡死"的地方 —— 比如失败消解的判据，松了会把无关步骤当成补齐，
紧了会让一次 count_mismatch 永远挂着、整轮撞死在门控上（T09 的 89 轮就是这么来的）。
"""
import sys

sys.path.insert(0, ".")

import os  # noqa: E402
import tempfile  # noqa: E402

import tools.ops  # noqa: E402,F401  注册工具
from agent.loop import _resolve_reason, _result_sids, SCOPE_LABELS, Loop  # noqa: E402
from core.runtime import Bus, Ledger  # noqa: E402

FAILED = 0


def check(group, label, got, want):
    global FAILED
    ok = got == want
    if not ok:
        FAILED += 1
    print("%s %-14s %-30s → %-14s (期望 %s)" % (
        "OK  " if ok else "FAIL", group, label, got, want))


def test_resolve():
    """失败消解的四条路径 + 两种必须拒绝的情形。"""
    # 典型失败：要 3 个门框料，只找到 2 个
    f = {"tool": "select", "want": 3, "got": 2, "sids": ["C008", "C009"], "resolved": False}
    cases = [
        ("同工具补对", f, "select", 3, 3, {"C008", "C009", "C010"}, "IDENTIFY", "same_tool"),
        ("换路径但含原结果", f, "clusters", None, 3, {"C008", "C009", "C010"}, "VERIFY", "superset"),
        ("凑够数但不含原结果", f, "clusters", None, 3, {"C004", "C005", "C017"}, "VERIFY", None),
        ("条数不够", f, "clusters", None, 2, {"C008", "C009"}, "VERIFY", None),
        ("原失败一件没找到", {"tool": "select", "want": 3, "got": 0, "sids": [],
                              "resolved": False},
         "clusters", None, 3, {"C004", "C005", "C017"}, "VERIFY", "superset"),
        ("已消解不再判", dict(f, resolved=True), "select", 3, 3, set(), "IDENTIFY", None),
        ("RECOVER 里空过", dict(f, resolved=False), "segment", None, None, set(),
         "RECOVER", "recover_move_on"),
        # 同一个工具、条数也对，但不是同一次判据 —— 仍然算同工具补对（want 一致即可）
        ("同工具 want=None", {"tool": "select", "want": None, "sids": [],
                              "resolved": False}, "select", None, None, set(), "VERIFY", None),
    ]
    for label, ff, name, want, n, sids, ph, exp in cases:
        check("消解", label, _resolve_reason(dict(ff), name, want, n, sids, ph), exp)


def test_result_sids():
    """从各种返回结构里取被指认的对象。"""
    check("取 sid", "groups 内层", _result_sids({"groups": [{"items": [{"sid": "C1"}]}]}), {"C1"})
    check("取 sid", "item 单件", _result_sids({"item": {"sid": "C3"}}), {"C3"})
    check("取 sid", "items 列表", _result_sids({"items": [{"sid": "C1"}, {"sid": "C2"}]}), {"C1", "C2"})
    check("取 sid", "非 dict 返回", _result_sids([1, 2, 3]), set())
    check("取 sid", "没有 sid 字段", _result_sids({"items": [{"x": 1}]}), set())


def _bare_loop():
    """一个不跑 LLM 的 Loop，只用来调纯逻辑方法（审计、门控）。"""
    led = Ledger(os.path.join(tempfile.mkdtemp(), "l.jsonl"))
    return Loop(Bus(led), led, policy=None, card_text="x", card_path="x",
                model_path="x", max_turns=5)


def test_audit_empty():
    """空答案必须被拒 —— 它曾经换来一个绿色的 OK（T09 54 轮后交 {}）。"""
    loop = _bare_loop()
    loop.results["num"] = {"x": 0.5}
    loop.phase = "REPORT"
    loop._audit("s9", {"values": {}, "evidence": ["num"]})
    check("审计", "空 values 被拒", loop.answer is None, True)
    check("审计", "记了一次拒绝", loop.audit_rejects, 1)

    loop2 = _bare_loop()
    loop2.results["num"] = {"x": 0.5}
    loop2.phase = "REPORT"
    loop2._audit("s9", {"values": {"d": 0.5}, "evidence": ["num"]})
    check("审计", "有内容的答案放行", bool(loop2.answer), True)


def test_audit_pointer():
    """只交 note 等于交了个指针 —— T09 用它绕过空答案检查。"""
    loop = _bare_loop()
    loop.results["x"] = {"v": 1}
    loop.phase = "REPORT"
    loop._audit("s9", {"values": {"note": "见 entities"}, "evidence": ["x"]})
    check("审计", "只有 note 被拒", loop.answer is None, True)

    loop2 = _bare_loop()
    loop2.results["x"] = {"v": 1}
    loop2.phase = "REPORT"
    loop2._audit("s9", {"values": {"has_hole": "否"}, "evidence": ["x"]})
    check("审计", "短字符串答案放行", bool(loop2.answer), True)


def test_audit_null():
    """键都在、值全是 null —— 看起来像交了答案，实际一个量都没给。"""
    for vals, want_reject in (({"box_edge_m": None, "r": None}, True),
                              ({"chair_shift_m": "未知"}, True),
                              ({"box_edge_m": 2.5}, False),
                              ({"positions": [1, 2]}, False)):
        loop = _bare_loop()
        # 证据里要有这些数，否则「数字无法追溯」会把它拒掉，测不到 null 这条规则
        loop.results["x"] = {"box_edge_m": 2.5, "positions": [1, 2], "v": 1}
        loop.phase = "REPORT"
        loop._audit("s9", {"values": dict(vals), "evidence": ["x"]})
        check("审计", "null 检测 %s" % list(vals)[0],
              loop.answer is None, want_reject)


def test_audit_precision():
    """模型照抄的是「观察里显示的 3 位小数」，审计必须按同一精度比对。

    写死 4 位时：投影给 0.523、真值 0.523074 → 判「编造」，T09 连拒三次 AUDIT_FAILED。
    """
    loop = _bare_loop()
    loop.results["wall"] = {"fill": 0.523074, "thickness": 0.2}
    loop.phase = "REPORT"
    loop._audit("s9", {"values": {"fill": 0.523}, "evidence": ["wall"]})
    check("审计", "照抄 3 位小数放行", bool(loop.answer), True)

    loop2 = _bare_loop()
    loop2.results["wall"] = {"fill": 0.523074}
    loop2.phase = "REPORT"
    loop2._audit("s9", {"values": {"fill": 0.42}, "evidence": ["wall"]})
    check("审计", "编造的数被拒", loop2.answer is None, True)


def test_scope_labels():
    """语义范围表与审计共用同一份词表，缺一个就会出现「范围外的词审计看不见」。"""
    check("词表", "四个标签", set(SCOPE_LABELS), {"门", "窗", "落地窗", "无法区分"})
    check("词表", "无重复", len(SCOPE_LABELS), len(set(SCOPE_LABELS)))


if __name__ == "__main__":
    test_resolve()
    test_result_sids()
    test_audit_empty()
    test_audit_precision()
    test_audit_pointer()
    test_audit_null()
    test_scope_labels()
    print("\n%s" % ("全部通过" if FAILED == 0 else "%d 项失败" % FAILED))
    sys.exit(1 if FAILED else 0)
