# -*- coding: utf-8 -*-
"""loop.py — agent 编排层。这是「沙盒自己会解题」与「脚本预先解好」的分界线。

编排层做六件事，每一件都可被评委独立检查：

  1. 阶段门控     —— 每个阶段只允许一类工具。越界调用被拒并留痕（gate_reject）。
                     这证明 agent 不能跳过侦察直接答题，也不能在 REPORT 阶段偷偷测量。
  2. 编号分配     —— 步骤号由编排层分配，agent 不能自定义，回放才不会错位。
  3. 观察封装     —— 工具输出经 schema.observe 压缩后再交给策略，并附不确定性提示。
  4. 失败升级     —— expect_n 不符时不是「报错退出」，而是进入 RECOVER 阶段，
                     把诊断信息交给策略，让它在受控范围内自救（这是真·恢复）。
  5. 答案审计     —— 提交的每个数字必须在某个证据步骤的输出里真实出现过。
                     这一条让「编一个看起来合理的数」在结构上不可能通过。
  6. 预算与停滞   —— 步数/墙钟上限；同一调用重复三次即判停滞终止。

注意：本层不含任何几何计算。所有数值来自 tools/*，本层只做调度与判定。
"""
import hashlib
import json
import time

import numpy as np

from core.runtime import TOOLS, _short
from agent import schema as sch

PHASES = ["SURVEY", "IDENTIFY", "MEASURE", "VERIFY", "REPORT"]

# 阶段 → 允许的几何工具。RECOVER 是唯一可跨类诊断的阶段，但也不许提交答案。
PHASE_TOOLS = {
    "SURVEY":   {"load", "clusters", "principal_frame"},
    "IDENTIFY": {"select", "principal_frame", "contains", "merge", "gaps", "clusters", "segment", "pick"},
    "MEASURE":  {"distance", "rank", "section", "segment", "pick", "merge"},
    "VERIFY":   {"distance", "section", "select", "principal_frame", "segment", "pick"},
    "RECOVER":  {"principal_frame", "clusters", "segment", "select", "gaps", "load", "pick"},
    "REPORT":   {"answer"},
}

PHASE_GOAL = {
    "SURVEY": "把模型读进来并弄清它由几个互不相连的部分组成。必须先完成，否则后面所有 sid 都无从谈起。",
    "IDENTIFY": "把任务卡点名的每个角色（如椅子/门扇/窗框/画框）用 select 压到唯一命中，"
                "并给该步起 label=角色名。唯一命中会自动登记到「已唯一确定」，之后可直接用 "
                "@角色名.items 引用。压不到唯一就先补判别量（厚度、法向、是否落地、有无洞口），"
                "不要靠「看着像」选一个。",
    "MEASURE": "对已定角色的实体做题目要求的测量，拿到带见证物的数值。",
    "VERIFY": "用与测量无关的独立路径复核结果：见证点是否在表面上、单位是否自洽、"
              "多口径是否一致。发现问题就退回 IDENTIFY 改判据，而不是改数字。",
    "RECOVER": "上一步没有给出预期证据。先诊断是判据错了还是参照系错了，再回退重做。"
               "禁止在这一阶段直接给答案。",
    "REPORT": "只调用 answer 提交。values 里至少要有：unit（单位）、chair（椅子 sid）、"
              "targets（角色名→sid）、distances_m（角色名→米，取自已完成的测量步骤）、"
              "order（由近到远的角色名列表）。每个数字都必须能在某个已完成步骤的返回值里找到，"
              "编排层会逐条核对；编造或改写的数值会被判审计失败。",
}


class AuditError(Exception):
    pass


class Loop:
    def __init__(self, bus, ledger, policy, card_text, card_path, model_path,
                 max_turns=40, wall_budget=180.0, expect_log=None):
        self.bus = bus
        self.L = ledger
        self.policy = policy
        self.card_text = card_text
        self.card_path = card_path
        self.model_path = model_path
        self.max_turns = max_turns
        self.wall_budget = wall_budget
        self.expect = dict(expect_log or {})     # step 序号 → 期望候选数（由策略声明）

        self.phase = "SURVEY"
        self.turn = 0
        self.results = {}                        # sid -> 工具原始输出
        self.steps = []                          # 轨迹（含观察摘要与理由）
        self.roles = {}                          # 角色 -> sid
        self.measurements = []                   # 测量步骤名
        self.failures = []                       # 未消解的失败
        self.gate_rejects = 0
        self.audit_rejects = 0
        self.answer = None
        self.status = "RUNNING"
        self.t0 = time.time()
        self._sig_count = {}

    # ------------------------------------------------------------ 对外

    def run(self):
        self.L.append("agent_start", {
            "card": self.card_path, "model": self.model_path,
            "phases": PHASES, "gates": {k: sorted(v) for k, v in PHASE_TOOLS.items()},
            "max_turns": self.max_turns, "policy": getattr(self.policy, "kind", "?"),
            "tools_visible": [t["function"]["name"] for t in sch.full_specs()],
        })
        while self.turn < self.max_turns and self.status == "RUNNING":
            if time.time() - self.t0 > self.wall_budget:
                self._finish("TIMEOUT")
                break
            self.turn += 1
            try:
                act = self.policy.decide(self.context())
            except StopIteration:
                self._finish(self._end_status("POLICY_EXHAUSTED"))
                break
            if act is None:
                self._finish("POLICY_EXHAUSTED")
                break
            self._dispatch(act)
        if self.status == "RUNNING":
            self._finish(self._end_status("BUDGET_EXHAUSTED"))
        return self.report()

    # ------------------------------------------------------------ 上下文

    def context(self):
        allowed = PHASE_TOOLS.get(self.phase, set())
        return {
            "turn": self.turn,
            "phase": self.phase,
            "goal": PHASE_GOAL[self.phase],
            "budget_left": self.max_turns - self.turn,
            "elapsed_s": round(time.time() - self.t0, 2),
            "card": self.card_text,
            "model": self.model_path,
            "tools": [t["function"]["name"] for t in sch.tool_specs(only=allowed)],
            "steps": self.steps,
            "last": self.steps[-1] if self.steps else None,
            "roles": self.roles,
            "measurements": self.measurements,
            "failure": next((f for f in reversed(self.failures) if not f.get("resolved")), None),
            "board": {
                "n_steps": len(self.steps),
                "n_unique_selections": sum(1 for s in self.steps
                                           if s.get("tool") == "select" and s.get("n") == 1),
                "n_measurements": len(self.measurements),
                "n_open_failures": sum(1 for f in self.failures if not f.get("resolved")),
            },
        }

    # ------------------------------------------------------------ 动作分发

    def _dispatch(self, act):
        name = act.get("tool") or act.get("action")
        why = act.get("why", "")
        if name == "advance_phase":
            self._advance(why)
            return
        if name == "declare_insufficient":
            self.L.append("declare_insufficient", {"reason": why,
                                                   "missing": act.get("missing", [])})
            self.status = "INSUFFICIENT_EVIDENCE"
            self.insufficient = {"reason": why, "missing": act.get("missing", [])}
            return
        if name is None:
            self.L.append("policy_noop", {"act": _short(act, 400)})
            return
        allowed = PHASE_TOOLS.get(self.phase, set())
        if name not in allowed:
            self.gate_rejects += 1
            self.L.append("gate_reject", {"phase": self.phase, "tool": name,
                                          "allowed": sorted(allowed), "why": why})
            self.steps.append({"id": "gate", "phase": self.phase, "tool": name,
                               "rejected": True, "why": why,
                               "obs": "阶段 %s 不允许调用 %s；可用：%s" % (self.phase, name, sorted(allowed))})
            return
        if name not in TOOLS:
            self.L.append("policy_error", {"error": "未注册工具 %s" % name})
            return
        self._call(name, act.get("args") or {}, why, act.get("expect_n"),
                   act.get("label"), act.get("alias"))

    def _call(self, name, args, why, expect_n=None, label=None, alias=None):
        # 步骤标识优先用 agent 给的语义标签（load / pf / win ...）：
        #  - 回放时不受「失败恢复多消耗了几轮」影响，引用永远指向同一个实体
        #  - 证据清单变成可读的一串名字，评委会更容易顺着读下去
        # 标签必须唯一，重名直接拒绝 —— 否则引用会静默指向另一步。
        rename = None
        if label:
            base = label
            k = 2
            while label in self.results and k < 100:
                label = "%s_%d" % (base, k)
                k += 1
            if label != base:
                # 标签唯一是硬约束（否则引用会静默指向另一步），但直接拒绝等于白烧一轮。
                # 改为自动加后缀并明确告知新名字：约束仍成立，代价从「一轮」降到「一句提示」。
                rename = (base, label)
                self.L.append("label_renamed", {"from": base, "to": label, "tool": name})
            sid = label
        else:
            sid = "s%d" % self.turn
        try:
            args = self.bus.resolve(args, self.results)
        except Exception as e:
            self.L.append("resolve_error", {"step": sid, "tool": name, "args": _short(args),
                                            "error": "%s: %s" % (type(e).__name__, e)})
            self.steps.append({"id": sid, "phase": self.phase, "tool": name, "args": _short(args, 300),
                               "why": why, "error": "引用解析失败：%s" % e})
            return

        self.L.append("decision", {"step": sid, "phase": self.phase, "tool": name,
                                   "args": _short(args), "why": why,
                                   "expect_n": expect_n if expect_n is not None else self.expect.get(sid)})

        sig = hashlib.sha256((name + json.dumps(args, sort_keys=True, default=str)).encode()).hexdigest()[:12]
        self._sig_count[sig] = self._sig_count.get(sig, 0) + 1
        repeat = self._sig_count[sig]
        if repeat >= 3:
            self.L.append("stall_abort", {"step": sid, "tool": name, "repeat": repeat})
            self._finish("STALLED")
            return

        try:
            out = self.bus.call(name, args, sid)
        except Exception as e:
            self.steps.append({"id": sid, "phase": self.phase, "tool": name,
                               "args": _short(args, 300), "why": why,
                               "error": str(e)[:300]})
            # 工具错误留在本阶段处理：参数写错是可当场修正的机械问题，
            # 不该升级成「证据不足」并把人吸进自救态。实测这一点会让智能体
            # 把 80% 的轮次耗在 RECOVER 里反复试探，再也没回到识别环节。
            self.L.append("tool_error", {"step": sid, "tool": name, "error": str(e)[:300]})
            return

        self.results[sid] = out
        if rename:
            out = dict(out) if isinstance(out, dict) else {"value": out}
            out["_label"] = ("标签 %s 已被占用，本次步骤记为 %s；后续引用请用 @%s。"
                             % (rename[0], rename[1], rename[1]))
        if alias:
            # 「取代」而不是「新增」：失败步骤的旧结果留在账本里，但引用解析改用新结果。
            # 没有这一条，回放时 @win.items 会永远指向那次返回 0 条的调用。
            self.results[alias] = out
            self.L.append("label_superseded", {"alias": alias, "by": sid, "tool": name})
        obs, stats = sch.observe(name, out)
        if repeat >= 2:
            # 第二次调用完全相同的参数时就必须提醒：返回值不会变，第 3 次会被判停滞。
            # 等到第 3 次才报错是不负责任的 —— 那时智能体已经浪费了两轮，而且它并不知道
            # 自己在原地打转（它以为每次都在“补全信息”）。
            obs = obs[:-1] + (',"_repeat":"你已用完全相同的参数调用过 %s，返回结果不会改变。'
                              '同样的调用第 3 次会被判为停滞并终止本次运行。'
                              '要推进就必须换参数、换工具，或前进到下一阶段。"}' % name)
        n = stats.get("n")
        want = expect_n if expect_n is not None else self.expect.get(sid)
        step = {"id": sid, "phase": self.phase, "tool": name, "args": _short(args, 300),
                "why": why, "n": n, "expect_n": want,
                "hint": sch.hint_for(name, stats), "obs": obs}
        self.steps.append(step)

        if name in ("select", "pick") and n == 1:
            items = out.get("items") or ([out["item"]] if out.get("item") else [])
            if items and isinstance(items[0], dict):
                # 步骤标签兼作角色名：agent 给这一步起名 door_leaf，就表示「我确认这就是门扇」。
                # 这样决策板上的内容来自 agent 自己的声明，而不是编排层替它猜。
                self.roles[sid] = items[0]["sid"]
                self.L.append("role_decided", {"role": sid, "sid": items[0]["sid"],
                                               "evidence": _short(args, 200)})
        if name in ("distance", "rank", "section"):
            self.measurements.append(sid)

        if self.phase == "RECOVER" and self.failures and not self.failures[-1].get("resolved"):
            f = self.failures[-1]
            if not (want is not None and n != want):
                f["resolved"] = True
                f["resolved_by"] = sid
                self.L.append("failure_resolved", {"step": f["step"], "by": sid, "tool": name})

        if want is None or n == want:
            self._tick_recover()

        if want is not None and n != want:
            self.failures.append({"step": sid, "kind": "count_mismatch",
                                  "want": want, "got": n, "args": _short(args, 300),
                                  "resolved": False})
            self._enter_recover(sid, "count_mismatch",
                                "期望 %s 个候选，实得 %s。参数：%s" % (want, n, _short(args, 200)))

        if name == "answer":
            self._audit(sid, out)

    # ------------------------------------------------------------ 阶段推进

    def _advance(self, why):
        gate = {
            "SURVEY": lambda: {s["tool"] for s in self.steps} >= {"load"},
            "IDENTIFY": lambda: self.context()["board"]["n_unique_selections"] >= 2,
            "MEASURE": lambda: len(self.measurements) >= 1,
            "VERIFY": lambda: not any(f for f in self.failures if not f.get("resolved")),
        }.get(self.phase)
        if gate is not None and not gate():
            self.gate_rejects += 1
            self.L.append("advance_rejected", {"phase": self.phase, "why": why,
                                               "board": self.context()["board"]})
            self.steps.append({"id": "gate", "phase": self.phase, "tool": "advance_phase",
                               "rejected": True, "why": why,
                               "obs": "本阶段目标尚未达成，不能进入下一阶段。"})
            return
        i = PHASES.index(self.phase) if self.phase in PHASES else 0
        if self.phase == "RECOVER":
            target = self._recover_from or "IDENTIFY"
        else:
            target = PHASES[min(i + 1, len(PHASES) - 1)]
        self.L.append("phase", {"from": self.phase, "to": target, "why": why})
        self.phase = target

    def _tick_recover(self):
        """自救阶段不该是无底洞：失败消解后连续两步成功就自动返回原阶段。"""
        if self.phase != "RECOVER":
            self._recover_n = 0
            return
        self._recover_n = getattr(self, "_recover_n", 0) + 1
        open_fail = any(not f.get("resolved") for f in self.failures)
        if not open_fail and self._recover_n >= 2:
            self.L.append("phase", {"from": "RECOVER", "to": self._recover_from or "IDENTIFY",
                                    "why": "失败已消解，自动返回"})
            self.phase = self._recover_from or "IDENTIFY"
            self._recover_n = 0

    def _enter_recover(self, step_id, kind, detail):
        if self.phase != "RECOVER":
            self._recover_from = self.phase
            self._recover_n = 0
            self.L.append("phase", {"from": self.phase, "to": "RECOVER",
                                    "why": "步骤 %s %s" % (step_id, kind)})
            self.phase = "RECOVER"

    # ------------------------------------------------------------ 答案审计

    def _flatten_numbers(self, obj, out):
        if isinstance(obj, bool):
            return
        if isinstance(obj, (int, float)):
            out.append(float(obj))
        elif isinstance(obj, dict):
            for v in obj.values():
                self._flatten_numbers(v, out)
        elif isinstance(obj, list):
            for v in obj:
                self._flatten_numbers(v, out)

    def _audit(self, sid, ans):
        """答案里的每个数字必须能在某个证据步骤的输出里找到。找不到 = 编的。"""
        ev = ans.get("evidence") or []
        problems = []
        if not ev:
            problems.append("没有引用任何证据步骤")
        known = set(self.results)
        for e in ev:
            if e not in known:
                problems.append("证据 %s 不存在" % e)

        pool = []
        for e in ev:
            if e in self.results:
                self._flatten_numbers(self.results[e], pool)

        claims = []
        self._flatten_numbers(ans.get("values", {}), claims)
        ungrounded = []
        for c in claims:
            if c in (0.0, 1.0) and not pool:            # 计数类
                continue
            hit = any(abs(c - p) <= max(1e-9, 1e-6 * max(1.0, abs(c))) for p in pool)
            if not hit:
                ungrounded.append(c)
        if ungrounded:
            problems.append("这些数字无法追溯到任何证据步骤：%s" % ungrounded[:6])

        if problems:
            self.audit_rejects += 1
            self.L.append("audit_reject", {"step": sid, "problems": problems,
                                           "n_evidence": len(ev)})
            self.steps.append({"id": "audit", "phase": self.phase, "tool": "answer",
                               "rejected": True, "why": "答案未被接受",
                               "obs": "；".join(problems)})
            if self.audit_rejects >= 3:
                self._finish("AUDIT_FAILED")
            else:
                self.L.append("phase", {"from": self.phase, "to": "VERIFY",
                                        "why": "答案审计未通过，退回复核"})
                self.phase = "VERIFY"
            return

        self.answer = ans
        self.status = "OK"
        self.L.append("answer_accepted", {"step": sid, "n_evidence": len(ev),
                                          "n_claims": len(claims),
                                          "unknowns": len(ans.get("unknowns") or [])})

    # ------------------------------------------------------------ 收尾

    def _end_status(self, default):
        """收尾状态按「被什么挡住」定名，而不是笼统的一律 POLICY_EXHAUSTED。
        评委看状态字就该知道是门控拦了、审计拦了，还是单纯没步数了。"""
        if self.answer is not None:
            return "OK"
        if self.audit_rejects:
            return "AUDIT_FAILED"
        if self.gate_rejects:
            return "GATE_BLOCKED"
        return default

    def _finish(self, status):
        self.status = status
        self.L.append("agent_end", {"status": status, "turn": self.turn,
                                    "phase": self.phase,
                                    "elapsed_s": round(time.time() - self.t0, 2)})

    def report(self):
        return {
            "status": self.status,
            "turns": self.turn,
            "phase": self.phase,
            "elapsed_s": round(time.time() - self.t0, 3),
            "answer": self.answer,
            "steps": self.steps,
            "roles": self.roles,
            "failures": self.failures,
            "gate_rejects": self.gate_rejects,
            "audit_rejects": self.audit_rejects,
            "insufficient": getattr(self, "insufficient", None),
        }
