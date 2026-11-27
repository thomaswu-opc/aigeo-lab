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
import copy
import hashlib
import json
import re
import time

import numpy as np

from core.runtime import TOOLS, _short
from agent import schema as sch

PHASES = ["SURVEY", "IDENTIFY", "MEASURE", "VERIFY", "EDIT", "REVALIDATE", "REPORT"]

# 阶段 → 允许的几何工具。RECOVER 是唯一可跨类诊断的阶段，但也不许提交答案。
#
# EDIT / REVALIDATE 是写层的两个阶段，它们存在的理由只有一条：
# 编辑不许自证。改完必须落盘、重新读回、用与编辑无关的只读工具重量一遍。
# 所以 REVALIDATE 里没有编辑工具 —— 验证阶段继续改，就永远存在一个没被验证的中间态。
# 语义范围表用的词。它既给出"每族允许哪些词"，也给出审计要认的"语义标签"集合 ——
# 两处用同一份，才不会出现"范围外的词审计看不见"。
SCOPE_LABELS = ("门", "窗", "落地窗", "无法区分")

# 「结构类」只读工具：它们的返回是**集合或形状**，同一份返回就意味着同一个问题。
# 换个措辞（items 从 sid 换成指纹、label 换个名）签名会变，但产物一字不差 ——
# 这正是现有「参数相同」检测的盲区。测量类工具不在此列：同一个距离在两个对称面上
# 重复出现是合法的，那里用「有没有新数字」判断更准（见 _note_fresh_numbers）。
STRUCT_TOOLS = {"select", "pick", "clusters", "principal_frame", "parts", "profile",
                "pockets", "levels", "openings", "assembly", "merge", "gaps", "contains"}


def _out_fp(obs):
    """返回内容指纹：去掉编排层追加的引导字段后再哈希。"""
    if not obs:
        return None
    t = re.sub(r'"(?:_repeat|_next|_reload|_same_hits|_phase_forced|_label|_ref)"\s*:\s*'
               r'("(?:[^"\\]|\\.)*"|\[[^\]]*\]|\{[^{}]*\})', '"~"', obs)
    return hashlib.sha256(t.encode("utf-8")).hexdigest()[:16]

PHASE_TOOLS = {
    "SURVEY":   {"load", "clusters", "principal_frame"},
    "IDENTIFY": {"select", "principal_frame", "contains", "merge", "gaps", "clusters", "segment", "pick",
                 # 拆部件与看截面：一个实体里到底有几件东西、某高度上截面是什么形状
                 # 顶点高度分层：特征平面（承托面、槽底、插脚区间）只能靠它找
                 "parts", "profile", "pockets", "levels",
                 # 面内缺料：洞口是不是「一个盒子」不重要，重要的是这面墙上哪里没料
                 "openings",
                 # 洞口族：谁挂在洞口上（接触+连通）。识别门窗的第一步不是看形状，是看归属
                 "assembly"},
    "MEASURE":  {"distance", "rank", "section", "segment", "mesh_health",
                 # select 必须在测量阶段可用：测量天然要「先收窄到一组候选，再逐项量」。
                 # 之前只在 IDENTIFY 放行，导致模型在 MEASURE 里想按条件重取子集时被门控拒绝，
                 # 只能反复重切截面 —— 实测 T01 第 1 问就死在这一步（52 轮 STALLED）。
                 "select", "footprint", "clearance", "opening_edge", "pick", "merge", "contains",
                 # 运动层：接触/穿插判定、旋转与平移扫掠；配准
                 "collide", "sweep_rotate", "sweep_translate", "regid2d",
                 # 形状层：截面轮廓、轴向凹坑、顶点分层
                 "profile", "pockets", "levels",
                 # 反解与拟合：理想球的半径、投影面积、按面积反求高度、派生量算术
                 # angle：两个平面/两根轴的夹角（门扇平面 vs 墙面就靠它，法向来自 dominant）
                 "proj_area", "fit_sphere", "solve_section_height", "calc", "angle",
                 # 面内缺料：洞口位置与下沿标高（量「距门多远」「洞口宽高」都要它）
                 "openings",
                 # 洞口族：族的成员清单与接触证据（fill / edge / linked）
                 "assembly",
                 # 区域关系与贴合面积：球在区域内的体积、面接触 vs 边角接触
                 "sphere_region", "contact_area"},
    # answer 在这里也放行：复核本来就可能「最后一次复核就是最后一次确认」，
    # 逼它先跨到 REPORT 才能交卷，实测会变成「不敢交卷 → 反复复核同一个调用 → 停滞」。
    "VERIFY":   {"answer", "distance", "section", "select", "principal_frame", "segment",
                 "mesh_health", "footprint", "clearance", "pick", "contains",
                 "collide", "sweep_rotate", "sweep_translate", "profile", "pockets", "levels",
                 "proj_area", "fit_sphere", "solve_section_height", "calc", "angle",
                 "openings", "assembly", "sphere_region", "contact_area"},
    "EDIT":     {"translate", "stretch_axis", "cut_plane", "rotate_axis", "reopen_round", "emit",
                 "mesh_health", "footprint", "clearance", "opening_edge", "segment",
                 "select", "pick", "merge", "section", "distance", "gaps", "contains",
                 # 改的过程中要能随时回看形状和碰撞，否则只能「改完再赌一次」
                 "profile", "collide", "sweep_rotate", "sweep_translate", "pockets", "levels"},
    "REVALIDATE": {"load", "clusters", "principal_frame", "select", "pick", "merge",
                   "mesh_health", "footprint", "clearance", "opening_edge",
                   "section", "distance", "rank", "segment", "contains", "gaps",
                   # 复核「有没有撞上」和「形状对不对」必须用只读工具，且必须在读回的文件上做
                   "collide", "sweep_rotate", "sweep_translate", "profile", "pockets", "levels",
                   "proj_area", "fit_sphere", "solve_section_height", "calc", "angle",
                   "openings", "assembly", "sphere_region", "contact_area"},
    "RECOVER":  {"principal_frame", "clusters", "segment", "select", "gaps", "load", "pick"},
    "REPORT":   {"answer"},
}

# 会改变几何的工具。一旦调用，所有「同参数 = 同结果」的缓存与重复判定都必须失效。
EDIT_TOOLS = {"translate", "stretch_axis", "cut_plane", "rotate_axis"}

PHASE_GOAL = {
    "SURVEY": "把模型读进来并弄清它由几个互不相连的部分组成。必须先完成，否则后面所有 sid 都无从谈起。",
    "IDENTIFY": "把任务卡点名的每个角色（如椅子/门扇/窗框/画框）用 select 压到唯一命中，"
                "并给该步起 label=角色名。唯一命中会自动登记到「已唯一确定」，之后可直接用 "
                "@角色名.items 引用。压不到唯一就先补判别量（厚度、法向、是否落地、有无洞口），"
                "不要靠「看着像」选一个。",
    "MEASURE": "对已定角色的实体做题目要求的测量，拿到带见证物的数值。",
    "VERIFY": "用与测量无关的独立路径复核结果：见证点是否在表面上、单位是否自洽、"
              "多口径是否一致。发现问题就退回 IDENTIFY 改判据，而不是改数字。",
    "EDIT": "按任务卡改动几何：translate 整体平移 / rotate_axis 绕指定轴旋转 / "
            "stretch_axis 沿轴分段重映射（加宽、拉高）/ cut_plane 平面切割并封口。"
            "每步都会返回改动前后的可核对量。改完必须 emit 落盘。"
            "若任务本来就只要求测量、不需要改任何几何，直接 advance_phase —— 它会跳到 REPORT，"
            "不要为了「待在这一阶段」去动一下模型。",
    "REVALIDATE": "读回刚 emit 的文件，用只读工具重新量一遍改动前后的量（体积、尺寸、开口面积、封闭性、"
                  "距离、接触与扫掠）。禁止在这一阶段继续改几何 —— 只有落盘后的文件被独立量过，编辑才算成立。",
    "RECOVER": "上一步没有给出预期证据。先诊断是判据错了还是参照系错了，再回退重做。"
               "禁止在这一阶段直接给答案。",
    "REPORT": "只调用 answer 提交。values 按任务卡要求的口径填（数值 + 单位 + 参照对象），"
              "evidence 列出支撑每个数值的已完成步骤 id。每个数字都必须能在某个已完成步骤的返回值里"
              "找到（1e-6 相对容差，或与该步骤的展示精度一致），"
              "编排层会逐条核对；编造或改写的数值会被判审计失败。",
}


def _result_sids(out):
    """从任意工具返回里取出「被指认的对象」清单（items / solids / groups[].items / item）。

    用途：判断后来的某一步是不是**补齐了**某次失败当时缺的那几件东西。
    只认里层 dict 的 sid 字段，不猜结构 —— 认不出来的返回空集，宁可不消解。
    """
    if not isinstance(out, dict):
        return set()
    got = set()
    for key in ("items", "solids", "rows", "results"):
        v = out.get(key)
        if isinstance(v, list):
            for it in v:
                if isinstance(it, dict) and it.get("sid"):
                    got.add(it["sid"])
    it = out.get("item")
    if isinstance(it, dict) and it.get("sid"):
        got.add(it["sid"])
    for g in (out.get("groups") or []):
        if isinstance(g, dict):
            for m in (g.get("items") or []):
                if isinstance(m, dict) and m.get("sid"):
                    got.add(m["sid"])
    return got


def _resolve_reason(f, name, want, n, cur_sids, phase):
    """这一步能否消解失败 f。返回 same_tool / superset / recover_move_on / None。

    抽成纯函数是为了能单测：这四条判据的松紧直接决定「会不会把无关步骤当成补齐」。
    """
    if f.get("resolved"):
        return None
    if f.get("tool") in (None, name) and want is not None and n == want:
        return "same_tool"
    if (f.get("want") is not None and n == f["want"]
            and set(f.get("sids") or ()) <= set(cur_sids)):
        # 换路径补齐：条数到位，且原来已经找到的那几件仍在结果里。
        # 只比条数会把「碰巧也是 3 个」的无关步骤当成消解。
        return "superset"
    if phase == "RECOVER" and want is None:
        return "recover_move_on"
    return None


class AuditError(Exception):
    pass


class Loop:
    def __init__(self, bus, ledger, policy, card_text, card_path, model_path,
                 max_turns=40, wall_budget=180.0, expect_log=None, read_only=False,
                 on_tick=None):
        self.bus = bus
        # 每轮末尾的回调：只用来出「进行中的画面」（run_agent 里的 live 预览）。
        # 不参与决策、不改场景，异常一律吞掉 —— 渲染失败不该让解题中断。
        self.on_tick = on_tick
        # 只读任务（原话只要求量、不要求改）不该被赶进 EDIT：那一阶段没有任何能用的工具，
        # 模型只能「用嘴交卷」——实测它会输出一段 Markdown 结论、没有工具调用，
        # 编排层只好判成 declare_insufficient。既然一句几何都没改，写层对它是空集。
        self.read_only = bool(read_only)
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
        self.facts = {}                          # 语义事实：sid -> {grounded, kind, thick, ...}
        self.results_memo = []                   # 结构化结果记忆：每步提炼的 {sid, tool, result} 摘要
        self.measurements = []                   # 测量步骤名
        self.failures = []                       # 未消解的失败
        self.gate_rejects = 0
        self.audit_rejects = 0
        self.edits = []                          # 成功的编辑步骤（translate/stretch/cut）
        self.emitted = set()                     # emit 写出的文件（绝对路径）
        self.reloaded = set()                    # 读回并确认是 emit 产物的文件
        self.answer = None
        self.status = "RUNNING"
        self.t0 = time.time()
        self._sig_count = {}
        self._sig_obs = {}          # 同一参数的成功返回：重复只读调用直接复用
        self._sig_out = {}          # 缓存对应的原始返回，用来把新步骤登记进 results
        self._sig_first = {}        # 同一签名的第一次步骤 id：重复时直接告诉它去引用那一步
        self._sig_err = {}          # 同一签名失败时的报错原文：失败调用没有返回可缓存，但要能重放提示
        self._out_seen = {}         # 返回值指纹 -> 首次产生的步骤：抓「换措辞问同一件事」
        self.tool_of = {}           # 步骤 id -> 工具名（引用别名要用：@load → 实际那一步）
        self._closing = False       # 是否已进入收口窗口（预算将尽，必须交卷）
        self._sid_n = 0             # 步骤号计数器（与 turn 解耦）
        self._free = 0              # 已退还的轮次数（重复只读调用不烧预算）
        self._null_rejects = 0      # 「量到了却填 null」被拒的次数（只拒一次，防死锁）
        self.closing_note = None    # 收口提示（进上下文，也进账本）
        self._last_select = None    # 上一步 select 的命中集合（判断"换条件有没有换来新信息"）
        self._loaded = {}           # 已读过哪些模型文件（重复 load 不重算，也不清去重计数）
        # 「这一步到底带回了新数字吗」—— 换参数而不换判据的打转，签名完全相同那条抓不到。
        # 实测 T01 第 1 问在旋转副本上就是这样耗掉 60 轮的：op_w1b / op_final / op_final2 /
        # op_o005 换着名字反复查同一面墙，每次的参数都不一样，返回的数一字不差。
        self._seen_nums = set()     # 本次运行已出现过的量（1e-6 归并）
        self._meas_fresh = []       # 每个测量步骤带回的新量个数
        self._forced_from = set()   # 已被预算纪律代为推进过的阶段
        self.forced = 0             # 代为推进的次数（进报告，不藏）
        self._spin = 0              # 连续「没有带回新东西」的步数（换皮重问、命中缓存都算）
        self.role_evidence = []     # 每个「唯一命中」角色附带的判据反例覆盖
        self.families = {}          # 洞口族：族 id -> {wall, facts, label_scope}（审计依据）
        self.assembly_steps = []    # 算出族的那些步骤 id（语义标签必须引用它们）

    # ------------------------------------------------------------ 对外

    def run(self):
        self.L.append("agent_start", {
            "card": self.card_path, "model": self.model_path,
            "phases": PHASES, "gates": {k: sorted(v) for k, v in PHASE_TOOLS.items()},
            "max_turns": self.max_turns, "policy": getattr(self.policy, "kind", "?"),
            "read_only": self.read_only,
            "tools_visible": [t["function"]["name"] for t in sch.full_specs()],
        })
        while self.turn < self.max_turns and self.status == "RUNNING":
            if time.time() - self.t0 > self.wall_budget:
                self._finish("TIMEOUT")
                break
            self._maybe_close()
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
            if self.on_tick is not None:
                try:
                    self.on_tick(self)
                except Exception:
                    pass
        if self.status == "RUNNING":
            self._finish(self._end_status("BUDGET_EXHAUSTED"))
        return self.report()

    # ------------------------------------------------------------ 上下文

    # 测量类工具的返回里，这些键才是「结论」；其余是过程量或噪声。
    NUM_KEYS = ("value", "distance", "angle_deg", "acute_deg", "area", "volume",
                "dot", "clearance", "gap", "width", "height", "depth", "shift")
    NOISE_KEYS = {"n", "ntri", "edge_pairs_evaluated", "share_a", "share_b", "k", "idx", "unit"}

    def _draft_board(self, limit=12):
        """把「已经量到的数」摆成一张可交付的草稿板。

        针对的是最贵的一类失败：**算对了没交**。live_1126_103948 把三道 distance
        （0.6965 / 2.2767 / 3.8019）和一次 angle（acute_deg=65）都量出来了，
        交卷时这几个键全填 null，unknowns 还写着「工具未返回可用数值」。
        原因不是它不肯填：那些数字已经滑出「最近观察」窗口，总览表里只有步骤名和工具名，
        不含数值 —— 它在组装答案的那一刻手里真的没有这些数。

        草稿板只做搬移：哪一步、什么工具、量出了哪些键值。不做语义判断，
        也不替它决定哪个数填哪个键（那是它的活，也是审计要认的东西）。
        """
        rows = []
        for sid in self.measurements[-limit:]:
            st = next((x for x in reversed(self.steps) if x.get("id") == sid), None)
            if not st:
                continue
            pairs, seen = [], set()
            for m in re.finditer(r'"([a-z_][a-z0-9_]*)"\s*:\s*(-?\d+(?:\.\d+)?)', st.get("obs") or ""):
                k, v = m.group(1), m.group(2)
                if k in self.NOISE_KEYS or (k, v) in seen:
                    continue
                seen.add((k, v))
                pairs.append((k, v))
            if not pairs:
                continue
            # 结论量排前面：模型一眼要看到的是 value / acute_deg，不是 witness 的坐标分量
            pairs.sort(key=lambda kv: 0 if kv[0] in self.NUM_KEYS else 1)
            rows.append("  %s  %s(%s) → %s" % (
                sid, st.get("tool"), (st.get("args") or "")[:60],
                ", ".join("%s=%s" % kv for kv in pairs[:5])))
        if not rows:
            return None
        return ("你已经量到这些数（工具原样返回，直接引用，不要重算、不要凭记忆写）：\n"
                + "\n".join(rows))

    def _has_numbers(self):
        """本次运行是否真的量出过数（有测量步骤、且其观察里带数值）。"""
        return any(s.get("id") in self.measurements and re.search(r":\s*-?\d", s.get("obs") or "")
                   for s in self.steps)

    def _maybe_close(self):
        """收口窗口：预算将尽时，把「继续取数」和「交卷」的收益差写在明处。

        实测最贵的一类失败不是算错，而是**算对了没交**：T06 已经量到净宽 6.0，
        然后在剩余 40 轮里反复复核，最后 99/100 用完、状态 BUDGET_EXHAUSTED，
        答案里一个数都没有。终局是「什么也没交」，而它手里其实有数。
        所以到了窗口就明确说：现在唯一会改变结果的动作是交卷。
        阶段一并推到 VERIFY（那里 answer 是放行的）—— 这不是替它思考，是把
        「还剩多少预算」这个它看不见的事实告诉它。
        """
        if self._closing or self.answer is not None:
            return False
        # 回放录制的验收链不能被预算启发式扰动：那 9 条记录的意义是「同一串决定
        # 必然得到同一个结果」。回放本来就不吃模型，也不该吃收口提示。
        if getattr(self.policy, "kind", "") == "recorded":
            return False
        # 一步测量都没有的时候不催交卷：那不是「算对没交」，是还没开始算。
        # 催它只会得到一份空答案，反而把 BUDGET_EXHAUSTED 变成 AUDIT_FAILED。
        if not self.measurements:
            return False
        left_turn = self.max_turns - self.turn
        left_time = self.wall_budget - (time.time() - self.t0)
        # 下限取 4 而不是 6：小预算（10 轮）下取 6 会在第 4 轮就催交卷，
        # 那不是收口，是中途打断。真实 bench 是 100 轮，窗口落在第 85 轮。
        near = (left_turn <= max(4, int(self.max_turns * 0.15))
                or left_time <= max(20.0, self.wall_budget * 0.15))
        if not near:
            return False
        self._closing = True
        before = self.phase
        # EDIT 不动：改完必须 emit，跑到一半收口等于什么都没做。
        if before in ("SURVEY", "IDENTIFY", "MEASURE", "RECOVER", "REVALIDATE"):
            self.phase = "VERIFY"
        self.closing_note = (
            "收口窗口：只剩 %d 轮 / %.0f 秒。从现在起，**取数不再改变结论** ——"
            "把已经量到的写进 answer.values（每个数都要能指到某个步骤），"
            "确实没量出来的写进 unknowns 并说明缺什么；已经有数却写 null 会被判「不算交卷」。"
            "如果还有一步没做完，先用一句话判断它会不会改变某个数：不会，就现在交卷。"
            % (left_turn, max(0.0, left_time)))
        board = self._draft_board()
        if board:
            self.closing_note += "\n" + board
        self.L.append("closing_window", {"from": before, "to": self.phase,
                                         "left_turn": left_turn,
                                         "left_s": round(max(0.0, left_time), 1)})
        return True

    def context(self):
        allowed = PHASE_TOOLS.get(self.phase, set())
        return {
            "turn": self.turn,
            "phase": self.phase,
            "goal": PHASE_GOAL[self.phase],
            "budget_left": self.max_turns - self.turn,
            "closing": self.closing_note,
            # 复核与报告阶段常驻一张「已量到的数」：交卷时最需要的就是它。
            # 只在这两个阶段给，是因为识别/测量阶段的任务是「找下一个量」，
            # 提前贴出来会让它以为已经够了、跳过本该做的测量。
            "measured": (self._draft_board(limit=14)
                         if self.phase in ("VERIFY", "REPORT") else None),
            "elapsed_s": round(time.time() - self.t0, 2),
            "card": self.card_text,
            "model": self.model_path,
            "tools": [t["function"]["name"] for t in sch.tool_specs(only=allowed)],
            "steps": self.steps,
            "last": self.steps[-1] if self.steps else None,
            "roles": self.roles,
            "facts": self.facts,
            "memo": self.results_memo,
            "measurements": self.measurements,
            "failure": next((f for f in reversed(self.failures) if not f.get("resolved")), None),
            "board": {
                "n_steps": len(self.steps),
                "n_unique_selections": sum(1 for s in self.steps
                                           if s.get("tool") == "select" and s.get("n") == 1),
                # 全量枚举类的问题（「一共有几个/每个实体多大」）没有要压到唯一命中的角色，
                # 它的「知道在量什么」等于「全集已确认」：一次 select/pick 返回的条数
                # 正好等于场景里的实体数。没有这一条，这类问题会永远卡在 IDENTIFY。
                "n_whole_set": (sum(1 for s in self.steps
                                    if s.get("tool") in ("select", "pick")
                                    and s.get("n") and len(self.bus.scene) > 1
                                    and s["n"] >= len(self.bus.scene))
                                if getattr(self, "bus", None) is not None else 0),
                "n_measurements": len(self.measurements),
                "n_open_failures": sum(1 for f in self.failures if not f.get("resolved")),
            },
        }

    # ------------------------------------------------------------ 动作分发

    def _dispatch(self, act):
        # 参数不是合法 JSON：这是一个可当场修正的机械错误，记一步、不动阶段、不改状态。
        # 之前这里会走 declare_insufficient，等于把「格式写错」判成「证据不足」，
        # 一次笔误就让整道题白跑（实测 A/B 里就丢过一次 OK）。
        if act.get("tool") == "__invalid_args__":
            self.steps.append({"id": "s%d" % self.turn, "phase": self.phase,
                               "tool": "（参数非法）", "error": act.get("why", "")[:300],
                               "hint": "参数要写成合法 JSON：字符串带引号、对象之间带逗号、"
                                       "列表用方括号。改完重发同一个调用即可。"})
            self.L.append("bad_args", {"turn": self.turn, "why": act.get("why", "")[:300]})
            return

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
            # 步骤号与轮次解耦：重复只读调用会把轮次退还（见 _refund），
            # 若用 turn 当编号，退还后会生成重复的 s12，证据引用会静默指向另一条。
            self._sid_n += 1
            sid = "s%d" % self._sid_n
        self.bus.notes = []
        try:
            args = self.bus.resolve(args, self.results, self.tool_of)
        except Exception as e:
            self.L.append("resolve_error", {"step": sid, "tool": name, "args": _short(args),
                                            "error": "%s: %s" % (type(e).__name__, e)})
            self.steps.append({"id": sid, "phase": self.phase, "tool": name, "args": _short(args, 300),
                               "why": why, "error": "引用解析失败：%s" % e})
            return
        # 别名纠正（@load → 实际步骤 all、.items → .solids）：已经按意图解析了，
        # 但必须把「正确的写法」说出来，否则它下一次还写错。
        ref_note = ""
        if self.bus.notes:
            parts = []
            for nt in self.bus.notes:
                if "field" in nt:
                    parts.append("%s 的字段名是 %s，不是 %s" % (nt["ref"], nt["to"], nt["field"]))
                else:
                    parts.append("%s 已按最近一次 %s 步骤解析为 @%s，以后请直接写 @%s"
                                 % (nt["ref"], nt.get("tool") or "同名", nt["to"], nt["to"]))
            ref_note = "；".join(parts)
            for nt in self.bus.notes:
                self.L.append("ref_alias", nt)
            self.bus.notes = []

        self.L.append("decision", {"step": sid, "phase": self.phase, "tool": name,
                                   "args": _short(args), "why": why,
                                   "expect_n": expect_n if expect_n is not None else self.expect.get(sid)})

        sig = hashlib.sha256((name + json.dumps(args, sort_keys=True, default=str)).encode()).hexdigest()[:12]
        self._sig_count[sig] = self._sig_count.get(sig, 0) + 1
        repeat = self._sig_count[sig]
        if repeat >= 3:
            # 只读调用的重复不等于「原地打转」：fit_sphere / pick / mesh_health 这些
            # 同一个参数再算一遍，结果一个比特都不会变。实测 agent 会在复核阶段
            # 把同一个 fit_sphere 连调三次，然后整轮被判停滞 —— 它其实只差一次提交。
            # 所以：只读调用第 3 次起直接返回缓存 + 明确提示推进；连重复 9 次才终止。
            # 会改几何的调用不改这个规则 —— 重复编辑是真的危险。
            pure = name not in EDIT_TOOLS and name != "emit"
            cached = self._sig_obs.get(sig)
            if pure and not cached and sig in self._sig_err and repeat < 9:
                # 失败过的调用没有可复用的返回，但「同样的参数再调一次」也只会得到同一个报错。
                # 一次失败就判整轮停滞太狠（实测是参数格式问题，改一下就好），
                # 所以这里把报错原样再摆一次，并指出去改哪里。
                err = self._sig_err[sig]
                obs = json.dumps({"_repeat": "第 %d 次完全相同的调用，上次就失败了：%s。"
                                             "同样的参数不会得到不同结果 —— 改参数（列表要写成真正的数组，"
                                             "或直接用 @步骤.items 引用），或者换一条路。" % (repeat, err[:200])},
                                 ensure_ascii=False)
                step = {"id": sid, "phase": self.phase, "tool": name, "args": _short(args, 300),
                        "why": why, "error": err, "hint": "重复了上次失败的调用", "obs": obs}
                self.steps.append(step)
                self.L.append("stall_warn", {"step": sid, "tool": name, "repeat": repeat,
                                             "action": "same_error"})
                self._refund(sid)
                return
            if pure and cached and repeat < 9:
                obs, stats = cached
                first = self._sig_first.get(sig, "")
                obs = obs[:-1] + (
                    ',"_repeat":"第 %d 次完全相同的只读调用，返回的是缓存结果（没有重算）。'
                    '同样内容已在步骤 %s 返回过，需要它时写 @%s.items 引用即可，不必再传一次参数。'
                    '再重复不会带来新信息：换判据，或者 advance_phase 推进到下一阶段。"}'
                    % (repeat, first or "（无）", first or "step"))
                step = {"id": sid, "phase": self.phase, "tool": name, "args": _short(args, 300),
                        "why": why, "n": stats.get("n"),
                        "expect_n": expect_n if expect_n is not None else self.expect.get(sid),
                        "hint": "重复的只读调用：已返回缓存，请推进阶段",
                        "obs": obs, "repeat": repeat, "rejected": False}
                self.steps.append(step)
                # 缓存命中也要登记。不登记的话，这一步在账本里有一个成功的 step id，
                # 却不在 results 里：之后 @它 引用不到、把它写进 evidence 也会被判「不存在」。
                # 实测 agent 就因为引用了缓存步骤，答案被审计拒掉、然后在原地打转。
                if sig in self._sig_out:
                    self.results[sid] = copy.deepcopy(self._sig_out[sig])
                self.L.append("stall_warn", {"step": sid, "tool": name, "repeat": repeat,
                                             "action": "cached"})
                # 命中缓存 = 这一步什么新东西都没带来，计入打转计数。
                # 实测 T01 第 1 问在旋转副本上连查 6 次同一面墙、每次都被缓存接住、
                # 最后一次直接判 STALLED —— 它其实早已有全部数字，只差收口。
                self._spin += 1
                self._refund(sid)
                self._maybe_force(sid, step, "连续 %d 步没有带回新数字（这一步命中缓存）" % self._spin)
                return
            self.L.append("stall_abort", {"step": sid, "tool": name, "repeat": repeat})
            self._finish("STALLED")
            return

        try:
            out = self.bus.call(name, args, sid)
        except Exception as e:
            step = {"id": sid, "phase": self.phase, "tool": name,
                    "args": _short(args, 300), "why": why,
                    "error": str(e)[:300]}
            if ref_note:
                step["hint"] = "引用已自动纠正：" + ref_note
            self.steps.append(step)
            # 工具错误留在本阶段处理：参数写错是可当场修正的机械问题，
            # 不该升级成「证据不足」并把人吸进自救态。实测这一点会让智能体
            # 把 80% 的轮次耗在 RECOVER 里反复试探，再也没回到识别环节。
            self.L.append("tool_error", {"step": sid, "tool": name, "error": str(e)[:300]})
            self._sig_err[sig] = str(e)[:300]
            return

        self.results[sid] = out
        self.tool_of[sid] = name
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
        self._sig_obs[sig] = (obs, stats)
        self._sig_out[sig] = out
        self._sig_first.setdefault(sig, sid)

        # 「换措辞问同一个东西」：参数不同、返回一字不差。这类步骤在账本里占
        # 全部只读步的 7%，而现有签名检测一条都抓不到（签名按参数算）。
        # 处理方式是提示而不是拦截：结果照给（它可能确实要看），但明确说「与第 X 步完全相同」，
        # 并计入打转计数 —— 连续几步这样就说明该换阶段了。
        same_as = None
        if name in STRUCT_TOOLS:
            fp = _out_fp(obs)
            if fp:
                same_as = self._out_seen.get(fp)
                if same_as and same_as != sid:
                    self._spin += 1
                    self.L.append("dup_output", {"step": sid, "tool": name, "same_as": same_as})
                elif not same_as:
                    self._out_seen[fp] = sid

        if same_as:
            obs = obs[:-1] + (',"_same_result":"这一步的返回与步骤 %s 完全相同 —— 换措辞不会换结果。'
                              '需要这个结果时写 @%s.字段 引用；要继续就必须换正交判据，'
                              '或者推进阶段。"}' % (same_as, same_as))
        if ref_note:
            obs = obs[:-1] + (',"_ref":"%s"}' % ref_note.replace('"', "'"))
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
        if same_as:
            step["hint"] = "返回与步骤 %s 完全相同" % same_as
            step["dup_of"] = same_as
        self.steps.append(step)
        if same_as:
            self._maybe_force(sid, step, "连续 %d 步返回与已有步骤完全相同" % self._spin)

        if name in ("select", "pick") and n == 1:
            items = out.get("items") or ([out["item"]] if out.get("item") else [])
            # 只登记「实体命中」。pick 也可以拿来取一个洞口矩形、一个数 —— 那不是角色，
            # 没有 sid，硬登记会 KeyError 把整轮打断（实测 T01-1 在 VERIFY 里就这样崩过一次）。
            if items and isinstance(items[0], dict) and "sid" in items[0]:
                # 步骤标签兼作角色名：agent 给这一步起名 door_leaf，就表示「我确认这就是门扇」。
                # 这样决策板上的内容来自 agent 自己的声明，而不是编排层替它猜。
                self.roles[sid] = items[0]["sid"]
                self.L.append("role_decided", {"role": sid, "sid": items[0]["sid"],
                                               "evidence": _short(args, 200)})
                # 「唯一命中」只是必要条件。这里顺手记一条：这次唯一的命中，
                # 有几条判据能指出具体反例（哪个候选满足其余条件、偏偏不满足这一条）。
                # 纯计算、不占轮次；报告里会显示成 8/12 这样的数字，判据密度一眼可见。
                if name == "select":
                    try:
                        from tools.ops import witness_check
                        ev = witness_check(args.get("items"), args.get("where") or {},
                                           items[0]["sid"])
                    except Exception:
                        ev = None
                    if ev:
                        self.role_evidence.append({"role": sid, **ev})
                        self.L.append("role_witness", {"role": sid, **ev})
                        if ev["n_witnessed"] < ev["n_cond"]:
                            step["hint"] = ("这次唯一命中里有 %d/%d 条判据指不出反例 —— "
                                            "它们不构成独立证据；要更稳就用 select(probe=true) 看明细"
                                            % (ev["n_cond"] - ev["n_witnessed"], ev["n_cond"]))
        # 轮次的两个大头都在这里：
        #  ① select 换条件反复试 —— 命中集合根本没变，说明判据没换，只是措辞换了；
        #  ② 重复 load —— load 会清空去重计数，所以"再读一遍"永远不会被判成重复调用。
        if name == "select" and isinstance(out, dict):
            hit = frozenset(i.get("sid") for i in (out.get("items") or [])
                            if isinstance(i, dict))
            if self._last_select is not None and hit == self._last_select:
                self._spin += 1
                msg = ("命中集合与上一步完全相同（%d 个），换条件没带来新信息 —— "
                       "要么换一条正交判据，要么推进阶段。" % len(hit))
                if isinstance(obs, str) and obs.rstrip().endswith("}"):
                    step["obs"] = obs.rstrip()[:-1] + ',"_same_hits":%s}' % json.dumps(
                        msg, ensure_ascii=False)
                step["hint"] = "命中集合没变（%d 个）" % len(hit)
                self._maybe_force(sid, step, "select 命中集合连续不变")
            else:
                self._spin = 0
            self._last_select = hit
            if len(hit) == 1:
                step["hint"] = ("已唯一命中 %s —— 别再筛同类条件，接着量或直接推进"
                                % next(iter(hit)))
        if name == "load" and isinstance(out, dict):
            p_ = (out.get("meta") or {}).get("path")
            n_seen = self._loaded.get(p_, 0)
            if n_seen >= 1:
                # 同一路径重复读：不重算、不清去重计数，直接把"没变化"讲明白
                msg = ("这个文件已经读过 %d 次，场景没有变化 —— 不要重复 load；"
                       "要继续就用已返回的 sid（@第一步.solids）。" % (n_seen + 1))
                if isinstance(obs, str) and obs.rstrip().endswith("}"):
                    step["obs"] = obs.rstrip()[:-1] + ',"_reload":%s}' % json.dumps(
                        msg, ensure_ascii=False)
                step["hint"] = "重复读入同一文件"
                self._spin += 1
            else:
                self._loaded[p_] = 0
            self._loaded[p_] = n_seen + 1

        if name in ("distance", "rank", "section", "mesh_health",
                    "footprint", "clearance", "opening_edge",
                    # 运动层与形状层的产物同样是「可复核的量」：扫掠角度、接触/重叠判定、
                    # 截面轮廓、凹坑、配准结果。不登记的话，靠碰撞判定收尾的任务
                    # 会卡在 MEASURE 阶段推不动 —— 它测的都是「会不会撞」，不是「多远」。
                    "collide", "sweep_rotate", "sweep_translate",
                    "profile", "pockets", "regid2d",
                    # angle / openings 也是「产出一个可复核的量」：一个夹角、一组洞口矩形。
                    # 漏登记的直接后果是 MEASURE 门控永远判「还没量过」，模型交不了卷。
                    "angle", "openings", "assembly"):
            # 凡是「产出一个可复核的量」的工具都算测量。早先只认 distance/rank/section，
            # 于是编排型任务（占地、距门、体检）全都不算数，MEASURE 阶段永远推不动。
            self.measurements.append(sid)
            self._note_fresh_numbers(sid, out)
            # 预算纪律：这一量到底带回了新数字吗？没有就往打转计数上记一笔。
            # 判据取「有没有新数字」而不是「量了几步」—— 步数阈值会误伤真有活干的题
            # （四小问合并的长题要量十几步），而「连续 5 步一个新数都没有」在任何题上
            # 都只意味着一件事：现在的判据问不出新东西了，该换个阶段做事。
            if self._meas_fresh and self._meas_fresh[-1][1] == 0:
                self._spin += 1
            else:
                self._spin = 0
            if len(self.measurements) >= 5:
                self._maybe_force(sid, step, "测量已连续 %d 步没有带回新数字" % self._spin)
        # 查出洞口之后，下一步该做的是「定族」，而不是接着用 select 试尺寸。
        # 这一句只是流程提示（不含任何语义判断）：族给出的是接触证据，不是标签。
        if name == "openings" and isinstance(out, dict) and out.get("openings"):
            if isinstance(obs, str) and obs.rstrip().endswith("}"):
                step["obs"] = obs.rstrip()[:-1] + (
                    ',"_next":"洞口已经找到。下一步用 assembly(walls=[这面墙]) 一次性把'
                    '「谁挂在洞口上」定下来 —— 它会给出每族的 grounded / 下框 / 扇 / 语义范围，'
                    '比逐个 select 试形状省得多。族定完之后再判语义，不要再筛尺寸。"}')

        # 语义范围收窄：assembly 已经把「谁挂在哪个洞口上」算清楚了，就不再让它翻全场景。
        # 这里只收窄**范围**（可能的词），不替它选词 —— 选词仍是模型的事，理由是几何 facts。
        if name == "assembly" and isinstance(out, dict) and out.get("families"):
            scopes = []
            for i, f in enumerate(out["families"], 1):
                ft = f.get("facts") or {}
                g, rail, sash = ft.get("grounded"), ft.get("has_bottom_rail"), ft.get("has_sash")
                if g is None:
                    opts = [SCOPE_LABELS[0], SCOPE_LABELS[1], SCOPE_LABELS[3]]
                elif not g:
                    opts = [SCOPE_LABELS[1], SCOPE_LABELS[3]]
                elif rail:
                    opts = [SCOPE_LABELS[2], SCOPE_LABELS[3]]
                elif sash:
                    opts = [SCOPE_LABELS[0], SCOPE_LABELS[3]]
                else:
                    opts = [SCOPE_LABELS[0], SCOPE_LABELS[2], SCOPE_LABELS[3]]
                scopes.append({"family": "F%d" % i, "wall": f.get("wall"),
                               "opening": [f["opening"]["width"], f["opening"]["height"]],
                               "facts": {"grounded": g, "rail": rail, "sash": sash,
                                         "sash_angle_deg": ft.get("sash_angle_deg")},
                               "label_scope": opts})
            loose = out.get("loose") or []
            note = ("族已定：%s。语义范围按 SYSTEM 第 4c 条收窄 —— 每个族只能从 label_scope 里选一个词，"
                    "选定后直接把答案写出来，不要再用尺寸/面积筛一遍。"
                    "未列入任何族的 %d 件构件是非门窗，不要给它们贴门窗标签。"
                    % ("；".join("%s 墙%s→%s" % (x["family"], x["wall"], "/".join(x["label_scope"]))
                                 for x in scopes), len(loose)))
            if isinstance(obs, str) and obs.rstrip().endswith("}"):
                step["obs"] = obs.rstrip()[:-1] + ',"_scope":%s,' \
                              '"_scope_note":%s}' % (json.dumps(scopes, ensure_ascii=False),
                                                     json.dumps(note, ensure_ascii=False))
            step["hint"] = "语义范围已收窄到 %d 个族" % len(scopes)
            for x in scopes:
                self.families[x["family"]] = {"wall": x["wall"], "facts": x["facts"],
                                              "label_scope": x["label_scope"]}
            self.assembly_steps.append(sid)
            self.L.append("scope_narrowed", {"step": sid, "n_families": len(scopes),
                                             "scopes": scopes})

        if name in EDIT_TOOLS:
            self.edits.append({"step": sid, "tool": name, "args": _short(args, 200)})
            self.L.append("geometry_edited", {"step": sid, "tool": name, "phase": self.phase})
            # 几何变了，「相同参数 = 相同结果」这个前提就不成立了。
            # 不清空的话，编辑前量一次、编辑后再量一次会被判成「原地打转」，
            # 而它恰恰是本层最该做的动作：改前改后各量一遍。
            self._sig_count.clear()
            self._out_seen.clear()
            # 「这个数以前出现过」的前提同样失效：改完之后 3.2 m 还是 3.2 m，但指的是新的东西。
            self._seen_nums.clear()
            self._meas_fresh.clear()
        if name == "emit" and isinstance(out, dict) and out.get("path"):
            self.emitted.add(out["path"])
        if name == "load" and isinstance(out, dict):
            # load 会把整个场景换掉，「同参数 = 同结果」这个前提随之失效：
            # 同一句 sweep_rotate 在状态1 和状态2 上本来就该给出不同角度。
            self._sig_count.clear()
            self._out_seen.clear()
            self._seen_nums.clear()
            self._meas_fresh.clear()
            p = (out.get("meta") or {}).get("path")
            if p in self.emitted:
                # 只有「读回的正是自己刚写出去的那个文件」才算复核。
                # 重读原始模型不算 —— 那是在看没改过的东西。
                self.reloaded.add(p)
                self.L.append("reloaded_own_output", {"step": sid, "path": p})

        # 「失败被消解」不该只在 RECOVER 里成立。
        # 实测 T01 第 1 问：模型在 VERIFY 里自己换了组参数把墙数补对（3 面），
        # 但那条 count_mismatch 仍挂着未消解 → advance_phase 被门控拒 → 被迫进 RECOVER
        # 再绕一圈 → 70 轮耗尽。同一个工具、同一个期望数，补对了就是补对了，
        # 与它当时站在哪个阶段无关。
        cur_sids = _result_sids(out)
        for f in self.failures:
            if f.get("resolved"):
                continue
            how = _resolve_reason(f, name, want, n, cur_sids, self.phase)
            if how:
                f["resolved"] = True
                f["resolved_by"] = sid
                f["how"] = how
                self.L.append("failure_resolved", {"step": f["step"], "by": sid,
                                                   "tool": name, "phase": self.phase,
                                                   "how": how})

        if want is None or n == want:
            self._tick_recover()

        if want is not None and n != want:
            self.failures.append({"step": sid, "kind": "count_mismatch", "tool": name,
                                  "want": want, "got": n,
                                  # 记下当时找到了哪几件：后来者要「包含这些」才算补齐，
                                  # 只比条数会被一次无关的 n=3 骗过去。
                                  "sids": sorted(cur_sids),
                                  "args": _short(args, 300), "resolved": False})
            self._enter_recover(sid, "count_mismatch",
                                "期望 %s 个候选，实得 %s。参数：%s" % (want, n, _short(args, 200)))

        if name == "answer":
            self._audit(sid, out)

    # ------------------------------------------------------------ 阶段推进

    def _note_fresh_numbers(self, sid, out):
        """记下这一步带回了几「个从未出现过的数」。

        签名去重只能抓「参数一模一样」的打转。真实的长尾是另一种：换标签、换筛选阈值、
        换一个 pick 目标，参数每次都不一样，返回的数字一字不差 —— 签名比对对它是瞎的。
        这里换一个与签名无关的判据：**这一步有没有让已知量的集合变大**。
        连续几步都不变大，就说明「再量也量不出新东西」，这跟量了几步无关。
        """
        nums = []
        self._flatten_numbers(out, nums)
        q = {round(x, 6) for x in nums}
        fresh = q - self._seen_nums
        self._seen_nums |= q
        self._meas_fresh.append((sid, len(fresh)))
        # 结果必须保留：每一步的结论提炼成结构化 JSON 记忆，随每一轮进 LLM 上下文。
        # 观察可以压成一行表格，但**结论不能压** —— 压掉语义结论就是矛盾的根源。
        self._memo_result(sid, out)
        # 语义事实固化：从 openings / assembly / select(dims) 等工具的返回里提取
        # 参照物身份判据（落地=门/有窗台=窗、厚=框/薄=玻璃），按 sid 持久化。
        # agent 自己算出的结论，这里只做**搬移**：不改语义、不替它选词，只是让这个
        # 结论不会随着原始观察滑出「最近 4 步」窗口而丢失。
        self._extract_facts(sid, out)

    def _memo_result(self, sid, out):
        """提炼这一步的结构化结果，存入 results_memo（每轮随 ctx 传给 LLM）。

        过程可以压缩（obs 压成表格），但**结果必须保留**。原始 obs 里塞了太多噪声
        （ntri、share、edge_pairs 等），LLM 用不上；这里只留「这一步到底产出了什么结论」
        的核心键值对，供下一轮直接引用。与 _draft_board 的区别：_draft_board 只服务
        VERIFY/REPORT 阶段且只认测量步骤；memo 每步都记、每轮都进上下文。

        只记**可复核的结论键**，不记过程量（ntri/idx/unit 等噪声）。避免把整份 obs
        原样塞进记忆——那等于没压缩，上下文会爆。
        """
        if not isinstance(out, dict):
            return
        KEEP = {
            # 洞口：谁落地=门、谁有窗台=窗
            "openings": ("grounded", "sill_z", "width", "height", "area"),
            # 族：哪面墙、什么性质
            "assembly": ("wall", "facts", "opening", "label_scope"),
            # 距离/夹角/间隙：核心测量值
            "distance": ("value", "a", "b", "witness_a", "witness_b"),
            "angle": ("angle_deg", "acute_deg", "n_a", "n_b", "a", "b", "dot"),
            "clearance": ("value", "p1", "p2"),
            "collide": ("a", "b", "deepest"),
            "opening_edge": ("edge", "wall", "opening"),
            "select": ("items",),
            "pick": ("item", "items"),
            "footprint": ("rect", "center", "dims"),
            "section": ("area", "height"),
            "mesh_health": ("closed", "boundary_loops", "duplicate_faces"),
            "parts": ("parts", "removed"),
            "principal_frame": ("axes", "origin"),
        }
        keys = KEEP.get(sid.split("_")[0] if "_" in sid else sid)
        # 工具名从 step 里拿（更可靠），但这里用 sid 前缀兜底。实际 tool 名通过调用方传入更准。
        memo = {"sid": sid, "result": {}}
        for k in keys or ():
            if k in out:
                v = out[k]
                # 只留 JSON 可序列化的精简形式
                if isinstance(v, dict):
                    memo["result"][k] = {kk: vv for kk, vv in v.items()
                                         if kk in ("grounded", "sill_z", "width", "height",
                                                   "wall", "family", "opening", "facts",
                                                   "label_scope") and isinstance(vv, (bool, int, float, str, list, dict))}
                elif isinstance(v, list):
                    memo["result"][k] = v[:6]
                elif isinstance(v, (bool, int, float, str)):
                    memo["result"][k] = v
        # 只要还有非空结果就记（避免塞一堆空 memo）
        if memo["result"]:
            # 同一步可能被多次调用（重复），只保留最新
            for i, m in enumerate(self.results_memo):
                if m["sid"] == sid:
                    self.results_memo[i] = memo
                    return
            self.results_memo.append(memo)

    def _extract_facts(self, sid, out):
        """从工具返回里提取「参照物身份」语义事实，按 sid 固化到 self.facts。

        这是解决「自己跟自己矛盾」的根源：agent 在第 5 步推出了「门墙是 C005」，
        到第 12 步算夹角时这个结论丢了（openings 的完整观察滑出了最近 4 步窗口）。
        这里把**结论**（不是原始观察）持久化，每轮都进上下文。

        只做搬移 + 标准化，不替 agent 选词：
          - openings: 落地(grounded=True)→门洞、有窗台(grounded=False)→窗洞
          - assembly: 族里每面墙的 grounded 与 opening 属性
          - select:   dims 最薄边 > 0.05m → 框/结构；<= 0.05m → 透光件/薄板
        纯几何判据，不依赖名称/类别。与 roomtruth 是两套实现（这里只从 agent
        实际调用的工具返回里搬移，不做独立重算）。
        """
        if not isinstance(out, dict):
            return
        tool = sid.split("_")[0] if "_" in sid else sid
        try:
            if tool == "openings":
                for o in (out.get("openings") or []):
                    if not isinstance(o, dict) or "center" not in o:
                        continue
                    # openings 不带 sid，无法按实体归类 —— 用洞口中心近似锚定。
                    # 真正要的是「这面墙是门墙还是窗墙」，需要知道洞口属于哪面墙，
                    # 所以这里只记通用事实：这个洞口是门(落地)还是窗(有窗台)。
                    key = "opening@%.2f_%.2f" % (o.get("center", [0, 0])[0],
                                                 o.get("center", [0, 0])[1])
                    self.facts[key] = {
                        "grounded": o.get("grounded"),
                        "sill_z": o.get("sill_z"),
                        "width": o.get("width"),
                        "height": o.get("height"),
                    }
            elif tool == "assembly":
                for f in (out.get("families") or []):
                    if not isinstance(f, dict):
                        continue
                    wall = f.get("wall")
                    if wall:
                        ft = f.get("facts") or {}
                        self.facts[wall] = {
                            "grounded": ft.get("grounded"),
                            "sill_z": (f.get("opening") or {}).get("sill_z"),
                            "opening_w": (f.get("opening") or {}).get("width"),
                            "opening_h": (f.get("opening") or {}).get("height"),
                            "label_scope": f.get("label_scope"),
                            "_src": "assembly",
                        }
            elif tool == "select":
                for it in (out.get("items") or []):
                    if not isinstance(it, dict) or "sid" not in it:
                        continue
                    dims = it.get("dims")
                    if isinstance(dims, (list, tuple)) and len(dims) >= 3:
                        thinnest = float(min(dims))
                        # 厚薄判据：厚=框/结构，薄=玻璃/画心。阈值 0.05m 是工程经验值
                        #（窗框 10-15cm、画框 5-8cm、玻璃 3-5mm、画心 2-3cm）。
                        self.facts[it["sid"]] = {
                            "thinnest_m": round(thinnest, 4),
                            "kind": "frame_or_struct" if thinnest > 0.05 else "thin_plate",
                            "_src": "select",
                        }
        except Exception:
            # facts 提取失败不影响主流程 —— 它只是辅助记忆，不是审计依据。
            pass


    def _refund(self, sid):
        """重复的只读调用不消耗轮次预算。

        理由：一次「同样的参数、同样的返回」没有产生任何信息，把它记进轮次等于
        用预算惩罚一个无效动作 —— 实测 T02 在 VERIFY 里这样耗掉 30/41 轮，
        而它真正需要的动作只有「交卷」。
        退还是**有上限**的（默认 6 次）：无限退还会把「原地打转」变成「永远不结束」，
        而且每次仍要花一次 LLM 调用和真实时间。上限用完后按普通轮次计费，
        后续由停滞判定收尾。
        """
        if self._free >= 6:
            return False
        self._free += 1
        self.turn -= 1
        self.L.append("turn_refunded", {"step": sid, "n": self._free})
        return True

    def _maybe_force(self, sid, step, why):
        """连续 5 步「没有带回新东西」→ 编排层代它推进阶段，并把这件事写在明处。

        这是预算纪律，不是替它思考：推进只改「站在哪个阶段」，不改答案、不改证据。
        敢这么做的前提很硬：模型手上已经有它自己要的全部数字（这正是「没有新数字」的含义），
        继续留在本阶段的唯一效果是把轮次烧完 —— 那时结局是 STALLED，什么也交不出来。
        每个阶段只代推一次；代推后还停不住，就交给停滞判定，不无限重来。
        """
        if self._spin < 5:
            return False
        if self.phase in self._forced_from or self.phase not in ("IDENTIFY", "MEASURE", "VERIFY"):
            return False
        if not self._gate_pass():
            # 门控没过（例如还有未消解的失败）说明这不是打转，是在补前置条件。
            self._spin = 0
            self.L.append("force_rejected", {"step": sid, "phase": self.phase,
                                             "why": "门控未过，代为推进不成立"})
            return False
        before = self.phase
        self._advance("连续 5 步没有带回新数字（%s），编排层代为推进" % why)
        if self.phase == before:
            return False
        self._forced_from.add(before)
        self.forced += 1
        self._spin = 0
        note = ("你已连续 5 步没有量到任何新数字（参数在换、结果不变），"
                "编排层按预算纪律把阶段从 %s 推到 %s。这不是新的证据，只是节奏裁决："
                "需要前面那些数时用 @步骤名 引用，不要再取一遍。"
                "现在做与当前阶段相符的事 —— 若已能作答，直接 answer 交卷。" % (before, self.phase))
        obs = step.get("obs")
        if isinstance(obs, str) and obs.rstrip().endswith("}"):
            step["obs"] = obs.rstrip()[:-1] + ',"_phase_forced":%s}' % json.dumps(note, ensure_ascii=False)
        step["hint"] = "已代为推进阶段：%s → %s" % (before, self.phase)
        self.L.append("phase_forced", {"step": sid, "from": before, "to": self.phase, "why": why})
        return True

    def _gate_pass(self):
        """当前阶段的推进门控是否成立（无副作用版，供 _advance 与代为推进共用）。"""
        gate = {
            "SURVEY": lambda: {s["tool"] for s in self.steps} >= {"load"},
            # 门槛是「有一条可复核的识别结果」，不是「至少两个」：只点名一个角色的题目
            # （「这个模型里的球多大」）永远凑不到两个，>=2 会把这类题锁死在 IDENTIFY。
            # 多角色任务漏识别不会被这条放过 —— 漏了就在测量时量错，审计会拦住。
            "IDENTIFY": lambda: (self.context()["board"]["n_unique_selections"] >= 1
                                 or self.context()["board"]["n_whole_set"] >= 1),
            "MEASURE": lambda: len(self.measurements) >= 1,
            "VERIFY": lambda: not any(f for f in self.failures if not f.get("resolved")),
            # 写层的第一条纪律：改了几何就必须落盘。
            # 「改完只在内存里体检一下就交卷」是这类任务最容易出的事 —— 报告很漂亮，
            # 但没有任何可交付的模型。没改几何则允许空过（纯测量任务走这条）。
            "EDIT": lambda: (not self.edits) or bool(self.emitted),
            # 第二条纪律：落盘之后必须读回。这条是整个写层的地基 ——
            # 没有它，agent 可以在内存里改完、自报体检通过，而磁盘上什么都没变。
            "REVALIDATE": lambda: bool(self.emitted) and bool(self.reloaded & self.emitted),
        }.get(self.phase)
        return True if gate is None else bool(gate())

    def _advance(self, why):
        if not self._gate_pass():
            self.gate_rejects += 1
            self.L.append("advance_rejected", {"phase": self.phase, "why": why,
                                               "board": self.context()["board"]})
            need = {
                "SURVEY": "至少 load 一次",
                "IDENTIFY": "要么把题目点名的角色各压到唯一命中（至少 1 个），"
                            "要么用一次 select/pick 把全集确认下来（条数 = 实体总数）",
                "MEASURE": "至少完成一次测量（MEASURE 阶段的白名单工具）",
                "VERIFY": "没有未消解的失败（补齐原失败缺的条目就算消解：条数要到位，"
                          "且原来那几件仍在结果里 —— 换一条路径证明同一件事同样算数）",
                "EDIT": "改了几何就必须 emit 落盘；没改几何可以空过",
                "REVALIDATE": "必须读回自己刚写出的那个文件再量",
            }.get(self.phase, "阶段目标未达成")
            self.steps.append({"id": "gate", "phase": self.phase, "tool": "advance_phase",
                               "rejected": True, "why": why,
                               "obs": "还不能进入下一阶段。这一阶段的判定是：%s。" % need})
            return
        i = PHASES.index(self.phase) if self.phase in PHASES else 0
        if self.phase == "RECOVER":
            target = self._recover_from or "IDENTIFY"
        else:
            target = PHASES[min(i + 1, len(PHASES) - 1)]
            # 离开 EDIT 时才知道要不要写层：这一步一次几何都没动，就没有落盘与回读可言，
            # 直接去报告。判据是「这一阶段实际做了什么」，不是「卡片看起来像哪类题」——
            # 后者要把题目知识塞进门控，正是这套东西一直想避免的。
            if self.phase == "EDIT" and not self.edits:
                target = "REPORT"
                self.L.append("phase_skip", {"from": self.phase, "skipped": ["REVALIDATE"],
                                             "why": "本次运行没有编辑动作，跳过复核"})
        if self.read_only and target in ("EDIT", "REVALIDATE"):
            self.L.append("phase_skip", {"from": self.phase,
                                         "skipped": ["EDIT", "REVALIDATE"],
                                         "why": "本轮只要求测量（原话没有改动几何的要求），写层没有可做的事"})
            target = "REPORT"
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
        elif isinstance(obj, str):
            # 模型经常把数写成字符串（"18.75"），早先直接跳过 —— 于是「写成字符串」
            # 就成了绕过审计的口子。这里把它当数字收进来，只认纯粹的数。
            s = obj.strip().replace(",", "")
            try:
                if s and (s[0] in "+-.0123456789"):
                    out.append(float(s))
            except ValueError:
                pass
        elif isinstance(obj, dict):
            for v in obj.values():
                self._flatten_numbers(v, out)
        elif isinstance(obj, list):
            for v in obj:
                self._flatten_numbers(v, out)

    @staticmethod
    def _norm_evidence(ev):
        """evidence 必须是步骤 id 的列表。模型有时把它序列化成一个字符串交过来，
        早先会按字符遍历，报出「证据 [ 不存在」「证据 \" 不存在」几十条 ——
        一个格式问题变成六十条假指控，把真正的错因埋掉了。"""
        if ev is None:
            return []
        if isinstance(ev, str):
            s = ev.strip()
            try:
                j = json.loads(s)
                ev = j if isinstance(j, list) else [j]
            except Exception:
                ev = [x for x in re.split(r"[,，;；\s]+", s) if x]
        if not isinstance(ev, (list, tuple)):
            ev = [ev]
        out = []
        for e in ev:
            if isinstance(e, dict):
                e = e.get("step") or e.get("id") or ""
            e = str(e).strip().strip("\"'")
            if e and e not in out:
                out.append(e)
        return out

    def _audit(self, sid, ans):
        """答案里的每个数字必须能在某个证据步骤的输出里找到。找不到 = 编的。"""
        ev = self._norm_evidence(ans.get("evidence"))
        # evidence 收的是「步骤 id」，但整套提示都在教模型用 @步骤.字段.下标 的引用语法，
        # 于是它把 "@op_w5c.items" 直接写进 evidence 也就顺理成章。照字面判会把一个写法问题
        # 报成「证据不存在」——实测 T01 第 1 问在旋转副本上连续三次因此被拒、整轮判失败。
        # 这里按「先认全名，再退到 @ 后面的根 id」解析；改写过的也记一笔，不偷偷改。
        if ev:
            fixed = []
            for e in ev:
                if e in self.results:
                    fixed.append(e)
                    continue
                root = e.lstrip("@").split(".")[0].strip()
                if root and root in self.results:
                    self.L.append("evidence_normalized", {"from": e, "to": root})
                    fixed.append(root)
                else:
                    fixed.append(e)
            ev = fixed
        problems = []
        if not ev:
            problems.append("没有引用任何证据步骤")
        known = set(self.results)
        for e in ev:
            if e not in known:
                # 报出"最像的那几个现有 id"：模型常常引用一个自己以为做过、实际失败的步骤，
                # 只说"不存在"它会换个名字再试，三轮就把整轮判死。
                near = [k for k in known if k.startswith(e[:4]) or e.startswith(k[:4])]
                problems.append("证据 %s 不存在（那一步可能失败了，没有返回值）；"
                                "现有可引用的相近步骤：%s"
                                % (e, "、".join(sorted(near)[:3]) or "（无）"))

        pool = []
        for e in ev:
            if e in self.results:
                self._flatten_numbers(self.results[e], pool)

        ans = dict(ans, evidence=ev)          # 交出去的答案也统一成列表，别让下游再猜一次

        claims = []
        self._flatten_numbers(ans.get("values", {}), claims)
        ungrounded = []
        for c in claims:
            if c in (0.0, 1.0) and not pool:            # 计数类
                continue
            # 容差有两个来源，取宽的那个：
            #   a) 数值本身：相对 1e-6（够严，能抓住改写与错位）
            #   b) 观察精度：观察里的浮点被压到 4 位小数，模型照抄就是 4.1846。
            #      若只认 a)，模型「看得见的数」和「交得上的数」永远不一致 ——
            #      审计会稳定地判它编造，而它其实一字未改地抄了证据。实测踩过。
            #   b) 观察精度：观察层把浮点压到 sch.SLIM_ND 位（3 位），模型照抄就是 0.523。
            #      两边都按这个精度取整再比 —— 之前这里写死 4 位，而投影是 3 位，
            #      于是「copied 值」永远对不上，T09 连拒三次判 AUDIT_FAILED。
            # 观察有两条精度通道：批量投影是 3 位（SLIM_ND），单件直通是原值。
            # 模型会在两者之间混用 —— 同一个 0.814508… 它可能写 0.815（抄投影）
            # 或 0.8145（抄单件的原值再四舍五入）。两条都要认，否则同一个数
            # 会被拒到第三次（实测 T09 每次都因 0.8145 被拒 1~2 轮）。
            hit = any(abs(c - p) <= max(1e-9, 1e-6 * max(1.0, abs(c)))
                      or abs(round(p, sch.SLIM_ND) - round(c, sch.SLIM_ND)) <= 1e-9
                      or abs(round(p, 4) - c) <= 1e-9
                      for p in pool)
            if not hit:
                ungrounded.append(c)
        if ungrounded:
            problems.append("这些数字无法追溯到任何证据步骤：%s —— 用 calc 把派生量算一遍再提交；"
                            "审计只认工具返回过的数，你在脑子里算的减法不算证据" % ungrounded[:6])

        # 语义标签（门/窗/落地窗/无法区分…）的审计依据是族的 facts：
        # 标签集合不是我们写死的词表，而是"这些族的 facts 允许哪些词"的并集 ——
        # 所以这条规则对任何"分类型"任务都成立，也不含题目知识。
        vocab = set(SCOPE_LABELS)
        claims_lab = []
        for v in (ans.get("values") or {}).values():
            if isinstance(v, str) and v in vocab:
                claims_lab.append(v)
            elif isinstance(v, dict):
                claims_lab += [x for x in v.values() if isinstance(x, str) and x in vocab]
        if claims_lab and self.families:
            allowed = sorted({lab for f in self.families.values() for lab in f["label_scope"]})
            bad = [c for c in claims_lab if c not in allowed]
            if bad:
                detail = "；".join("%s→%s" % (k, "/".join(v["label_scope"]))
                                   for k, v in sorted(self.families.items()))
                problems.append("语义标签 %s 不在任何族允许的范围里（%s）—— 范围由族的 facts 定"
                                "（grounded / 下框 / 扇），不是由形状定的；按 SYSTEM 4c 重新选词，"
                                "或补一次覆盖所有墙的 assembly" % (bad, detail))
            if not any(e in self.assembly_steps for e in ev):
                problems.append("给了语义标签却没引用算族的那个步骤（%s）—— 标签的证据是族的 facts，"
                                "把族那一步写进 evidence" % (self.assembly_steps[-1] if self.assembly_steps else "assembly"))

        # 空答案不算交卷。审计原来只管「数字有没有出处」，于是「什么都不写」天然通过 ——
        # 实测 T09 有一次 54 轮后交了 {"values": {}}，状态还是 OK。这不是证据不足，是没干活：
        # 真的量不出来应该走 declare_insufficient（终态 INSUFFICIENT_EVIDENCE），
        # 而不是用空 dict 换一个绿色的 OK。走统一出口，三次都空就判 AUDIT_FAILED。
        # values 里只有 note/notes 这类自述字段，等于交了一个「指针」而不是结论。
        # 实测 T09 交过 {"note": "见 entities / multi_part / ..."} —— 它以为前面已经写过，
        # 实际一次都没交。空答案被拒之后，这是它对审计的第二条绕路。
        _META = {"note", "notes", "说明", "comment", "remark"}

        def _substantive(xs):
            """values 里有没有真正的结论：至少一个不是 None / 空串 / 空容器。

            实测 T02 交过 box_edge_m=null、sphere_radius_m=null —— 键名齐全，值全是 null。
            审计只管「数字有没有出处」，null 不是数字，于是拦不住；它看起来像交了答案，
            实际一个量都没给。T08 交了 chair_shift_m="未知"，同一类。
            """
            for x in xs.values():
                if isinstance(x, (list, dict)):
                    if x:
                        return True
                elif isinstance(x, str):
                    if x.strip() and x.strip() not in ("未知", "无", "none", "null", "None"):
                        return True
                elif x is not None:
                    return True
            return False
        _vals = ans.get("values")
        if isinstance(_vals, dict) and _vals and set(_vals) <= _META:
            problems.append("values 里只有说明文字（%s），没有结论本身 —— 把结果写到具体键上"
                            "（键名见任务卡的「答案字段名」），note 只能作为附注"
                            % "/".join(sorted(_vals)))
        if isinstance(_vals, dict) and _vals and not _substantive(_vals):
            problems.append("values 里的键都在，但值全是 null / 空 / 「未知」—— 这不算交卷。"
                            "量出来就写数（每个数要有工具出处），量不出来就把这一项写进 unknowns")
        # 「量到了却填 null」：这不是编造（审计本来就拦编造），是**交付断层**。
        # live_1126_103948 三道 distance 全成功、angle 给出 acute_deg=65，交卷时这几个键却
        # 全是 null，unknowns 的理由是「工具未返回可用数值」—— 与账本直接矛盾。
        # 只拒一次：有些项确实量不出来，第二次允许它带着真实理由放过。
        if isinstance(_vals, dict) and _vals and self._has_numbers() and self._null_rejects < 1:
            nulls = [k for k, x in _vals.items()
                     if x is None or (isinstance(x, str) and x.strip() in ("", "未知", "无", "null", "None"))]
            if nulls:
                self._null_rejects += 1
                problems.append(
                    "这些键是 null：%s。但本次运行已经量出过数（下面这些是工具原样返回的）：\n%s\n"
                    "能对上就直接填进去；确实没有对应量的，把 unknowns 的理由改写成"
                    "「缺哪个工具/哪一步做不到」——不要写「工具未返回可用数值」，账本里它们返回了。"
                    % ("/".join(nulls[:6]), self._draft_board(limit=8) or "（无）"))

        if not isinstance(_vals, dict) or not _vals:
            problems.append("答案的 values 是空的 —— 空答案不算交卷。把结论写进 values"
                            "（键名见任务卡的「答案字段名」）；确实量不出来就调 "
                            "declare_insufficient 说明缺什么，那是一个诚实的终态")

        problems = list(dict.fromkeys(problems))[:8]

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
            "forced": self.forced,
            "role_evidence": self.role_evidence,
            "insufficient": getattr(self, "insufficient", None),
        }
