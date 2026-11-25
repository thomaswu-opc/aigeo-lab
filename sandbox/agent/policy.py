# -*- coding: utf-8 -*-
"""policy.py — 决策策略。可替换，是「agent 在内」与「agent 在外」的分界点。

三种实现，用途不同但接口相同：

  RecordedPolicy  —— 回放一份已记录的决策序列。评委没有 API key 也能跑，
                     且逐条对照「当时的观察 → 当时的判断」；记录本身有哈希可校验。
  LLMPolicy       —— 运行时真调 LLM（OpenAI 兼容协议）。这才是生产路径；
                     接上 key 就换掉 RecordedPolicy，编排层与工具层一行不改。
  FallbackPolicy  —— 无 key、无记录时的规则回退。明确标注为降级路径，
                     只用来证明「架构在无模型时也能跑完」，不作为方法主体。

接口只有一个方法的理由：编排层需要的是「下一个动作」，不是「一段思考」。
让策略返回文本再由编排层解析，等于把控制权交回给不确定的自然语言；
返回结构化动作则让每一步都可审计、可重放、可被门控拒绝。
"""
import json
import os
import time
import urllib.error
import urllib.request

from agent import schema as sch

SYSTEM = """你是一个几何求解智能体的决策部分。你不做算术，你只做判断与选择。

硬性规则：
1. 任何数值都必须来自工具返回。你只能引用、比较、排序，绝不能自己算出一个数字放进答案。
2. 任何关于「这是门/窗/墙/椅子」的判断，只能用几何描述量（尺寸、厚度、法向、内环、落地标高），
   不得使用名称、类别、材质、父子关系、自定义属性 —— 即使它们出现在返回值里也一律忽略。
3. 识别角色时追求「唯一命中」。命中 0 个：判据太严或参照系不对；命中多个：判据不正交，
   需要再加一个正交条件。不要靠「看起来最像」挑一个。
   命中一个不等于找对了。判据要站得住，得能回答两件事，`select(probe=true)` 会逐条给出：
   ① 反例检验 —— 这条判据排除了谁？（能点名「C008 满足其余条件、偏偏不满足这一条」才算数；
      指不出反例的是凑数条件，删掉结论不变，它不构成证据）
   ② 余量检验 —— 阈值还能挪多少而不改变结论？（正好压在某个候选的实测值上，说明这个「唯一」
      是巧合，稍微一动就不唯一）
4. 「洞口、开口」不是构件，是构件上缺的那块料：用 openings 直接在墙面/板面上查缺料，
   它给每个洞口的矩形中心、宽高、下沿标高、四个角点、以及下沿是否通底（门洞通底、窗洞不通底）。
   文件里也可能另有一个贯穿墙厚的盒状体表示洞口体积，那是旁证（assembly 会把它标成 void）。
4b. 「谁属于这个洞口」用 assembly 判，**不要用重合度**：重合是位置关系，一开门扇就不重合了。
   它用接触+连通：面接触（构件落在洞口棱柱内）、线接触（构件贴洞口边界环，开着的扇靠铰链边
   也算）、连通（玻璃经框、把手经扇）。每个成员都带 in_prism / edge_gap / via，判据可复核。
   注意：窗框与画框本身也是带洞的环，它们的"洞"会被 plate_ratio 挡掉，不构成独立的一族。
4c. 门窗的**语义范围**由 assembly 返回的 facts 收窄，你只在这个范围里选一个词，不要用尺寸筛：
       grounded=false                    → {窗, 无法区分}
       grounded=true  且 有下框(has_bottom_rail)   → {落地窗, 无法区分}
       grounded=true  且 无下框 且有扇(has_sash)   → {门, 无法区分}（扇的开合角只是状态，不是判据）
       grounded=true  且 无下框 且 无扇            → {门, 落地窗, 无法区分}
   一律允许答「无法区分」，但要写明缺哪条证据。不要用宽高、面积、把手当判据 ——
   反例是现成的：落地窗也通底、固定窗没有扇、平开窗有扇、有的门有门槛。
   **没有出现在任何 family 里的构件 = 非门窗**（例如挂在整墙上的画框），不要硬凑一个标签。
   交卷时：语义标签必须以**族那一步**作为 evidence（审计会查），且标签必须落在该族的 label_scope 里 ——
   范围外的词不是"更准确"，是"没依据"。
   注意 plate_ratio 会把「窗框自身的洞」挡掉（skipped 里能看到），别把它当成一族。
5. 若模型整体不与坐标轴对齐（principal_frame 返回 is_axis_aligned=false），
   后续描述量一律改用 items_frame，不要继续用世界轴下的 dims。
6. 证据不足时调用 declare_insufficient 并说明缺什么，不要猜。猜错比说不知道扣分更多。
7. 每次调用工具都必须给 why：一句话说明「我要用这个返回值确认什么」。
   这句 why 会进账本，是评审理解你推理过程的唯一入口。

工作节奏：
- SURVEY 只做 load 和 clusters，先弄清场景由几部分组成。
- IDENTIFY 用 select 把每个角色压到唯一命中。**但只认题目点名的对象和它们的参照物** ——
  没被问到的构件不要逐个识别。每多认一个角色就多几轮、多一条要复核的证据，而它不在答案里。
- 只要题里出现门窗、洞口、「墙上挂着的东西」，IDENTIFY 的第一件事是
  `assembly(walls=[...能查出洞口的板...])` —— 它一次给出「谁属于哪个洞口」和
  每族的 grounded / 下框 / 扇 / 语义范围。**先拿族，再在族内认角色**；
  不要再靠「薄片 + 尺寸 1.0~1.4 m」这种形状筛选去试，那正是认错对象的老路。
- MEASURE 只对已确定的实体做测量。
- VERIFY 用与测量不同的路径复核（见证点、单位、多口径）。
- 阶段目标达成后调用 advance_phase。

实体的几何指纹字段（select / load 返回的每个元素就是这些，仅此这些）：
- sid          实体编号（仅用于指代，不含任何语义）
- bbox_lo/hi   世界坐标下轴对齐包围盒的角点，[x,y,z]
- d0/d1/d2     三个边长，**降序**；d0 是最长边，d2 是最短边。注意它们不说是哪根轴
- sx/sy/sz     沿世界 x / y / z 的跨度。问「哪一维是净宽、哪一维是净深」只能看这三个，
               因为模型整体旋转后，同一个 4.8 会从 sx 跑到 sy 去，而 d0 永远是 4.8
- h_long_axis  水平面内较长的那条边落在哪根轴上（"x" 或 "y"），h_long/h_short 是两条边的长度。
               这是纯比较结论，不带你猜：地板的长边方向就是房间的「横向」。
               要在房间里说「左右」，先看这个字段，再决定用哪根轴做坐标差 ——
               不要假设世界 x 就是房间的横向
- dims         与 d0/d1/d2 相同的三个数（冗余，便于阅读）
- min_thickness 各主要面法向方向上，顶点投影跨度的最小值 —— 也就是「板厚」
- z_min        最低点标高；z_min≈0 表示落地
- axis_aligned 所有主法向是否都平行于坐标轴；false 说明这件是斜置的（如开着的门扇）
- dominant     主要法向簇：[{n:[法向], area:面积, share:占该实体总面积比例, thickness:沿该法向跨度}]
- volume/bbox_volume/fill   实体体积、包围盒体积、体积占满率
- area/max_patch_area/ntri  表面积、最大单块共面区域面积、三角面数
- cover        物体在自身包围盒底面上的投影占满率

关于引用（重要，别自己搬运清单）：
- 集合参数可以直接写引用字符串，指向已完成步骤的返回值。语法 @步骤标签.字段.下标：
  "@load.solids"、"@clu.groups[0].items"、"@thin.items"、"@rank.rows"。
- 每步都可以带 label（如 win / leaf），之后用 @win.items 指代它。不要重新 load 来「找回清单」。
- 想从大集合里挑子集，正确做法是 select 加 where 条件，不是把清单复制进参数。

关于 where 条件：
- 写法一：{"min_thickness<=": 0.06}；写法二：{"min_thickness": {"$lte": 0.06}}，两者等价。
- 可用逻辑组合：{"$and": [c1, c2]}、{"$or": [c1, c2]}、{"$not": c}。
- 字段在该实体上不存在时，该条件判假（不会静默忽略）—— 所以筛出 0 条时，先确认字段名拼对了。

关于上下文：
- 长清单在观察里会被截断，并出现 _truncated 字段说明截断了什么、原有多少条。
  看到它不要怀疑数据缺失、更不要重复调用同一工具；用 select 的 where 条件收窄集合。
- 同一个工具 + 同样的参数连续调用三次会被判为停滞并终止。想重试就必须改变参数。
- 改参数也可能是在原地打转：同一面墙换个标签再查一遍、同一个中心换个工具再取一遍，
  返回的数一字不差。**判断有没有进展看「有没有带回新的数」，不看换了几个参数名。**
  连续 5 步测量都带不回新数字，编排层会直接代为推进阶段，并在那一步的记账里写明。
  想把结论坐实，靠的是换一条**原理不同的路径**（见证点、另一种分解、独立的第二口径），
  不是把同一件事再问一遍。

你在几何上是没有直觉的：不要想象形状，去读字段。"""


class Policy:
    kind = "base"

    def decide(self, ctx):
        raise NotImplementedError


# ------------------------------------------------------------ 回放


class ReplayDrift(Exception):
    pass


def _slug(tool, i):
    return {"load": "load", "clusters": "clusters", "principal_frame": "pf"}.get(tool, "%s_%d" % (tool, i))


def _subst_model(act, model):
    """把记录里的 $MODEL 换成实际模型路径。记录与模型解耦，才能一稿多跑。"""
    def walk(o):
        if isinstance(o, str):
            return model if o == "$MODEL" else o
        if isinstance(o, dict):
            return {k: walk(v) for k, v in o.items()}
        if isinstance(o, list):
            return [walk(v) for v in o]
        return o
    return walk(act)


class RecordedPolicy(Policy):
    """回放已记录的决策。每条记录 = 一次「观察 → 判断 → 动作」。"""

    kind = "recorded"

    def __init__(self, episode):
        self.turns = list(episode.get("turns") or [])
        self.note = episode.get("note", "")
        self.i = 0
        self.applied = {}                        # 编排层 turn 号 → 记录下标
        self.recovery = []
        self.last_on_fail = []

    def _turn_of_step(self, sid):
        try:
            t = int(str(sid).lstrip("s"))
        except ValueError:
            return None
        k = self.applied.get(t)
        return self.turns[k] if k is not None else None

    def decide(self, ctx):
        # 1) 失败进入 RECOVER：取该步预置的补救动作（等价于 LLM 当场重新决策）
        if ctx["phase"] == "RECOVER" and not self.recovery:
            f = ctx["failure"] or {}
            self.recovery = list(self.last_on_fail or [])
            self.last_on_fail = []
            if not self.recovery:
                return {"tool": "declare_insufficient",
                        "why": "步骤 %s 失败且没有可用的补救方案" % f.get("step"),
                        "missing": ["失败步骤 %s：%s" % (f.get("step"), f.get("kind"))]}
        if self.recovery:
            return self.recovery.pop(0)

        # 2) 正常推进
        if self.i >= len(self.turns):
            raise StopIteration
        t = self.turns[self.i]
        want = t.get("phase")
        if want and want != ctx["phase"]:
            raise ReplayDrift("记录第 %d 条要求阶段 %s，实际在 %s（回放已漂移，拒绝继续）"
                              % (self.i, want, ctx["phase"]))
        self.i += 1
        self.applied[ctx["turn"]] = self.i - 1
        self.last_on_fail = list(t.get("on_fail") or [])
        act = dict(t)
        act.pop("phase", None)
        act.pop("on_fail", None)
        act.setdefault("label", _slug(t.get("tool", ""), self.i))
        return _subst_model(act, ctx["model"])


# ------------------------------------------------------------ 真 LLM


# ------------------------------------------------------------ 端点解析


_PROVIDER_ENV = {
    # 同一台机器上常有多个厂商的 key。按模型名挑对应的那一个，
    # 否则会出现「用 OpenAI 的 key 打 DeepSeek 的域名」这种 401，且很难看懂。
    "deepseek": ("DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY", "https://api.deepseek.com"),
    "openai": ("OPENAI_BASE_URL", "OPENAI_API_KEY", "https://api.openai.com/v1"),
}


def _provider(model):
    m = (model or "").lower()
    for k in _PROVIDER_ENV:
        if k in m:
            return k
    return "openai"


def _endpoint(provider, base_url=None, api_key=None):
    env_base, env_key, default = _PROVIDER_ENV.get(provider, _PROVIDER_ENV["openai"])
    base = (base_url or os.environ.get("LLM_BASE_URL") or os.environ.get(env_base)
            or default).rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get(env_key)
    return base, key


def available_models():
    """探测本机可用的模型，供命令行 `--llm auto` 使用。"""
    out = []
    for prov, (_, env_key, _) in _PROVIDER_ENV.items():
        if os.environ.get(env_key):
            out.append({"deepseek": "deepseek-chat", "openai": "gpt-4o"}[prov])
    return out


class LLMPolicy(Policy):
    """OpenAI 兼容 chat/completions。temperature=0，保证同一观察给出同一动作。"""

    kind = "llm"

    def __init__(self, model, base_url=None, api_key=None, temperature=0.0,
                 timeout=90, keep_full=4, max_retry=3):
        self.model = model
        self.provider = _provider(model)
        self.base, self.key = _endpoint(self.provider, base_url, api_key)
        self.temperature = temperature
        self.timeout = timeout
        self.keep_full = keep_full
        self.max_retry = max_retry
        self.calls = []

    @property
    def ready(self):
        return bool(self.key)

    def usage(self):
        tot = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for c in self.calls:
            for k in tot:
                tot[k] += (c.get("usage") or {}).get(k) or 0
        return dict(tot, calls=len(self.calls))

    # --- 把 ctx 渲染成消息。旧观察压成一行，只保留最近几条完整内容（上下文预算）
    def _messages(self, ctx):
        steps = ctx["steps"]
        cut = max(0, len(steps) - self.keep_full)
        hist = ["| 步 | 阶段 | 工具 | 命中 | 意图 |", "|---|---|---|---|---|"]
        for s in steps[:cut]:
            # 失败/被拒的步骤打叉：它们没有返回值，写进 evidence 会被判「不存在」。
            # 不打叉时模型会照抄步骤名去引用，然后被审计拒三次、整轮 STALLED（实测 T09）。
            mark = "✗" if (s.get("error") or s.get("rejected")) else ""
            hist.append("| %s%s | %s | %s | %s | %s |" % (
                s.get("id"), mark, s.get("phase", ""), s.get("tool", ""),
                s.get("n", "—"), (s.get("why") or "")[:60]))
        recent = []
        for s in steps[cut:]:
            recent.append("### %s  %s  %s\nwhy: %s\nobservation:\n```json\n%s\n```\n%s" % (
                s.get("id"), s.get("phase", ""), s.get("tool", ""), s.get("why", ""),
                s.get("obs", s.get("error", "")), ("提示：" + s["hint"]) if s.get("hint") else ""))

        body = [
            "任务（这是你收到的全部要求）：\n" + ctx["card"].strip(),
            "模型路径：" + ctx["model"],
            "",
            "已完成步骤总览：\n" + "\n".join(hist),
            "",
            "最近观察：\n" + ("\n\n".join(recent) if recent else "（无）"),
            "",
            "当前阶段：%s（第 %d 轮，剩余 %d 轮）" % (ctx["phase"], ctx["turn"], ctx["budget_left"]),
            "本阶段目标：" + ctx["goal"],
            "本阶段可用工具：" + ", ".join(ctx["tools"]),
            "已唯一确定的角色：" + json.dumps(ctx["roles"], ensure_ascii=False),
        ]
        # 阶段内的「该收手了」提示。不是任务知识，是节奏纪律：
        # 实测 T01 第 1 问已经把两个距离量出来、还复核了三遍，仍在 VERIFY 里换路径继续核对，
        # 60 轮耗尽而从未提交。复核的价值在「换路径」，不在「遍数」。
        nph = sum(1 for s in steps if s.get("phase") == ctx["phase"])
        if ctx["phase"] == "VERIFY" and nph >= 1:
            body.append("提示：本阶段已有 %d 步。一条测量 + 一条复核就够了，第三条路径不会改变结论。"
                        "现在唯一有意义的动作是 advance_phase 进 REPORT 把数交上去；"
                        "重复同一个调用会被判停滞并终止本次运行。" % nph)
        elif ctx["phase"] == "MEASURE" and nph >= 8:
            body.append("提示：本阶段已走 %d 步。关键量都已量到的话，就 advance_phase 去复核；"
                        "不要对同一面墙、同一个截面反复重切。"
                        "判据换汤不换药（同一面墙换名字再查一遍、同一个中心换 pick 再取一遍）"
                        "不会带来新的数 —— 连续 5 步量不出新数字，编排层会直接代为推进阶段，"
                        "那一步的记账里会写明「代为推进」。" % nph)

        f = ctx["failure"]
        if f:
            body.append("未消解的失败：" + json.dumps(f, ensure_ascii=False))
        return [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "\n".join(body)}]

    def decide(self, ctx):
        payload = {
            "model": self.model,
            "messages": self._messages(ctx),
            "tools": [t for t in sch.full_specs()
                      if t["function"]["name"] in set(ctx["tools"]) | {"advance_phase",
                                                                      "declare_insufficient"}],
            "tool_choice": "auto",
            "temperature": self.temperature,
        }
        if not self.ready:
            raise RuntimeError("没有可用的 API key：请设 %s 或传 --api-key"
                               % _PROVIDER_ENV.get(self.provider, ("", "LLM_API_KEY"))[1])
        # 正文作答的补救：模型有时把结论直接写在 content 里、不给 tool_call，
        # 旧行为是一次就判 declare_insufficient —— 实测跑出了正确答案 65.0° 却整轮判失败。
        # 这不是「证据不足」，是「没走接口」：回它一句「用工具提交」再问一次，只重试一次。
        tcs, txt = [], ""
        for attempt in range(2):
            data = self._post(payload)
            msg = data["choices"][0]["message"]
            self.calls.append({"turn": ctx["turn"], "phase": ctx["phase"],
                               "usage": data.get("usage"), "raw": msg})
            tcs = msg.get("tool_calls") or []
            if tcs:
                break
            txt = (msg.get("content") or "").strip()
            try:
                return json.loads(txt)          # 有的模型直接回一段 JSON，那也认
            except Exception:
                pass
            if attempt == 0:
                payload = dict(payload, messages=payload["messages"] + [
                    {"role": "assistant", "content": txt[:1200]},
                    {"role": "user", "content":
                     "你把结论写在正文里了，但沙盒只接受**工具调用**。已得结论就调用 answer 提交"
                     "（values 放结论、evidence 放步骤 id）；还缺证据就继续调用工具。不要再用正文作答。"}])
                continue
            return {"tool": "declare_insufficient",
                    "why": "两次都没有给出工具调用：" + txt[:200],
                    "missing": ["可执行动作"]}
        fn = tcs[0]["function"]
        # 模型偶尔吐出非法 JSON（少个引号、多个逗号），直接 json.loads 会让整轮抛异常退出。
        # 这不是「证据不足」，是格式问题：当成可修正的参数错误报回去，比崩掉整个 run 好。
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args)
            assert isinstance(args, dict)
        except Exception:
            self.calls.append({"turn": ctx["turn"], "phase": ctx["phase"],
                               "bad_args": str(raw_args)[:300]})
            # 不要直接 declare_insufficient —— 那会让一次格式错误变成「证据不足」的结论，
            # 整个任务白跑。交给编排层记一条可修正的错误步骤，下一轮它自己重发。
            return {"tool": "__invalid_args__",
                    "why": "工具参数不是合法 JSON：%s" % str(raw_args)[:160]}
        name = fn["name"]
        if name == "advance_phase":
            return {"tool": "advance_phase", "why": args.get("reason", "")}
        if name == "declare_insufficient":
            return {"tool": "declare_insufficient", "why": args.get("reason", ""),
                    "missing": args.get("missing", [])}
        why = args.pop("why", None) or ""
        exp = args.pop("expect_n", None)
        label = args.pop("label", None)
        return {"tool": name, "args": args, "why": why,
                "expect_n": exp, "label": label}

    def _post(self, payload):
        """带退避重试。429/5xx 是网络噪声，不该让整条 episode 判失败。"""
        body = json.dumps(payload, ensure_ascii=False).encode()
        last = None
        for k in range(self.max_retry):
            req = urllib.request.Request(
                self.base + "/chat/completions", data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": "Bearer %s" % self.key})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                detail = e.read()[:400].decode("utf-8", "replace")
                last = RuntimeError("LLM HTTP %s: %s" % (e.code, detail))
                if e.code in (400, 401, 403, 404):
                    raise last                      # 配置错，重试无意义
            except Exception as e:
                last = RuntimeError("LLM 连接失败：%s" % e)
            time.sleep(1.5 * (k + 1))
        raise last


# ------------------------------------------------------------ 降级


class FallbackPolicy(Policy):
    """无 key 无记录时的规则回退。只保证架构可跑通，不作为方法主体。"""

    kind = "fallback"

    def __init__(self):
        self.done = 0

    def decide(self, ctx):
        ph = ctx["phase"]
        if ph == "SURVEY":
            if self.done == 0:
                self.done += 1
                return {"tool": "load", "args": {"path": ctx["model"]}, "why": "先读模型"}
            if self.done == 1:
                self.done += 1
                return {"tool": "clusters", "args": {"items": "@s1.solids", "tol": 1.0},
                        "why": "看场景由几个部分构成", "expect_n": None}
            return {"tool": "advance_phase", "why": "已读入并分离场景"}
        return {"tool": "declare_insufficient",
                "why": "降级策略不具备几何判据，无法继续",
                "missing": ["决策策略（请接 LLMPolicy 或提供回放记录）"]}
