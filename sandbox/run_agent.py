# -*- coding: utf-8 -*-
"""run_agent.py — 沙盒入口（agent 版）。

与 run.py 的区别：run.py 执行一份外部写好的 plan；这里只给任务卡与模型，
由 agent 自己看观察、自己选工具、自己判证据够不够。决策序列回放进账本。

用法：
  # 1) 回放一份已记录的决策（无需 API key，评委可直接跑）
  python run_agent.py --card tasks/T01_2.md --model <model> --episode episodes/T01_2.agent.json --tag ag

  # 2) 真 LLM 在线决策
  python run_agent.py --card tasks/T01_2.md --model <model> --llm gpt-4o --tag ag_llm

  # 3) 无 key 无记录，验证架构能跑（预期停在 SURVEY 之后并报证据不足）
  python run_agent.py --card tasks/T01_2.md --model <model> --fallback --tag ag_fb
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from core.runtime import Bus, Ledger, TOOLS, save_json  # noqa: E402
import tools.ops  # noqa: F401,E402
from agent import schema as sch  # noqa: E402
from agent.loop import Loop, PHASES, PHASE_TOOLS  # noqa: E402
from agent.policy import FallbackPolicy, LLMPolicy, RecordedPolicy, available_models  # noqa: E402
from viz import render  # noqa: E402

PALETTE = [(243, 156, 62), (66, 118, 214), (32, 168, 138), (188, 84, 160),
           (214, 168, 40), (120, 120, 200), (200, 96, 96), (96, 176, 96)]


def sha(path, n=16):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def load_policy(a):
    if a.llm:
        name = a.llm
        if name == "auto":
            # 按本机已有的 key 挑模型：机器上常同时有多个厂商的令牌，
            # 让 --llm auto 直接可跑，省掉一个必填项（评委最容易卡在这）。
            cand = available_models()
            if not cand:
                raise SystemExit("--llm auto 找不到可用 key（DEEPSEEK_API_KEY / OPENAI_API_KEY）")
            name = cand[0]
            print("（auto）使用模型：%s" % name)
        return LLMPolicy(name, a.base_url, a.api_key, temperature=a.temperature)
    if a.episode:
        return RecordedPolicy(json.load(open(a.episode, encoding="utf-8")))
    return FallbackPolicy()


def build_scenes(bus, res, orbits=3):
    """把 agent 的决策轨迹变成镜头。标题写阶段，副标题写 agent 自己的 why。"""
    by = {s.sid: s for s in bus.scene}
    scenes = []
    ans = res.get("answer") or {}
    values = ans.get("values") or {}
    targets = values.get("targets") or {}
    colors = {}
    # 从 PALETTE[1] 起分配：PALETTE[0] 留给椅子，否则椅子与第一个目标同色，特写里分不清
    for i, sid in enumerate([v for v in targets.values() if isinstance(v, str)]):
        colors[sid] = PALETTE[(i + 1) % len(PALETTE)]
    if isinstance(values.get("chair"), str):
        colors[values["chair"]] = PALETTE[0]

    scenes.append(dict(title="① SURVEY · 读入 %d 个实体" % len(bus.scene),
                       caption="没有名称、没有类别、没有属性可用；此后一切判断只依据几何描述量",
                       highlight={}, lines=[], points=[], steps=[]))

    for st in res["steps"]:
        if st.get("rejected"):
            scenes.append(dict(title="门控拒绝 · %s 阶段" % st.get("phase", ""),
                               caption=(st.get("obs") or "")[:110],
                               highlight={}, lines=[], points=[],
                               steps=[st.get("id")]))
            continue
        t = st.get("tool")
        if t == "principal_frame":
            scenes.append(dict(title="② 建立场景自身的参照系",
                               caption=_frame_caption(st),
                               highlight={}, lines=[], points=[], steps=[st.get("id")]))
        elif t == "select" and st.get("n") == 1:
            sid = _first_sid(st)
            if sid and sid in by:
                scenes.append(dict(title="③ IDENTIFY · 唯一命中 %s" % sid,
                                   caption=(st.get("why") or "")[:110],
                                   highlight={sid: PALETTE[len(scenes) % len(PALETTE)]},
                                   lines=[], points=[], zoom=None, steps=[st.get("id")]))
        elif t in ("rank", "distance") and targets:
            scenes.extend(_measure_scenes(by, st, colors, values.get("chair"),
                                          cross=(t == "distance"), step_id=st.get("id")))

    if ans:
        d = values.get("distances_m") or {}
        L = ["%s %.3f m" % (k, v) for k, v in d.items() if isinstance(v, (int, float))]
        scenes.append(dict(title="⑤ REPORT · 提交并接受审计（审计：每个数字都必须能在证据步骤里找到）",
                           caption="最短表面距离：" + "   ".join(L) if L else "见报告",
                           highlight=colors, lines=[], points=[],
                           hud=["椅子 " + str(values.get("chair"))] + L,
                           steps=[(res["steps"][-1].get("id") if res.get("steps") else "answer")]))
    return scenes


def _pick(st, key, dflt):
    """从观察里取一个标量字段。

    观察是按字符预算截断过的，principal_frame 的 info 很长，json.loads 常常在
    尾部断掉（is_axis_aligned / dev_from_world_deg 正好在末尾）。所以先解析，
    失败再退到正则扫原文 —— 字幕宁可少一个数，也不能把「已对齐」写成「未对齐」。
    """
    s = st.get("obs") or ""
    try:
        j = json.loads(s)
        info = j.get("info") or {}
        v = {"aligned": info.get("is_axis_aligned"), "coverage": info.get("coverage"),
             "dev": info.get("dev_from_world_deg")}.get(key)
        if v is not None:
            return str(v) if key == "aligned" else v
    except Exception:
        pass
    pat = {"coverage": r'"coverage"\s*:\s*([0-9.]+)',
           "dev": r'"dev_from_world_deg"\s*:\s*(-?[0-9.]+)',
           "aligned": r'"is_axis_aligned"\s*:\s*(true|false)'}[key]
    m = re.search(pat, s)
    if not m:
        return dflt
    return m.group(1)



def _axes_of(st):
    """从观察里读主轴的三个方向向量（截断也能读：axes 在 JSON 最前面）。"""
    s = st.get("obs") or ""
    m = re.search(r'"axes"\s*:\s*\[\s*\[([^\]]+)\][^,]*,\s*\[([^\]]+)\][^,]*,\s*\[([^\]]+)\]', s)
    if not m:
        return None
    try:
        return [[float(x) for x in g.split(",")] for g in m.groups()]
    except Exception:
        return None


def _frame_caption(st):
    """参照系镜头的字幕。dev/aligned 在 info 尾部，常被预算截掉 —— 用 axes 现算，
    避免把「已对齐」写成「未对齐」（这条字幕会直接出现在录像里）。"""
    cov = _pick(st, "coverage", "?")
    dev = _pick(st, "dev", None)
    al = _pick(st, "aligned", None)
    if dev in (None, "?") or al in (None, "?"):
        ax = _axes_of(st)
        if ax is not None:
            dev = max(math.degrees(math.acos(min(1.0, max(abs(c) for c in row)))) for row in ax)
        else:
            dev = "?"
    try:
        dev_s = "%.1f" % float(dev)
        aligned = float(dev) < 1.0
    except Exception:
        dev_s = str(dev)
        aligned = (al == "True")
    if al == "True":
        aligned = True
    return "覆盖率 %s，与坐标轴偏差 %s° —— %s" % (
        cov, dev_s, "已对齐，直接用世界轴" if aligned else "未对齐，切换到场景参照系")


def _first_sid(st):
    try:
        j = json.loads(st.get("obs") or "")
        it = (j.get("items") or [{}])[0]
        return it.get("sid")
    except Exception:
        return None


def _measure_scenes(by, st, colors, chair=None, cross=False, step_id=None):
    """测量镜头：整间房 + 两个物体分别高亮 + 连线。

    早期版本 fit 到两个见证点，结果镜头钻进墙里、满屏灰色 —— 只看得到两个点却不知道它们在哪。
    教训：见证点是「数值的凭据」，不是「构图的中心」；给证据配图要保住位置语境。
    """
    out = []
    try:
        j = json.loads(st.get("obs") or "")
        rows = j.get("rows") or ([j] if "witness_a" in j else [])
    except Exception:
        return out
    for r in rows[:4]:
        a, b = r.get("witness_a"), r.get("witness_b")
        if not a or not b:
            continue
        sid = r.get("sid") or r.get("b")
        hl = {k: v for k, v in (colors or {}).items() if k in by}
        if chair and sid and sid in hl:
            hl = {chair: PALETTE[0], sid: hl[sid]}
        label = "椅子" if chair else "A"
        fit = None
        if chair in by and sid in by:
            fit = np.vstack([by[chair].P, by[sid].P])   # 取两个实体全部顶点：构图给位置语境，不放到见证点上
        out.append(dict(title="④ MEASURE · 最短表面距离 %.4f m" % r.get("value", float("nan")),
                        caption="见证点 %s — %s   （两点必须落在各自表面上，这是数值唯一的自证）" % (
                            _fmt(a), _fmt(b)),
                        highlight=hl, lines=[(a, b)],
                        points=[{"p": a, "label": label}, {"p": b, "label": sid or "B"}],
                        fit=fit, pitch=32, steps=[step_id]))
    return out


def _fmt(p):
    return "(" + ", ".join("%.3f" % x for x in p) + ")"


def scene_scope(results):
    """取 clusters 返回的最大一簇作为「主场景」。第一帧仍画全场景（说明文件里不止一个模型），
    其后只画主场景 —— 否则球体、立方体、火箭会占掉大半画面，房间反而看不清。

    注意取数来源是原始 results，不是压缩后的观察：观察里的长清单会被截到 8 条，
    用它去还原场景范围会静默丢实体。
    """
    for out in (results or {}).values():
        if isinstance(out, dict) and out.get("groups"):
            return set(out["groups"][0].get("sids") or [])
    return None


def render_all(bus, res, outdir, orbits=3, video=True, results=None):
    scenes = build_scenes(bus, res, orbits)
    scope = scene_scope(results)
    os.makedirs(os.path.join(outdir, "frames"), exist_ok=True)
    manifest, paths = [], []
    for si, sc in enumerate(scenes):
        for k in range(orbits):
            yaw = 34 + 360.0 * k / orbits
            p = os.path.join(outdir, "frames", "seq_%03d.png" % len(paths))
            if sc.get("subset"):
                ds = [s for s in bus.scene if s.sid in sc["subset"]]
            elif si > 0 and scope:
                ds = [s for s in bus.scene if s.sid in scope]
            else:
                ds = bus.scene
            render.frame(ds, p, yaw=yaw, pitch=sc.get("pitch", 26),
                         highlight=sc.get("highlight"), zoom=sc.get("zoom"), fit=sc.get("fit"),
                         lines=sc.get("lines"), points=sc.get("points"),
                         title=sc.get("title", ""), caption=sc.get("caption", ""),
                         hud=sc.get("hud"))
            paths.append(p)
            manifest.append({"i": len(paths) - 1, "scene": si, "yaw": yaw,
                             "title": sc.get("title"), "caption": sc.get("caption"),
                             "steps": sc.get("steps") or []})
    save_json(os.path.join(outdir, "frames", "manifest.json"), manifest)
    vid = None
    if video and paths:
        vid = render.encode(os.path.join(outdir, "frames", "seq_%03d.png"),
                            os.path.join(outdir, "video.mp4"), fps=1.6)
    return paths, vid


def motion_table(res):
    """写层任务的自证表：动作量 + 落位后的接触判定 + 扫掠区间。

    为什么不是「见证点」：写层交的不是一段距离，而是一组动作和它们造成的状态。
    能自证的只有三件事 —— 动作量有没有出处、落位后是「贴合」还是「穿插」、
    以及扫掠在哪个参数区间上真的发生了接触。所以这张表只列这三样，
    数值一律从答案里取（答案的每个数字都已被答案审计逐项追溯到证据步骤）。
    """
    v = (res.get("answer") or {}).get("values") or {}
    if not v:
        return ""
    L = ["", "| 项 | 值 | 含义 |", "|---|---|---|"]
    def put(k, label, meaning):
        if k in v:
            L.append("| %s | `%s` | %s |" % (label, json.dumps(v[k], ensure_ascii=False), meaning))
    # 对接类：动作量
    put("rotation_axis", "旋转轴", "绕哪根轴调朝向")
    put("rotation_angle_deg", "转角 (°)", "由对应点闭式解出，非目测")
    put("horizontal_shift_m", "水平位移 (m)", "平面内分量；垂直方向另行一步负责")
    put("horizontal_direction_unit", "水平方向（单位向量）", "方向与距离分开给，方向不靠读者自己除")
    put("horizontal_distance_m", "水平距离 (m)", "位移向量的长度，工具给出")
    put("turn", "转向", "右手定则：正角绕轴的正向逆时针")
    put("fit_residual_max_m", "配准残差 (m)", "残差大即配对错，不是拟合差")
    put("drop_m", "下降量 (m)", "两级台阶高差；落位后承托面贴合")
    d = v.get("docked") or {}
    if d:
        L.append("| 落位接触 | `%s` | 相交对数 >0 才算贴上；穿刺对数必须为 0 |"
                 % json.dumps(d, ensure_ascii=False))
    # 扫掠类：门扇开合。转轴是个复合量，摊成几行 —— 把整个 dict 塞进一个单元格，
    # 表格就没法读了，而这几项恰恰是要逐条对着题面勾的。
    hg = v.get("hinge") or {}
    if hg:
        for k, label in (("pivot", "转轴轴心 (m)"), ("axis_vector", "转轴轴向"),
                         ("handle_side_end", "把手侧那条窄面中心线")):
            if k in hg:
                L.append("| %s | `%s` | 门扇远离把手一侧的竖直窄面中心线 |"
                         % (label, json.dumps(hg[k], ensure_ascii=False)))
        if hg.get("reason"):
            L.append("| 取法 | %s | 两条窄面中心线取距把手较远者 |" % hg["reason"])
    put("chair_center_before", "椅子占地中心（改前）", "与原始模型对照")
    put("chair_shift_m", "椅子位移 (m)", "把占地中心搬到指定位置")
    put("chair_center_after", "椅子占地中心", "搬动后的位置，与目标值对照")
    for k, meaning in (("state1_first_contact_deg", "状态 1 首次接触角 (°)"),
                       ("state1_overlap_intervals_deg", "状态 1 重叠区间 (°)"),
                       ("state2_first_contact_deg", "状态 2 首次接触角 (°)"),
                       ("state2_overlap_intervals_deg", "状态 2 重叠区间 (°)"),
                       ("sweep_down_first_contact_t", "下降首次接触位置（t）"),
                       ("sweep_down_intervals_t", "下降全程的接触区间（t∈[0,1]）"),
                       ("sweep_up_intervals_t", "反向上升的接触区间，应只在零位移附近")):
        put(k, meaning, "由扫掠 + 边界二分给出，不是只查端点")
    for k, meaning in (("scan_step_deg", "粗扫步长 (°)"), ("refine_deg", "二分精度 (°)"),
                       ("n_probes", "扫掠取样点数")):
        put(k, meaning, "精度与代价必须一起报，否则「没查到」不等于「不存在」")
    # 只有表头没有数据行 = 这道题里没有任何运动/接触量。此时不能返回一张空表，
    # 否则调用方会以为「有运动量但都是空的」，反而把真正的测量表挤掉。
    if len(L) <= 3:
        return ""
    L.append("")
    L.append("- 判定口径：`n_pairs=0` → 分离；`n_pairs>0 且 n_pierce=0 且 depth≤1e-6` → "
             "相切/贴合；`n_pierce>0 或 depth>1e-6` → 重叠（真穿入）。")
    return "\n".join(L)


def measure_table(v):
    """把测量类的答案摊成表。单位从字段后缀读，不从值里猜 ——
    「这个数是长度还是面积」必须由字段名本身说清，否则 19.2 可以读成 19.2 m。"""
    if not v:
        return ""
    unit_of = {"_m2": "m²", "_m3": "m³", "_m": "m", "_ratio": "—", "_deg": "°"}
    skip = {"unit", "methods", "notes", "floor_extent_note"}
    rows = []
    for k, val in v.items():
        if k in skip or isinstance(val, (dict, list)):
            continue
        u = ""
        for suf, lab in (("_m2", "m²"), ("_m3", "m³"), ("_ratio", "—"), ("_m", "m")):
            if k.endswith(suf):
                u = lab
                break
        rows.append("| `%s` | %s | %s |" % (k, json.dumps(val, ensure_ascii=False), u))
    if not rows:
        return ""
    return "\n| 量 | 值 | 单位 |\n|---|---|---|\n" + "\n".join(rows)


def write_report(outdir, res, env, paths, vid):
    who = "本轮任务（由自然语言编译）" if env.get("brief") else os.path.basename(env["card"])
    L = ["# agent 运行报告 · %s\n" % who]
    L.append("> 本报告由沙盒在运行时生成。任务只以文本进入，模型只以文件进入；"
             "每个数字都能追到某一步工具调用。\n")
    L.append("| 项 | 值 |\n|---|---|")
    for k in ("input_kind", "card", "brief", "card_sha256", "model", "model_sha256",
              "policy", "episode", "episode_sha256", "n_tools", "t0"):
        if env.get(k) is not None:
            L.append("| %s | `%s` |" % (k, env[k]))
    L.append("| 状态 | **%s** |" % res["status"])
    L.append("| 轮次 | %d |" % res["turns"])
    L.append("| 耗时 | %.2f s |" % res["elapsed_s"])
    _u = res.get("llm_usage") or {}
    if _u.get("total_tokens"):
        L.append("| 用量 | prompt %s / completion %s / total %s tok（%d 次请求） |" % (
            _u.get("prompt_tokens"), _u.get("completion_tokens"),
            _u.get("total_tokens"), res.get("llm_calls") or 0))
    L.append("")

    L.append("## 一、决策链\n")
    L.append("| 步骤 | 阶段 | 工具 | 命中 | 判断依据 |")
    L.append("|---|---|---|---|---|")
    for s in res["steps"]:
        if s.get("rejected"):
            L.append("| — | %s | ~~%s~~ | 拒绝 | %s |" % (s.get("phase", ""), s.get("tool", ""),
                                                        (s.get("obs") or "")[:70]))
        else:
            L.append("| %s | %s | `%s` | %s | %s |" % (
                s.get("id"), s.get("phase", ""), s.get("tool", ""),
                s.get("n", "—"), (s.get("why") or "")[:80]))

    if res.get("answer"):
        a = res["answer"]
        L.append("\n## 二、答案（附证据引用）\n")
        L.append("```json\n%s\n```" % json.dumps(a.get("values"), ensure_ascii=False, indent=2))
        L.append("\n- 证据步骤：%s" % a.get("evidence"))
        L.append("- 置信度：%s" % json.dumps(a.get("confidence") or {}, ensure_ascii=False))
        if a.get("unknowns"):
            L.append("- 未确定项：")
            for u in a["unknowns"]:
                L.append("  - " + u)
    elif res.get("insufficient"):
        L.append("\n## 二、证据不足（按约定停止，不猜）\n")
        L.append("- 原因：%s" % res["insufficient"].get("reason"))
        for m in res["insufficient"].get("missing", []):
            L.append("  - 缺：%s" % m)

    L.append("\n## 三、编排层做了什么\n")
    L.append("| 机制 | 本次运行的表现 |")
    L.append("|---|---|")
    L.append("| 阶段门控 | 拒绝 %d 次越界调用 |" % res["gate_rejects"])
    L.append("| 答案审计 | 拒绝 %d 次无法溯源的答案 |" % res["audit_rejects"])
    L.append("| 失败升级 | %d 次进入 RECOVER |" % len(res["failures"]))
    L.append("| 预算纪律 | 代为推进阶段 %d 次（测量连续 5 步无新数字时触发，见账本 `phase_forced`）"
             % res.get("forced", 0))
    _re = res.get("role_evidence") or []
    if _re:
        _full = sum(1 for r in _re if r.get("n_witnessed") == r.get("n_cond"))
        L.append("| 判据反例 | %d/%d 个角色的每条判据都能指出反例（其余角色的判据里有凑数的，见 `role_witness`）"
                 % (_full, len(_re)))
    for f in res["failures"]:
        L.append("| ↳ %s | %s：%s |" % (f.get("step"), f.get("kind"),
                                       str(f.get("detail") or ("期望 %s 实得 %s" % (f.get("want"), f.get("got"))))[:90]))

    L.append("\n## 四、复核\n")
    v = res.get("verify") or {}
    if v:
        L.append("- 账本哈希链：%s（%d 条）" % ("完好" if v["ledger"]["ok"] else "断裂 %s" % v["ledger"]["broken"],
                                          v["ledger"]["n"]))
        if v.get("witness"):
            L.append("\n| 目标 | 报告值 | 见证点残差 | |wa−wb|−值 |")
            L.append("|---|---|---|---|")
            for w in v["witness"]:
                L.append("| %s | %.6f | %.2e | %.2e |" % (
                    w["sid"], w["value"],
                    max(w["residual_on_chair"], w["residual_on_target"]), w["gap_minus_value"]))
        _ans = (res.get("answer") or {}).get("values") or {}
        L.append("\n- 单位裁决：%s" % (v.get("unit") or _ans.get("unit")))
        ws = v.get("witness") or []
        # 阈值说明：报告值保留 6 位小数，比较本身就有 5e-7 的舍入差；
        # 所以取 2e-6（≈ 2 µm），比任何施工公差都严，又不会被显示精度冤枉。
        ok = bool(ws) and all(max(w["residual_on_chair"], w["residual_on_target"]) < 1e-5
                              and w["gap_minus_value"] < 2e-6 for w in ws)
        if ws:
            L.append("- 见证点复核：%s" % (
                "通过（%d 对，残差 <1e-5 m，|wa−wb| 与报告值差 <2e-6 m）" % len(ws) if ok
                else "**未通过**"))
        else:
            # 写层任务没有「两点之间距离」这种东西可见证：答案不是一段距离，
            # 而是一个动作（转多少、移多远、降多少）加一个状态（贴合还是穿插）。
            # 硬套见证点口径只会印出「未通过」，把一件本来没做的事说成失败。
            # 这里换成它真正的自证方式：动作量 + 落位后的接触判定 + 扫掠区间。
            mt = motion_table(res)
            if mt:
                L.append("- 见证点复核：不适用（写层任务无点对距离；改由下面的运动层量自证）")
                L.append(mt)
            else:
                # 纯测量任务既没有见证点、也没有运动：它的自证方式是「同一个量用
                # 两种算法各算一次」与「派生量由工具算出」。这里把答案原样摊开，
                # 让读的人能逐条对回决策链里的那一步。
                L.append("- 见证点复核：不适用（纯测量任务无点对距离、无运动量；"
                         "自证方式见下面的测量表与交叉核对）")
                L.append(measure_table(_ans))
    if vid:
        L.append("\n## 五、过程录像\n\n`frames/` %d 帧 → `video.mp4`（%s）。"
                 "每帧标题是阶段，副标题是 agent 当时的判断依据，可与决策链逐条对照。"
                 % (len(paths), "成功" if vid["ok"] else "失败：" + str(vid.get("err"))))
    open(os.path.join(outdir, "report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", help="任务卡（自然语言）。与 --brief 二选一")
    ap.add_argument("--brief",
                    help="管线由原话编译出的任务规格 JSON（agent/brief.py 产出，含 prompt / raw / 结构化要点）。"
                         "这是控制台的正常入口：用户只给一段话，结构化部分是这条管线自己算出来的。")
    ap.add_argument("--model", required=True)
    ap.add_argument("--episode", help="回放已记录的决策序列")
    ap.add_argument("--llm", help="在线 LLM 模型名（OpenAI 兼容）；auto = 按本机已有 key 挑")
    ap.add_argument("--base-url")
    ap.add_argument("--api-key")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--fallback", action="store_true")
    ap.add_argument("--tag", default="agent")
    ap.add_argument("--orbits", type=int, default=3)
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--wall-budget", type=float, default=180.0)
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    if not (a.card or a.brief):
        raise SystemExit("必须给 --card（一段自然语言）或 --brief（管线编译出的规格）之一")
    for flag, what in (("model", "模型"), ("card", "任务卡"), ("brief", "任务规格")):
        p = getattr(a, flag)
        if p and not os.path.exists(p):
            raise SystemExit(
                "找不到%s文件：%s\n赛方的模型包与任务卡不随仓库分发。请下载 "
                "https://hack26.ctrlpath.co/api/tasks/download 后解压到 pkg/，\n"
                "使路径形如 pkg/AI-Geometry-Models/IFC/AI_Geometry_Input.ifc（见 README「一分钟上手」）。"
                % (what, p))

    outdir = os.path.join(HERE, "runs", a.tag)
    os.makedirs(outdir, exist_ok=True)
    brief = None
    if a.brief:
        # 控制台路径：用户只给了原话，card_text 是管线从原话编译出来的说明，
        # 不是人写的题卡。brief 原样留在 run 目录里，页面和审计都读它。
        brief = json.load(open(a.brief, encoding="utf-8"))
        card_text = brief["prompt"]
        card_rel = os.path.relpath(a.brief, HERE)
    else:
        card_text = open(a.card, encoding="utf-8").read()
        card_rel = os.path.relpath(a.card, HERE)
    ledger = Ledger(os.path.join(outdir, "events.jsonl"))
    policy = load_policy(a)
    if brief:
        # 规格跟着这一轮走：runs/ 被拷走、_cards 被清掉，这一轮的「题目从哪来」都还在。
        save_json(os.path.join(outdir, "brief.json"), brief)

    env = {"card": card_rel,
           "card_sha256": hashlib.sha256(card_text.encode()).hexdigest()[:16],
           "brief": (os.path.join("runs", a.tag, "brief.json") if brief else None),
           "input_kind": ("natural_language" if brief else "task_card"),
           "model": os.path.relpath(a.model, HERE), "model_sha256": sha(a.model),
           "policy": policy.kind, "episode": a.episode,
           "llm": getattr(policy, "model", None),
           "llm_usage": (policy.usage() if hasattr(policy, "usage") else None),
           "episode_sha256": sha(a.episode), "n_tools": len(TOOLS),
           "tools": sorted(TOOLS), "phases": PHASES,
           "t0": __import__("time").strftime("%Y-%m-%d %H:%M:%S")}

    bus = Bus(ledger)
    loop = Loop(bus, ledger, policy, card_text, env["card"], a.model,
                max_turns=a.max_turns, wall_budget=a.wall_budget,
                # 交付类型来自管线对原话的编译：只要求测量的任务不必进写层。
                # 用 --card 跑既有任务卡时保持原行为（None → False）。
                read_only=bool(brief and brief.get("kind") == "measure"))
    res = loop.run()
    if hasattr(policy, "usage"):
        # 用量必须在跑完之后再取一次：env 在开始时构建，那时还没有任何请求。
        env["llm_usage"] = policy.usage()
        env["llm_calls"] = len(getattr(policy, "calls", []))
        # 同时写进 result.json：成本要能被事后统计，不然"多少次调用、多少钱"只能靠估。
        res["llm_usage"] = env["llm_usage"]
        res["llm_calls"] = env["llm_calls"]

    ans = res.get("answer") or {}
    verify = {"ledger": Ledger.verify(os.path.join(outdir, "events.jsonl"))}
    if ans.get("values", {}).get("targets"):
        verify["witness"] = witness_check(loop, ans)
        meta = {}
        for out in loop.results.values():
            if isinstance(out, dict) and isinstance(out.get("meta"), dict):
                meta = out["meta"]
                break
        verify["unit"] = meta.get("scale_source")
    res["verify"] = verify
    res["env"] = env
    save_json(os.path.join(outdir, "result.json"), res)
    save_json(os.path.join(outdir, "steps.json"), res["steps"])

    paths, vid = [], None
    if bus.scene:
        paths, vid = render_all(bus, res, outdir, a.orbits, not a.no_video, loop.results)
    write_report(outdir, res, env, paths, vid)

    print("\n=== agent · %s ===" % a.tag)
    print("  状态 %s   轮次 %d   耗时 %.2fs   门控拒绝 %d   审计拒绝 %d" % (
        res["status"], res["turns"], res["elapsed_s"], res["gate_rejects"], res["audit_rejects"]))
    for s in res["steps"]:
        tag = "拒绝" if s.get("rejected") else (s.get("n") if s.get("n") is not None else "")
        print("   %-3s %-9s %-16s %-4s %s" % (s.get("id"), s.get("phase", ""), s.get("tool", ""),
                                              tag, (s.get("why") or "")[:60]))
    if ans.get("values"):
        print("  答案:", json.dumps(ans["values"], ensure_ascii=False)[:200])
    print("  账本:", verify["ledger"])


def witness_check(loop, ans):
    """见证点复核：与测量完全独立的第二路径 —— 把见证点投影回各自三角面。

    注意取数来源：从工具返回里取，不从答案里取。答案里的数字若被改过，
    这里就对不上 —— 这正是「答案不能自证」的落地方式。
    """
    from tools import geom
    by = {s.sid: s for s in loop.bus.scene}
    targets = (ans.get("values") or {}).get("targets") or {}
    rows = []
    # rank 把见证点放在 rows[i] 里，distance 放在顶层；两种形态都要收，
    # 否则批量测量出来的数会「没有见证点可复核」—— 那是复核对功能的反向依赖，必须避免。
    cands, seen = [], set()
    for out in loop.results.values():
        if not isinstance(out, dict):
            continue
        if out.get("witness_a") and out.get("b"):
            cands.append((out.get("a"), out["b"], out))
        for r in out.get("rows") or []:
            if isinstance(r, dict) and r.get("witness_a"):
                cands.append((out.get("a"), r.get("sid"), r))
    for a_id, b_id, out in cands:
        if not b_id or (a_id, b_id) in seen:
            continue
        seen.add((a_id, b_id))
        B = by.get(b_id)
        A = by.get(a_id) if a_id else None
        if A is None or B is None:
            continue
        wa, wb = np.array(out["witness_a"], float), np.array(out["witness_b"], float)
        TA = np.stack(geom.tri_arrays(A), axis=1)
        TB = np.stack(geom.tri_arrays(B), axis=1)
        _, da = geom._closest_on_tri_batch(wa[None, :], TA)
        _, db = geom._closest_on_tri_batch(wb[None, :], TB)
        rows.append({"sid": b_id, "value": out["value"],
                     "residual_on_chair": float(da.min()), "residual_on_target": float(db.min()),
                     "gap_minus_value": float(abs(np.linalg.norm(wa - wb) - out["value"]))})
    return rows


if __name__ == "__main__":
    main()
