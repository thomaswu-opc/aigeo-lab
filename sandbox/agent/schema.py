# -*- coding: utf-8 -*-
"""schema.py — 把工具注册表翻译成 LLM 能调用的规格，并把工具输出压缩成「可决策的观察」。

两个东西必须分开，否则会掉进两个坑：

  1. schema 必须来自注册表，不能手写。
     手写 schema = 手写工具清单 = 工具清单里会不知不觉混进「第几题用哪个」的知识。
     这里全部从 core.runtime.TOOLS 反射出来，注册表里没有任务编号，schema 里也就没有。

  2. 观察不等于输出。
     工具返回的原始 dict 直接塞给 LLM，会让它淹没在 22 个实体的 17 个字段里，
     然后开始编造。压缩的目标是「决策所需的最小充分证据」：
     保留判据会用到的字段，保留计数，保留不确定度与降级标注，丢掉其余。
"""
import inspect
import json
import os
import re

from core.runtime import TOOLS, schema

_TYPE = {"str": "string", "list": "array", "dict": "object",
         "int": "integer", "float": "number", "bool": "boolean"}

# 判据会用到、且不泄漏语义的字段。src_type / name 一律不进观察。
_KEEP = ("sid", "d0", "d1", "d2", "min_thickness", "z_min", "axis_aligned",
         "dominant", "thickness", "is_axis_aligned", "coverage", "dev_from_world_deg",
         "n_clusters", "clusters", "found", "key", "a", "b", "up_confident", "where",
         "fill", "cover", "volume", "value", "aabb_gap", "cid", "n", "span", "bbox",
         "share", "tilt_from_world_deg", "area", "holes", "reflex", "notch_area",
         "inside", "outside", "groups", "rows", "items", "meta", "info", "axes",
         "origin", "unknowns", "confidence", "evidence", "method", "uncertainty",
         "n_segments", "n_groups", "transform", "witness_a", "witness_b", "limits",
         "skipped", "scale_source", "unit", "note", "notes", "direction", "axis",
         "bbox_lo", "bbox_hi", "ntri", "max_patch_area", "bbox_volume",
         # 包围盒中心：位置判别（哪面墙、哪个洞口）只能靠这三个标量
         "cx", "cy", "cz", "dims", "sx", "sy", "sz", "span",
         # 水平长边落在哪根轴上（h_long_axis / h_long / h_short）
         "h_long_axis", "h_long", "h_short",
         # 网格体检（mesh_health）：编辑类任务的判据全在这里，少一个字段就等于
         # 让模型在「看不见封闭性」的情况下判断封口成功没有。
         "closed", "watertight", "n_boundary_edges", "n_boundary_loops",
         "n_duplicate_faces", "n_degenerate", "n_nonmanifold_edges",
         "n_flipped_edges", "euler", "surface_area", "signed_volume", "dims",
         "n_vert", "n_tri", "n_used_vert", "note_unused_vert",
         # 洞口族（assembly）：归属证据一条都不能压掉，否则模型只能看结论、无法复核
         "families", "members", "rel", "in_prism", "cover_uv", "edge_gap",
         "spans_wall", "solid_fill", "via", "loose", "prism", "corners",
         "rect", "u_dir", "v_dir", "n_dir", "u_len", "v_len", "u_range", "v_range", "depth",
         # 编辑与落盘的返回量
         "kept_side", "plane", "stats", "cut", "health_before", "health_after",
         "dangling", "loops", "n_cap_tri", "anchors", "max_shift",
         "bbox_before", "bbox_after", "order", "bytes", "format", "n_solids", "path",
         # 占地 / 距边 / 洞口底边（checks.py）。少了这些，模型只知道「量过了」，
         # 不知道量出来是多少，于是只能凭平移量反推答案 —— 那正是审计要抓的东西。
         "center_world", "rect_world_xy", "rect", "center", "size", "grounded",
         "z_min", "z_max", "axis_map", "nearest_on_footprint", "nearest_on_edge",
         "footprint_rect", "edge", "width", "height", "sill_z", "inward_normal",
         "axis_u", "axis_thin", "wall", "opening", "floor", "rule", "n_tri_checked",
         # 碰撞 / 扫掠（collide.py）。这几个字段是「会不会撞」的全部可观测证据：
         # 相交对数为 0 才叫分离，depth 超过容差才叫重叠 —— 少了 depth，
         # 「相切」和「撞进去 2 mm」在观察里长得一模一样。
         "hit", "contact", "overlap", "n_pairs", "depth", "n_in_a", "n_in_b",
         "deepest", "closed_a", "closed_b", "tol",
         "first_contact_deg", "first_contact_n_pairs", "first_overlap_deg",
         "intervals", "overlap_intervals", "n_probes", "scan_step_deg", "refine_deg",
         "first_contact_t", "first_contact_m", "intervals_t", "intervals_m",
         "pivot", "moving", "fixed", "length", "a0", "a1",
         # 形状分解与配准（shape.py）
         # thin_ends 的元素是「竖直线」，只留 mid 与边长；少了 mid，两条候选转轴线
         # 在观察里就是两个空对象 —— 那正是判「哪条远离把手」的唯一依据。
         "mid", "edge_len",
         "parent", "parts", "share", "kept_original", "polygon", "sides", "simple",
         "top", "pockets", "bottom", "size_axes", "rings", "inner", "perimeter",
         "n_vert", "n_tri", "n_pairs", "angle_deg", "rotation_axis", "translate",
         # 夹角（angle）：acute_deg 是平面之间的较小夹角，dot 与两根法向一起给，
         # 少一个模型就会自己取补角，把 65° 报成 115°。
         "acute_deg", "dot", "n_a", "n_b", "share_a", "share_b",
         # 洞口（openings）：缺料的矩形、下沿标高、是否通底 —— 门洞与窗洞就是靠这三个量分开的
         "openings", "touches_top", "resolution",
         "residual", "rms", "max",
         # 配准解的平面分量。少了 translate_plane，调用方只能拿到含轴向分量的
         # 三维平移，一不小心就把「插脚中心搬到槽底」当成水平移动量用掉。
         "translate_plane", "shift_xy_length", "shift_axis_len", "direction_unit", "turn",
         # 顶点高度分层（levels）：特征平面靠它找，层高与顶点数一个都不能丢
         "levels", "h", "level",
         # 反解与拟合（measure.py）。这几个字段共同回答「这个数是解出来的还是编的」：
         # 没有 monotonic/crossings，二分给出的高度看起来永远是对的；
         # 没有 roundness，拟合半径就成了另一条先验。
         "ok", "error", "radius", "ideal_volume", "ideal_area", "roundness",
         "hull", "hull_n", "bbox_area", "fill_ratio", "n_points",
         "monotonic", "crossings", "n_solutions_brackets", "bracket", "samples",
         "search", "n_iter", "probe_height", "n_inner", "gross", "op",
         # 区域关系与贴合面积（contact.py）。relation 与 signed_gap 必须成对出现：
         # 只有距离没有关系名，读的人会把「内切」当「分离」；只有关系名没有体积，
         # 「整颗在内」与「凸出 1 mm」在观察里没有区别。
         "relation", "signed_gap", "center_gap", "inside_volume", "outside_volume",
         "sphere_volume", "ratio", "cap_height", "sphere_center", "box_lo", "box_hi",
         "min_distance", "max_patch", "normal",
         # 平面包含（contains.plane）：三维包含会把「坐在地板上的椅子」判成越界，
         # 不把 plane 与 margin 一起带出来，这个误判在观察里看不出来。
         "plane", "margins", "margin", "lo_gap", "hi_gap", "all_inside",
         "n_inside", "n_outside",
         # 答案里成对呈现的量。配对必须能一眼看出「拿哪个跟哪个比」，
         # 只剩空对象就等于把配对依据从报告里删掉了。
         "plug_area", "pocket_area", "plug_sides", "pocket_sides")

# 控制类工具：不进几何总线，由编排层拦截。LLM 用它们表达「阶段推进 / 放弃」。
CONTROL = [
    {"type": "function", "function": {
        "name": "advance_phase",
        "description": "当前阶段的目标已达成，进入下一阶段。必须写明达成依据。",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string", "description": "为什么可以进入下一阶段（引用步骤号与计数）"}},
            "required": ["reason"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "declare_insufficient",
        "description": "证据不足以确定答案，停止并说明缺什么。宁可报未知，不要猜数字。",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string"},
            "missing": {"type": "array", "items": {"type": "string"},
                        "description": "逐条列出缺失的信息"}},
            "required": ["reason", "missing"], "additionalProperties": False}}},
]


# ------------------------------------------------------------ 参数说明瘦身
# 实测：一次 MEASURE 调用的 prompt 里，工具 schema 占 74%（参数说明 15.0k 字符 +
# 工具描述 6.0k 字符）。钱花在「教模型怎么调工具」上，不在「它看到的东西」上。
# 所以这里对**发给模型的渲染结果**做精简，源码里的完整说明一个字不动 ——
# 页面、README、评审看到的是原文，模型看到的是能做出正确选择的最小信息。
# 想对照原文：WB_SCHEMA=full 启动即可回到精简前的渲染。
_TERSE = os.environ.get("WB_SCHEMA", "terse").lower() != "full"

# 观察压缩的总开关。WB_OBS=full 时基本不压（预算放大 9 倍、一条不截、不投影），
# 用来在没有 git 回退的前提下做「压缩前/后」对照 —— 压缩逻辑和修复逻辑是两件事，
# 想判断「质量不稳是不是压缩害的」，只需要把这一层关掉，不该把轮次修复、审计改进一起退掉。
_OBS_FULL = os.environ.get("WB_OBS", "").lower() == "full"
_DEFAULT_RE = re.compile(r"[（(]默认[^）)]*[）)]")


def _split_top(text, seps="，,；;。"):
    """按标点切句，但**不切开括号内部** —— 否则「（mm，默认 15）」会被切成两半，
    单位那半截被丢掉，模型就可能把 0.015 当 15 传进来。"""
    out, buf, depth = [], [], 0
    for ch in text:
        if ch in "（(":
            depth += 1
        elif ch in "）)":
            depth = max(0, depth - 1)
        if ch in seps and depth == 0:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    out.append("".join(buf))
    return out


def _brief(desc, n=26):
    """把「类型 + 长说明」压成「类型 + 一句话」，保留单位与默认值。

    "list  实体清单，或引用字符串（如 @load.solids、@clu.groups[0].items）"
      → "list 实体清单"
    "float 连通阈值（mm，默认 15）：装配件之间的功能缝…"
      → "float 连通阈值（mm）（默认 15）"
    单位与默认值都必须留：丢了前者会把 mm 当 m，丢了后者会多传一遍参数。
    """
    if not _TERSE or not isinstance(desc, str):
        return desc
    typ, _, rest = desc.strip().partition(" ")
    rest = rest.strip()
    if not rest:
        return typ
    unit = dflt = ""
    m = re.search(r"[（(]([^）)]*默认[^）)]*)[）)]", rest)
    if m:
        inner = m.group(1)
        pre = inner.split("默认")[0].strip(" ，,、")
        val = inner.split("默认", 1)[1].strip(" ：:，,、")
        if pre:
            unit = "（%s）" % pre
        if val:
            dflt = "（默认 %s）" % val
        rest = rest.replace(m.group(0), unit)
    body = re.sub(r"[（(][^）)]{16,}[）)]", "", rest)
    out = ""
    for part in _split_top(body):
        part = part.strip()
        if not part:
            continue
        if out and len(out) + len(part) + 1 > n:
            break
        out = (out + "，" + part) if out else part
    if not out:
        out = body.strip()[:n]
    if out.count("（") > out.count("）") and "（" in out:
        out = out[: out.rindex("（")].rstrip("，,、 ")
    if len(out) > n:
        out = out[:n].rstrip("，,、 ") + "…"
    return (typ + " " + out).strip() + dflt


def _fields(returns, n=46):
    """把「返回：{a, b, [c]…}」压成字段名清单 —— 模型要的是"能拿到什么"，不是散文。

    取花括号里第一层的键名，超过预算就截断。字段名本身就是契约，去掉它等于让模型
    猜返回结构（它就会去重新 load 一遍）。
    """
    if not _TERSE or not isinstance(returns, str):
        return returns
    t = returns.strip()
    m = re.search(r"[{（(](.*)[}）)]", t)
    body = m.group(1) if m else t
    keys = [k.strip() for k in re.split(r"[,，、]", body) if k.strip()]
    keys = [k.split(":")[0].split("=")[0].strip(" []") for k in keys]
    out = "{" + ", ".join(k for k in keys if k and not k.startswith("…")) + "}"
    return out[:n] if len(out) > n else out


def tool_specs(only=None):
    """返回 OpenAI function-calling 规格。only 可限定子集（用于阶段门控）。"""
    out = []
    for t in schema():
        if only is not None and t["name"] not in only:
            continue
        fn = TOOLS[t["name"]]["fn"]
        sig = inspect.signature(fn)
        props, req = {}, []
        for pname, pdesc in t["parameters"].items():
            ty = pdesc.split(None, 1)[0].rstrip("：:").lower()
            props[pname] = {"type": _TYPE.get(ty, "string"), "description": _brief(pdesc)}
            if ty.startswith("list"):
                # 集合参数必须同时接受「引用字符串」。只声明 array 时模型没法写 @load.solids，
                # 只能把整份清单逐字复制进参数 —— 实测它复制了 19 个 sid，既贵又易错。
                # 引用由编排层解析，模型不必搬运数据。
                props[pname] = {
                    "anyOf": [{"type": "array", "items": {}}, {"type": "string"}],
                    "description": _brief(pdesc) + (
                        ("；或写 @步骤.field 引用" if _TERSE else
                         "。也可以直接写引用字符串，如 \"@load.solids\"、\"@clu.groups[0].items\"；"
                         "引用指向已完成步骤的返回值，不必自己复制清单内容。")
                        if ("不要写" not in (pdesc or "")) and (
                            "引用" in (pdesc or "") or "@" in (pdesc or "")
                            or pname in ("items", "sids", "walls", "solids", "targets"))
                        else "")}
            sp = sig.parameters.get(pname)
            if sp is not None and sp.default is inspect.Parameter.empty:
                req.append(pname)
        # 这三项不传给工具函数，由编排层消费：
        #   why      —— 判断依据，进账本，是评审读懂推理链的入口
        #   label    —— 语义标签，决定后续引用 @label.field 的稳定性
        #   expect_n —— 声明的预期命中数，不符时编排层判失败并触发自救
        # 不给 LLM 这三项，RECOVER 机制在 LLM 路径上等于不存在 —— 这是必须补的口子。
        props["why"] = {"type": "string", "description": _brief(
            "一句话说明这次调用要确认什么（会写进账本）", 20)}
        props["label"] = {"type": "string", "description": _brief(
            "本步的语义标签（如 win / leaf），供后续用 @label.field 引用", 22)}
        props["expect_n"] = {"type": "integer", "description": _brief(
            "预期返回的条目数；不符时编排层判定失败并进入自救。计数类工具才需要", 24)}
        if _TERSE:
            # 前置条件不进 schema：写错会被门控拒（拒绝信息本身就带「这一阶段的判定是什么」），
            # 比事先读一段规则更省。失败时的降级提示保留（它防的是"信了一个不可信的量"）。
            # 工具的一句话用途压到 44 字：够回答「这工具干嘛的」，不够讲它的来历。
            # 散文式背景（"例如把某实体的测量行对齐到最终报告…"）对选择没有增量。
            desc = "%s｜失败时：%s｜返回：%s" % (
                _brief(t["description"], 44), _brief(t["degrade"], 30), _fields(t["returns"]))
        else:
            desc = "%s｜前置条件：%s｜失败时：%s｜返回：%s" % (
                t["description"], t["preconditions"], t["degrade"], t["returns"])
        out.append({"type": "function", "function": {
            "name": t["name"],
            "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req,
                           "additionalProperties": False}}})
    return out


def full_specs():
    return tool_specs() + CONTROL


# ------------------------------------------------------------ 观察压缩


def _trim(o, nd=4, in_list=False, depth=0):
    """只对「清单里的元素」做字段白名单，嵌套的元信息保持完整。

    早期版本对任意深度都过滤，结果把 principal_frame 的 info（coverage / is_axis_aligned）
    整个删空了 —— 那正是 agent 判断要不要切参照系的关键证据。
    教训：压缩要按「体积来自哪里」下手（长清单），不要按深度一刀切。
    """
    # 深度上限 8 而不是 6：截面→simple→thin_ends→元素→mid 这条链正好 7 层，
    # 6 层会把 mid 压成「…」，于是「两条候选转轴线在哪」在观察里彻底消失。
    if depth > 8:
        return "…"
    if isinstance(o, float):
        # 四舍五入会把「穿进去 2.4e-8」和「完全贴合」压成同一个 0.0，
        # 也会把 1e-6 的容差压成 0。判「相切还是重叠」用的就是这两个数，
        # 压成 0 之后观察里就只剩「看着像贴上了」，等于没有证据。
        # 所以只有当这个数会被舍成 0 时才改用科学计数法。
        if o != 0.0 and round(o, nd) == 0.0:
            return float("%.2e" % o)
        return round(o, nd)
    if isinstance(o, dict):
        keep = (lambda k: k in _KEEP) if in_list else (lambda k: True)
        return {k: _trim(v, nd, False, depth + 1) for k, v in o.items() if keep(k)}
    if isinstance(o, list):
        return [_trim(v, nd, isinstance(v, dict), depth + 1) for v in o]
    return o


# 观察压缩时优先保留的字段：够回答「几个、多大、落地没有」，不够回答「这是什么」。
# 后者的信息量大但极占字符（法向簇一条就上百字符），留给按 sid 单独取回。
# min_thickness 必须带上：d2 是「最短边」，min_thickness 是「板厚」，两者对开着的门扇
# 差了整整一个量级（0.41 vs 0.04）。只给 d2 时，模型分不清「薄板」和「细长条」，
# 于是把门套横档当成门扇候选、把贯穿墙厚的洞口盒当成实心块，判据从一开始就歪了。
_SLIM_KEYS = ("sid", "n", "d0", "d1", "d2", "sx", "sy", "sz", "min_thickness",
              # 只留「长边落在哪根轴」这个比较结论（h_long_axis）；h_long/h_short 与
              # max(sx,sy) 重复，压掉它们能让 20 条实体一次全列出来 —— 对「逐项列举」
              # 类任务来说，看得全比看得细重要。
              "h_long_axis",
              "z_min", "axis_aligned")


def _slim_rows(rows, nd=3):
    """把一串实体指纹压成「一行一个实体」。只在原样塞不下时才用。

    数值压到 3 位小数：几何判据的分辨率是毫米，第 4 位以后只有浮点噪声
    （"1.999999" 这种）。不压的话 20 条实体总长 3554 字符，刚好越过 3224 的预算，
    于是退化成「只显示前 5 条」—— 而「逐项列举」类任务就再也凑不齐 16 个实体。
    """
    def _r(v):
        if isinstance(v, bool) or not isinstance(v, float):
            return v
        r = round(v, nd)
        return int(r) if abs(r - int(r)) < 1e-12 else r

    out = []
    for it in rows:
        if isinstance(it, dict) and "sid" in it and ("d0" in it or "dims" in it):
            out.append({k: _r(it[k]) for k in _SLIM_KEYS if k in it})
        else:
            out.append(it)
    return out


def observe(name, out, max_items=None, max_chars=None):
    """把工具输出压成观察。返回 (text, stats)。

    stats 里的 n / n_items 是编排层做「唯一性判定」的依据 —— 注意是编排层判定，
    不是 LLM 判定：LLM 只看到数字，改不动它。

    默认预算是「够 20 个实体一行一个」；WB_OBS=full 时预算放大到基本不触发压缩。
    """
    if max_items is None:
        max_items = 400 if _OBS_FULL else 24
    if max_chars is None:
        max_chars = 48000 if _OBS_FULL else 5200

    stats = {}
    if isinstance(out, dict):
        for key in ("n", "n_groups", "n_segments", "n_faces", "n_regions"):
            if key in out:
                stats[key] = out[key]
        for key in ("items", "solids", "rows", "results", "groups"):
            if isinstance(out.get(key), list):
                stats["n_" + key] = len(out[key])
                if "n" not in stats:
                    stats["n"] = len(out[key])

    t = _trim(out)
    if isinstance(t, dict):
        notes = []
        for key in list(t):
            if not isinstance(t[key], list):
                continue
            total = len(t[key])
            raw_sz = len(json.dumps(t[key], ensure_ascii=False, separators=(",", ":")))
            # 条数少不等于塞得下：20 个实体各带法向簇与体积，JSON 能到十几 KB。
            # 只看条数会让它走「整体按字符截断」那条路，截出来的是半截 JSON，
            # 模型没法用、只能反复重取 —— 实测有一半轮次耗在这上面。
            if total <= max_items and raw_sz <= max_chars * 0.62:
                continue
            # 按「字符预算」而不是「条数」截断。按条数截断时，一条紧凑记录和一条巨大记录
            # 代价相同，于是 20 个 150 字符的实体只给看 12 个 —— 模型看不到全貌，就会反复
            # 想「补齐被截断的部分」，实测把一半轮次耗在这上面。按字符给预算，
            # 紧凑的结构自然能一次看全，臃肿的结构才被截。
            kept, acc = [], 0
            for it in t[key]:
                sz = len(json.dumps(it, ensure_ascii=False, separators=(",", ":")))
                if kept and acc + sz > max_chars * 0.62:
                    break
                kept.append(it)
                acc += sz + 1
            if len(kept) < total:
                # 截断之前先试一次「投影」：20 个实体的完整指纹塞不下，但一行一个实体的
                # 三边 + 落地标高塞得下。模型要的是全貌时，给全貌；要某一件的细节时，
                # 让它按 sid 单独取 —— 比让它反复重读同一个被截断的清单好。
                slim = _slim_rows(t[key]) if not _OBS_FULL else t[key]
                ssz = sum(len(json.dumps(it, ensure_ascii=False, separators=(",", ":")))
                          for it in slim)
                if ssz <= max_chars * 0.62:
                    t[key] = slim
                    notes.append("%s 有 %d 条，已压成一行一个实体（只留 %s）"
                                 % (key, total, "/".join(_SLIM_KEYS)))
                    stats.setdefault("slim", []).append("%s×%d" % (key, total))
                else:
                    t[key] = kept
                    notes.append("%s 只显示了前 %d 条（实际 %d 条）" % (key, len(kept), total))
                    stats.setdefault("truncated", []).append("%s→%d/%d" % (key, len(kept), total))
        if notes:
            # 截断必须写进观察本身。只把它记在 stats 里，模型会看到 n=20 却只有 8 条，
            # 于是合理地怀疑数据有问题，然后反复重读同一个工具直到停滞 —— 实测确实发生过。
            # 上下文管理器对自己被压缩这件事必须诚实，否则模型的所有推理都建立在一个假前提上。
            slimmed_only = bool(stats.get("slim")) and not stats.get("truncated")
            tail = ("。条数是完整的，只是字段少了；要某一件的完整指纹（法向簇/体积/覆盖）"
                    "用 select 带 where 按 sid 取回，不要重复调用同一工具。"
                    if slimmed_only else
                    "。需要完整集合时用 select 按条件收窄，或按 sid 重新查询，"
                    "不要重复调用同一工具。")
            t["_truncated"] = "；".join(notes) + tail
    s = json.dumps(t, ensure_ascii=False, separators=(",", ":"))
    if len(s) > max_chars:
        s = s[:max_chars] + "…（已截断 %d 字符；完整内容在账本里）" % (len(s) - max_chars)
        stats["truncated_chars"] = True
    stats["chars"] = len(s)
    return s, stats


def hint_for(name, stats):
    """给 LLM 的一句「不确定性提示」。这是编排层对工具语义的封装，不是数据。"""
    n = stats.get("n")
    if stats.get("truncated"):
        return ("观察已截断（%s）。不要据此认为数据缺失：用 select 的 where 条件把集合收窄到"
                "你能完整看到为止。" % ", ".join(stats["truncated"]))
    if name in ("select",) and n is not None:
        if n == 0:
            return "0 个候选：判据太严或参照系不对。不要放宽到「看着像」的程度；先考虑换参照系或换判据。"
        if n == 1:
            return "唯一命中：这是本轮最强的证据形态。"
        return "%d 个候选：判据不足以区分，需要加一个正交判据把它压到 1。" % n
    if name == "principal_frame":
        return "若 is_axis_aligned=false，说明模型整体不与坐标轴对齐，后续描述量应改用 items_frame。"
    if name == "clusters":
        return "多簇说明文件里有多个互相分离的模型；先确认要处理的是哪一簇再往下走。"
    if name == "distance":
        return "复核 witness 是否落在各自表面上；这是这个数字唯一的自证方式。"
    return ""
