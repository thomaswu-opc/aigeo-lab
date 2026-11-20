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
         "cx", "cy", "cz", "dims",
         # 网格体检（mesh_health）：编辑类任务的判据全在这里，少一个字段就等于
         # 让模型在「看不见封闭性」的情况下判断封口成功没有。
         "closed", "watertight", "n_boundary_edges", "n_boundary_loops",
         "n_duplicate_faces", "n_degenerate", "n_nonmanifold_edges",
         "n_flipped_edges", "euler", "surface_area", "signed_volume", "dims",
         "n_vert", "n_tri", "n_used_vert", "note_unused_vert",
         # 编辑与落盘的返回量
         "kept_side", "plane", "stats", "cut", "health_before", "health_after",
         "dangling", "loops", "n_cap_tri", "anchors", "max_shift",
         "bbox_before", "bbox_after", "order", "bytes", "format", "n_solids", "path",
         # 占地 / 距边 / 洞口底边（checks.py）。少了这些，模型只知道「量过了」，
         # 不知道量出来是多少，于是只能凭平移量反推答案 —— 那正是审计要抓的东西。
         "center_world", "rect_world_xy", "rect", "center", "size", "grounded",
         "z_min", "z_max", "axis_map", "nearest_on_footprint", "nearest_on_edge",
         "footprint_rect", "edge", "width", "height", "sill_z", "inward_normal",
         "axis_u", "axis_thin", "wall", "opening", "floor", "rule", "n_tri_checked")

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
            props[pname] = {"type": _TYPE.get(ty, "string"), "description": pdesc}
            if ty.startswith("list"):
                # 集合参数必须同时接受「引用字符串」。只声明 array 时模型没法写 @load.solids，
                # 只能把整份清单逐字复制进参数 —— 实测它复制了 19 个 sid，既贵又易错。
                # 引用由编排层解析，模型不必搬运数据。
                props[pname] = {
                    "anyOf": [{"type": "array", "items": {}}, {"type": "string"}],
                    "description": pdesc + "。也可以直接写引用字符串，如 "
                                   "\"@load.solids\"、\"@clu.groups[0].items\"、\"@thin.items\"；"
                                   "引用指向已完成步骤的返回值，不必自己复制清单内容。"}
            sp = sig.parameters.get(pname)
            if sp is not None and sp.default is inspect.Parameter.empty:
                req.append(pname)
        # 这三项不传给工具函数，由编排层消费：
        #   why      —— 判断依据，进账本，是评审读懂推理链的入口
        #   label    —— 语义标签，决定后续引用 @label.field 的稳定性
        #   expect_n —— 声明的预期命中数，不符时编排层判失败并触发自救
        # 不给 LLM 这三项，RECOVER 机制在 LLM 路径上等于不存在 —— 这是必须补的口子。
        props["why"] = {"type": "string",
                        "description": "一句话说明这次调用要确认什么（会写进账本）"}
        props["label"] = {"type": "string",
                          "description": "本步的语义标签（如 win / leaf），供后续用 @label.field 引用"}
        props["expect_n"] = {"type": "integer",
                             "description": "预期返回的条目数；不符时编排层判定失败并进入自救。计数类工具才需要"}
        out.append({"type": "function", "function": {
            "name": t["name"],
            "description": "%s｜前置条件：%s｜失败时：%s｜返回：%s" % (
                t["description"], t["preconditions"], t["degrade"], t["returns"]),
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
    if depth > 6:
        return "…"
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, dict):
        keep = (lambda k: k in _KEEP) if in_list else (lambda k: True)
        return {k: _trim(v, nd, False, depth + 1) for k, v in o.items() if keep(k)}
    if isinstance(o, list):
        return [_trim(v, nd, isinstance(v, dict), depth + 1) for v in o]
    return o


def observe(name, out, max_items=24, max_chars=5200):
    """把工具输出压成观察。返回 (text, stats)。

    stats 里的 n / n_items 是编排层做「唯一性判定」的依据 —— 注意是编排层判定，
    不是 LLM 判定：LLM 只看到数字，改不动它。
    """
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
            if total <= max_items:
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
                t[key] = kept
                notes.append("%s 只显示了前 %d 条（实际 %d 条）" % (key, len(kept), total))
                stats.setdefault("truncated", []).append("%s→%d/%d" % (key, len(kept), total))
        if notes:
            # 截断必须写进观察本身。只把它记在 stats 里，模型会看到 n=20 却只有 8 条，
            # 于是合理地怀疑数据有问题，然后反复重读同一个工具直到停滞 —— 实测确实发生过。
            # 上下文管理器对自己被压缩这件事必须诚实，否则模型的所有推理都建立在一个假前提上。
            t["_truncated"] = ("；".join(notes) +
                               "。需要完整集合时用 select 按条件收窄，或按 sid 重新查询，"
                               "不要重复调用同一工具。")
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
