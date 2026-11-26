# -*- coding: utf-8 -*-
"""accuracy_check.py — 区分「有答案」和「答对」，回看历史 run。

为什么需要它：bench.py 的状态 OK / audit_reject=0 只证明「交出了数、数有出处」，
证明不了「数答对了」。实测 live_1126_143151 状态 OK、审计通过，三个最短距离全对，
但门扇夹角交了 25°（真值 65°）—— 它量错了参照墙（量了与门墙垂直的另一面墙），
审计放行是因为 25° 确实是 angle 工具返回过的。审计验证「数有出处」，不是「数答对」。

这跟求解器的区别（守住边界，别把它做成脚本求解器）：
  * 它**不产生任何答案**：不调 LLM、不调 agent、不在核验路径上新写几何算法。
    求解器回答「答案是什么」，核验器只回答「你交的数落不落在独立期望值里」。
  * 期望值**全部复用 bench.py 已有的独立判据**（loose_wants / check_tXX），
    不在这里新写一条。bench 的期望值来源有三层、都可辩护：
      闭式（题面给的 2.5/18.75）· 独立几何实现（SVD vs tools.angle，两套实现算同一量）·
      内部自洽（题面相对位移）。准确性继承自 bench，已被重复实验验证
      （T01_3 连跑四次 agent 错过一次 25°，独立判据四次都稳定给 65°）。
  * **判据强度的代价如实写明**：宽松判据只看「期望值有没有出现在答案里」（任意键），
    理论上可能巧合出现 —— 这是 bench 为「现场 run 键名不固定」付出的代价，这里照单全收。
  * **核不了的题绝不硬判**：一个 run 若归类不到任何带独立期望值的赛题，
    只报「有答案 · 无法独立核验」，不猜、不现场解题。这条红线保证它不退化成求解器。

映射靠**意图归类**（看 brief.raw 的措辞选判据），不是看它交了什么数 —— 先按意图选
判据、再用判据评答案；顺序反了就成了抄答案。

用法：
  python accuracy_check.py                 # 扫全部 runs/live_*，逐 run 报 + 汇总
  python accuracy_check.py --tag live_1126 # 只看 tag 前缀
  python accuracy_check.py --all           # 连 runs/ 下所有（含 rp_/bench_）一起看
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bench  # noqa: E402   复用它的独立判据，不重写


# ---------------------------------------------------------------- 意图归类
# 关键词 → bench 里带独立期望值的赛题。只用于「选哪条判据」，不用于给答案。
# 一道现场题可能同时问几件事（三距离 + 夹角），命中多条就逐条核。
INTENT_KEYWORDS = {
    "T01_1": ["门洞为参照", "左还是右", "前还是后", "相对门洞", "位置关系"],
    "T01_3": ["门扇与门所在墙", "门扇与墙的夹角", "门扇与墙面", "门扇与门所在墙面"],
    "T02": ["五个位置", "球与正方体", "完全在内", "外切", "内切", "完全在外"],
    "T03": ["净宽", "净深", "地板面积", "面积和体积"],
    "T04": ["占地是否落在", "占地边缘", "家具占地", "是否穿墙"],
    "T05": ["三种摆放", "椅子三", "平移修正"],
    "T06": ["加宽", "开洞", "改圆窗", "5.4"],
    "T07": ["抬高", "切顶", "切平", "顶面"],
    "T08": ["开门", "首次接触", "碰椅子"],
    "T10": ["对接", "插脚", "凹槽", "旋转角"],
}


def classify(raw):
    """按题目措辞归类到 bench 里带独立判据的赛题（可能命中多条）。"""
    raw = raw or ""
    return [cid for cid, kws in INTENT_KEYWORDS.items() if any(k in raw for k in kws)]


def has_answer(r):
    """是否「交出了答案」：answer.values 非空，或明确 insufficient（那是诚实交代，也算交代）。

    注意区分：unknowns 不算答案（它是「没答上来的项」）；insufficient 算一种交代。
    """
    a = r.get("answer") or {}
    vals = a.get("values") or {}
    insuf = a.get("insufficient")
    return bool(vals), bool(insuf), vals


def grade_run(r):
    """返回 (有答案, 交代类型, 核验结果, 明细)。

    核验结果三档：
      "答对"        —— 命中的判据全部通过（宽松口径：期望值都在答案里出现）
      "部分答对"    —— 命中了多条判据，部分通过
      "答错"        —— 有命中判据但全部不通过
      "无法独立核验" —— 交了答案但归类不到任何带独立期望值的赛题
      "无答案"      —— 没交 values 也没 insufficient
    """
    has, insuf, vals = has_answer(r)
    if not has and not insuf:
        return False, "无答案", "无答案", []

    cids = classify((r.get("brief") or {}).get("raw"))
    if not cids:
        note = "无法独立核验" if has else ("诚实交代 insufficient" if insuf else "无答案")
        return has, ("交代" if insuf else "有答案"), note, []

    # 对命中的每条判据核两遍：先严格（键名匹配，高置信），不中再宽松（值出现，低置信）。
    # 现场题键名不固定，严格常全判缺失 —— 那是「它没按我们格式写」，不是「它算错了」。
    # 但宽松只扫全 blob 数字，无法分语义，理论上可能巧合命中 —— 必须如实降一档置信度。
    details = []
    strict_pass = strict_fail = loose_pass = loose_fail = 0
    for cid in cids:
        strict_fn = getattr(bench, "check_" + cid.lower(), None)
        if strict_fn is not None:
            bad = strict_fn(vals)
            if not bad:
                strict_pass += 1
                details.append("%s ✓严格（键名匹配，高置信）" % cid)
                continue
            # 严格判了缺失（键名不对）→ 降宽松；严格判了值不符（键在值错）→ 直接算错
            # 注意：`xxx=None≠yyy` 形式的 bad 项，`None` 表示键压根不在（agent 用了别的键名）
            # —— 不是「它写了但算错了」。所以 None≠ 必须归为缺失，否则会制造假阳性
            # （live_1126_143151 本来 T01_1 的宽松没中是因为换了表述，却因 =None≠ 被算成值错）。
            miss = [b for b in bad if "缺失" in b or "=None≠" in b]
            wrong = [b for b in bad if "缺失" not in b and "=None≠" not in b]
            if wrong:
                loose_fail += 1
                details.append("%s ✗严格（键在但值错）%s" % (cid, "; ".join(wrong)[:100]))
                continue
        wants = bench.loose_wants(cid)
        if not wants:
            continue
        bad = bench._loose(vals, wants)
        if bad:
            # 宽松「未中」不硬判答错。诚实的边界在这里：
            #   期望值没出现在答案里，两种可能分不清 ——
            #     a) agent 真答错了（live_1126_143151 门扇夹角交 25°，真值 65°）
            #     b) agent 换了表述（T02 例：交体积 4.185/15.625，没交期望的盒边 2.5）
            # 要区分 a/b，得知道「哪个键/哪个实体对应哪个物理量」——那就得自己做实体识别，
            # 等于替 agent 解题，越界成求解器。也试过「同域冲突」检测（看 blob 里有没有
            # 同域的另一组值），但数值范围框不住语义：0.45/0.67/4.0 这些长度会被误当角度，
            # 噪声盖过信号。所以宁可标「判据未覆盖 + 提示人工」，也不硬判。这是本工具为
            # 守住「不当求解器」这条边界付出的能力上限，写进文档。
            loose_fail += 1
            details.append("%s 判据未覆盖（%s）" % (cid, "; ".join(bad)[:110]))
        else:
            loose_pass += 1
            details.append("%s ✓宽松（仅值出现，可能巧合，低置信）" % cid)

    if strict_pass == 0 and loose_pass == 0 and loose_fail == 0:
        return has, "有答案", "无法独立核验（命中的赛题没有可核的期望值）", details
    # 判定分层：宽松「未中」不等于答错 —— 可能是 agent 换了表述（没直接交那个数）
    # 但用了等价的量（如 T02 交了体积 4.185/15.625 而非盒边 2.5）。那种不能硬判错。
    if strict_pass > 0 and loose_fail == 0:
        return has, "有答案", "答对（严格口径，高置信）", details
    if loose_fail == 0 and loose_pass > 0:
        return has, "有答案", "疑似答对（宽松口径，需人工确认）", details
    if strict_pass == 0 and loose_pass == 0 and loose_fail == 0:
        # 命中了赛题、但所有判据都没中期望值。这里有个必须如实交代的边界：
        # 「未中」有两种原因，本工具**分不清** ——
        #   a) agent 换了表述（T02 例：交了体积 4.185/15.625，没交期望的盒边 2.5）→ 其实答对了
        #   b) agent 真答错了（live_1126_143151 例：门扇夹角交 25°，真值 65°）→ 答错
        # 要区分 a/b，得知道「哪个键对应哪个物理量」——那就得自己做实体识别，
        # 等于替 agent 解题，越界成求解器。所以宁可标「未覆盖 + 提示人工」，
        # 也不硬判对错。这是本工具为守住边界付出的能力上限，写进文档。
        return has, "有答案", "判据未覆盖（期望值未出现，可能换表述也可能答错，需人工）", details
    return has, "有答案", "部分答对", details


def main():
    ap = argparse.ArgumentParser(description="区分「有答案」和「答对」，回看历史 run")
    ap.add_argument("--tag", default="live_", help="只看 runs/<tag>* 前缀")
    ap.add_argument("--all", action="store_true", help="扫 runs/ 下全部（含 rp_/bench_）")
    ap.add_argument("--verbose", action="store_true", help="打印每条判据明细")
    a = ap.parse_args()

    if a.all:
        pattern = os.path.join(HERE, "runs", "*")
    else:
        pattern = os.path.join(HERE, "runs", a.tag + "*")

    runs = sorted(glob.glob(pattern), key=os.path.getmtime)
    rows = []
    for d in runs:
        rp = os.path.join(d, "result.json")
        if not os.path.exists(rp):
            continue
        tag = os.path.basename(d)
        if a.tag and not a.all and not tag.startswith(a.tag):
            continue
        try:
            r = json.load(open(rp, encoding="utf-8"))
        except Exception as e:
            rows.append((tag, "?", "解析失败", "-", "-", str(e)[:60]))
            continue
        # 把 brief 挂上供归类
        bp = os.path.join(d, "brief.json")
        r["brief"] = json.load(open(bp, encoding="utf-8")) if os.path.exists(bp) else {}
        has, kind, verdict, details = grade_run(r)
        status = r.get("status", "?")
        rows.append((tag, status, verdict, kind,
                     json.dumps((r.get("answer") or {}).get("values") or {}, ensure_ascii=False).count('":'),
                     details))

    # ------------------------------------------------------------- 逐 run 报
    print("run                          状态            判定                    有答案  期望项")
    print("-" * 110)
    for tag, status, verdict, kind, nkeys, details in rows:
        mark = {"答对（严格口径，高置信）": "✓",
                "疑似答对（宽松口径，需人工确认）": "◐",
                "部分答对": "◑",
                "判据未覆盖（期望值未出现，可能换表述，需人工）": "?",
                "答错": "✗"}.get(verdict, "·")
        short = verdict.split("（")[0] if "（" in verdict else verdict
        print("%-28s %-13s %s %-22s %-6s %s" % (
            tag, status, mark, short, "是" if kind in ("有答案", "交代") else "否", nkeys))

    # ------------------------------------------------------------- 汇总：两档分开
    n_total = len(rows)
    n_answer = sum(1 for x in rows if x[3] in ("有答案", "交代"))
    n_noanswer = sum(1 for x in rows if x[2] == "无答案")
    n_right = sum(1 for x in rows if x[2] == "答对（严格口径，高置信）")
    n_suspect = sum(1 for x in rows if x[2] == "疑似答对（宽松口径，需人工确认）")
    n_partial = sum(1 for x in rows if x[2] == "部分答对")
    n_uncov = sum(1 for x in rows if x[2] == "判据未覆盖（期望值未出现，可能换表述，需人工）")
    n_wrong = sum(1 for x in rows if x[2] == "答错")
    n_unverif = sum(1 for x in rows if "无法独立核验" in x[2])
    n_verifiable = n_right + n_suspect + n_partial + n_uncov + n_wrong

    print()
    print("== 汇总（%d 个 run）==" % n_total)
    print("  有答案（交了数或诚实交代）：      %d" % n_answer)
    print("  无答案（没交数也没交代）：        %d" % n_noanswer)
    print("  ── 上面只是「交没交」，下面才是「对不对」──")
    print("  命中带期望值赛题（可进核验）：    %d" % n_verifiable)
    print("     ✓ 答对（严格口径，高置信）：   %d" % n_right)
    print("     ◐ 疑似答对（宽松口径，可能巧合）：%d" % n_suspect)
    print("     ◑ 部分答对：                   %d" % n_partial)
    print("     ? 判据未覆盖（可能换表述，需人工）：%d" % n_uncov)
    print("     ✗ 答错（键在但值错）：         %d" % n_wrong)
    print("  交了答案但无法独立核验：          %d" % n_unverif)
    if n_verifiable:
        print()
        print("  高置信答对率（严格口径）：%d/%d" % (n_right, n_verifiable))

    if a.verbose:
        print()
        print("== 判据明细 ==")
        for tag, status, verdict, kind, nkeys, details in rows:
            if details:
                print("  %s [%s]" % (tag, verdict))
                for d in details:
                    print("     - %s" % d)


if __name__ == "__main__":
    main()