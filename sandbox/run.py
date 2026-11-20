# -*- coding: utf-8 -*-
"""run.py — 沙盒入口。三件事：跑 plan、留痕、出可复核的产物。

用法：
  python run.py --card tasks/T01_2.md --model <model> --plan plans/T01_2.json --tag ifc

产出（runs/<tag>/）：
  events.jsonl   哈希链账本：每次工具调用一行，改一个字节即断链
  result.json    最终答案 + 每步摘要 + 环境指纹
  report.md      人读报告（含见证点残差复核）
  frames/*.png   帧；video.mp4 由帧编码（同一份账本驱动，不是录屏）
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from core.runtime import Bus, Ledger, TOOLS, save_json, schema  # noqa: E402
import tools.ops  # noqa: F401,E402  （注册工具）
from tools import geom  # noqa: E402
from viz import render  # noqa: E402

PALETTE = {
    "chair": (243, 156, 62),
    "door_leaf": (66, 118, 214),
    "window_frame": (32, 168, 138),
    "picture_frame": (188, 84, 160),
}


def sha(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def subst(obj, params):
    if isinstance(obj, str) and obj.startswith("$"):
        key = obj.strip("${}")
        if key in params:
            return params[key]
    if isinstance(obj, dict):
        return {k: subst(v, params) for k, v in obj.items()}
    if isinstance(obj, list):
        return [subst(v, params) for v in obj]
    return obj


class StopRun(Exception):
    """证据不足时停止：宁可报未知，也不要猜。"""

    def __init__(self, step, want, got, payload):
        self.step, self.want, self.got, self.payload = step, want, got, payload
        super().__init__("步骤 %s 期望 %s 个候选，实得 %s" % (step, want, got))


def execute(plan, ledger, bus, model):
    results, steps_meta = {}, []
    params = dict(plan.get("params", {}))
    params["MODEL"] = model
    for st in plan["steps"]:
        sid, tname = st["id"], st["tool"]
        raw = subst(st.get("args", {}), params)
        args = bus.resolve(raw, results)
        out = bus.call(tname, args, sid)
        results[sid] = out
        n = out.get("n") if isinstance(out, dict) else None
        steps_meta.append({"id": sid, "tool": tname, "why": st.get("why", ""), "n": n,
                           "expect": st.get("expect_n")})
        if st.get("expect_n") is not None and n != st["expect_n"]:
            ledger.append("stop_insufficient", {"step": sid, "tool": tname,
                                                "expect_n": st["expect_n"], "got_n": n,
                                                "payload": st.get("why", "")})
            raise StopRun(sid, st["expect_n"], n, out)
    return results, steps_meta


def witness_residual(bus, ranking, chair_sid):
    """见证点复核：见证点必须落在各自表面上，且两点距离等于报告值。"""
    by = {s.sid: s for s in bus.scene}
    rows = []
    for r in ranking:
        A = by[chair_sid]
        B = by[r["sid"]]
        wa = np.array(r["witness_a"], float)
        wb = np.array(r["witness_b"], float)
        TA = np.stack(geom.tri_arrays(A), axis=1)
        TB = np.stack(geom.tri_arrays(B), axis=1)
        _, da = geom._closest_on_tri_batch(wa[None, :], TA)
        _, db = geom._closest_on_tri_batch(wb[None, :], TB)
        rows.append({"sid": r["sid"], "value": r["value"],
                     "residual_on_chair": float(da.min()),
                     "residual_on_target": float(db.min()),
                     "witness_gap": float(np.linalg.norm(wa - wb)),
                     "gap_minus_value": float(abs(np.linalg.norm(wa - wb) - r["value"]))})
    return rows


def make_frames(bus, ans, outdir, orbits=4, scope=None):
    by = {s.sid: s for s in bus.scene}
    draw = [s for s in bus.scene if scope is None or s.sid in scope]
    chair = ans["values"]["chair"]
    tg = ans["values"]["targets"]
    rank = ans["values"]["distance_ranking"]
    color = {chair: PALETTE["chair"]}
    for role, sid in tg.items():
        color[sid] = PALETTE[role]
    focus_ids = [chair] + list(tg.values())
    focus_pts = np.vstack([by[s].P for s in focus_ids])

    dist_of = {r["sid"]: r["value"] for r in rank}
    hud = ["%s  %.3f m" % (("椅子→" + role), dist_of.get(sid, float("nan")))
           for role, sid in tg.items()]
    hud.sort(key=lambda s: float(s.split()[-2]))

    scenes = []
    scenes.append(dict(title="AI 几何沙盒 · Task 01.2",
                       caption="① 读入模型：%d 个实体，无名称、无类别、无属性" % len(draw),
                       highlight={}, zoom=None, lines=[], points=[]))
    scenes.append(dict(title="② 仅凭几何描述量完成角色识别",
                       caption="橙=椅子  蓝=门扇  绿=窗框  紫=画框",
                       highlight=color, zoom=None, lines=[], points=[]))
    for i, role in enumerate(["door_leaf", "window_frame", "picture_frame"], 1):
        r = next(x for x in rank if x["sid"] == tg[role])
        pair = np.vstack([by[chair].P, by[tg[role]].P])
        scenes.append(dict(
            title="③-%d 椅子 → %s" % (i, {"door_leaf": "门扇", "window_frame": "窗框", "picture_frame": "画框"}[role]),
            caption="最短表面距离 %.3f m   见证点 (%.2f, %.2f, %.2f) — (%.2f, %.2f, %.2f)" % (
                r["value"], *r["witness_a"], *r["witness_b"]),
            highlight={chair: PALETTE["chair"], tg[role]: color[tg[role]]},
            zoom=tuple((np.array(r["witness_a"]) + np.array(r["witness_b"])) / 2),
            fit=pair,
            lines=[(r["witness_a"], r["witness_b"])],
            points=[{"p": r["witness_a"], "label": "椅子"},
                    {"p": r["witness_b"], "label": {"door_leaf": "门扇", "window_frame": "窗框", "picture_frame": "画框"}[role]}],
            subset=[chair, tg[role]]))
    scenes.append(dict(title="④ 排序与复核",
                       caption="按由近到远：" + " < ".join("%s %.3f" % (s.split("→")[1].split()[0], float(s.split()[-2])) for s in hud),
                       highlight=color, zoom=None, lines=[], points=[], hud=hud))

    paths, manifest = [], []
    for si, sc in enumerate(scenes):
        for k in range(orbits):
            yaw = 34 + 360.0 * k / orbits
            p = os.path.join(outdir, "frames", "seq_%03d.png" % len(paths))
            ds = draw if not sc.get("subset") else [by[x] for x in sc["subset"]]
            render.frame(ds, p, yaw=yaw, pitch=42 if sc.get("subset") else (26 if si != 1 else 30),
                         highlight=sc["highlight"], zoom=sc.get("zoom"), fit=sc.get("fit"),
                         lines=sc["lines"], points=sc["points"],
                         title=sc["title"], caption=sc["caption"], hud=sc.get("hud"))
            paths.append(p)
            manifest.append({"i": len(paths) - 1, "scene": si, "yaw": yaw,
                             "title": sc["title"], "caption": sc["caption"]})
    save_json(os.path.join(outdir, "frames", "manifest.json"), manifest)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--orbits", type=int, default=3)
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    for flag, what in (("model", "模型"), ("card", "任务卡")):
        p = getattr(a, flag)
        if not os.path.exists(p):
            raise SystemExit(
                "找不到%s文件：%s\n赛方的模型包与任务卡不随仓库分发。请下载 "
                "https://hack26.ctrlpath.co/api/tasks/download 后解压到 pkg/，\n"
                "使路径形如 pkg/AI-Geometry-Models/IFC/AI_Geometry_Input.ifc（见 README「一分钟上手」）。"
                % (what, p))

    tag = a.tag or os.path.splitext(os.path.basename(a.model))[0].lower()
    outdir = os.path.join(HERE, "runs", tag)
    os.makedirs(outdir, exist_ok=True)

    plan = json.load(open(a.plan, encoding="utf-8"))
    card_text = open(a.card, encoding="utf-8").read()
    ledger = Ledger(os.path.join(outdir, "events.jsonl"))

    env = {
        "card": os.path.relpath(a.card, HERE), "card_sha256": hashlib.sha256(card_text.encode()).hexdigest()[:16],
        "model": os.path.relpath(a.model, HERE), "model_sha256": sha(a.model),
        "plan": os.path.relpath(a.plan, HERE), "plan_sha256": sha(a.plan),
        "tools": sorted(TOOLS), "n_tools": len(TOOLS),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "cwd": HERE, "t0": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    ledger.append("run_start", env)

    bus = Bus(ledger)
    t0 = time.time()
    stop = None
    try:
        results, steps = execute(plan, ledger, bus, a.model)
    except StopRun as e:
        stop = e
        results, steps = {}, []
        ledger.append("run_stopped", {"step": e.step, "expect_n": e.want, "got_n": e.got})
    except Exception as e:
        ledger.append("run_failed", {"error": str(e)})
        print("RUN FAILED:", e)
        raise
    dt = time.time() - t0

    if stop is not None:
        r = {"env": env, "status": "INSUFFICIENT_EVIDENCE",
             "stopped_at": {"step": stop.step, "expect_n": stop.want, "got_n": stop.got,
                            "returned": stop.payload},
             "elapsed_s": round(dt, 2),
             "verify": {"ledger": Ledger.verify(os.path.join(outdir, "events.jsonl"))}}
        save_json(os.path.join(outdir, "result.json"), r)
        print("\n=== %s === 停止：%s（期望 %s，实得 %s）" % (tag, stop.step, stop.want, stop.got))
        L = ["# 运行报告（未完成）\n", "状态：**证据不足，按约定停止，不猜答案**\n",
             "| 项 | 值 |\n|---|---|",
             "| 卡 | `%s` |" % env["card"], "| 模型 | `%s` |" % env["model"],
             "| 停止步骤 | %s |" % stop.step,
             "| 期望候选数 | %s |" % stop.want, "| 实得候选数 | %s |" % stop.got,
             "\n```json\n%s\n```" % json.dumps(stop.payload, ensure_ascii=False, indent=2)[:2500],
             "\n## 为什么停\n", "该步的判据依赖「模型整体与坐标轴对齐」。变体把整个场景绕竖轴转了 45°，",
             "所有实体的包围盒都变成斜置的，于是「薄/长/方」这些基于轴对齐包围盒的描述量失效。",
             "这是**已知限制**，不是随机失败：45° 时至少 3 个角色判据同时命中错误集合。",
             "\n修复方向：把判据从「与坐标轴对齐」换成「与场景主方向对齐」（用墙面法向做参照系），",
             "或先做一次主轴估计再做包围盒描述。"]
        open(os.path.join(outdir, "report.md"), "w", encoding="utf-8").write("\n".join(L))
        return

    ans = results[plan["steps"][-1]["id"]]
    verify = {"ledger": Ledger.verify(os.path.join(outdir, "events.jsonl")),
              "witness": witness_residual(bus, ans["values"]["distance_ranking"], ans["values"]["chair"]),
              "unit": results["s1"]["meta"].get("scale_source"),
              "ambiguous_steps": [s["id"] for s in steps
                                  if s.get("expect") and s["n"] != s["expect"]]}
    verify["witness_ok"] = all(w["residual_on_chair"] < 1e-5 and w["residual_on_target"] < 1e-5
                               and w["gap_minus_value"] < 1e-6 for w in verify["witness"])

    result = {"env": env, "answer": ans, "steps": steps, "verify": verify,
              "elapsed_s": round(dt, 2), "n_solids": len(bus.scene)}
    save_json(os.path.join(outdir, "result.json"), result)
    ledger.append("run_end", {"answer_digest": sha(os.path.join(outdir, "result.json")),
                              "ok": verify["witness_ok"] and verify["ledger"]["ok"]})

    scope = None
    if plan.get("scope"):
        scope = {it["sid"] for it in bus.resolve(plan["scope"], results)}
    frames = make_frames(bus, ans, outdir, a.orbits, scope)
    vid = None
    if not a.no_video and frames:
        vid = render.encode(os.path.join(outdir, "frames", "seq_%03d.png"),
                            os.path.join(outdir, "video.mp4"), fps=2.0)

    write_report(outdir, result, frames, vid, a)

    print("\n=== %s ===" % tag)
    for r in ans["values"]["distance_ranking"]:
        print("  %-8s %8.3f m   witness %s ↔ %s" % (r["sid"], r["value"], r["witness_a"], r["witness_b"]))
    print("  账本链完好: %s   见证点复核: %s   耗时 %.2fs" % (verify["ledger"]["ok"], verify["witness_ok"], dt))
    print("  产物:", outdir, "| frames:", len(frames), "| video:", bool(vid and vid["ok"]))


def write_report(outdir, res, frames, vid, a):
    ans, v = res["answer"], res["verify"]
    L = []
    L.append("# 运行报告 · %s\n" % res["env"]["card"])
    L.append("| 项 | 值 |\n|---|---|")
    for k in ("card_sha256", "model", "model_sha256", "plan_sha256", "n_tools", "t0"):
        L.append("| %s | `%s` |" % (k, res["env"][k]))
    L.append("| 耗时 | %.2f s |" % res["elapsed_s"])
    L.append("| 实体数 | %d |\n" % res["n_solids"])

    L.append("## 结论\n")
    L.append("| 目标 | 距离 (m) | 见证点（椅子侧） | 见证点（目标侧） |")
    L.append("|---|---|---|---|")
    for r in ans["values"]["distance_ranking"]:
        L.append("| %s | **%.3f** | %s | %s |" % (r["sid"], r["value"], r["witness_a"], r["witness_b"]))
    L.append("\n角色指派：" + json.dumps(ans["values"]["targets"], ensure_ascii=False) +
             "；椅子 " + ans["values"]["chair"])
    L.append("\n> " + ans.get("notes", "") + "\n")

    L.append("## 复核\n")
    L.append("| 目标 | 报告值 | 见证点到椅子表面残差 | 到目标表面残差 | |wa−wb|−值 |")
    L.append("|---|---|---|---|---|")
    for w in v["witness"]:
        L.append("| %s | %.6f | %.2e | %.2e | %.2e |" % (
            w["sid"], w["value"], w["residual_on_chair"], w["residual_on_target"], w["gap_minus_value"]))
    L.append("\n- 账本哈希链：%s（%d 条记录）" % ("完好" if v["ledger"]["ok"] else "断裂 %s" % v["ledger"]["broken"], v["ledger"]["n"]))
    L.append("- 单位裁决：%s" % v["unit"])
    L.append("- 选择不唯一的步骤：%s" % (v["ambiguous_steps"] or "无"))

    L.append("\n## 执行轨迹\n")
    L.append("| 步 | 工具 | 返回条目 | 意图 |\n|---|---|---|---|")
    for s in res["steps"]:
        L.append("| %s | `%s` | %s | %s |" % (s["id"], s["tool"], s["n"], s["why"]))

    if ans.get("unknowns"):
        L.append("\n## 未确定项\n")
        for u in ans["unknowns"]:
            L.append("- " + u)
    if vid:
        L.append("\n## 录像\n\n`frames/` %d 帧 → `video.mp4`（%s）" %
                 (len(frames), "编码成功" if vid["ok"] else "编码失败：" + vid["err"]))
    open(os.path.join(outdir, "report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
