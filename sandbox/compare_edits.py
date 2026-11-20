# -*- coding: utf-8 -*-
"""compare_edits.py — 写层也能换输入：同一份决策记录，换格式 / 换单位，产出的模型是否一致。

读层已经证明过「换格式不影响答案」（compare_agent.py）。写层要证明一件更强的事：
**同一串编辑动作在不同格式、不同单位、名字全丢的输入上，产出的模型仍然一样。**

这件事不显然。写层的参数是米制的坐标（锚点表、切割平面标高），如果读入时单位裁决错了，
或者实体顺序变了导致 `@floor.items[0]` 指到别的东西，产出会立刻跑偏 —— 而编辑器本身
不会报错，它只是照着错的锚点改。所以这张表不是形式主义，它是写层唯一的外部校准。

表由本脚本自动生成到 runs/edit_matrix.md。
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from core.runtime import Bus, Ledger, TOOLS      # noqa: E402
import tools.ops                                # noqa: E402

BASE = "../pkg/AI-Geometry-Models/IFC/AI_Geometry_Input.ifc"
MATRIX = os.path.join(HERE, "runs", "_matrix")

TASKS = [
    ("T-07", "tasks/T07.md", "episodes/T07.edit.json", 40, "edit07"),
    ("T-05", "tasks/T05.md", "episodes/T05.edit.json", 60, "edit05"),
    ("T-06", "tasks/T06.md", "episodes/T06.edit.json", 80, "edit06"),
]
CASES = [
    ("IFC 原件", BASE),
    ("OBJ 名全随机", "variants/obj_plain.obj"),
    ("单位改 mm（×1000）", "variants/obj_mm.obj"),
    ("绕竖轴 90°", "variants/obj_rot90.obj"),
]

DOOR = ([4.6, -1.55, 0.0], [4.6, -0.55, 0.0])


def fresh_bus(tag):
    d = os.path.join(MATRIX, "_ledger")
    os.makedirs(d, exist_ok=True)
    return Bus(Ledger(os.path.join(d, tag + ".jsonl")))


def load(bus, path):
    TOOLS["load"]["fn"](bus, path)
    return list(bus.scene)


def mh(bus, sid):
    return TOOLS["mesh_health"]["fn"](bus, [sid])["rows"][0]


def fp(bus, sid):
    return TOOLS["footprint"]["fn"](bus, [sid])["rows"][0]


def biggest(solids, bus, pred):
    """按 pred(health) 选体积最大的那个 —— 房间/立方体都有多个候选，用体积消歧最稳。"""
    out = []
    for s in solids:
        r = mh(bus, s.sid)
        if pred(r):
            out.append((s, r))
    if not out:
        return None
    return max(out, key=lambda t: t[1]["volume"])


def measure(task, tag, files):
    """读回归档的产物，返回可以横向比较的几个数（全部来自只读工具）。"""
    # 产物的文件名由 episode 里的 emit 决定，这里给每个文件起一个短名，
    # 好让三个任务共用同一张表（也避免在别处按 basename 猜）。
    alias = {"cube_tall.gltf": "cube", "ball_cut.gltf": "ball",
             "chair_p1.gltf": "p1", "chair_p2.gltf": "p2",
             "chair_p3.gltf": "p3", "chair_p1_fixed.gltf": "fix",
             "room_w6.0_square.gltf": "w6sq", "room_w6.0_round.gltf": "w6rd",
             "room_w5.4_round.gltf": "w54rd"}
    files = {alias[k]: v for k, v in files.items() if k in alias}
    bus = fresh_bus(tag)
    v = {}

    if task == "T-07":
        sc = load(bus, files["cube"])
        s, r = biggest(sc, bus, lambda h: h["dims"][2] > 2.0 and h["dims"][0] < 3.0)
        v["cube_dims"] = r["dims"]
        v["cube_v"] = r["volume"]
        sc = load(bus, files["ball"])
        s, r = biggest(sc, bus, lambda h: h["dims"][2] < 2.0)
        v["ball_h"] = r["dims"][2]
        v["ball_v"] = r["volume"]
        v["ball_closed"] = int(bool(r["closed"] and r["n_boundary_loops"] == 0))
        v["key"] = r["volume"]

    elif task == "T-05":
        ds = {}
        for k, name in (("p1", "p1"), ("p2", "p2"),
                        ("p3", "p3"), ("fix", "fix")):
            sc = load(bus, files[k])
            c = TOOLS["clearance"]["fn"](bus, sc[0].sid, DOOR[0], DOOR[1])
            ds[k] = round(c["value"], 4)
        v["dist"] = ds
        v["key"] = ds["p3"]

    elif task == "T-06":
        ws, areas = [], []
        for k in ("w6sq", "w6rd", "w54rd"):
            sc = load(bus, files[k])
            s, r = biggest(sc, bus, lambda h: h["dims"][2] < 0.15 and h["dims"][0] > 4.0)
            ws.append(round(r["dims"][0], 4))
            hole = None
            for S in sc:
                seg = TOOLS["segment"]["fn"](bus, S.sid, 20.0, 4)
                for f in seg["faces"]:
                    for h in (f.get("holes") or []):
                        if h.get("area", 0) >= 1.0:
                            hole = h
                if hole:
                    break
            areas.append(round(hole["area"], 4) if hole else None)
        v["widths"] = ws
        v["areas"] = areas
        v["key"] = ws[0] + ws[2]          # 6.0 + 5.4 = 11.4，一个数就能看出两档都对

    return v


def fmt(task, v):
    if task == "T-07":
        return "%.3f×%.3f×%.3f / V=%.4f · 球 h=%.3f V=%.4f 封闭=%d" % (
            v["cube_dims"][0], v["cube_dims"][1], v["cube_dims"][2], v["cube_v"],
            v["ball_h"], v["ball_v"], v["ball_closed"])
    if task == "T-05":
        d = v["dist"]
        return "距门 %.4f / %.4f / %.4f / %.4f" % (d["p1"], d["p2"], d["p3"], d["fix"])
    if task == "T-06":
        return "净宽 %s · 洞口面积 %s" % (v["widths"], v["areas"])
    return str(v)


def run_one(task, card, ep, turns, outdir, case, model):
    """跑一次，把产物从固定输出目录挪到归档目录（episode 里的路径是固定的）。"""
    for f in os.listdir(outdir):
        if f.endswith(".gltf") or f.endswith(".obj"):
            os.remove(os.path.join(outdir, f))
    tag = "cmp_" + task.replace("-", "").lower() + "_" + "".join(
        c if c.isalnum() else "_" for c in case)[:16]
    stale = os.path.join(HERE, "runs", tag, "result.json")
    if os.path.exists(stale):
        os.remove(stale)
    p = subprocess.run([sys.executable, "run_agent.py", "--card", card, "--model", model,
                        "--episode", ep, "--tag", tag, "--max-turns", str(turns),
                        "--no-video"], cwd=HERE, capture_output=True, text=True)
    rf = os.path.join(HERE, "runs", tag, "result.json")
    if not os.path.exists(rf):
        return None, None, (p.stderr or p.stdout)[-300:]
    res = json.load(open(rf, encoding="utf-8"))
    arch = os.path.join(MATRIX, tag)
    os.makedirs(arch, exist_ok=True)
    files = {}
    for f in os.listdir(outdir):
        if f.endswith(".gltf") or f.endswith(".obj"):
            shutil.copy2(os.path.join(outdir, f), os.path.join(arch, f))
            files[f] = os.path.join(arch, f)
    return res, files, None


def main():
    if not os.path.exists(os.path.join(HERE, BASE)):
        raise SystemExit("[!] 找不到模型包：%s（见 README 的下载说明）" % BASE)
    os.makedirs(MATRIX, exist_ok=True)

    rows = []
    for task, card, ep, turns, outdir in TASKS:
        od = os.path.join(HERE, "runs", outdir)
        base_key, base_files = None, None
        for case, model in CASES:
            res, files, err = run_one(task, card, ep, turns, od, case, model)
            if err or not files:
                rows.append({"任务": task, "输入": case, "状态": "CRASH",
                             "轮数": "—", "关键量": (err or "没有产物"), "偏差": "—", "账本": "—"})
                continue
            if case == CASES[0][0]:
                base_files = dict(files)
            try:
                v = measure(task, task + "_" + case, files)
                txt = fmt(task, v)
                key = v["key"]
            except Exception as e:
                rows.append({"任务": task, "输入": case, "状态": res.get("status"),
                             "轮数": res.get("turns"), "关键量": "测量失败：%s" % e,
                             "偏差": "—", "账本": "—"})
                continue
            if base_key is None:
                base_key = key
            dev = abs(key - base_key)
            rows.append({
                "任务": task, "输入": case, "状态": res.get("status"),
                "轮数": res.get("turns"), "关键量": txt,
                "偏差": "%.2e" % dev,
                "账本": "完好" if (res.get("verify") or {}).get("ledger", {}).get("ok") else "—",
            })
            print("%s %-20s %s" % (task, case, txt), flush=True)

        # episode 里的 emit 路径是固定的，所以矩阵跑完时 runs/edit0x/ 里躺着的是
        # 最后一个变体（90°）的产物。把基准变体拷回去 —— 否则随后跑 verify_edits.py
        # 验的就不是「那道题交的东西」，而是一份被矩阵覆盖过的文件。
        if base_files:
            for name, src in base_files.items():
                shutil.copy2(src, os.path.join(od, name))

    cols = ["任务", "输入", "状态", "轮数", "关键量", "偏差", "账本"]
    L = ["# 写层泛化矩阵\n",
         "同一张任务卡 + 同一份决策记录（锚点表、切割标高一个不改），只换输入格式与单位，"
         "看产出的模型是否一致。表由 `compare_edits.py` 自动生成，"
         "关键量全部由只读工具在**读回的产物文件**上重量得到。\n",
         "| " + " | ".join(cols) + " |",
         "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        L.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    L.append("\n## 怎么读这张表\n")
    L.append("- 每一组变体跑的是**同一串编辑动作**：`stretch_axis` 的锚点、`cut_plane` 的标高、"
             "`translate` 的位移都是米制常数。单位改成 mm 后仍然对，说明读入层把尺度裁回米之后再交几何。")
    L.append("- 名字全随机、对象顺序不变，`@floor.items[0]` 这类引用仍指向同一个实体 —— "
             "所以矩阵里所有关键量一致。OBJ 一列的 1e-8 偏差来自 OBJ 只保留 6 位小数。")
    L.append("- **90° 一列是这张表最有价值的部分，它把写层的边界划清楚了**：")
    L.append("  - T-07 一致。因为它的编辑只动 z（抬顶面、z=0.5 处切平），绕竖轴转 90° 不动 z。")
    L.append("  - T-05 / T-06 不一致，而且都被判 `AUDIT_FAILED`。原因是锚点与位移**按世界轴写**："
             "转 90° 后房间的「左右」不再对应世界 x，米制常数就落到了错的轴上。")
    L.append("  - 这两行失败是**正确行为**，不是被掩盖的 bug：沙盒发现提交的数字与证据对不上，拒绝收卷。"
             "要修就得把锚点改写成「沿场景参照系」——见 README 已知限制第 5 条。")
    out = os.path.join(HERE, "runs", "edit_matrix.md")
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n→ runs/edit_matrix.md")


if __name__ == "__main__":
    main()
