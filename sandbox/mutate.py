# -*- coding: utf-8 -*-
"""mutate.py — 造「同一问题、不同模型」的变体，用来证明方法不是对着原文件写死的。

变体只改：格式(→OBJ)、对象名(随机化)、整体刚体变换、单位尺度。
不改：相对位置、形状、比例。所以正确答案必须「跟着变换走」而不是「保持不变」。
"""
import argparse
import os
import random
import string
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tools import world  # noqa: E402


def rot_z(deg, pivot):
    a = np.radians(deg)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
    t = pivot - R @ pivot
    return R, t


def write_obj(solids, path, name_len=6, seed=0):
    rng = random.Random(seed)
    lines, base = [], 1
    for s in solids:
        nm = "".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(name_len))
        lines.append("o %s" % nm)
        for p in s.P:
            lines.append("v %.6f %.6f %.6f" % tuple(p))
        for f in s.F:
            lines.append("f %d %d %d" % (f[0] + base, f[1] + base, f[2] + base))
        base += len(s.P)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return {"n_solids": len(solids), "n_v": base - 1, "bytes": os.path.getsize(path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rot-z", type=float, default=0.0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--pivot", default="bbox", choices=["bbox", "origin"])
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    solids, meta = world.load(a.src)          # 已经做过单位裁决，坐标系=米
    allP = np.vstack([s.P for s in solids])
    pivot = allP.mean(0) if a.pivot == "bbox" else np.zeros(3)
    if a.rot_z:
        R, t = rot_z(a.rot_z, pivot)
        for s in solids:
            s.P = s.P @ R.T + t
    if a.scale != 1.0:
        for s in solids:
            s.P = s.P * a.scale
    info = write_obj(solids, a.out, seed=a.seed)
    print("wrote %s  rotZ=%s scale=%s  %s" % (a.out, a.rot_z, a.scale, info))


if __name__ == "__main__":
    main()
