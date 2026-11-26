# -*- coding: utf-8 -*-
"""geom.py — 把场景导成「浏览器能直接画」的紧凑几何。

为什么不再只给 PNG：快照有三个改不掉的问题 —— 遮挡关系依赖画家算法（会错）、
视角固定（看不到侧面/顶部）、不能点（看不出这块是哪个实体）。
本机实测：20 个实体、9245 顶点、18426 三角面，扁平 JSON 约 500 KB，
gzip 后 ~150 KB，本机局域网传输是零感知的量级。所以「重不重」不是问题，
问题只在「要不要为此写一套前端」。

格式刻意压平（不嵌套 [[x,y,z],...]）：
  v  顶点扁平数组 x,y,z,x,y,z...
  f  索引扁平数组 a,b,c,a,b,c...
  g  实体分组 {sid, v0, vn, f0, fn, dim, c}
这样前端 `new Float32Array(g.v)` 直接进 BufferGeometry，不需要再 map 一遍。
"""
import json
import os

import numpy as np

# 与 viz/render.py 的 PALETTE 对齐，两张视图里同一个 sid 颜色一致
PALETTE = [
    (150, 162, 178), (176, 186, 200), (128, 142, 160),
    (214, 158, 106), (198, 132, 96), (226, 186, 128),
    (108, 168, 148), (140, 150, 200), (196, 150, 186),
]


def dump(solids, path, tag=None, step=None, turn=None, note=None):
    """写一份 geom.json。原子替换：页面 800 ms 轮询，不能读到半个文件。"""
    if not solids:
        return None
    order = sorted(solids, key=lambda s: s.sid)
    v, f, groups = [], [], []
    color_of = {}
    for i, s in enumerate(order):
        color_of[s.sid] = PALETTE[i % len(PALETTE)]

    voff = foff = 0
    for s in order:
        P = np.asarray(s.P, float)
        F = np.asarray(s.F, int)
        # 4 位小数 = 0.1 mm，足够"看"，也让文件比全精度小一半
        v.extend(np.round(P, 4).reshape(-1).tolist())
        f.extend((F + voff).reshape(-1).tolist())
        dim = (P.max(0) - P.min(0)).tolist() if len(P) else [0, 0, 0]
        groups.append({"sid": s.sid, "v0": voff, "vn": len(P), "f0": foff, "fn": len(F),
                       "dim": [round(x, 4) for x in dim], "c": list(color_of[s.sid])})
        voff += len(P)
        foff += len(F)

    payload = {
        "tag": tag, "step": step, "turn": turn, "note": note,
        "n_solids": len(order), "n_tri": len(f) // 3,
        "v": v, "f": f, "g": groups,
    }
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    os.replace(tmp, path)
    return {"bytes": os.path.getsize(path), "n_solids": len(order), "n_tri": len(f) // 3}


def live_geometry(bus, outdir, every=4, min_gap=1.5):
    """跑动中更新 runs/<tag>/geom.json。与 live_preview 分开节流：

    预览图是给人看「有没有在动」，几何是给人转着看「现在是什么样」，
    后者更有用，所以刷得比图勤一点，但同样不能每步都写。
    """
    import time
    path = os.path.join(outdir, "geom.json")
    state = {"n": -1, "t": 0.0}

    def _cb(loop):
        if not bus.scene:
            return
        n = len(loop.steps)
        now = time.time()
        if state["n"] >= 0 and (n - state["n"] < every and now - state["t"] < min_gap):
            return
        last = loop.steps[-1] if loop.steps else {}
        try:
            dump(bus.scene, path, step=last.get("id"), turn=loop.turn,
                 note="%s · %s" % (loop.phase, (last.get("tool") or "—")))
            state["n"], state["t"] = n, now
        except Exception:
            pass

    return _cb
