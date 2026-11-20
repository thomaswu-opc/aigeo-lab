# -*- coding: utf-8 -*-
"""verify_edits.py — 不经过智能体，直接验收 EDIT 层产出的模型文件。

这个脚本的立场是评委的立场：**只认磁盘上的文件**。
它不读 result.json、不读 episode、不看智能体说过什么，只把 runs/edit0*/ 下的
.gltf 读回来，用与编辑无关的只读工具重量一遍，再对照任务卡的要求逐条勾。

期望值来自任务卡里的约束（净宽 6.0 / 5.4、窗洞 1.4、面积相同等），
不是从智能体的答案里抄的 —— 抄答案的验收没有任何意义。
"""
import glob
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from core.runtime import Bus, Ledger, TOOLS      # noqa: E402
import tools.ops                                # noqa: E402

TOL_LEN = 1e-3          # 赛方：长度容差 1 mm
TOL_VOL = 3e-3          # 体积相对容差（球的网格量 vs 理想球体允许 0.3%）

_VDIR = os.path.join(HERE, "runs", "_verify")
os.makedirs(_VDIR, exist_ok=True)
bus = Bus(Ledger(os.path.join(_VDIR, "ledger.jsonl")))


def load(path):
    """读文件，返回这次的实体列表。

    load 会整场替换 bus.scene（不是追加），所以每次读完直接取全量即可；
    但必须**当场**取，下一次 load 会把上一次的壳冲掉。
    """
    TOOLS["load"]["fn"](bus, path)
    return list(bus.scene)


def mh(sid):
    return TOOLS["mesh_health"]["fn"](bus, [sid])["rows"][0]


def fp(sid):
    return TOOLS["footprint"]["fn"](bus, [sid])["rows"][0]


def holes_of(sid, min_area=0.2):
    """把实体的所有内环拿出来（面积、bbox 中心）。"""
    out = []
    seg = TOOLS["segment"]["fn"](bus, sid, 20.0, 4)
    for f in seg["faces"]:
        for h in (f.get("holes") or []):
            if h.get("area", 0) >= min_area:
                out.append({"area": h["area"], "w": h.get("w"), "h": h.get("h"),
                            "center": h.get("center"), "face_n": f["n"]})
    return out


def find_floor(solids):
    best = None
    for s in solids:
        r = mh(s.sid)
        if r["dims"][2] < 0.15 and r["dims"][0] > 4.0 and r["dims"][1] > 3.0:
            if best is None or r["volume"] > best[1]["volume"]:
                best = (s.sid, r)
    return best


def find_wall_with_hole(solids, min_area):
    for s in solids:
        hs = holes_of(s.sid, min_area)
        if hs:
            return s.sid, hs
    return None, []


ROWS = []


def check(group, item, got, want, ok, note=""):
    ROWS.append({"组": group, "检查项": item, "实测": got, "要求": want,
                 "结论": "通过" if ok else "不通过", "备注": note})


# ---------------------------------------------------------------- T-07
CUBE_V0, CUBE_V1 = 2.5 ** 3, 2.5 * 2.5 * 3.0
BALL_R = 1.0
BALL_CAP_H = 0.5
BALL_V_KEEP = 4 * math.pi / 3 - math.pi * BALL_CAP_H ** 2 * (3 * BALL_R - BALL_CAP_H) / 3
CUT_A = math.pi * (BALL_R ** 2 - BALL_CAP_H ** 2)

p = os.path.join(HERE, "runs", "edit07", "cube_tall.gltf")
if os.path.exists(p):
    sc = load(p)
    r = mh(sc[0].sid)
    check("T07 长方体", "尺寸 (m)", [round(x, 6) for x in r["dims"]], "[2.5, 2.5, 3.0]",
          all(abs(r["dims"][i] - v) <= TOL_LEN for i, v in enumerate([2.5, 2.5, 3.0])))
    check("T07 长方体", "体积 (m³)", r["volume"], "%.4f" % CUBE_V1,
          abs(r["volume"] - CUBE_V1) <= TOL_VOL * CUBE_V1, "编辑前 %.4f" % CUBE_V0)
    check("T07 长方体", "封闭 / 破洞 / 重复面",
          "%s / %d / %d" % (r["closed"], r["n_boundary_loops"], r["n_duplicate_faces"]),
          "封闭 / 0 / 0",
          r["closed"] and r["n_boundary_loops"] == 0 and r["n_duplicate_faces"] == 0)

p = os.path.join(HERE, "runs", "edit07", "ball_cut.gltf")
if os.path.exists(p):
    sc = load(p)
    r = mh(sc[0].sid)
    check("T07 球", "保留高度 (m)", round(r["dims"][2], 6), "1.5",
          abs(r["dims"][2] - 1.5) <= TOL_LEN, "球最低点在 z=0，切面 z=0.5")
    check("T07 球", "体积 (m³)", r["volume"], "%.4f（理想）" % BALL_V_KEEP,
          abs(r["volume"] - BALL_V_KEEP) <= TOL_VOL * BALL_V_KEEP,
          "网格球本身比理想球小 0.100%")
    check("T07 球", "封闭 / 破洞 / 重复面",
          "%s / %d / %d" % (r["closed"], r["n_boundary_loops"], r["n_duplicate_faces"]),
          "封闭 / 0 / 0",
          r["closed"] and r["n_boundary_loops"] == 0 and r["n_duplicate_faces"] == 0,
          "切口已封口（欧拉 %d，与球同胚）" % r["euler"])

# ---------------------------------------------------------------- 基准：房间从哪来
# 期望值不许从智能体的答案里抄。房间的基准（左墙内表面、地板前边界）直接从原始模型
# 的地板上读 —— 那是题面给的输入，不是解。拿不到原始模型时才回落到常量并注明。
SRC = os.path.join(HERE, "..", "pkg", "AI-Geometry-Models", "IFC", "AI_Geometry_Input.ifc")
BASE_SRC = "原始模型地板"
if os.path.exists(SRC):
    TOOLS["load"]["fn"](bus, SRC)
    base = None
    for s in bus.scene:
        r0 = mh(s.sid)
        if r0["dims"][2] < 0.15 and r0["dims"][0] > 4.0 and r0["dims"][1] > 3.0:
            if base is None or r0["volume"] > base[1]["volume"]:
                base = (s.sid, r0)
    fr0 = fp(base[0])
    LEFT_IN, FRONT_Y, RIGHT_IN = fr0["rect_world_xy"][0], fr0["rect_world_xy"][1], fr0["rect_world_xy"][2]
else:
    LEFT_IN, FRONT_Y, RIGHT_IN = 4.6, -2.0, 9.4
    BASE_SRC = "常量回落（未找到原始模型）"
print("房间基准（%s）：左墙内表面 x=%.4f，地板前边界 y=%.4f，右墙内表面 x=%.4f\n"
      % (BASE_SRC, LEFT_IN, FRONT_Y, RIGHT_IN))

# ---------------------------------------------------------------- T-05
# 题面给的三个位置是**占地中心**到左墙内侧平面 / 地板前边界的距离，
# 所以期望中心坐标可以由这两个基准直接算出来：center = (LEFT_IN + a, FRONT_Y + b)。
# 期望距门距离再由「占地矩形到门洞底边线段」闭式算，全程不碰智能体的数。
DOOR_EDGE = ([4.6, -1.55, 0.0], [4.6, -0.55, 0.0])
POS = {"chair_p1.gltf": (0.70, 1.62), "chair_p2.gltf": (1.60, 1.62),
       "chair_p3.gltf": (1.20, 2.40), "chair_p1_fixed.gltf": None}   # None = 平移副本，不查位置


def rect_to_seg(rect, p1, p2):
    """轴对齐矩形 [x0,y0,x1,y1] 到线段 p1-p2 的最短平面距离（闭式）。"""
    (ax, ay), (bx, by) = (p1[0], p1[1]), (p2[0], p2[1])
    # 线段是 x 恒定的竖线（门洞底边在左墙内侧平面上），退化成「x 距离 + y 区间间隙」
    if abs(ax - bx) < 1e-9:
        dx = max(ax - rect[2], rect[0] - ax, 0.0)
        ylo, yhi = min(ay, by), max(ay, by)
        dy = max(ylo - rect[3], rect[1] - yhi, 0.0)
        return math.hypot(dx, dy)
    raise NotImplementedError("门洞底边不是竖线，需要补一般线段距离")


for name, pos in POS.items():
    p = os.path.join(HERE, "runs", "edit05", name)
    if not os.path.exists(p):
        continue
    sc = load(p)
    c = TOOLS["clearance"]["fn"](bus, sc[0].sid, DOOR_EDGE[0], DOOR_EDGE[1])
    f = fp(sc[0].sid)
    rect = f["rect_world_xy"]
    want_c = None
    if pos is not None:
        want_c = (LEFT_IN + pos[0], FRONT_Y + pos[1])
        check("T05 " + name, "占地中心 (m)",
              "(%.4f, %.4f)" % (f["center_world"][0], f["center_world"][1]),
              "(%.4f, %.4f)" % want_c,
              abs(f["center_world"][0] - want_c[0]) <= TOL_LEN and
              abs(f["center_world"][1] - want_c[1]) <= TOL_LEN,
              "题面位置 %.2f / %.2f 加基准推出" % pos)
    want_d = rect_to_seg(rect, DOOR_EDGE[0], DOOR_EDGE[1])
    check("T05 " + name, "距门距离 (m)", round(c["value"], 4), "%.4f（闭式）" % want_d,
          abs(c["value"] - want_d) <= 10 * TOL_LEN,
          "≥1 m " + ("成立" if c["value"] >= 1.0 - TOL_LEN else "不成立（需提平移方案）"))
    in_floor = (rect[0] >= LEFT_IN - TOL_LEN and rect[2] <= RIGHT_IN + TOL_LEN and
                rect[1] >= FRONT_Y - TOL_LEN and rect[3] <= 2.0 + TOL_LEN)
    check("T05 " + name, "占地在地板内 / 落地", "%s / %s" % (in_floor, f["grounded"]),
          "True / True", in_floor and f["grounded"])

# ---------------------------------------------------------------- T-06
ROOM = {"room_w6.0_square.gltf": (6.0, "square"),
        "room_w6.0_round.gltf": (6.0, "round"),
        "room_w5.4_round.gltf": (5.4, "round")}
for name, (want_w, kind) in ROOM.items():
    want_cx = LEFT_IN + want_w / 2.0          # 左墙内表面不动，洞口居中于新净跨
    p = os.path.join(HERE, "runs", "edit06", name)
    if not os.path.exists(p):
        continue
    sc = load(p)
    fid, fr = find_floor(sc)
    check("T06 " + name, "净宽 / 净深 (m)",
          "%.4f / %.4f" % (fr["dims"][0], fr["dims"][1]),
          "%.1f / 4.0" % want_w,
          abs(fr["dims"][0] - want_w) <= TOL_LEN and abs(fr["dims"][1] - 4.0) <= TOL_LEN,
          "左墙内表面不动，右墙外移")
    wid, hs = find_wall_with_hole(sc, 1.0)
    if hs:
        h = hs[0]
        # 洞口的 bbox 中心用 w/h 与 center 反推不准，这里用「洞口两端 x 的均值」= 面中心
        cx = h["center"][0]
        check("T06 " + name, "洞口面积 (m²)", round(h["area"], 6), "1.960 = 1.4²",
              abs(h["area"] - 1.96) <= 1e-3)
        check("T06 " + name, "洞口中心 x / 净跨中点", "%.3f / %.3f" % (cx, want_cx),
              "%.3f / %.3f" % (want_cx, want_cx), abs(cx - want_cx) <= 0.02,
              "居中判据：洞口中心 = (左墙内表面 + 右墙内表面)/2")
        if kind == "square":
            check("T06 " + name, "洞口形状 (宽×高)", "%.4f×%.4f" % (h["w"], h["h"]),
                  "1.4×1.4", abs(h["w"] - 1.4) <= TOL_LEN and abs(h["h"] - 1.4) <= TOL_LEN)
        else:
            check("T06 " + name, "洞口形状 (宽×高)", "%.4f×%.4f" % (h["w"], h["h"]),
                  "≈√1.96 的正方形外框（圆）",
                  abs(h["w"] - h["h"]) <= 2 * TOL_LEN and abs(h["area"] - 1.96) <= 1e-3,
                  "圆用 48 段多边形，bbox≈外接正方形")
    wall = mh(wid)
    check("T06 " + name, "后墙封闭 / 破洞", "%s / %d" % (wall["closed"], wall["n_boundary_loops"]),
          "封闭 / 0", wall["closed"] and wall["n_boundary_loops"] == 0)

# ---------------------------------------------------------------- 输出
if not ROWS:
    print("没有找到 runs/edit0*/ 下的产物。先跑：python run_edit_all.py")
    raise SystemExit(1)

cols = ["组", "检查项", "实测", "要求", "结论", "备注"]
w = {c: max(len(c), max(len(str(r[c])) for r in ROWS)) for c in cols}
line = "  ".join("-" * w[c] for c in cols)
print("EDIT 层产物验收（只读磁盘文件，不读智能体的报告）\n")
print("  ".join(c.ljust(w[c]) for c in cols))
print(line)
for r in ROWS:
    print("  ".join(str(r[c]).ljust(w[c]) for c in cols))
bad = [r for r in ROWS if r["结论"] != "通过"]
print("\n%d 项检查，通过 %d，不通过 %d" % (len(ROWS), len(ROWS) - len(bad), len(bad)))
raise SystemExit(1 if bad else 0)
