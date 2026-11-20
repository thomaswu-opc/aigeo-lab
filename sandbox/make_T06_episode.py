# -*- coding: utf-8 -*-
"""make_T06_episode.py — 生成 T06（加宽房间 + 方窗改圆窗 + 再改净宽）的决策记录。

手写 40 多条 turn 容易写错，也不利于复核「哪些量是参数」。这里把
「一组锚点 = 一组参数」显式写成数据，再展开成步骤序列 ——
同一份锚点表既是 episode 的来源，也是回答「可继续修改参数」时的依据。

关键：不靠 sid 序号猜哪面墙。墙按包围盒中心 cx / 跨度 d0 区分，
这两个量都能被 select 的 where 直接筛。
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- 现状（几何实测）
X_L_IN, X_R_IN, X_R_OUT = 4.6, 9.4, 9.6      # 左墙内表面 / 右墙内表面 / 右墙外表面
X_B_NEG, X_B_POS = 4.4, 9.6                  # 后墙沿 x 的两端
WIN_LO, WIN_HI = 6.3, 7.7                    # 窗洞 x 范围（中心 7.0 = 净跨中点）
WIN_W = WIN_HI - WIN_LO                      # 1.4
WIN_AREA = WIN_W * WIN_W                     # 1.96

# ---------------------------------------------------------------- 目标参数
W1, W2 = 6.0, 5.4
R1_IN, R1_OUT = X_L_IN + W1, X_L_IN + W1 + 0.2    # 10.6 / 10.8
R2_IN, R2_OUT = X_L_IN + W2, X_L_IN + W2 + 0.2    # 10.0 / 10.2
W1_MID, W2_MID = (X_L_IN + R1_IN) / 2, (X_L_IN + R2_IN) / 2   # 7.6 / 7.3
WIN1 = (W1_MID - WIN_W / 2, W1_MID + WIN_W / 2)   # 6.9 / 8.3
WIN2 = (W2_MID - WIN_W / 2, W2_MID + WIN_W / 2)   # 6.6 / 8.0

WALL = {"d2": {"$lte": 0.25}, "d0": {"$gte": 3.5},
        "z_min": {"$lte": 0.006}, "z_min>=": -0.006}


def st(label, sid, anchors, why):
    return {"phase": "EDIT", "label": label, "tool": "stretch_axis",
            "args": {"sids": [sid], "axis": 0, "anchors": anchors}, "why": why}


def widen(tag, r_in, r_out, win_lo, win_hi, head):
    """一次改净宽所需的全部锚点。这张表就是「参数联动关系」。"""
    pic0, pic1 = r_in - 0.08, r_in          # 画框贴右墙内侧，厚 0.08
    pb0, pb1 = r_in - 0.04, r_in - 0.015    # 画板在画框前
    return [
        st(tag + "_floor", "@floor.items[0].sid",
           [[X_L_IN, X_L_IN], [X_R_IN, r_in]],
           head + "地板：左端固定，右端随新净宽。它的 x 跨度就是净宽，所以它同时是校验量。"),
        st(tag + "_wallB", "@wallB.items[0].sid",
           [[X_B_NEG, X_B_NEG], [WIN_LO, win_lo], [WIN_HI, win_hi], [X_B_POS, r_out]],
           head + "后墙：必须四段 —— 洞左拉伸、洞身整体平移（宽度不变）、洞右拉伸。"
           "只用一段线性映射会把窗洞一起拉宽，尺寸就变了。"),
        st(tag + "_wallR", "@wallR.items[0].sid",
           [[X_R_IN, r_in], [X_R_OUT, r_out]],
           head + "右墙：纯平移，厚度与高度不变。"),
        st(tag + "_winF", "@winframe.items[0].sid",
           [[WIN_LO, win_lo], [WIN_HI, win_hi]],
           head + "窗框：随洞口平移，尺寸不变。"),
        st(tag + "_picF", "@picture.items[0].sid",
           [[X_R_IN - 0.08, pic0], [X_R_IN, pic1]],
           head + "画框：随右墙移动并继续贴墙。"),
    ]


turns = [
    {"phase": "SURVEY", "label": "load", "tool": "load",
     "args": {"path": "$MODEL", "keep_metadata": False},
     "why": "读入。keep_metadata=false：改形状只能用几何，不许看名称与类别。"},
    {"phase": "SURVEY", "label": "clusters", "tool": "clusters",
     "args": {"items": "@load.solids", "tol": 1.0},
     "why": "文件装着三组互不相连的模型；房间是实体最多的那一簇。"},
    {"phase": "SURVEY", "tool": "advance_phase",
     "why": "已读入并分出房间簇。进入角色识别。"},

    {"phase": "IDENTIFY", "label": "wallL", "tool": "select",
     "args": {"items": "@clusters.groups[0].items",
              "where": {"$and": [{"d2": {"$lte": 0.25}}, {"d0": {"$gte": 3.5}},
                                 {"z_min": {"$lte": 0.006}}, {"z_min>=": -0.006},
                                 {"cx": {"$lte": 5.0}}]}},
     "why": "薄、长、落地的竖向板，且包围盒中心靠左 = 左墙。加宽时它必须一动不动。"},
    {"phase": "IDENTIFY", "label": "wallR", "tool": "select",
     "args": {"items": "@clusters.groups[0].items",
              "where": {"$and": [{"d2": {"$lte": 0.25}}, {"d0": {"$gte": 3.5}},
                                 {"z_min": {"$lte": 0.006}}, {"z_min>=": -0.006},
                                 {"cx": {"$gte": 9.0}}]}},
     "why": "同类板，中心靠右 = 右墙。加宽时随新净宽平移。"},
    {"phase": "IDENTIFY", "label": "wallB", "tool": "select",
     "args": {"items": "@clusters.groups[0].items",
              "where": {"$and": [{"d2": {"$lte": 0.25}}, {"d0": {"$gte": 4.5}},
                                 {"z_min": {"$lte": 0.006}}, {"z_min>=": -0.006}]}},
     "why": "同类板，但沿 x 的跨度最长（5.2 > 4.0）= 后墙。窗洞开在它上面。"},
    {"phase": "IDENTIFY", "label": "floor", "tool": "select",
     "args": {"items": "@clusters.groups[0].items",
              "where": {"$and": [{"z_min": {"$lte": -0.01}}, {"d2": {"$lte": 0.15}},
                                 {"d0": {"$gte": 4.5}}]}},
     "why": "薄、水平、底面低于 z=0 的大板 = 地板。顶面边界就是房间范围，x 跨度即净宽。"},
    {"phase": "IDENTIFY", "label": "winhole", "tool": "select",
     "args": {"items": "@clusters.groups[0].items",
              "where": {"$and": [{"d0": {"$gte": 1.35}}, {"d0": {"$lte": 1.45}},
                                 {"d1": {"$gte": 1.35}}, {"d1": {"$lte": 1.45}},
                                 {"d2": {"$gte": 0.55}}, {"d2": {"$lte": 0.65}}]}},
     "why": "两向约 1.4、厚度约 0.6（贯穿墙体）的盒状几何 = 窗洞。"
            "窗框也是 1.4×1.4，但厚度只有 0.24，不贯穿 —— 靠这一点区分。"},
    {"phase": "IDENTIFY", "label": "winframe", "tool": "select",
     "args": {"items": "@clusters.groups[0].items",
              "where": {"$and": [{"d0": {"$gte": 1.35}}, {"d0": {"$lte": 1.45}},
                                 {"d1": {"$gte": 1.35}}, {"d1": {"$lte": 1.45}},
                                 {"d2": {"$lte": 0.3}}]}},
     "why": "同为 1.4×1.4 但更薄 = 窗框。改圆窗后它需要重新匹配，所以现在就要定下来。"},
    {"phase": "IDENTIFY", "label": "picture", "tool": "select",
     "args": {"items": "@clusters.groups[0].items",
              "where": {"$and": [{"d2": {"$lte": 0.1}}, {"d0": {"$gte": 1.0}},
                                 {"d0": {"$lte": 1.3}}, {"d1": {"$gte": 0.8}}]}},
     "why": "薄、高约 1 m、贴在墙面上的板 = 画框。它贴右墙，所以加宽时要随墙走。"},
    {"phase": "IDENTIFY", "tool": "advance_phase",
     "why": "三面墙按位置分别唯一命中，地板/窗洞/窗框/画框也已确定。进入基线测量。"},

    {"phase": "MEASURE", "label": "mh_floor", "tool": "mesh_health",
     "args": {"sids": "@floor.items"},
     "why": "量地板三向尺寸：x 跨度即净宽 4.8、y 跨度即净深 4.0，同时确认闭合。"},
    {"phase": "MEASURE", "label": "wall_h", "tool": "mesh_health",
     "args": {"sids": "@wallL.items"},
     "why": "量左墙高度（3.0）。净宽/净深/墙高是本题唯一允许变化的三个量里的两个不变量。"},
    {"phase": "MEASURE", "label": "hole0", "tool": "segment",
     "args": {"sid": "@wallB.items[0].sid", "sharp_deg": 20.0},
     "why": "量后墙大面的内环：内环就是窗洞（不贴地的开口才会成为内环，门洞贴地所以没有）。"
            "这一步给出洞口尺寸、中心与所属面。"},
    {"phase": "MEASURE", "tool": "advance_phase",
     "why": "净宽、净深、墙高与窗洞内环都拿到。进入编辑。"},
    {"phase": "VERIFY", "tool": "advance_phase",
     "why": "编辑前的判据只有现状尺寸，已在测量阶段取得，无需额外复核。"},
]

# ---- 06.1 加宽到 6.0
turns += widen("w1", R1_IN, R1_OUT, WIN1[0], WIN1[1], "净宽 4.8→6.0（左墙内表面不动、向右加宽）：")
turns += [
    {"phase": "EDIT", "label": "mh_6_0", "tool": "mesh_health",
     "args": {"sids": ["@floor.items[0].sid", "@wallR.items[0].sid"]},
     "why": "核对加宽后地板 x 跨度是否 6.0、净深是否仍 4.0，右墙厚度是否仍 0.2。"},
    {"phase": "EDIT", "label": "hole1", "tool": "segment",
     "args": {"sid": "@wallB.items[0].sid", "sharp_deg": 20.0},
     "why": "量加宽后窗洞内环：尺寸必须仍是 1.4×1.4，中心必须落在新净跨中点 7.6。"},
    {"phase": "EDIT", "label": "emit_6_0", "tool": "emit",
     "args": {"path": "runs/edit06/room_w6.0_square.gltf"},
     "why": "加宽后方窗版落盘。"},

    # ---- 06.2 方窗改圆窗
    {"phase": "EDIT", "label": "round", "tool": "reopen_round",
     "args": {"sid": "@wallB.items[0].sid", "center": [W1_MID, 2.1, 1.6],
              "area": WIN_AREA, "segments": 48},
     "why": "把后墙方洞改成同面积圆洞：面积 1.96 不变、中心不动，半径由面积反算。"
            "板与孔壁整体重建，不是把方洞压成椭圆。"},
    {"phase": "EDIT", "label": "mh_round", "tool": "mesh_health",
     "args": {"sids": "@wallB.items[0].sid"},
     "why": "重建后体检：必须仍封闭、无破洞、无重复面；体积应等于「板体积 − 开口面积×板厚」。"},
    {"phase": "EDIT", "label": "emit_round", "tool": "emit",
     "args": {"path": "runs/edit06/room_w6.0_round.gltf"},
     "why": "圆窗版落盘。"},

    # ---- 06.3 再改到 5.4，验证圆窗自动居中
    {"phase": "EDIT", "label": "s54_floor", "tool": "stretch_axis",
     "args": {"sids": ["@floor.items[0].sid"], "axis": 0,
              "anchors": [[X_L_IN, X_L_IN], [R1_IN, R2_IN]]},
     "why": "净宽 6.0→5.4：右墙内表面 10.6→10.0，左端固定。地板随之收窄。"},
    {"phase": "EDIT", "label": "s54_wallR", "tool": "stretch_axis",
     "args": {"sids": ["@wallR.items[0].sid"], "axis": 0,
              "anchors": [[R1_IN, R2_IN], [R1_OUT, R2_OUT]]},
     "why": "右墙纯平移。左墙不动 —— 这是「左墙内表面位置不变」的直接体现。"},
    {"phase": "EDIT", "label": "s54_winF", "tool": "stretch_axis",
     "args": {"sids": ["@winframe.items[0].sid"], "axis": 0,
              "anchors": [[WIN1[0], WIN2[0]], [WIN1[1], WIN2[1]]]},
     "why": "窗框随新中心平移到 6.6..8.0（中心 7.3）。"},
    {"phase": "EDIT", "label": "s54_picF", "tool": "stretch_axis",
     "args": {"sids": ["@picture.items[0].sid"], "axis": 0,
              "anchors": [[R1_IN - 0.08, R2_IN - 0.08], [R1_IN, R2_IN]]},
     "why": "画框随右墙，继续贴墙。"},
    {"phase": "EDIT", "label": "s54_wallB", "tool": "stretch_axis",
     "args": {"sids": ["@wallB.items[0].sid"], "axis": 0,
              "anchors": [[X_B_NEG, X_B_NEG], [WIN1[0], WIN2[0]],
                          [WIN1[1], WIN2[1]], [R1_OUT, R2_OUT]]},
     "why": "后墙按新净跨收窄、洞口区域平移到新中心。此时圆会被压成椭圆 —— 下一步重建回来。"},
    {"phase": "EDIT", "label": "round54", "tool": "reopen_round",
     "args": {"sid": "@wallB.items[0].sid", "center": [W2_MID, 2.1, 1.6],
              "area": WIN_AREA, "segments": 48},
     "why": "在新净跨下重建圆孔（中心 7.3）。这一步就是「圆窗仍能自动居中」的实现："
            "中心由净跨中点算出，不需要手工挪洞。"},
    {"phase": "EDIT", "label": "mh_5_4", "tool": "mesh_health",
     "args": {"sids": ["@floor.items[0].sid", "@wallB.items[0].sid"]},
     "why": "核对净宽是否 5.4，以及重建后的后墙是否仍封闭。"},
    {"phase": "EDIT", "label": "emit_5_4", "tool": "emit",
     "args": {"path": "runs/edit06/room_w5.4_round.gltf"},
     "why": "5.4 m 圆窗版落盘。"},
    {"phase": "EDIT", "tool": "advance_phase",
     "why": "三段编辑都已改完并各自落盘（3 个文件）。进入复核。"},
]

for tag, f, note in (("w6sq", "room_w6.0_square.gltf", "加宽后的方窗版"),
                     ("w6rd", "room_w6.0_round.gltf", "圆窗版"),
                     ("w54", "room_w5.4_round.gltf", "5.4 m 圆窗版")):
    turns.append({"phase": "REVALIDATE", "label": "rl_" + tag, "tool": "load",
                  "args": {"path": "runs/edit06/" + f},
                  "why": "读回%s文件。此后所有量来自磁盘，不来自内存。" % note})
    turns.append({"phase": "REVALIDATE", "label": "hv_" + tag, "tool": "mesh_health",
                  "args": {"sids": "@rl_%s.solids" % tag},
                  "why": "读回后重量尺寸与封闭性，必须与编辑时的值一致。"})

turns.append({"phase": "REVALIDATE", "tool": "advance_phase",
              "why": "三个文件都已读回并独立量过，进入报告。"})

turns.append({
    "phase": "REPORT", "tool": "answer",
    "args": {
        "values": {
            "net_width_before_m": 4.8,
            "net_depth_m": 4.0,
            "wall_height_m": 3.0,
            "net_width_6_m": 6.0,
            "net_width_54_m": 5.4,
            "window_width_m": 1.4,
            "window_height_m": 1.4,
            "window_center_x_6_m": 7.6,
            "window_center_x_54_m": 7.3,
            "round_opening_area_m2": 1.96,
            "round_opening_diameter_m": 1.5799,
        },
        "evidence": ["mh_floor", "wall_h", "hole0", "mh_6_0", "hole1", "round",
                     "mh_round", "round54", "mh_5_4", "hv_w6sq", "hv_w6rd", "hv_w54"],
        "confidence": {"net_width_6_m": "high", "window_center_x_6_m": "high",
                       "round_opening_area_m2": "high"},
        "unknowns": [],
        "notes": "加宽只改了一处：沿 x 的分段线性锚点表。同一份表同时决定地板、后墙、右墙、"
                 "窗框、画框的位移，所以「联动」是数据而不是散落在代码里的特例；"
                 "改净宽只需换表里的两个数。后墙必须用四段锚点（洞左拉伸/洞身平移/洞右拉伸），"
                 "只用一段会把窗洞一起拉宽。方窗改圆窗按面积 1.96 反算半径，48 段多边形使开口面积"
                 "精确等于 1.96 而非近似。改到 5.4 m 时后墙拉伸会把圆压成椭圆，故重新执行一次圆孔重建 ——"
                 "这正是「参数化」的含义：形状由参数决定，改参数只需重算。",
    },
    "why": "提交。净宽、洞口尺寸与中心均可追溯到体检与内环步骤；三个文件都读回复核过。",
})

ep = {"note": "T06 的决策记录。一次加宽 = 替换一张锚点表；一次开圆孔 = 重建一块板。"
              "两次改净宽用同一套锚点结构、只是数值不同，这就是「可继续修改参数的模型」的证据。",
      "card": "tasks/T06.md", "turns": turns}

out = os.path.join(HERE, "episodes", "T06.edit.json")
json.dump(ep, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("wrote", out, len(turns), "turns")
