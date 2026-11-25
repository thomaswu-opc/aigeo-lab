#!/usr/bin/env bash
# 一条命令复现：造变体 -> agent 自行求解（含录像）-> 对照组 -> 对抗样本 -> 矩阵 -> 观测页
set -e
cd "$(dirname "$0")"

MODEL=../pkg/AI-Geometry-Models/IFC/AI_Geometry_Input.ifc
CARD=tasks/T01_2.md
EP=episodes/T01_2.agent.json

if [ ! -f "$MODEL" ]; then
  echo "[!] 找不到模型包：$MODEL"
  echo "    赛方模型不随仓库分发。下载 https://hack26.ctrlpath.co/api/tasks/download"
  echo "    解压后应存在 pkg/AI-Geometry-Models/IFC/AI_Geometry_Input.ifc（见 README）。"
  exit 1
fi

echo "== 1. 造变体 =="
python mutate.py --src "$MODEL" --out variants/obj_plain.obj
python mutate.py --src "$MODEL" --out variants/obj_rot90.obj --rot-z 90
python mutate.py --src "$MODEL" --out variants/obj_rot45.obj --rot-z 45
python mutate.py --src "$MODEL" --out variants/obj_mm.obj   --scale 1000

echo "== 2. agent 自行求解（IFC 原件，出录像）=="
python run_agent.py --card $CARD --model "$MODEL" --episode $EP --tag ag_ifc --orbits 3

echo "== 3. 对照组：同一套工具，外部 plan 驱动 =="
python run.py --card $CARD --model "$MODEL" --plan plans/T01_2.json --tag ifc --orbits 2 --no-video

echo "== 4. agent 换模型重跑 + 对抗样本 + 汇总 =="
python compare_agent.py

echo "== 5. 降级路径（无 key 无记录）=="
python run_agent.py --card $CARD --model "$MODEL" --fallback --tag ag_fallback --no-video

echo "== 6. 读层四题（02 球与盒 / 03 截面与体积 / 04 占地与净距 / 09 实体识别）+ 写层五题 + 独立验收 =="
python run_edit_all.py
python compare_edits.py

echo "== 7. 观测页：把关键 run 导成单文件 HTML（双击即看）=="
for tag in ag_ifc ag_fallback adv_gate adv_fab; do
  python view.py export "$tag" --no-open
done
for tag in edit02 edit03 edit04 edit05 edit06 edit07 edit08 edit09 edit10; do
  python view.py export "$tag" --inline --no-open
done
# 控制台的两页（含 3 个 run 的数据与画面）：双击 docs/demo/studio.html 即可，不用起服务
python studio.py export --runs ag_ifc,edit09,edit02 --out ../docs/demo/studio.html

echo
echo "产物："
echo "  runs/ag_ifc/viewer.html      观测页：逐步回放 + 画面随步骤切换 + 数字追证据（双击打开）"
echo "  docs/demo/studio.html        控制台两页（主页 + 实验场）：出题入口、工具映射、存档/读档"
echo "  runs/ag_ifc/video.mp4        过程录像（帧由账本驱动，非录屏）"
echo "  runs/ag_ifc/report.md        决策链 + 答案 + 审计记录"
echo "  runs/ag_ifc/events.jsonl     哈希链账本"
echo "  runs/agent_matrix.md         泛化 + 防作弊矩阵"
echo "  runs/edit03/result.json      读层答案：净宽 4.8 / 净深 4.0、地板 19.2、剩余 18.97、球 4.1846（理想 4.1888）"
echo "  runs/edit02/result.json      读层答案：五位置关系（内切与外切的有符号距离都是 0，靠体积分开）"
echo "  runs/edit04/result.json      读层答案：占地余量 0.47、距门 0.47 → 1 m 净距不通过"
echo "  runs/edit09/result.json      读层答案：16 实体分类、三墙填充率 0.825/0.874/1.000"
echo "  runs/edit07/ball_cut.gltf    写层产物：切平并封口的球（切面面积 2.3536）"
echo "  runs/edit06/room_w6.0_round.gltf  写层产物：净宽 6.0、等面积圆窗的房间"
echo "  runs/edit05/chair_p1_fixed.gltf   写层产物：平移后距门 1.02 m 的椅子副本"
echo "  runs/edit08/state2.gltf      写层产物：门关闭 + 椅子搬到「距左墙 0.55、距前边界 0.95」的副本"
echo "  runs/edit10/docked.gltf      写层产物：三脚入槽、承托面贴合、无穿插的对接结果"
echo
echo "交互式观测：python view.py          （127.0.0.1:8765，可跨 run 对比）"
echo "终端观测：  python view.py ls  /  python view.py show ag_ifc"
