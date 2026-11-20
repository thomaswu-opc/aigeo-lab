#!/usr/bin/env bash
# 一条命令复现：造变体 -> agent 自行求解（含录像）-> 对照组 -> 对抗样本 -> 矩阵 -> 观测页
set -e
cd "$(dirname "$0")"

MODEL=../pkg/AI-Geometry-Models/IFC/AI_Geometry_Input.ifc
CARD=tasks/T01_2.md
EP=episodes/T01_2.agent.json

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

echo "== 6. 观测页：把关键 run 导成单文件 HTML（双击即看）=="
for tag in ag_ifc ag_fallback adv_gate adv_fab; do
  python view.py export "$tag" --no-open
done

echo
echo "产物："
echo "  runs/ag_ifc/viewer.html      观测页：逐步回放 + 画面随步骤切换 + 数字追证据（双击打开）"
echo "  runs/ag_ifc/video.mp4        过程录像（帧由账本驱动，非录屏）"
echo "  runs/ag_ifc/report.md        决策链 + 答案 + 审计记录"
echo "  runs/ag_ifc/events.jsonl     哈希链账本"
echo "  runs/agent_matrix.md         泛化 + 防作弊矩阵"
echo
echo "交互式观测：python view.py          （127.0.0.1:8765，可跨 run 对比）"
echo "终端观测：  python view.py ls  /  python view.py show ag_ifc"
