#!/usr/bin/env bash
# 第 01 题四小问：1) 左右/前后 2) 三距离（另见 reproduce.sh 的 ag_ifc）3) 门扇与墙面夹角
# 4) 整个房间绕地板前左角竖轴转 90° 后重算。
#
# 前两问要 LLM（现场求解）；第 4 问的旋转模型由 mutate.py 现造，绕点是「地板前左角」，
# 不是包围盒中心 —— 用 --pivot-xy 指定。
#
# 用法：
#   bash run_t01.sh            # 需要 DEEPSEEK_API_KEY 或 OPENAI_API_KEY
#   python verify_edits.py     # 只读磁盘产物验收（含 T01 四小问）
set -e
cd "$(dirname "$0")"

MODEL=../pkg/AI-Geometry-Models/IFC/AI_Geometry_Input.ifc
ROT=variants/room_rot90_corner.obj

[ -f "$MODEL" ] || { echo "[!] 缺模型包：$MODEL（见 README）"; exit 1; }

echo "== 0. 造「绕地板前左角转 90°」的副本 =="
if [ -f "$ROT" ]; then
  echo "   已有 $ROT，跳过（要重造先删掉）"
else
  python mutate.py --src "$MODEL" --out "$ROT" --rot-z 90 --pivot-xy "4.6,-2"
fi

echo "== 1. 第 1 问：左右 / 前后 + 两个方向的距离（原件）=="
python run_agent.py --card tasks/T01_1.md --model "$MODEL" --llm auto --tag t01_1k --max-turns 100 --wall-budget 320 --no-video

echo "== 2. 第 3 问：门扇平面与门所在墙面的较小夹角（原件）=="
python run_agent.py --card tasks/T01_3.md --model "$MODEL" --llm auto --tag t01_3b --max-turns 70 --no-video

echo "== 3. 第 4 问：同一套问法，在旋转副本上重算 =="
python run_agent.py --card tasks/T01_3.md --model "$ROT" --llm auto --tag t01_3_rot90 --max-turns 70 --no-video
# 第 1 问在旋转副本上要能拿到与原件一字不差的四个结论（右 0.800 / 后 0.670）。
# 这条以前是已知限制：方位命题在旋转后会被写歪。改成「交带符号的偏移、标签由符号翻译」之后过了。
python run_agent.py --card tasks/T01_1.md --model "$ROT" --llm auto --tag t01_1_rot --max-turns 100 --wall-budget 320 --no-video || true

echo "== 4. 验收（只读产物，另一套算法复算）=="
python verify_edits.py | tail -20
