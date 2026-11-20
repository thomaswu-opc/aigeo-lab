# AI 几何沙盒（Geometry Sandbox）

给智能体一张**任务卡**和一个**模型文件**，它自己观察、自己选工具、自己判断证据够不够、
自己提交并接受审计。仓库里没有针对某道题写的代码：题目只以文本（card）存在，
判断只以决策记录（episode）存在，数值只由几何内核产生。

## 两条执行路径，别混淆

| 入口 | 谁在决策 | 用途 |
|---|---|---|
| `run.py` | 外部写好的 `plans/*.json` | plan 执行器。用于回归基线、复算数值 |
| **`run_agent.py`** | **沙盒内的 agent（`agent/loop.py` + `agent/policy.py`）** | **参赛形态：没有预先解好的 plan，只有观察→决策→执行→复核的循环** |

`run.py` 存在的意义是对照组：同一套工具，一个由外部指令驱动、一个由 agent 自己驱动，
两者数值必须一致 —— 这证明编排层没有引入偏差，也证明工具层是确定性的。

## 一条命令复现

```bash
bash reproduce.sh
```

造 4 个变体 → 用 `run_agent.py` 跑 agent（含录像）→ 跑 2 个对抗样本 → 生成 `runs/agent_matrix.md`。

单次运行 agent：

```bash
python run_agent.py --card tasks/T01_2.md \
                    --model ../pkg/AI-Geometry-Models/IFC/AI_Geometry_Input.ifc \
                    --episode episodes/T01_2.agent.json --tag ag_ifc
```

三种决策策略，接口相同（`agent/policy.py`）：

| 策略 | 命令 | 说明 |
|---|---|---|
| `RecordedPolicy` | `--episode episodes/…json` | 回放已记录的决策序列。**没有 API key 也能跑**，逐条对照「当时的观察 → 当时的判断」 |
| `LLMPolicy` | `--llm gpt-4o` | 运行时真调 LLM（OpenAI 兼容，`LLM_API_KEY` / `--base-url`）。生产路径，编排层与工具层一行不改 |
| `FallbackPolicy` | `--fallback` | 无 key 无记录时的规则回退。只证明架构能起来，会诚实地停在「证据不足」 |

## 人怎么看这个过程（观测层）

沙盒是无人值守跑的，所以「过程」必须能被人当场复核，否则录像只是装饰。四条通道，全部只读
（不触发任何工具、不需要 API key、不改 runs/ 一个字节）：

| 形式 | 命令 | 用途 |
|---|---|---|
| 终端列表 | `python view.py ls` | 所有 run 的状态 / 轮次 / 答案一行一个，`--kind all` 连开发记录一起看 |
| 终端时间线 | `python view.py show ag_ifc` | 逐步打印「阶段 · 步骤 · 工具 · 为什么」，`--full 400` 附带原始观察 |
| 本地观测台 | `python view.py` | 起 `127.0.0.1:8765`：左选 run，中间逐步回放，右侧画面随步骤切换 |
| 单文件导出 | `python view.py export ag_ifc` | 生成 `runs/ag_ifc/viewer.html`，**双击即看**，不起服务、不装依赖；加 `--inline` 连帧一起内嵌，得到一个可随手带走的 HTML |

观测台读四份既有产物，不为展示多写一份数据：`events.jsonl`（账本：谁在哪一步做了什么、为什么）、
`steps.json`（每步的完整观察）、`frames/manifest.json`（每帧属于哪一步）、`result.json`
（答案 / 审计 / 门控 / 失败升级）。

四个刻意的设计：

1. **画面跟着步骤走**。`manifest` 的每帧带 `steps`，游标落在第 N 步就切到该步所属镜头 ——
   不是播一段视频，是「这一步 agent 看到的几何」。`run_agent.py` 里为此把 step id 写进了场景。
2. **数字可追**。答案面板里每个距离点一下，跳到**第一次出现这个数**的那次工具返回；
   审计规则（1e-6、必须出现在证据步骤里）在界面上被复述一遍，看的人不必读源码。
3. **失败面不藏**。`GATE_BLOCKED` / `AUDIT_FAILED` / `STALLED` 各有可读解释；被拒的数字逐条列出
   （例：「这些数字无法追溯到任何证据步骤：[0.42, 1.85, 3.1]」）。
4. **账本在界面上自检**。逐条核对 `prev` 是否等于上一条 `hash`，断了标红 —— 改一条记录就会露。

## 编排层做了什么（这是比赛要的那部分）

`agent/loop.py` 只 300 行，但它决定 agent 是不是「会解题」：

| 机制 | 位置 | 为什么必须有 |
|---|---|---|
| **六阶段门控** | `PHASE_TOOLS` | SURVEY 阶段调不了 `distance`，REPORT 阶段调不了 `select`。越界被拒并留痕 `gate_reject`。**从结构上排除「跳过识别直接答题」** |
| **步骤语义标签** | `_call(label=…)` | 引用用 `@win.items` 而不是 `@s4.items`。失败恢复多耗几轮也不会让引用错位；证据清单变成可读的一串名字 |
| **观察压缩** | `agent/schema.observe` | 只对「长清单的元素」做字段白名单，嵌套元信息保持完整。工具输出原样灌给 LLM 会让它淹没后开始编 |
| **失败升级而非退出** | `_enter_recover` | `expect_n` 不符 → 进 RECOVER，把诊断交给策略自救，成功后 `failure_resolved`。45° 变体就是靠这条自愈的 |
| **答案审计** | `_audit` | 提交的每个数字必须能在某个证据步骤的输出里真实出现过（1e-6 相对容差）。**「编一个看起来合理的数」在结构上不可能通过** |
| **预算 / 停滞** | `max_turns` / `_sig_count` | 同一调用重复三次判停滞；墙钟上限。收尾状态按「被什么挡住」命名：`GATE_BLOCKED` / `AUDIT_FAILED` / `INSUFFICIENT_EVIDENCE` |

## 结构与文件

```
run_agent.py       agent 入口：跑循环、留痕、出报告与录像
agent/loop.py      编排层：六阶段状态机 + 门控 + 失败升级 + 答案审计 + 预算
agent/policy.py    决策层：Recorded / LLM / Fallback 三种策略，同一接口
agent/schema.py    工具 → LLM 规格；工具输出 → 可决策的观察
episodes/*.json    决策记录（可回放）；adv_*.json 是故意做坏的对抗样本

run.py             plan 执行器（对照组）
core/runtime.py    哈希链账本 + 工具总线 + 引用解析（无 eval）
tools/world.py     6→1 读入层：IFC/OBJ/glTF/USD → 世界坐标三角网格；裁决单位与朝向
tools/geom.py      度量内核：距离（含见证点）、体积、截面、法向聚类、场景主方向
tools/brep.py      三角汤 → 面结构：共面区域、边界环、内环（开口）、缺口面积
tools/ops.py       13 个注册工具，全部与任务无关
viz/render.py      无头渲染（PIL 正交投影 + 画家算法），不需要显卡
view.py            观测层：CLI 列表 / 终端时间线 / 本地观测台 / 单文件导出
viz/viewer.html    观测台前端（单文件、零外部依赖，导出时内嵌数据）
tasks/*.md         任务卡：智能体收到的全部任务信息
plans/*.json       执行器用的 plan：只有工具调用与引用，没有控制流
mutate.py          造变体：换格式 / 随机化对象名 / 刚体旋转 / 改单位
compare.py         执行器的泛化矩阵
compare_agent.py   agent 的泛化 + 防作弊矩阵
```

## 13 个工具

`load` `clusters` `principal_frame` `select` `pick` `merge` `contains` `gaps`
`distance` `rank` `section` `segment` `answer`

+ 2 个控制动作（不进几何总线，由编排层拦截）：`advance_phase` `declare_insufficient`

每个工具都带**前置条件 / 失败时的降级路径 / 返回物说明**，三项一起进 LLM 的 tool schema。
注册表里不存在任何任务编号。

## 怎么证明「不是预先解好的」

| 机制 | 落地位置 |
|---|---|
| 任务信息只以文本进入 | `tasks/*.md` 的 SHA-256 写进账本首条 |
| 工具注册表里没有任务编号 | `tools/ops.py` 全文件不含任何题目 id |
| 决策是数据不是代码 | `episodes/*.json` 只有工具调用与 `@label.field` 引用，无控制流、无算术 |
| 编排约束可被攻击测试 | `episodes/adv_gate.json` 试图跳过识别 → `GATE_BLOCKED` |
| 答案不可编造 | `episodes/adv_fabricate.json` 提交假数字 → `AUDIT_FAILED`，并逐条列出无法溯源的数值 |
| 每步可回溯 | `events.jsonl` 哈希链，改一字节即断链 |
| 换模型重跑不变 | `runs/agent_matrix.md`：格式/命名/单位/90° 全一致；45° 触发自救仍一致 |
| 失效就说不知道 | 证据不足时 `status=INSUFFICIENT_EVIDENCE`，不给错数 |
| 录像即账本 | `frames/manifest.json` 记录每帧对应的阶段与角度，视频由帧编码，非录屏 |


## LLM 策略：实测到哪一步，卡在哪一步

`--llm auto` 会按本机已有的 key 自动选模型（`DEEPSEEK_API_KEY` → `deepseek-chat`）。
跑一轮：

```bash
python run_agent.py --card tasks/T01_2.md --model <model> --llm auto --max-turns 80 --tag llm
```

**已做到**（10 轮实测迭代的记录在 `runs/llm*/`）：

| 轮 | 状态 | 观察到的瓶颈 | 修复 |
|---|---|---|---|
| 1 | STALLED | 观察把 20 个实体截成 8 条，`n` 仍写 20，agent 合理地怀疑数据缺失并反复重读 | 截断写进观察（`_truncated`），且改为按字符预算截断 |
| 2 | STALLED | `items` 在 schema 里写死为 `array`，模型无法写引用，只能把 19 个 sid 逐字复制进参数 | 集合参数改 `anyOf: [array, string]`，接受 `@load.solids` |
| 3 | STALLED | `select` 收到 sid 列表后原样回吐裸 sid，字段全丢，下一次 where 静默返回 0 | `select` 始终回填完整指纹 |
| 4 | STALLED | 工具报错都升级成「证据不足」，RECOVER 变成吸收态，80% 轮次耗在里面 | 工具错误留在本阶段；RECOVER 连续成功即自动返回 |
| 5 | STALLED | `bbox_lo/bbox_hi` 实际存在却被观察白名单滤掉，模型反复问一个它永远看不到的字段 | 补进白名单 + 把字段清单写进系统提示 |
| 6 | STALLED | `min_thickness` 对窗框报 1.4 m（等于板宽），厚度判据整体失效 | `dominant_dirs` 改为按「方向簇总面积」判定（窗框的环面由 16 块 0.04 m² 小三角拼成，合计 20%） |
| 7 | STALLED | `select` 无条件重算世界轴指纹，把 `principal_frame` 的成果覆盖掉 | 输入自带的量优先，缺什么补什么 |
| 8 | STALLED | 把数组序列化成字符串传来，解析器没去引号 → 「场景里没有 C011」 | 解析时去括号去引号 |
| 9 | STALLED | `segment` 在带凹口的面上一路抛异常（numpy 2.x 移除了二维 `np.cross`） | 手写二维叉积 |
| 10 | STALLED | 角色已能压对椅子/门扇，但窗框与画框（框体 vs 内嵌板）反复混淆，33 轮耗尽 | **未解决**，见下 |

**已稳定的部分**：椅子 `C017`、门扇 `C011` 每次都能压到唯一命中；`distance` 拿到 0.696547，
与我手算的基准一致；见证点残差 1e-7。协议层（引用、标签、门控、审计、账本）全程无异常。

**卡住的地方**（诚实的结论，不是调参问题）：

1. **框体 vs 内嵌板**。窗 = `C013`(框 1.4×1.4×0.24) + `C014`(玻璃 1.28×1.28×0.02)；
   画 = `C015`(框 1.2×0.9×0.08) + `C016`(板 1.04×0.74×0.025)。题目问「窗框 / 画框」，
   而内嵌板的尺寸特征更「干净」，模型很自然地选了板。要区分只能靠「框有贯通内环、板没有，
   且板嵌在框的内环里」这一条**关系判据**，而当前工具里 `contains` 只比包围盒，不够。
2. **它不擅长证伪自己的假设**。第 8 轮它认定「门扇应轴对齐」，理由是「门要么关着要么沿轴开」——
   而题面明写「打开的门」。这类错误不是工具缺失，是模型对几何情境的想象替代了读数。
3. **成本**：约 7.5k prompt tokens / 轮，80 轮 ≈ 60 万 tokens。回放路径 0.2 秒、0 成本，
   这就是为什么参赛提交以 `episodes/` 为主、LLM 路径作为「同一编排层可换策略」的证据。

**下一步要补的是工具，不是提示词**：`contains` 要升级成真正的「内环包含判定」
（谁的边界落在谁的内环里），并加一个 `enclosure` 判据把「框与其内嵌板」配成对。
这件事在建模上对 09/10 也有用（洞口归属、插脚与凹槽的对应）。

## 已知限制（写在明处）

1. **EDIT 层缺失** —— 06 的加宽联动、07 的拉伸与切平、05.4 的修正平移，需要网格布尔与流形检查，尚未实现。
2. **扫掠碰撞缺失** —— 08 的 0.1° 开门扫掠、10 的下降穿插，需要 CCD 或扫掠体求交。
3. **曲面只有二分类** —— 球/柱/锥尚未解析拟合，只能给「光滑闭合曲面」，给不出解析半径。
4. **`principal_frame` 只处理绕竖轴旋转** —— 它把 z′ 固定为世界 Z；真正倾斜（斜墙、坡屋面）会落进 `up_confident=false` 分支。
5. **`clusters` 的边界依赖主参照系** —— 已在主参照系下聚类（世界轴 AABB 只是真实距离的下界，
   旋转时最多膨胀 √2 倍：45° 时火箭与房间的间隙从 2.8 m 塌到 0.71 m，三簇并成一簇）。
   残留风险：若场景没有显著平面朝向（全曲面场景），`frame.used` 会退回 `world` 并标注，
   此时判据退化到世界轴。`consistency.agree=false` 就是给 agent 的显式信号。
6. **LLM 策略尚未收敛到「提交答案」** —— 见下一节，这是当前最大的未完成项。
7. **`distance` 对相交网格返回 0**，不区分「接触 / 穿透 / 含于内部」。
8. **`FBX` / `.blend` 不支持**，需离线转 OBJ。
