# v1.5 编排层改动：先把「不该失败的轮次」从账上抹掉

> 结论一句话：现场 46 次运行、2658 轮里的 **195 次报错，有 135 次（69%）是机械问题**，
> 不是几何推理失败。这 135 次可以在工具/编排层直接消掉，不需要更强的模型、也不需要更多轮次。

## 为什么不是「多跑几次看分数」

| 观测 | 95% 置信区间 | 能说明什么 |
|---|---|---|
| 9/22 = 41% | [23%, 61%] | 宽 38 个点；±3 次全在噪声里 |
| 10/24 = 42% | [24%, 61%] | 同上 |

要检出 41% → 60%，每组需 105 次运行（α=0.05、power=0.8）；41% → 70% 需 42 次。
按每题约 100 轮、每轮一次调用，一组 105 次约 6000 万 tokens —— 有限资源下做不了。

**所以换判据**：稳定性要求单次成功率 ≥95%（p=0.8 时「连跑 5 次全过」只有 33%）。
平均分抬 20 个点仍然是「五次里四次不稳」。真正要消灭的是**每次都以同样方式失败**的模式，
而这类模式在账本里可以被确定性数出来。

## 195 次报错的构成与处理

统计口径：`bench_*` / `raw_*` 共 46 次现场运行（`replay_check.py` 可复算）。

| 类别 | 次数 | 新规则 | 结果 |
|---|---|---|---|
| 引用写成工具名（`@load.solids`，实际步骤叫 `all`） | 37 | 按「最近一次同名工具的步骤」解析 + 观察里写明正确写法 | 37 全部可解析 |
| 字段名猜错（`@all.items`，实际 `solids`） | 16（load 类） | 字段别名 `items ↔ solids ↔ entities` + 回报正式字段名 | 16 可别名 |
| 编号失效（`C017` 被 `parts` 拆走） | 87 | `parts` 返回 `removed`；报错点名「已拆成 C017.1/C017.2」 | 82 可说明 |
| 其他（参数形态、门控、审计） | 55 | 未覆盖 | — |

合计 135/195 = **69% 被接住或说明**；其中 53 次这一步直接成功，82 次仍需重发但不必再猜。

## 五条改动

1. **引用别名**（`core/runtime.py`）：`@load` → 最近一次 `load` 的步骤；`.items` → `.solids`。
   只在意图唯一时纠正，纠正会写进账本与观察，不悄悄改。
2. **拆件映射**（`tools/shape.py` + `Bus.missing_msg`）：`parts` 返回 `removed`；
   只有 1 个连通分量时**不换号**（拆了等于没拆，换号只会让后续引用失效）。
3. **产物去重**（`agent/loop.py`）：参数不同、返回一字不差 → 明说「与第 X 步相同」并入空转计数。
   只对结构类工具生效，测量类交给「有没有新数字」判定。
4. **重复不烧预算**：重复只读调用退还轮次（上限 6 次）；步骤号与轮次解耦（避免退还后 s12 重复）。
5. **收口窗口 + 草稿板**：剩余 15% 预算且已有测量时进 VERIFY，并把「已量到的数」按步骤摆出来。
   只搬移数字，不决定填哪个键。

## 回归证据（离线，不花 token）

- `python test_orch.py`：22 项全过（别名、字段、拆件、单件不换号、收口、草稿板）。
- `python selftest.py`：全过。
- 五道难题的录制回放仍全部 `OK`，账本链完整、答案键数不变：

| 回放 | 状态 | 轮 | 答案键数 |
|---|---|---|---|
| rp_T03 | OK | 39 | 28 |
| rp_T05 | OK | 34 | 10 |
| rp_T06 | OK | 44 | 11 |
| rp_T08 | OK | 41 | 18 |
| rp_T10 | OK | 35 | 26 |

能力上限没有被改动破坏；改动只作用在「本来会失败的机械动作」上。

## 明细

现场运行 46 次｜总轮次 2658｜报错 195

| 运行 | 状态 | 轮 | 报错 | A 引用名→已解析 | B 字段名→已别名 | C 拆件→已说明 |
|---|---|---|---|---|---|---|
| bench_T03_2 | BUDGET_EXHAUSTED | 100 | 36 | 1 | 0 | 34 |
|   A clu | @load → @all | | | | |
|   C grounded_all | C017 已拆为 ['C017.1'] | | | | |
|   C room_all | C017 已拆为 ['C017.1'] | | | | |
|   C all_now | C017 已拆为 ['C017.1'] | | | | |
|   C all_now | C017 已拆为 ['C017.1'] | | | | |
|   C all_now | C017 已拆为 ['C017.1'] | | | | |
|   C all_now | C017 已拆为 ['C017.1'] | | | | |
|   C all_now | C017 已拆为 ['C017.1'] | | | | |
|   C room_now | C017 已拆为 ['C017.1'] | | | | |
|   C all_now | C017 已拆为 ['C017.1'] | | | | |
|   C room_now | C017 已拆为 ['C017.1'] | | | | |
|   C all_now | C017 已拆为 ['C017.1'] | | | | |
|   C all_now | C017 已拆为 ['C017.1'] | | | | |
|   C room_now | C017 已拆为 ['C017.1'] | | | | |
|   C all_now | C017 已拆为 ['C017.1'] | | | | |
|   C all_now | C017 已拆为 ['C017.1'] | | | | |
|   C all_now | C017 已拆为 ['C017.1'] | | | | |
|   C all_now | C017 已拆为 ['C017.1'] | | | | |
|   C all_now | C017 已拆为 ['C017.1'] | | | | |
|   C all_now | C017 已拆为 ['C017.1'] | | | | |
|   C all_now | C017 已拆为 ['C017.1'] | | | | |
|   C room_now | C017 已拆为 ['C017.1'] | | | | |
|   C all_now | C017 已拆为 ['C017.1'] | | | | |
|   C all_now | C017 已拆为 ['C017.1'] | | | | |
|   C all_now2 | C017 已拆为 ['C017.1'] | | | | |
|   C clu_all | C017 已拆为 ['C017.1'] | | | | |
|   C all20_check | C017 已拆为 ['C017.1'] | | | | |
|   C exist_check | C017 已拆为 ['C017.1'] | | | | |
|   C exist_check2 | C017 已拆为 ['C017.1'] | | | | |
|   C exist_probe | C017 已拆为 ['C017.1'] | | | | |
|   C exist_probe | C017 已拆为 ['C017.1'] | | | | |
|   C exist_probe3 | C017 已拆为 ['C017.1'] | | | | |
|   C exist_probe4 | C017 已拆为 ['C017.1'] | | | | |
|   C exist_now | C017 已拆为 ['C017.1'] | | | | |
|   C chair_fp | C017 已拆为 ['C017.1'] | | | | |
| bench_T03_1 | STALLED | 60 | 37 | 1 | 0 | 32 |
|   A clu | @load → @all | | | | |
|   C grounded_thin | C017 已拆为 ['C017.1'] | | | | |
|   C grounded_thin | C017 已拆为 ['C017.1'] | | | | |
|   C room3 | C017 已拆为 ['C017.1'] | | | | |
|   C all2 | C017 已拆为 ['C017.1'] | | | | |
|   C all2 | C017 已拆为 ['C017.1'] | | | | |
|   C all2 | C017 已拆为 ['C017.1'] | | | | |
|   C all2 | C017 已拆为 ['C017.1'] | | | | |
|   C all2 | C017 已拆为 ['C017.1'] | | | | |
|   C room_now | C017 已拆为 ['C017.1'] | | | | |
|   C room_now | C017 已拆为 ['C017.1'] | | | | |
|   C room_now | C017 已拆为 ['C017.1'] | | | | |
|   C all_now | C017 已拆为 ['C017.1'] | | | | |
|   C all_now | C017 已拆为 ['C017.1'] | | | | |
|   C all_now | C017 已拆为 ['C017.1'] | | | | |
|   C room_now | C017 已拆为 ['C017.1'] | | | | |
|   C room_now2 | C017 已拆为 ['C017.1'] | | | | |
|   C room_now3 | C017 已拆为 ['C017.1'] | | | | |
|   C room_now4 | C017 已拆为 ['C017.1'] | | | | |
|   C thin | C017 已拆为 ['C017.1'] | | | | |
|   C room_now5 | C017 已拆为 ['C017.1'] | | | | |
|   C room_now6 | C017 已拆为 ['C017.1'] | | | | |
|   C thin | C017 已拆为 ['C017.1'] | | | | |
|   C room16 | C017 已拆为 ['C017.1'] | | | | |
|   C thin | C017 已拆为 ['C017.1'] | | | | |
|   C thin | C017 已拆为 ['C017.1'] | | | | |
|   C thin | C017 已拆为 ['C017.1'] | | | | |
|   C thin | C017 已拆为 ['C017.1'] | | | | |
|   C all_now | C017 已拆为 ['C017.1'] | | | | |
|   C thin | C017 已拆为 ['C017.1'] | | | | |
|   C thin | C017 已拆为 ['C017.1'] | | | | |
|   C room16 | C017 已拆为 ['C017.1'] | | | | |
|   C thin | C017 已拆为 ['C017.1'] | | | | |
| raw_T10_2 | INSUFFICIENT_EVIDENCE | 92 | 19 | 1 | 0 | 16 |
|   A grp | @load → @all | | | | |
|   C grounded | C000 已拆为 ['C000.1'] | | | | |
|   C cur | C000 已拆为 ['C000.1'] | | | | |
|   C cur | C000 已拆为 ['C000.1'] | | | | |
|   C cur | C000 已拆为 ['C000.1'] | | | | |
|   C cur | C000 已拆为 ['C000.1'] | | | | |
|   C cur | C000 已拆为 ['C000.1'] | | | | |
|   C cur | C000 已拆为 ['C000.1'] | | | | |
|   C cur | C000 已拆为 ['C000.1'] | | | | |
|   C cur | C000 已拆为 ['C000.1'] | | | | |
|   C cur | C000 已拆为 ['C000.1'] | | | | |
|   C plate | C000 已拆为 ['C000.1'] | | | | |
|   C base_bot2 | C003 已拆为 ['C003.1'] | | | | |
|   C cur9 | C003 已拆为 ['C003.1'] | | | | |
|   C cur10 | C000 已拆为 ['C000.1'] | | | | |
|   C cur11 | C000 已拆为 ['C000.1'] | | | | |
|   C cur12 | C000 已拆为 ['C000.1'] | | | | |
| bench_T05_1 | AUDIT_FAILED | 86 | 3 | 1 | 2 | 0 |
|   A clu3 | @load → @all | | | | |
|   B wall_left | @all_2.items.items → .solids | | | | |
|   B chair_bb2 | @all_2.items.items → .solids | | | | |
| bench_T10_1 | BUDGET_EXHAUSTED | 100 | 8 | 0 | 3 | 0 |
|   B allf2 | @all3.items.items → .solids | | | | |
|   B grounded_solid3 | @all3.items.items → .solids | | | | |
|   B allf5 | @all3.items.items → .solids | | | | |
|   B pair2 | @allf7.solids.solids 仍无解 | | | | |
|   B pair_final | @allf7.solids.solids 仍无解 | | | | |
|   B pin_pick | @pins.faces.faces 仍无解 | | | | |
| bench_T10_2 | BUDGET_EXHAUSTED | 100 | 6 | 1 | 2 | 0 |
|   A clu2 | @load → @all | | | | |
|   B plate3 | @all2.items.items → .solids | | | | |
|   B stack | @all2.items.items → .solids | | | | |
| raw_T03_1 | OK | 59 | 3 | 1 | 2 | 0 |
|   A clu2 | @load → @all | | | | |
|   B floor_fp | @all_2.items.items → .solids | | | | |
|   B cube_v | @all_2.items.items → .solids | | | | |
| bench_T05_2 | BUDGET_EXHAUSTED | 100 | 2 | 2 | 0 | 0 |
|   A clu | @load → @all | | | | |
|   A clu2 | @load → @all | | | | |
| bench_T06_2 | INSUFFICIENT_EVIDENCE | 99 | 5 | 2 | 0 | 0 |
|   A clu2 | @load → @all | | | | |
|   A room_cluster | @load → @all | | | | |
| bench_T07_2 | INSUFFICIENT_EVIDENCE | 79 | 6 | 2 | 0 | 0 |
|   A cube | @load → @all | | | | |
|   A ball_before | @load → @reload | | | | |
| raw_T02_1 | OK | 52 | 2 | 1 | 1 | 0 |
|   A clu3 | @load → @all | | | | |
|   B modelA_pair | @all_2.items.items → .solids | | | | |
| raw_T02_2 | OK | 24 | 2 | 1 | 1 | 0 |
|   A clu3 | @load → @all | | | | |
|   B cube | @all_2.items.items → .solids | | | | |
| raw_T04_2 | OK | 32 | 2 | 1 | 1 | 0 |
|   A pf | @load → @all | | | | |
|   B clu3 | @all_2.items.items → .solids | | | | |
| raw_T05_1 | AUDIT_FAILED | 79 | 6 | 0 | 2 | 0 |
|   B all_fp | @all_2.items.items → .solids | | | | |
|   B room_fp3 | @all_2.items.items → .solids | | | | |
|   B fam_ref | @fam3.items.items 仍无解 | | | | |
| raw_T06_2 | INSUFFICIENT_EVIDENCE | 100 | 7 | 2 | 0 | 0 |
|   A clu2 | @load → @all | | | | |
|   A thin_high | @load → @all | | | | |
| bench_CASE_C_1 | OK | 54 | 3 | 1 | 0 | 0 |
|   A clu2 | @load → @all | | | | |
| bench_CASE_C_2 | OK | 44 | 2 | 1 | 0 | 0 |
|   A clu | @load → @all | | | | |
| bench_T01_1_2 | OK | 42 | 2 | 1 | 0 | 0 |
|   A clu2 | @load → @all | | | | |
| bench_T01_3_2 | OK | 18 | 1 | 1 | 0 | 0 |
|   A pf | @load → @all | | | | |
| bench_T02_2 | AUDIT_FAILED | 95 | 2 | 0 | 1 | 0 |
|   B grp5 | @all_2.items.items → .solids | | | | |
|   B p025 | @p_d025.items.items 仍无解 | | | | |
| bench_T04_1 | OK | 53 | 1 | 1 | 0 | 0 |
|   A clu3 | @load → @all | | | | |
| bench_T06_1 | BUDGET_EXHAUSTED | 100 | 6 | 1 | 0 | 0 |
|   A pf | @load → @all | | | | |
| bench_T07_1 | INSUFFICIENT_EVIDENCE | 39 | 2 | 1 | 0 | 0 |
|   A grp3 | @load → @all | | | | |
| bench_T08_2 | OK | 41 | 4 | 1 | 0 | 0 |
|   A clu | @load → @all | | | | |
| raw_T01_1_1 | OK | 26 | 1 | 1 | 0 | 0 |
|   A clu | @load → @all | | | | |
| raw_T01_1_2 | OK | 33 | 2 | 1 | 0 | 0 |
|   A clu | @load → @all | | | | |
| raw_T01_3_1 | OK | 15 | 1 | 1 | 0 | 0 |
|   A clu | @load → @all | | | | |
| raw_T01_3_2 | OK | 15 | 1 | 1 | 0 | 0 |
|   A clu | @load → @all | | | | |
| raw_T03_2 | OK | 59 | 1 | 1 | 0 | 0 |
|   A grp | @load → @all | | | | |
| raw_T04_1 | OK | 22 | 1 | 0 | 1 | 0 |
|   B furn2 | @all.items.items → .solids | | | | |
| raw_T05_2 | INSUFFICIENT_EVIDENCE | 53 | 2 | 1 | 0 | 0 |
|   A clu | @load → @all | | | | |
|   B all_edit | @all_now.solids.solids 仍无解 | | | | |
| raw_T06_1 | INSUFFICIENT_EVIDENCE | 99 | 4 | 1 | 0 | 0 |
|   A pf | @load → @all | | | | |
| raw_T07_1 | OK | 36 | 1 | 1 | 0 | 0 |
|   A clu2 | @load → @all | | | | |
| raw_T07_2 | OK | 39 | 1 | 1 | 0 | 0 |
|   A clu2 | @load → @all | | | | |
| raw_T08_1 | OK | 46 | 1 | 1 | 0 | 0 |
|   A clu | @load → @all | | | | |
| raw_T08_2 | OK | 63 | 2 | 1 | 0 | 0 |
|   A clu3 | @load → @all | | | | |
| raw_T10_1 | STALLED | 89 | 1 | 1 | 0 | 0 |
|   A clu | @load → @all | | | | |
| bench_T01_1_1 | OK | 40 | 2 | 0 | 0 | 0 |
| bench_T08_1 | INSUFFICIENT_EVIDENCE | 56 | 1 | 0 | 0 | 0 |
| bench_T09_1 | OK | 67 | 1 | 0 | 0 | 0 |
| raw_T09_1 | OK | 44 | 2 | 0 | 0 | 0 |
| raw_T09_2 | OK | 49 | 3 | 0 | 0 | 0 |

合计：报错 195｜引用名可自动解析 37｜字段名可自动别名 16｜拆件失效可说明 82｜合计能接住 135（69%）
仍无解：A 0｜C(非拆件) 4
