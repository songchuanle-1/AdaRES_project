# Research Pipeline Report - AdaRES_project
快照时间：2026-10-10T12:50:36.237033+00:00；生成报告时快照年龄：112 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-10 20:50:35
================================================================================
  Target: AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515.

## Phase 1: Metric Target
  Target: mIoU ≥ 0.5150
  [✗] Champion: AdaRES-Y36 — mIoU=0.5110 (gap to target: +0.0040)
       Epochs: ?, State: done_complete
  Phase 1: NOT YET — continue exploring ✗

## Baseline
  Baseline: AdaRES-B1 — mIoU=0.5020
  Champion vs Baseline: γ = +0.0090

## Phase 2: Story Verification
  (等待 Phase 1 达标后启动)

## Overall
  Phase 1: champion=0.5110, target=0.5150, gap=+0.0040
================================================================================
```

## 1. GPU Overview
- 仅列授权卡；未采集利用率、显存或进程时标为未知，不推断空闲。
| Server | GPU | Util | Memory | Process |
|---|---|---|---|---|
| server-2 | 0 | 未知 | 未知 | 未知 |
| server-2 | 1 | 未知 | 未知 | 未知 |
- 可用 GPU：未做实时占用确认，不能据此分配。
- 监控摘要（begin 已执行，不再重复探测）：
```text
================================================== 2026-10-10 20:50 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y37_pair_distribution_res] Best=0.4926 11/30 no_improve=1/30 gpu=0 ? 序列: 0.4623, 0.4697, 0.4747, 0.4926, 0.4920
  [adares_y39_point_distribution_res] 1/30 尚无验证 - gpu=1

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=2
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y37 | configs_song/model_paper5/adares_y37_pair_distribution_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | running |
| AdaRES-Y39 | configs_song/model_paper5/adares_y39_point_distribution_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 2；准入预算 0。

## 4. This Tick's Actions
- 本轮仅操作 AdaRES_project；当前环境任务 ID 与前台配置一致，GPU0/1 授权及冻结协议不变，Windows Pipeline 保持 Disabled。begin 仅执行一次，先读紧凑视图，不重复监控或目标 preflight。
- 快照为 2026-10-10 20:50:36.237033+08，真实截止为 21:50:36.237033+08；模式 event，变化为活跃队列、研究工件和研究状态。紧凑视图新终态 0、到期候选 0、运行 2、等待 0、目标容量 2、准入预算 0。
- 原始监控显示 Y37 在 GPU0 为 11/30，舍入最佳 mIoU 0.4926，无改善 1/30；Y39 在 GPU1 为 1/30，尚无验证。第 2 段结构化表缺失的指标仍为未知，不用原始日志猜补；当前 PID、GPU 利用率和显存未知，20:22:07 的历史启动证据不当作实时占用。
- 本轮确认前轮 Y39 准入及研究工件已进入当前前台快照，保持两项原有固定 30 轮训练，不重新启动、不额外派发、不重复已关闭候选评审或 GPU 测试。更新七段短记忆；仅补报告第 4/5 段，保留脚本事实表。
- 租约身份读取意外包含了历史字段；未据此重审闭合方向或改写历史证据，后续租约核实仅输出必要字段。本轮没有新增诊断、候选准入或训练启动。
- 执行状态：已获得本轮租约，finish 尚未执行，不能称完成；将以 CLOSURE_POSTFLIGHT_PASS、COMPLETE 和独立锁释放核实为准。科学状态：Y37/Y39 为 training，训练后的有效性、必要性、无伤害和 Pareto 结论未知。
- 第 4 段用量：本轮收据窗口从 begin 到收据记录；收据尚未生成，当前调用数、输入、缓存、输出、推理和实际产出统计未知。预期产出为监控确认、短记忆和六段报告，没有新增实验；最终统计须读取真实收据。缓存属于输入，推理属于输出，不重复相加，不累计前轮，不视为账单；释放后不改写项目报告。

## 5. Decisions Made
- 本轮无新增科学决策。容量已达到 2/2，且无终态或到期候选，按 event 合同收尾；不虚构资源故障、独立方向耗尽、冷却或测试通过。
- 目标仍未达到 0.515；CSV 的 Y36 舍入 0.5110 仅为自动数值冠军，不构成科学晋级。保留 W2 科学比较基准及 Y36 complete_inconclusive，Phase 2 未获授权。
- Y37/Y39 完成后才执行各自已冻结的全量重复评估、场景 bootstrap、子组伤害和训练后成本协议；出现真实崩溃、终态或到期候选时再处理，不盲目恢复历史任务。
- 不操作 AdaQ3D_project、其他项目、调度或后台，不新建任务、不嵌套 codex exec。下一调度时间未知，不猜测为一小时后；本轮若 finish 失败将保留证据、按真实原因 abort 并核实释放。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
