# Research Pipeline Report - AdaRES_project
快照时间：2026-10-18T22:49:50.932815+00:00；生成报告时快照年龄：320 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-19 06:49:50
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
================================================== 2026-10-19 06:49 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  无训练进程

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2:
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| 无 | 未知 | 未知 | 未知 | 未知 | 未知 | 未知 | 未知 | 未知 | 快照内无运行任务 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 0（运行 0 / 认领 0 / 等待 0）；目标 2；准入预算 2。
- 研究储备 2 / 2；储备准备预算 0，不增加训练并发或启动授权。
- 候选累计记录 107：complete=14 / rejected=79 / closed=12 / blocked=2。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 3；收据文件 30（含修订或重试，不代表成功率）。

## 4. This Tick's Actions
- 快照采集于上海2026-10-19 06:49:50；规范begin已执行一次monitor/goal，均返回0。模式design，运行/认领/等待均0，无新终态。历史CSV最佳Y36 0.5110、B1 0.5020，历史提升0.0090，距目标0.515还差0.0040；本轮没有训练指标提升。
- 绑定due0、shown为空、blocked_not_due2，储备2/2、准备预算0却仍required bootstrap。06:51:49只读核验原token/thread、context/pending完整，账本SHA仍2f133583bdd8f924f822be7f42743d5249b9163dc530c54cba3abd9ca1e33d63，共享foreground/closure源码与既有故障证据一致。没有重复依赖复查、原生/资源门禁或冷却回归，不改账本/design_cycle/retry/wake。
- Y52限定许可缺失仅有上海03:55:11缓存、Y54仅04:51:34缓存，本轮未到期不重查，当前授权变化均未知。保留原blocked/human_authorization、inconclusive、修订0及全部证据。GPU0/1实时占用未知，白名单和无项目训练都不能证明空闲；未额外探测或使用GPU。
- 科学状态未变：Y52原indicator前提反证、辅助score MSE/共享query有限原生证据，Y54 fine-support有效性mask既有因果/反向/固定输入资源证据均保留；训练收益、新颖性及全局Pareto未知。Y53/Y55反证和Y50实现拒绝不改。本轮科学测试、依赖复查、新候选、候选阶段/证据变化、模型改动、修订、提交、准入及训练启动均0。
- 实际产出仅规范状态记录、notes、七段短记忆及完整六段审计，事实表复用begin。收据用量仅上海06:49:49 begin至receipt的本地记录窗口，当前计数待收据、未知，不含更早读取/晚到记录/最终回复；缓存已含input、推理已含output，不重复相加、不作账单。
- 本轮没有满足required bootstrap的候选进展；旧CLOSURE_INCOMPLETE: Capacity-gap design made no durable candidate progress; create a stable diagnosis candidate and run its next test before stopping仅引用既有检查，不冒充本轮新执行。本轮不调用已知无效finish/postflight，拟规范abort保留INCOMPLETE，并独立核实context未提交及原租约/pending释放，不预写完成。

## 5. Decisions Made
- 延续20261017-145336_question.md：共享冷却将12条历史closed列入pending，既有缺陷源码未变。不重跑同义回归、不自动修共享代码、改历史/目标/门槛/deadline/调度或凑占位候选；留诚实审计后直接abort，不把状态记录包装成研究完成。
- 延续20261017-132812_question.md：Y52/Y54各需自身七个已测文件的明确限定提交许可或匹配commit，不能自授或扩大Y50许可。本轮未复查当前许可；明确解锁后仍需当前goal/源/六门禁/commit/回执及实时GPU核验，不自行入队或启动。
- 保持Y52 retry上海2026-10-19 07:55:11.701865和Y54 08:51:34.477821，均晚于本轮07:49:50 deadline；这些是复查条件，不是已核实调度。后续新快照真正到期、真实wake/明确授权变化或获准的限定共享修复才续做，不扩绑定清单、不重复预检、不等待凑截止，释放后不写项目。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
