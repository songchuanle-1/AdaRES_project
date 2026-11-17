# Research Pipeline Report - AdaRES_project
快照时间：2026-10-18T19:49:22.312626+00:00；生成报告时快照年龄：827 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-19 03:49:21
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
================================================== 2026-10-19 03:49 : AdaRES_project ==================================================

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
- 数据采集于上海2026-10-19 03:49:22，模式design；规范begin已执行一次monitor/goal且均返回0，运行/认领/等待均0，无新终态。历史CSV最佳Y36 0.5110、B1 0.5020，历史提升0.0090，目标0.515还差0.0040；本轮没有新训练指标。
- 本轮绑定due仅Y52，Y54在快照时未到期；储备2/2、准备预算0、准入预算2。03:55:11只对Y52执行一次真实限定许可/七个已测文件/HEAD与索引复查：许可文件不存在，七文件哈希一致且均未跟踪、索引空，HEAD仍26ee1cec57c53ee2f94d6969d0bb637565b6ed59。维持blocked/human_authorization、inconclusive、模型修订0，retry更新为上海07:55:11.701865，wake_paths不变。
- 依赖证据pipeline/analyses/y52_authorization_recheck_edb00de1a8d648d0bb2b6df206ce8a95.json，SHA256 09ec61a65ee997ceaf65d3646b374535820f9403a09cba785562dace3cbf41d5；03:59:14独立核实证据、原租约与context/pending完整。仅Y52证据/retry和design_cycle更新，其他106候选及历史负证据不变。未重复原生/资源门禁、源码路线审查或既有冷却回归。
- 科学状态未变：Y52原indicator前提已有反证，辅助score MSE/共享query仅有限原生证据；Y54 fine-support有效性mask有既有因果/反向/固定输入资源证据，训练收益、新颖性及全局Pareto未知。Y54许可缺失仍仅上海2026-10-18 23:53:24缓存，本轮当前变化未知；不因壁钟越过其retry扩展本轮绑定清单。GPU0/1白名单不等于实时空闲，占用未知，未额外探测或使用GPU。
- 实际产出为1个新的到期依赖证据文件、1项候选依赖续做及本轮notes/七段短记忆/完整六段审计；没有新科学候选、科学测试、模型改动、修订、提交、准入或训练启动。收据用量只含上海03:49:21 begin至receipt的本地记录窗口，当前计数待收据、未知，不含更早读取/晚到记录/最终回复；不作账单，缓存包含在input、推理包含在output，不重复相加。
- 本轮只拟基于真实新增依赖证据执行一次规范finish，尚未执行postflight、不预写PASS或完成。完成必须以明确PASS、FINALIZED、context已提交及原租约/pending释放独立核验为准；失败则保留证据并规范abort，不重复无变化finish，不放宽准入或硬撑deadline。

## 5. Decisions Made
- 延续20261017-132812_question.md，Y52/Y54各需七个已测文件的明确限定提交许可或匹配commit。Y52本轮实查仍缺，不能自授、扩大Y50许可或把用户询问当同意；明确解锁后仍须当前goal/源/六门禁/commit/回执和实时GPU核验。
- design_cycle用本轮Y52真实依赖证据记录external_resource_blocked；实际blocker_kind为human_authorization，不是GPU短缺或机制失败。Y52 retry为上海2026-10-19 07:55:11.701865；Y54保持03:53:24.443394，只有后续新快照真正到期或真实wake变化才处理。retry不是已核实调度，不另建候选或扩大绑定due清单。
- 延续20261017-145336_question.md的既有共享冷却故障，03:59:14只确认源码哈希未变，未重跑或修复；本轮若finish通过仅表示依赖检查与审计闭环，不证明冷却已修复或方法有效。不改历史closed、共享运行时、目标/门槛/deadline、其他项目或调度，释放后不写项目。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
