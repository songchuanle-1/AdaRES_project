# Research Pipeline Report - AdaRES_project
快照时间：2026-10-18T17:54:38.215184+00:00；生成报告时快照年龄：481 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-19 01:54:37
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
================================================== 2026-10-19 01:54 : AdaRES_project ==================================================

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
- 本轮快照采集于上海2026-10-19 01:54:38；规范begin已执行一次monitor及goal，均返回0。模式design，运行/认领/等待均0，无新终态；目标mIoU 0.515，历史CSV最佳Y36 0.5110、B1 0.5020，历史提升0.0090，目标差0.0040；不是本轮训练收益。
- 绑定研究清单due0、shown为空、blocked_not_due2；储备2/2、准备预算0，但仍要求bootstrap。上海01:58:14只读核实当前token/thread/deadline及context/pending哈希匹配；research_work账本SHA仍b4ea9876d247a804ca334326dc5fa7723605b7a9de2247c364b6c4011ba1b16f，共享foreground/closure源码与既有故障证据一致。未重复依赖复查、原生门禁、资源探测或已知回归。
- Y52许可缺失仅有上海2026-10-18 22:51:58缓存，Y54仅23:53:24缓存，本轮未到期不重查，当前许可变化未知；保持原blocked/human_authorization、修订0、证据及retry/wake_paths。未把用户询问或Y50既有许可扩为Y52/Y54许可。GPU0/1在授权白名单内，但实时利用率/显存/外来作业占用未知，不能据此分配。
- 科学状态不变：Y52原indicator前提已反证，剩余辅助score MSE/共享query仅有限原生证据；Y54 fine-support有效性mask有既有因果、反向及固定输入资源证据，训练收益/新颖性/全局Pareto仍未知。Y53/Y55反证与Y50实现拒绝全部保留；本轮无新科学结论、候选、修订、提交、准入或训练启动。
- 实际产出仅本轮规范状态记录、七段短记忆、notes与完整六段审计报告；第0-3段及GPU/运行/队列表复用begin事实。收据用量窗口仅上海2026-10-19 01:54:37的begin至receipt本地记录，不含更早读取、晚到记录和最终回复；当前计数待收据、未知，不当账单，缓存是input子集且推理是output子集，不重复相加。
- 本轮没有新增候选进展，不满足bootstrap闭环；旧CLOSURE_INCOMPLETE: Capacity-gap design made no durable candidate progress; create a stable diagnosis candidate and run its next test before stopping仅引用既有失败，不冒充本轮执行结果。本轮未调用finish或postflight，不预写PASS/完成/已释放；留档后拟规范abort，并独立核实INCOMPLETE、context未提交及原租约/pending释放。

## 5. Decisions Made
- 延续pipeline/decisions/20261017-145336_question.md的既有故障处置：历史12条closed进入pending阻断blocked-only冷却；本轮只确认源哈希未变，不重复同义测试，不自动修共享运行时、改历史账本/目标/门槛/deadline/调度或制造占位候选。
- 延续pipeline/decisions/20261017-132812_question.md：Y52/Y54待已测文件限定提交许可或匹配commit；本轮未复查许可，不能自授、入队或启动。授权/匹配commit真正到达后仍需当前goal、源、六门禁、回执及实时GPU核验。
- 保留Y52 retry上海2026-10-19 02:51:58.850032和Y54 03:53:24.443394；它们是重查条件，不是已核实调度时间。不等待到期扩展本轮due清单、不重做preflight；下一新快照出现真正到期项、真实wake/明确许可变化或获准的限定共享修复才继续对应动作。
- 按契约诚实保留本轮INCOMPLETE并直接abort，不重复已知无效finish，不硬撑至02:54:38 deadline，不把状态审计说成科学进展。完成状态以命令及收据/独立释放核验为准；释放后不写项目。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
