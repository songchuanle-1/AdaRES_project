# Research Pipeline Report - AdaRES_project
快照时间：2026-10-08T12:48:11.066691+00:00；生成报告时快照年龄：87 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-08 20:48:10
================================================================================
  Target: AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515.

## Phase 1: Metric Target
  Target: mIoU ≥ 0.5150
  [✗] Champion: AdaRES-W2 — mIoU=0.5100 (gap to target: +0.0050)
       Epochs: 30/30, State: done_complete
  Phase 1: NOT YET — continue exploring ✗

## Baseline
  Baseline: AdaRES-B1 — mIoU=0.5020
  Champion vs Baseline: γ = +0.0080

## Phase 2: Story Verification
  (等待 Phase 1 达标后启动)

## Overall
  Phase 1: champion=0.5100, target=0.5150, gap=+0.0050
================================================================================
```

## 1. GPU Overview
- 仅列授权卡；未采集利用率、显存或进程时标为未知，不推断空闲。
| Server | GPU | Util | Memory | Process |
|---|---|---|---|---|
| server-2 | 1 | 未知 | 未知 | 未知 |
- 可用 GPU：未做实时占用确认，不能据此分配。
- 监控摘要（begin 已执行，不再重复探测）：
```text
================================================== 2026-10-08 20:48 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y33_token_attention_res] Best=0.4419 4/30 no_improve=0/30 gpu=1 ? 序列: 0.3238, 0.3786, 0.4419

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y33 | configs_song/model_paper5/adares_y33_token_attention_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 1；准入预算 0。

## 4. This Tick's Actions
- 本轮仅执行AdaRES_project，核实实际任务ID与前台归属一致、Windows项目调度Disabled、closure_policy明确启用及前轮锁已消失；目标、冻结协议与GPU0保留授权不变，未操作其他项目、调度或后台。
- begin仅执行一次，首先读取紧凑视图。快照北京时间2026-10-08 20:48:11.066691，event模式，真实截止21:48:11.066691；monitor/goal返回码均0。容量1、活跃1、运行1、认领0、等待0、准入预算0、到期0、新终态0，原因仅research_artifact_changed。
- 监控摘要显示同一Y33在server-2 GPU1到4/30，早期Best mIoU=0.4419，no_improve=0/30，序列0.3238、0.3786、0.4419。该摘要为四舍五入训练日志，不是冻结全量评估或最终收益；规定事实表中的指标字段仍不可得，保持未知。当前利用率、显存与进程未知，未推断空闲；GPU0未查询或使用。
- 读取本轮报告与短记忆一次，仅更新七段短记忆的操作时间线和早期监控结果，补充中文动作/决策并复用六段事实报告。没有重复preflight、等价诊断、历史审查、额外候选或准入；没有更改研究台账、模型、冻结设计、训练设置或重启训练。
- 报告写入时本轮执行状态仍待finish，科学状态training；效果、必要性、创新性与训练后成本优势未知。前轮459bc48c已finish COMPLETE、收据FINALIZED/PASS及独立锁消失验证，不重放旧启动或Y23恢复。
- 本轮收据尚未生成，用量窗口/输入输出/变更计数未知；finish后核对收据与释放状态，不作释放后的文件写入。用量仅本地begin至收据的局部窗口，排除较早读取、晚到记录及最终答复；缓存已含输入、推理已含输出，不重复累加或当账单。实际产出为记忆更新与本轮六段报告，不视作新增科学证据。

## 5. Decisions Made
- 无新科学决策。继续同一Y33固定30epoch/seed1999/原生128queries/标准单次argmax；早期0.4419不用于调参、早停、晋升champion或修改0.515目标及门槛。
- 授权单任务容量已达活跃目标，预算0且无到期候选，保持现有训练，不重复无信息增益测试。出现实际故障或终态后再按既定冻结全量、整场景bootstrap、训练来源分组与配对成本方案处理。
- 本轮为例行在训进度，无需用户操作或主动提醒。沿用GPU0保留、server-2仅GPU1和其他隔离规则；完成真实postflight并独立核实释放后安静收尾。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
