# Research Pipeline Report - AdaRES_project
快照时间：2026-10-08T10:52:24.840506+00:00；生成报告时快照年龄：115 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-08 18:52:24
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
================================================== 2026-10-08 18:52 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y33_token_attention_res] 1/30 尚无验证 - gpu=1

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
- 本轮快照为北京时间2026-10-08 18:52:24.840506，event模式，租约96bfaa1855384aa7ba51b4cdd27e5e02，真实截止19:52:24.840506。核实当前任务归属与Windows项目调度Disabled；本轮只执行AdaRES_project，未操作其他项目、调度或后台。
- begin已一次完成监控与目标检查，返回码均为0；首先读取紧凑视图，然后读取短记忆、生成报告和同一Y33候选。容量1、活跃1、运行1、等待0、准入预算0、到期候选0、新终态0；变化来自前轮研究记录及Y33入队启动后的快照变化。
- 脚本监控显示Y33仍在server-2 GPU1训练，摘要为1/30且尚无验证结果。规定事实表中的epoch与指标不可得，保持未知；本轮未额外测量训练步数、PID、利用率、显存或进程，不把18:26的历史启动证据冒充当前实时证据，也不据未知值推断空闲。GPU0未查询或使用。
- 仅更新七段短记忆的操作时间线并保存本轮中文动作与决策，复用脚本事实表。未重复preflight、未重跑等价诊断、未修改研究台账或冻结设计、未新增候选/准入、未重启训练或重放旧Y23恢复。
- 前轮cde054a6收据已FINALIZED、postflight PASS、context_committed=true；前轮finish COMPLETE及独立锁消失证据成立。本轮执行状态在写入时仍待finish，不预称完成；科学状态为training，疗效、必要性、创新性与训练后成本优势仍未知。
- 本轮收据尚未生成，用量窗口、输入/输出和实际变更计数均未知；finish后以真实收据与锁状态为准。收据仅统计本地begin至生成收据的局部窗口，排除更早读取、晚到记录与最终答复；缓存已含在输入、推理已含在输出，不重复相加或视为账单。实际产出为操作记忆更新及本轮六段报告，不计作新的科学证据。

## 5. Decisions Made
- 无新科学决策。沿用同一Y33已准入的固定30epoch、seed1999、原生128queries及单次标准argmax训练，效果未定；资源已占满且无到期诊断，不进行额外准入或无信息增益复测。
- 继续按原冻结设计等待同一Y33的实际验证、故障或终态；到期后开展既定全量9508表达/141场景、Y33/W2/B1各两次及整场景bootstrap10000、训练集原始点数分组和训练后配对成本评估。目标0.515、W2历史champion0.5100及所有门槛保持不变。
- 遵守GPU0保留与server-2仅GPU1的授权。例行在训状态无需用户操作，本轮不发主动进展提醒；报告与证据保存在项目内，finish通过后独立核实释放，不作租约释放后的文件写入。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
