# Research Pipeline Report - AdaRES_project
快照时间：2026-10-09T01:28:14.208477+00:00；生成报告时快照年龄：102 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-09 09:28:13
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
| server-2 | 0 | 未知 | 未知 | 未知 |
| server-2 | 1 | 未知 | 未知 | 未知 |
- 可用 GPU：未做实时占用确认，不能据此分配。
- 监控摘要（begin 已执行，不再重复探测）：
```text
================================================== 2026-10-09 09:27 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y33_token_attention_res] Best=0.5051 23/30 no_improve=2/30 gpu=1 ? 序列: 0.4999, 0.5026, 0.5051, 0.5012, 0.5020
  [adares_y35_language_evidence_res] 1/30 尚无验证 - gpu=0

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=2
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y33 | configs_song/model_paper5/adares_y33_token_attention_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |
| AdaRES-Y35 | configs_song/model_paper5/adares_y35_language_evidence_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 2；准入预算 0。

## 4. This Tick's Actions
- AdaRES_project本轮快照北京时间2026-10-09 09:28:14.208477，mode=event，原截止10:28:14.208477未改；核实实际任务环境ID与配置owner01a07e32-78bf-7670-925b-370911ea48db一致。最新授权server2 GPU0/GPU1，各卡1独立任务，server6禁用；closure_policy启用、Pipeline-AdaRES_project仍Disabled，不操作其他项目或调度。
- begin一次且compactview先读，monitorRC0/goalRC0。变化原因是上一轮实际启动Y35、队列和研究产物更新；当前运行2/认领0/等待0/容量目标2/新准入预算0，到期候选0、新终态0。没有新增故障、训练终态或需要重试的候选；不重复preflight、诊断或历史审查。
- 复用脚本0-3事实表：Y33/GPU1、Y35/GPU0均running，表中未采集的GPU利用率/显存及指标仍未知。原始monitor另记录Y33为23/30、roundedbest.5051、no_improve2/30；Y35为1/30、尚无验证，不能补零或作为最终全量指标。科学状态均training，效益/必要性/新颖性/训练后Pareto未知。
- 承接上一轮已验证启动和完成记录：Y35原PID108331/原生30轮、Y33原PID64797/GPU1和源码不变；上一轮a8d0c77d具体哈希修复后finishPASS/COMPLETE，收据FINALIZED/contextcommitted，09:25:56独立核实前台锁不存在。首次失败归档50498dc8和修复证据原样保留，不把上一轮产出重新计成本轮。
- 本轮仅更新七段memory和完整六段快照报告，保留冻结协议与旧证据；无新候选、准入或启动。当前finish及本轮收据窗口/用量/产出计数待核实，缺失记未知；仅实际finish成功并独立核实释放后报告COMPLETED。缓存已包含在输入、推理已包含在输出，不重复相加或当账单。

## 5. Decisions Made
- 本轮没有新的科研决策或改路；延续已冻结的Y33/Y35训练，不重复启动、移动、重启或提前结束，也不创建无变化决策日志。
- 容量已满且准入预算0，不新增候选填位或放宽门槛。等待实际终态/故障/到期动作，再按各自冻结全量双次评估、场景聚类置信区间、子组伤害及成本协议判断。
- 保持mIoU目标.515、releasedbackbone、seed1999、fixed30、native128和单argmax；不验证集调参，区间跨0仍是不确定而非等价或无伤害。下一触发时间未读保存调度，未知，不承诺当前时间加一小时。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
