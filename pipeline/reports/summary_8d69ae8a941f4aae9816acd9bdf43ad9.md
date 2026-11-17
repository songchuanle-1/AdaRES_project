# Research Pipeline Report - AdaRES_project
快照时间：2026-10-05T17:51:03.536802+00:00；生成报告时快照年龄：173 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-06 01:51:03
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
================================================== 2026-10-06 01:51 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y28_mask_readback_res] Best=0.4869 9/30 no_improve=0/30 gpu=1 ? 序列: 0.4525, 0.4606, 0.4684, 0.4631, 0.4869

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y28 | configs_song/model_paper5/adares_y28_mask_readback_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 1；准入预算 0。

## 4. This Tick's Actions
- 仅检查 AdaRES_project。已核实实际当前任务与 execution_mode.json 归属一致、前台模式、Windows Pipeline-AdaRES_project 禁用、closure_policy 显式启用及 server-2 仅授权 GPU1；未修改调度或其他项目。begin 仅执行一次，先读精简模型视图；快照为北京时间2026-10-06 01:51:03.536802，真实截止为02:51:03.536802。
- 复用 begin 已执行的监控和目标检查：容量1、活跃1、运行1、等待0、准入预算0、到期候选0、新终态0。唯一事件原因为 research_artifact_changed，无本轮列出的终态、崩溃或启动验证故障；未重复 preflight、读取完整快照或重查关闭方向。
- 规范监控显示 Y28 当前第9/30轮，日志 Best mIoU=0.4869，no_improve=0/30，最近序列0.4525、0.4606、0.4684、0.4631、0.4869。轮次标记不等于已完成9轮或固定30轮结束；科学状态为 training，最终效果、必要性和训练后Pareto仍未知。当前利用率、显存、计算PID、Acc@50和spmIoU未采集，保留未知；2026-10-05 20:50独立进度证据是历史值，不用于本轮资源分配。
- 本轮仅更新七段记忆中的当前快照、运行摘要和Last Tick，并补充本报告动作与决策；研究账本SHA256仍为e90e6e986791852c903619200d1619a7a180ee026567347c1677545345603e4a。无新候选、候选阶段变化、新科学证据引用或训练启动；不额外进行无信息增益探针、不提前停止Y28、不查询或使用GPU0。
- 写入本说明时 report/finish及当前收据尚待执行，不能预称COMPLETED。当前用量窗口和计数未知，最终以finish收据和独立租约释放检查为准；收据仅覆盖本地begin至记录时段，排除前置读取、迟到记录及最终回复。缓存属于输入、推理属于输出，均不重复相加，也不是账单；实际产出为运行状态核对及记忆/报告维护，不是新增科学结论。

## 5. Decisions Made
- 本轮没有新科学决策。Y28已按真实准入占满目标容量，继续原固定30轮、seed1999、min_epochs30/patience30；不追加准入、重启、凑变体或修改损失权重、温度、阈值、特征、子群和epoch。
- 目标仍为ScanRefer mIoU>=0.515，历史W2冠军0.5100、差距0.0050，Phase1未达标、Phase2尚未开始。早期日志Best不用于冠军晋升或因果有效性判断；训练完成后执行已冻结Y28/W2/B1各两遍全量9508表达/141场景、整场景bootstrap及训练后资源协议。
- 保留既有不利对照、原始故障和全部证据，不重写历史、不重放旧Y23恢复逻辑。完成规范报告后仅执行本租约finish并独立核实释放；若无法诚实通过postflight则保留证据并用原token明确abort，不放宽检查。下一次调度时间未读取，记未知。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
