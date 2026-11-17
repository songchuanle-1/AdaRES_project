# Research Pipeline Report - AdaRES_project
快照时间：2026-10-06T17:48:18.490074+00:00；生成报告时快照年龄：105 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-07 01:48:18
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
================================================== 2026-10-07 01:48 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y30_query_trajectory_res] Best=0.4865 10/30 no_improve=0/30 gpu=1 ? 序列: 0.4558, 0.4682, 0.4752, 0.4757, 0.4865

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y30 | configs_song/model_paper5/adares_y30_query_trajectory_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 1；准入预算 0。

## 4. This Tick's Actions
- 快照UTC2026-10-06 17:48:18.490074，即本地2026-10-07 01:48:18.490074+08:00；event原因仅research_artifact_changed。核实实际线程与owner一致、foreground、closure启用、GPU1唯一授权及后台调度Disabled。begin仅一次并先读model view；活跃1/运行1/容量1/预算0/到期候选0/新终态0，未重复preflight。
- 复用begin监控及脚本0-3事实表：Y30为running/GPU1，日志10/30轮、Best=0.4865、no_improve=0/30，近期序列0.4558/0.4682/0.4752/0.4757/0.4865。记录是早期训练进度，不证明固定30轮已完成或最终效益；Acc50/spmIoU、当前设备利用率/显存/独立PID核验未知，不以历史设备观测替代。
- 容量已满、无到期候选或可操作故障，未重复进度探针、GPU诊断、准入门槛、封闭路线或全量评测，未新增候选/证据引用/准入/重启训练。仅更新119行七段记忆的三个操作段落及本轮动作/决策；research_work.json保持SHA256 2007494a46450b44d6ef920a569f14f8b06156a77685fd98f142f89707fef068，旧科学证据、失败/修复及不利对照不变。
- 科学状态training，最终效益/时序必要性/Pareto/新颖性未知。W2历史0.5100距目标0.515差0.005，Phase1未达/Phase2未授权。GPU0未查询或使用，旧Y23恢复授权已履行不重放；2026-10-06 20:51的独立占用记录为历史，不是当前占用或独立峰值。
- 本轮实际产出为监控结果整合、操作记忆更新与完整六段报告，无新科学实验或结论。报告生成时本轮收据未写入，窗口/用量未知；之后以FINALIZED收据为准，只统计本地begin至收据的局部记录，排除此前读取、迟到记录和最终回复。缓存包含在输入、reasoning包含在输出，不重复相加、不当账单。执行完成仅由finish成功和独立确认租约释放判断，释放后不回写报告。

## 5. Decisions Made
- 无新增科学决策。继续同一Y30/GPU1固定30轮，原冻结full9508表达式/141场景、Y30/W2/B1各两遍、整场景bootstrap10000/seed1999、TRAIN切分1487/3325、分组伤害及训练后串行成本计划不变，真实终态/故障/到期工作出现后才处理下一步。
- 常规训练仍在推进，不制造变体或重复验证填满tick，不把早期日志提升称为方法有效；保持GPU0用户保留、目标/门槛/调度不变，不操作其他项目或恢复后台。
- 完整六段报告复用脚本事实表；保存自动化的下次触发时间本轮未读取，未知。finish若失败只具体修复，不能诚实完成时保留证据并abort原token后核实释放，不将报告生成等同于tick完成。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
