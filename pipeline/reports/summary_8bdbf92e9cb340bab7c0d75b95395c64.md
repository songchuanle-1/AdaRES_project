# Research Pipeline Report - AdaRES_project
快照时间：2026-10-06T12:51:12.537840+00:00；生成报告时快照年龄：165 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-06 20:51:12
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
================================================== 2026-10-06 20:51 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y30_query_trajectory_res] Best=0.3383 2/30 no_improve=0/30 gpu=1 ?

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
- 快照2026-10-06 20:51:12.537840+08:00，event；核实实际线程等于owner、前台模式、closure启用、GPU1白名单和后台调度Disabled，begin仅一次，先读model view。活跃1/运行1/容量1/预算0/到期候选0/新终态0；事件仅研究产物及ledger变化，无新故障或终态，不重复preflight。复用脚本0-3事实表，不把未知指标写成0。
- 20:51:40.825870至20:51:48.074765+08:00复用既有只读脚本核实同一Y30原PID39709/cwd/argv/config/log/源码标记4f5c31f57994b8f9fb640d652c913863dc310a95/GPU1，Train3/30 steps24->34/4686正常增加。两次最多24KB尾部无Traceback/OOM/RuntimeError仅限这些尾部。GPU1设备23696MiB/83%是设备占用，不是独立候选峰值；未查询或使用GPU0。
- 新鲜证据pipeline/analyses/y30_progress_8bdbf92e9cb340bab7c0d75b95395c64.json，SHA256 faef5060e3ae0f9487e7d1087834dfe6465ae2721bb64bcf9abdb68055942965，绑定同一Y30并更新119行七段记忆中的三个操作段落，旧证据/失败/修复/不利对照保留。未重复封闭路线、准入门槛、GPU诊断或全量评测；未新增候选、重启或恢复旧实验。
- 科学状态training：begin监控2/30轮logBest=0.3383、no_improve=0/30仅为早期训练记录，后续独立观察已到Train3/30，不反写快照事实表。最终冻结全量mIoU、Acc50/spmIoU、时序必要性、效益/Pareto/新颖性未知。W2历史0.5100距目标0.515差0.005，Phase1未达，Phase2未授权。
- 本轮实际产出为同一训练的一项新鲜进度证据和操作状态更新，不是新科学结论。执行状态为report/finish待验证，只有finish成功并独立核实释放才能称COMPLETED。报告生成时本轮收据尚未写入，用量窗口/计数未知；收据只统计本地begin至生成收据的局部窗口，排除此前读取、迟到记录及最终回复，缓存包含在输入、reasoning包含在输出，不重复相加、不当账单，释放后不回写报告。

## 5. Decisions Made
- 无新增科学决策。继续同一Y30/GPU1原进程和固定30轮，保持原冻结full9508表达式/141场景、Y30/W2/B1各两遍、全场景bootstrap10000/seed1999、TRAIN大小切分1487/3325、分组伤害及训练后串行成本计划不变。只有真实终态/故障/可操作变化才处理下一步。
- 容量已满、预算0、无到期候选，不制造变体填数或重复设计。GPU0继续为用户保留；Y18完成后同一Y23迁移恢复的授权已履行，不重放旧恢复脚本，不修改目标/门槛/调度，不操作其他项目或恢复后台。
- 本轮完整六段报告保存在项目内；保存自动化的下一次触发时间本轮未读取，未知。若finish出现具体故障则保留证据、按原token修复或abort并核实释放，不将训练进度正常等同于tick完成。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
