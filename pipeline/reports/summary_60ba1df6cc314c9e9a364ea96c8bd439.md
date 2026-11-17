# Research Pipeline Report - AdaRES_project
快照时间：2026-10-04T01:50:01.217324+00:00；生成报告时快照年龄：119 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-04 09:50:00
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
================================================== 2026-10-04 09:49 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y25_mask_feedback_res] Best=0.4941 19/30 no_improve=2/30 gpu=1 ? 序列: 0.4917, 0.4857, 0.4941, 0.4870, 0.4782

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y25 | configs_song/model_paper5/adares_y25_mask_feedback_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 1；准入预算 0。

## 4. This Tick's Actions
- 本轮快照为北京时间2026-10-04 09:50:01.217324，模式event；归属、冻结目标、closure策略及GPU1单卡授权已核对，Windows后台任务保持Disabled。begin监控和目标检查各执行一次，不重复preflight。活跃/运行/容量均1，等待0、准入预算0、到期候选0、新终态0。
- 复用阶段感知只读脚本，实际观测09:50:21.154107至09:50:26.576437；核对原Y25 PID98611、配置、日志、工作目录、GPU1及源94918c1。原生Train19/30从2089推进至2100/4686，日志24290033增至24292959字节。观测完成和原生进度分别为真，并非仅凭complete标志推断。
- 只查询授权GPU1，设备显存22260MiB、利用率80%，不是候选隔离峰值。两个各24000字节日志尾段未见崩溃标记，仅覆盖这些尾段，不宣称全日志无异常；GPU0未查询或使用。快照best=.4941、19/30、no_improve=2/30为中间验证，科学状态training，最终效力未知。
- 保存pipeline/analyses/y25_phase_60ba1df6cc314c9e9a364ea96c8bd439.json和原始日志，SHA256=fac0c8806db961b87eaa690ef2a7232aaf7663f088101b3ed8a1f82a6f6a8e35。仅更新Y25下一步、证据引用及七段短记忆，保留所有历史故障、不利对照、科学结论和冻结协议；报告复用脚本0-3段事实表。
- 本轮收据用量窗口及postflight结果在报告生成时未知，finish后只读核验，不提前声称执行完成。用量仅包含本地begin至收据的部分日志，不含此前读取、迟到记录和最终回复；缓存已计入输入、推理已计入输出，不重复相加，不作为账单。实际产出为1份新哈希进度证据、原始日志及短记忆更新；新候选0、新训练启动0。

## 5. Decisions Made
- 准入目标已满足，无到期诊断或新终态，保持同一Y25固定30轮、seed1999和冻结配置继续训练；不重启、早停、调参、额外准入或重复审查已关闭路线。
- 中间best及no_improve变化不作为最终效力或目标达标证据；保留不利初始损失、训练软注意力/推理硬掩码分布差异和原生重复、resume数值噪声，不宣称可靠收益、损害、Pareto或创新性。
- 真实固定30轮结束后按原冻结设计执行9508表达式/141场景的Y25、W2、B1各两次原生评估、整场景bootstrap、冻结大小分组和成本比较。GPU1当前占用，不附加GPU测试；GPU0仍为用户保留，不操作其他项目或调度，下一次触发时间未读取，记未知。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
