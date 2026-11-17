# Research Pipeline Report - AdaRES_project
快照时间：2026-10-12T23:49:39.597067+00:00；生成报告时快照年龄：114 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-13 07:49:39
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
================================================== 2026-10-13 07:49 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y43_point_asymmetry_res] Best=0.4763 7/30 no_improve=0/30 gpu=0 ? 序列: 0.3669, 0.4291, 0.4444, 0.4597, 0.4763
  [adares_y44_coordinate_feature_res] Best=0.3334 2/30 no_improve=0/30 gpu=1 ?

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=2
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y43 | configs_song/model_paper5/adares_y43_point_asymmetry_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | running |
| AdaRES-Y44 | configs_song/model_paper5/adares_y44_coordinate_feature_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 2；准入预算 0。

## 4. This Tick's Actions
- 仅执行 AdaRES_project 本轮：当前环境任务 ID 与配置归属一致；目标文件哈希未变化，GPU0/1 授权和固定协议有效，Windows 后台调度仍为 Disabled，closure_policy 已明确启用。未操作其他项目、调度或后台。
- begin 仅执行一次，先读紧凑视图。北京时间快照为 2026-10-13 07:49:39.597067，event 原因仅为 research_artifact_changed；截止为 08:49:39.597067。无新增终态、到期候选或遗漏到期项，运行 2/目标 2，准入预算 0。
- 复用 begin 原始监控：Y43 在 GPU0 为 7/30，日志四舍五入最佳 mIoU=0.4763；Y44 在 GPU1 为 2/30，日志四舍五入最佳 mIoU=0.3334，均 no_improve=0。脚本事实表缺失的完整指标、利用率、显存和当前进程保持未知，不把日志训练进度当终态收益。
- 读取短期记忆和上一轮收据，确认上一轮 FINALIZED/postflight PASS、候选变更及新增证据/启动均为 0；未出现需要复查的新增终态或到期诊断，因此不再次读取全部候选证据、不重复原生测试或 preflight。
- 在核实本轮租约后仅更新七节记忆和例行报告动作说明；例行 notes 放在 reports 下，不创建新的研究分析或无操作决策日志。本轮实际新增候选、实验、测试、科学证据、准入及启动均为 0，原始远端训练持续。
- 报告渲染时执行状态为待 finish；只有显式 CLOSURE_POSTFLIGHT_PASS、COMPLETE 及独立锁释放核实后才称本轮完成。科学状态仍为 Y43/Y44 training_not_efficacy，Y42 complete_inconclusive，Phase 1 未达标、Phase 2 未获准。
- 本轮收据用量窗口从 07:49:19.175921 至实际收据记录时刻；渲染时结束时刻、用量及记录变更计数未知，finish 后以收据为准。窗口不含此前读取、迟到记录和最终回复；缓存属于输入，推理属于输出，不重复相加、不跨轮累加、不视为账单。

## 5. Decisions Made
- 无新增科学决策。容量目标已经满足且没有到期诊断，保持 Y43 GPU0 与 Y44 GPU1 原始固定 30 轮训练，不新增设计、重复变体、盲目重启或调参。
- 保留 W2 科学比较基准及所有既有冻结评测和分组伤害规则；CSV 中 Y36=0.5110 的排名不代表科学晋升，当前日志指标也不证明有效性或无害性。
- 真实终态或故障出现后再执行对应冻结分析：候选/W2/B1 各完整 9508 表达式、141 场景两次，以 FIRST 为主分析，做场景自助法、分组伤害和训练后成本评估。当前不重跑已闭合 Y42。
- 本轮不宣称研究耗尽或资源故障，不新增冷却，不更改目标、门槛、训练资源或自动化。准确下一次调度时间未知；finish 后不改写项目报告。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
