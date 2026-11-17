# Research Pipeline Report - AdaRES_project
快照时间：2026-10-29T18:46:20.565054+00:00；生成报告时快照年龄：100 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-30 02:46:20
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
| server-1 | 0 | 未知 | 未知 | 未知 |
| server-2 | 0 | 未知 | 未知 | 未知 |
| server-2 | 1 | 未知 | 未知 | 未知 |
- 可用 GPU：未做实时占用确认，不能据此分配。
- 监控摘要（begin 已执行，不再重复探测）：
```text
================================================== 2026-10-30 02:46 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y66_peer_state_feedback_res] Best=0.5012 18/30 no_improve=0/30 gpu=0 ? 序列: 0.4975, 0.4994, 0.5012, 0.5009, 0.5012

--- server-2 (192.168.1.163) ---
  [adares_y65_confidence_feedback_res] Best=0.5094 26/30 no_improve=4/30 gpu=0 ? 序列: 0.5094, 0.5051, 0.5041, 0.5057, 0.5037
  [adares_y67_stage_readout_norm_res] Best=0.4978 11/30 no_improve=0/30 gpu=1 ? 序列: 0.4644, 0.4713, 0.4710, 0.4894, 0.4978

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=2
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y65 | configs_song/model_paper5/adares_y65_confidence_feedback_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y67 | configs_song/model_paper5/adares_y67_stage_readout_norm_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y66 | configs_song/model_paper5/adares_y66_peer_state_feedback_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 3（运行 3 / 认领 0 / 等待 0）；目标 5；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 120：complete=16 / rejected=79 / closed=12 / blocked=10 / training=3。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 45（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve paused Y57 and its saved model/optimizer/scheduler. Do not resume or restart it under the generic GPU1 resource grant; wait for explicit Y57 continuation. Preserve current Y65/Y66/Y67. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 本轮b6bf canonical begin/design已核实际会话及Disabled后台调度，前8c69收据FINALIZED/PASS/完整释放。
- 本地2026-10-30 02:46三原PID观察通过：AdaRES-Y65 Train26/30 4601→4611/4686；AdaRES-Y67 Val11/30 168→190/363；AdaRES-Y66 Train18/30 4569→4577/4686。server-2使用已验证严格增量观察器，原六源码/YAML与当前db35新增Y67全部源码/config核等同；server-1原observer通过。三原准入/cwd/config/marker/独占/功耗核验通过，queue/running/goal/原收据保持。证据pipeline/checks/server2_y68_additive_adares-y65_b6bf846dc9544141b6eeb0d0f3b700af.json；pipeline/checks/server2_y68_additive_adares-y67_b6bf846dc9544141b6eeb0d0f3b700af.json；pipeline/checks/y66_server1_phase_progress_b6bf846dc9544141b6eeb0d0f3b700af.json。
- canonical gpu_occupation.py先server-2再server-1实测：GPU0 Y65/112356/22720MiB，GPU1 Y67/100811/22720MiB，server-1 GPU0 Y66/1973300/22836MiB，均有预约且无授权空槽。pipeline/checks/gpu_occupation_server-2_b6bf846dc9544141b6eeb0d0f3b700af.json SHA670270d4560b1e60e66c05c698d35705d8a6ea7bf5a13ba232b3ec1e40b7411a;pipeline/checks/gpu_occupation_server-1_b6bf846dc9544141b6eeb0d0f3b700af.json SHA27faab6495d69849e544f00b73ca314ed5850660fe5a663f455b036e15333f27。本轮tick_attempt为真实resource_check/blocked并归入匹配候选。
- Y68原CPU/七文件9af/双GPU原生计划/严格增量观察器保持，不重复已完成测试或准备。七标题记忆更新103行；无新增终态、故障、准入或新训练。

## 5. Decisions Made
- 三原运行继续，不以验证阶段单次低利用率当空闲；server-2 GPU0保持280W、GPU1既有390W和server-1既有350W不改。Y57实验自身继续暂停及原epoch27模型/optimizer/scheduler恢复要求，禁止server-6/跨项目/改频率。
- Y68及八项固定30原冻结后评/成本等待真实授权空槽；不挤占运行、不靠重训或静态/CPU替代native。真实终态归档/释放预约后推进既存Y68精确原生两步，通过后原完整协议冻结、准入并同轮尝试派发。
- 未来9af部署后仍需严格新增观察器实际核剩余keeper，不把当前db35观察代签未来。无新增模型或路线，不凑准备额度，不宣称方法有效。
- 健康训练和资源等待无需用户处理；report后finish并核FINALIZED/PASS/context committed/lease released，无实质变化时静默，释放后不写项目。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
