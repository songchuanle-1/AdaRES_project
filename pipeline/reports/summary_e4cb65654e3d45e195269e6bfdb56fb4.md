# Research Pipeline Report - AdaRES_project
快照时间：2026-10-30T08:50:52.241112+00:00；生成报告时快照年龄：500 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-30 16:50:51
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
================================================== 2026-10-30 16:50 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y69_coarse_geometry_bias_res] Best=0.4324 4/30 no_improve=0/30 gpu=0 ? 序列: 0.3343, 0.3561, 0.4324

--- server-2 (192.168.1.163) ---
  [adares_y67_stage_readout_norm_res] Best=0.5064 29/30 no_improve=7/30 gpu=1 ? 序列: 0.5020, 0.5030, 0.5029, 0.5054, 0.5047
  [adares_y68_language_mask_modulation_res] Best=0.4927 12/30 no_improve=0/30 gpu=0 ? 序列: 0.4644, 0.4720, 0.4861, 0.4894, 0.4927

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=2
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y67 | configs_song/model_paper5/adares_y67_stage_readout_norm_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y68 | configs_song/model_paper5/adares_y68_language_mask_modulation_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y69 | configs_song/model_paper5/adares_y69_coarse_geometry_bias_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 3（运行 3 / 认领 0 / 等待 0）；目标 5；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 122：complete=16 / rejected=79 / closed=12 / blocked=12 / training=3。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 47（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve paused Y57 and its saved model/optimizer/scheduler. Do not resume or restart it under the generic GPU1 resource grant; wait for explicit Y57 continuation. Preserve current Y65/Y66/Y67. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 本轮核Y69 Train4 1912→1919、Y68 Train12 2776→2784、Y67 Train29 910→920；三原PID/source/YAML/准入/marker/独占/既有功耗保持。队列无合格waiting，无新终态，前4a2收尾完成核实。
- 为既存Y70补齐原Y67/GPU1专用终态归档工具、真实slot/source/物理空闲和原生执行入口、实际结果准入与门控派发工具；12项生命周期+8项资源+14项准入负守卫共34项CPU测试通过，六工具源和原始日志留档。
- Y70实际GPU1 readiness check核原host/390W/精确冻结七文件与当前goal/HEAD39b，22720MiB/56%/PID100811及原Y67未canonical完成，ready=false；未初始化CUDA或执行native GPU步骤，保护原运行不早停。
- canonical三卡物理占用均忙且有预约，Y70原生两步与10项原完整冻结后评/成本继续等资源；保存七节记忆、旧记忆精确快照、哈希绑定接续handoff。本轮未签收据/入队/部署，不预签postflight。

## 5. Decisions Made
- 优先完成最近可接续Y70的缺失工具与实际资源核验，不新开科学版本造成既存冻结HEAD重绑定；原CPU/prepare/save/10项native计划守卫证据复用，实际GPU门槛不替代。
- Y67仍在第29轮且真实占卡，不能归档/重启/挤占或以观察进度当完成；待原完整30轮/last-best/model/optimizer/scheduler/source/原收据核实并canonical归档、无compute/无预约/idle后，Y70实际两步PASS才按原完整后评freeze准入并同轮启动。
- 门控派发额外限制只新增Y69/Y70共14文件并保护原Y68 GPU0，server-1 Y69和Y57暂停/epoch27专用恢复要求保持。科学goal字节、精确数据协议、原收据、seed1999/fixed30、既有350W/280W/390W、server-6禁用和调度频率不改，无需用户新增授权。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
