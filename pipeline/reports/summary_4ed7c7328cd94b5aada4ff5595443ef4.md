# Research Pipeline Report - AdaRES_project
快照时间：2026-10-30T14:46:55.862548+00:00；生成报告时快照年龄：450 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-30 22:46:55
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
================================================== 2026-10-30 22:46 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y69_coarse_geometry_bias_res] Best=0.4869 10/30 no_improve=0/30 gpu=0 ? 序列: 0.4579, 0.4697, 0.4700, 0.4823, 0.4869

--- server-2 (192.168.1.163) ---
  [adares_y68_language_mask_modulation_res] Best=0.5032 20/30 no_improve=0/30 gpu=0 ? 序列: 0.4967, 0.4923, 0.5024, 0.4940, 0.5032
  [adares_y70_query_scale_fusion_res] Best=0.4432 6/30 no_improve=1/30 gpu=1 ? 序列: 0.3225, 0.3637, 0.4182, 0.4432, 0.4423

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=2
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y68 | configs_song/model_paper5/adares_y68_language_mask_modulation_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y70 | configs_song/model_paper5/adares_y70_query_scale_fusion_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y69 | configs_song/model_paper5/adares_y69_coarse_geometry_bias_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 3（运行 3 / 认领 0 / 等待 0）；目标 5；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 123：complete=16 / rejected=79 / closed=12 / blocked=13 / training=3。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 48（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve paused Y57 and its saved model/optimizer/scheduler. Do not resume or restart it under the generic GPU1 resource grant; wait for explicit Y57 continuation. Preserve current Y65/Y66/Y67. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 观察原Y68 Train20 348→358、Y70 Train6 2439→2449、Y69 Val10 4→23，原PID/source/YAML/准入/marker/独占/既有功耗保持；无合格waiting、新终态或运行故障，前a785收尾释放核实。
- 为既存Y71补齐Y68/GPU0终态归档及精确source绑定、GPU0原生执行与真实结果准入/门控派发工具。12项生命周期+8项资源+14项准入共34项CPU负守卫通过，7工具源和三原log留档；不是实际终态或native GPU证据。
- 实际本地source contract核原Y68六文件/原YAML/原收据、限定生产9af或39b及本地87019七新增，未读取新终态checkpoint。真实GPU0 readiness为22800MiB/72%/原PID20742且Y68未canonical完成，ready=false；未初始化CUDA/执行native两步，保护运行。
- canonical三卡占用均忙且有预约，Y71原生两步与11项原完整冻结复评/成本等待资源。旧CPU模型/prepare/save/native计划证据复用；当前模型/YAML/HEAD87019、goal/数据/收据/功耗/调度不变，保存七节记忆和handoff，不预签postflight。

## 5. Decisions Made
- 推进最近可接续Y71的缺失工具及实际资源核验，避免新增科学版本造成既存冻结HEAD重绑定；真实空槽前不抢占/重启/迁移原Y68/Y70/Y69，不把候选局部资源hold当全局暂停。
- Y68第20轮仍运行，不能归档或以步数增长当完成；只有完整30轮/原last-best/model/optimizer/scheduler/source/receipt证明及canonical归档后，GPU0真实idle无预约才执行Y71实际native两步。PASS才按原完整后评freeze准入/同轮门控启动。
- Y57暂停及epoch27专用恢复、精确原数据协议/科学goal字节/旧运行收据、seed1999/fixed30/原loss-selector、既有350W/280W/390W和server-6禁用/调度频率保持。本轮未新增训练结果或需用户处理事项，例行健康进展静默通知。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
