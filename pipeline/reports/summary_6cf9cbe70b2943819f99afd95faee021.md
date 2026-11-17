# Research Pipeline Report - AdaRES_project
快照时间：2026-10-31T14:46:01.983056+00:00；生成报告时快照年龄：365 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-31 22:46:01
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
================================================== 2026-10-31 22:45 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y72_visual_mask_refinement_res] Best=0.4686 7/30 no_improve=0/30 gpu=0 ? 序列: 0.3841, 0.4378, 0.4490, 0.4628, 0.4686

--- server-2 (192.168.1.163) ---
  [adares_y71_visual_product_fusion_res] Best=0.4989 21/30 no_improve=1/30 gpu=0 ? 序列: 0.4949, 0.4985, 0.4921, 0.4989, 0.4985
  [adares_y73_initial_query_anchor_res] Best=0.4708 8/30 no_improve=0/30 gpu=1 ? 序列: 0.4227, 0.4423, 0.4582, 0.4628, 0.4708

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=2
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y71 | configs_song/model_paper5/adares_y71_visual_product_fusion_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y73 | configs_song/model_paper5/adares_y73_initial_query_anchor_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y72 | configs_song/model_paper5/adares_y72_visual_mask_refinement_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 3（运行 3 / 认领 0 / 等待 0）；目标 5；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 127：complete=16 / rejected=79 / closed=12 / blocked=17 / training=3。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 51（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve paused Y57 and its saved model/optimizer/scheduler. Do not resume or restart it under the generic GPU1 resource grant; wait for explicit Y57 continuation. Preserve current Y65/Y66/Y67. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 实际原生观察确认Y71第21、Y72第7、Y73第8轮原PID计数递增，source/YAML/收据/280W与350W及390W/独占和生产marker保持，无新终态或正式指标。
- 逐项核Y74/Y75现有四份native计划、原goal/七科学文件与保留版本/branch指针、原CPU native观察回调证据及源hash仍有效；只复核原准备，未重跑实现或CPU/GPU测试。
- 两台canonical occupation确认三授权卡均原任务busy并预约，真实resource_check/blocked入档；两候选真实GPU native和14项原完整后评待资源。

## 5. Decisions Made
- 保留Y74/Y75两独立待GPU检查项，暂无授权空槽，不挤占当前训练、不重复CPU/版本/计划准备或新增重复模型。
- 先canonical终态归档与fresh独占物理idle/no compute/no reservation，再精确计划实际native两步PASS，原完整freeze/准入/安全同步及门控启动；Y57独立暂停及原数据/goal字节/禁用项/调度保持。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
