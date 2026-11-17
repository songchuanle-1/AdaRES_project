# Research Pipeline Report - AdaRES_project
快照时间：2026-10-31T02:42:57.286013+00:00；生成报告时快照年龄：285 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-31 10:42:56
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
================================================== 2026-10-31 10:42 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y69_coarse_geometry_bias_res] Best=0.5015 24/30 no_improve=3/30 gpu=0 ? 序列: 0.4952, 0.5015, 0.4979, 0.4977, 0.5003

--- server-2 (192.168.1.163) ---
  [adares_y70_query_scale_fusion_res] Best=0.5029 22/30 no_improve=1/30 gpu=1 ? 序列: 0.4987, 0.4968, 0.4964, 0.5029, 0.5007
  [adares_y71_visual_product_fusion_res] Best=0.4473 5/30 no_improve=0/30 gpu=0 ? 序列: 0.3185, 0.3699, 0.4333, 0.4473

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=2
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y70 | configs_song/model_paper5/adares_y70_query_scale_fusion_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y71 | configs_song/model_paper5/adares_y71_visual_product_fusion_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y69 | configs_song/model_paper5/adares_y69_coarse_geometry_bias_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 3（运行 3 / 认领 0 / 等待 0）；目标 5；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 125：complete=16 / rejected=79 / closed=12 / blocked=15 / training=3。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 49（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve paused Y57 and its saved model/optimizer/scheduler. Do not resume or restart it under the generic GPU1 resource grant; wait for explicit Y57 continuation. Preserve current Y65/Y66/Y67. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 上海10:43确认Y71 Train5/30 2899→2908、Y70 Train22/30 4100→4109、Y69 Train24/30 4673→4681原生递增；原PID/source/YAML/准入/独占和280W/390W/350W核实，无新增终态/连接或训练故障及合格waiting。
- 核Y72 server2 GPU0/1及未激活server1、Y73 server2 GPU0/1共5份已准备计划：当前goal、科学源码/YAML/原native program与retained commit精确匹配，Y72原server1 CPU预检证据仍有效，active仍server2，未重放准备或原生GPU检查。
- fresh canonical occupation核三个授权GPU原53714/52561/1135633均忙且预约，保持当前训练和既存12项原完整冻结后评/成本等待；当前token真实resource_check/blocked及证据已保存。
- 本轮仅状态/版本/资源核验，无新增模型、准入、派发、原生GPU步骤或正式评估指标。备份记忆并更新55行，handoff=pipeline/checks/resource_handoff_e4bdf184bc5649dfb1c159fc8cbd4f93.json。状态正常，无需用户处理，保持静默通知。

## 5. Decisions Made
- 保留原三任务和已验证的五份准备计划；按真实资源/预约变化继续原生检查。Y69实际fixed30后先完整原产物/源码/收据核验与canonical归档，fresh独占idle/无预约/350W才允许inactive Y72 server1 activate/run。实际native PASS仍是原完整后评freeze/准入/安全版本选择/门控派发的必要条件。
- 保留原goal字节/数据/seed1999/fixed30/禁用项/原收据/功耗/频率，复用项目授权，不跨项目、不用server-6、不抢占或重启受保护任务。Y57独立暂停不变；无新增正式指标或必须用户处理项。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
