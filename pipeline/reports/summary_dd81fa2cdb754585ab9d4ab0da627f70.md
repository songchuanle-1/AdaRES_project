# Research Pipeline Report - AdaRES_project
快照时间：2026-10-31T22:49:57.792080+00:00；生成报告时快照年龄：1422 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-01 06:49:57
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
================================================== 2026-11-01 06:49 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y72_visual_mask_refinement_res] Best=0.5004 16/30 no_improve=1/30 gpu=0 ? 序列: 0.4985, 0.4943, 0.4945, 0.5004, 0.5002

--- server-2 (192.168.1.163) ---
  [adares_y73_initial_query_anchor_res] Best=0.4962 19/30 no_improve=1/30 gpu=1 ? 序列: 0.4925, 0.4894, 0.4892, 0.4962, 0.4941
  [adares_y71_visual_product_fusion_res] 进程未见 (deathwatch 1/3) — 暂不处置

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
- Y71自然完成固定30轮；原完整Train/Val1..30与稳定finite last30/best/model/optimizer/scheduler、source/YAML/原receipt/freeze核验，canonical done_complete归档。训练best0.498878628仅为训练监控；正式FIRST未更新。
- 释放的server2 GPU0保持280W并fresh核独占空闲，Y74原生YAML/tools.train真实TRAIN两步loss4.615511/6.321258、全部六门控与真实qqa消费、有限梯度/实际更新通过；14负守卫、原Y64完整后评协议核等、准入与安全additive部署，已门控启动且native计数递增。
- Y72/Y73原PID/源/配置/收据和训练保留；新精确observer覆盖已注册87019/c60/7da/3c及七文件版本边。Y57实验暂停继续保留。
- 新增Y76置信度阶段输入1792参数；真实CPU原生模型/发布backbone/TRAIN162738点8表达、独立公式/零/RNG/七阶段W2锚点/梯度/真实消费/仅对应confidence改变通过；七文件独立保留，双卡native计划/11证据负守卫和实际CPU callback两步通过。两封装依赖失败原证据保留，原模型/native计划未改；未声称GPU任务PASS。
- 两台当前canonical occupation确认三授权卡busy且预约；Y75/Y76原native检查与15项完整冻结后评/cost等待独占资源，已保存当前状态和七段记忆。

## 5. Decisions Made
- 本轮实际新增候选1/准入1/启动1/完成1，均在snapshot budget2内；有真实native PASS即本轮入队启动。当前资源被三项既有训练占用，停止新增工作并保留两项储备。
- 保留原数据/验证边界/全部输入/seed1999/fixed30/禁用项和科学goal字节；不把训练best或CPU回调当正式FIRST、GPU PASS或效益。
- server1 GPU0与server2 GPU0/1一卡一实验和既有350/280/390W不变；Y57续训仍须单独明确请求；不触碰server6、其他项目、调度频率。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
