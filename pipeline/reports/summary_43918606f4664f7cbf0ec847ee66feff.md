# Research Pipeline Report - AdaRES_project
快照时间：2026-11-01T04:51:26.468523+00:00；生成报告时快照年龄：434 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-01 12:51:25
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
================================================== 2026-11-01 12:51 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y72_visual_mask_refinement_res] Best=0.5004 23/30 no_improve=8/30 gpu=0 ? 序列: 0.4967, 0.4943, 0.5003, 0.4981, 0.4922

--- server-2 (192.168.1.163) ---
  [adares_y74_language_update_gate_res] Best=0.4720 8/30 no_improve=0/30 gpu=0 ? 序列: 0.4352, 0.4466, 0.4543, 0.4699, 0.4720
  [adares_y73_initial_query_anchor_res] Best=0.4974 28/30 no_improve=6/30 gpu=1 ? 序列: 0.4965, 0.4949, 0.4871, 0.4922, 0.4907

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=2
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y73 | configs_song/model_paper5/adares_y73_initial_query_anchor_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y74 | configs_song/model_paper5/adares_y74_language_update_gate_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y72 | configs_song/model_paper5/adares_y72_visual_mask_refinement_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 3（运行 3 / 认领 0 / 等待 0）；目标 5；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 128：complete=16 / rejected=79 / closed=12 / blocked=18 / training=3。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 52（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve paused Y57 and its saved model/optimizer/scheduler. Do not resume or restart it under the generic GPU1 resource grant; wait for explicit Y57 continuation. Preserve current Y65/Y66/Y67. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 核实际owner/foreground配置和Disabled旧后台任务，先查原三运行与waiting0。Y72 Train23、Y73 Train28、Y74 Train8原PID计数真实增长，source/YAML/原receipt/功耗与独占均核，无新终态或运行故障。
- 针对将完成的Y73提前准备独立完整fixed30终态核验与canonical归档helper，20项生命周期/原GPU1/390W/原PID/完整epochs/模型优化器调度器/原mapping与Y57暂停/精确7da到3c七A文件部署守卫通过。
- 实际在原server2只读验证Y73六源码/YAML/原receipt/freeze/3c生产marker和原native声明目录20261031_051122。没有加载checkpoint、执行完整终态核验、归档或停止训练；不是训练完成或效益结论。
- Y75/Y76原四native计划、七文件和retained commits/goal/body/原CPU callbacks完整性有效；两台canonical实际GPU/compute PID显示三卡busy且预约，真实resource_check/blocked保存。两储备和15项原完整后评/cost继续等待独占资源。
- 保存准备plan及依赖、当前handoff、原记忆精确备份和七段记忆；无新候选/原生GPU任务/准入/启动/归档/正式指标，Y57暂停和调度频率保持。

## 5. Decisions Made
- 下一次Y73真实自然fixed30退出后才按准备helper完整验证并显式GPU1 canonical归档。保持原日志/checkpoint/source/receipt/freeze；不将计划、unit fixtures或PID当完成证据。
- GPU1实际释放后fresh matching idle/no compute/no reservation/390W才执行Y75既有GPU1原生YAML/TRAIN两步；PASS则当轮原完整freeze/准入/新明确source契约/安全门控启动，既有运行/Y57暂停不动。
- 原数据/验证边界/seed1999/fixed30/禁用项/goal科学字节和原全部后评协议保持；server1 GPU0/server2 GPU0/1一卡一实验及既有350/280/390W不变，不用server6、不跨项目或改调度。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
