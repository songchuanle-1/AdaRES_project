# Research Pipeline Report - AdaRES_project
快照时间：2026-10-31T08:53:54.363604+00:00；生成报告时快照年龄：1416 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-31 16:53:53
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
================================================== 2026-10-31 16:53 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  无训练进程
  [adares_y69_coarse_geometry_bias_res] 进程未见 (deathwatch 1/3) — 暂不处置

--- server-2 (192.168.1.163) ---
  [adares_y71_visual_product_fusion_res] Best=0.4937 13/30 no_improve=2/30 gpu=0 ? 序列: 0.4712, 0.4817, 0.4937, 0.4888, 0.4931
  [adares_y70_query_scale_fusion_res] 进程未见 (deathwatch 1/3) — 暂不处置

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
- 本轮Y69/Y70原进程自然退出，完整Train/Val1..30及稳定finite last30/best/model/optimizer544/scheduler核实，canonical归档；训练best分别0.5014769434928894/0.5029489994049072，均非正式FIRST，原checkpoint保留。
- Y72在server1 GPU0/350W真实TRAIN两步loss4.615511417388916、6.384006500244141通过；Y73在server2 GPU1/390W两步loss4.615511417388916、6.345763206481934通过。实际六新路径/预测消费/原生数据及新梯度真实更新确认，12项准入负守卫通过。
- 两项原完整9508表达/141场景/FIRST/重复/scene-bootstrap10000/TRAINcut/harm/同卡cost协议深复制核等并签准入；本地只选各七已测试additive文件保留其他index/工作文件，现有门控sync/dispatch同轮启动，分别PID311445和62829，第1轮实际步数递增。
- 新精确版本observer核Y71原PID53714/source/YAML/receipt保持，第14轮递增；server1生产c60/server2生产7da。两台canonical occupation确认三授权槽busy+reserved，14项完整冻结后评等待资源。
- 初只读terminal-readiness脚本转义SyntaxError原源/log保留；独立raw-string+AST attempt2修复通过，无训练改动。

## 5. Decisions Made
- 本轮2项准入/启动已达快照预算，保留现有三训练；本轮没有新的正式主指标或效益判断，不从训练best晋升。
- 以后只用scripts/y72_y73_deployment.py观察当前三运行；旧87019-only observer留历史，不放宽任意marker。未来新版本须重新核精确additive与存活契约。
- 保留Y57独立暂停与原checkpoint；不改科学goal字节/原数据/禁用项/功耗硬件/调度频率，不跨项目、不触server6、不重放已完成操作。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
