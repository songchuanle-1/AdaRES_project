# Research Pipeline Report - AdaRES_project
快照时间：2026-11-02T00:45:58.953289+00:00；生成报告时快照年龄：1349 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-02 08:45:58
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
================================================== 2026-11-02 08:45 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y76_stage_confidence_conditioning_res] Best=0.4992 14/30 no_improve=0/30 gpu=0 ? 序列: 0.4890, 0.4928, 0.4948, 0.4940, 0.4992

--- server-2 (192.168.1.163) ---
  [adares_y77_geometry_query_init_res] Best=0.3876 3/30 no_improve=0/30 gpu=0 ? 序列: 0.3329, 0.3876
  [adares_y75_language_query_init_res] Best=0.5045 25/30 no_improve=1/30 gpu=1 ? 序列: 0.4999, 0.5019, 0.4957, 0.5045, 0.5004

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=2
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y75 | configs_song/model_paper5/adares_y75_language_query_init_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y77 | configs_song/model_paper5/adares_y77_geometry_query_init_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y76 | configs_song/model_paper5/adares_y76_stage_confidence_conditioning_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 3（运行 3 / 认领 0 / 等待 0）；目标 5；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 130：complete=16 / rejected=79 / closed=12 / blocked=20 / training=3。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 55（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve paused Y57 and its saved model/optimizer/scheduler. Do not resume or restart it under the generic GPU1 resource grant; wait for explicit Y57 continuation. Preserve current Y65/Y66/Y67. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 运行：原Y75/server2GPU1、Y76/server1GPU0、Y77/server2GPU0实际训练计数递增，PID/原源码YAML/收据/版本/功耗保持；最新Train25/30、15/30、3/30。
- 新准备：新增独立Y79查询集合到语言反馈；原生YAML完整模型/发布backbone/真实TRAIN加载、公式和全部锚点控制及两合成CPU解码器优化步通过，七测试文件隔离保留版本，双GPU原生计划11守卫与16冻结回调CPU执行通过。
- 完成：本轮新候选1/准入0/启动0/归档0，无正式FIRST/cost结果；GPU检查未执行，CPU成功不作native任务PASS或效益证明。
- 已更新精确记忆备份、七节短记忆与哈希handoff；Y75原专用终态计划全部依赖/收据/freeze保持，未加载checkpoint或提前归档。

## 5. Decisions Made
- fresh canonical occupation确认三授权卡均busy+reserved，Y78优先Y79等待匹配独占空槽；不挤占现有运行，不重复已通过的CPU和计划准备。
- Y75自然fixed30结束后按其专用完整终态程序canonical显式GPU1归档，再在fresh idle/unreserved390W上执行Y78精确原native两步；实际PASS后当轮原完整冻结后评协议/新准入/精确source契约/安全派发。
- 保持科学goal与原收据、原数据协议/独立环境/部署映射、server2GPU0 280W与GPU1 390W、server1GPU0 350W；Y57独立实验继续暂停，server6禁止，不跨项目或修改调度。
- 需处理：当前无新增用户操作事项；Y57继续暂停，无须重复确认已有项目内操作授权。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
