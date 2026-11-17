# Research Pipeline Report - AdaRES_project
快照时间：2026-11-01T16:47:15.549575+00:00；生成报告时快照年龄：341 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-02 00:47:14
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
================================================== 2026-11-02 00:47 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y76_stage_confidence_conditioning_res] Best=0.4466 5/30 no_improve=0/30 gpu=0 ? 序列: 0.3371, 0.3782, 0.4353, 0.4466

--- server-2 (192.168.1.163) ---
  [adares_y74_language_update_gate_res] Best=0.5054 24/30 no_improve=0/30 gpu=0 ? 序列: 0.4983, 0.5004, 0.4990, 0.4972, 0.5054
  [adares_y75_language_query_init_res] Best=0.4948 14/30 no_improve=0/30 gpu=1 ? 序列: 0.4946, 0.4896, 0.4918, 0.4919, 0.4948

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=2
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y74 | configs_song/model_paper5/adares_y74_language_update_gate_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y75 | configs_song/model_paper5/adares_y75_language_query_init_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
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
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 54（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve paused Y57 and its saved model/optimizer/scheduler. Do not resume or restart it under the generic GPU1 resource grant; wait for explicit Y57 continuation. Preserve current Y65/Y66/Y67. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 先查三原运行并尝试已有队列派发，waiting0。Y74 Train24、Y75 Train14、Y76 Train5持续原生计数增长，原PID/source/YAML/receipt及280/390/350W和独占保持；无新终态或故障，Y57保持暂停。
- 现有Y77/Y78四份原native计划及测试版本/七源码YAML/goal绑定核验有效；实时canonical occupation确认三授权卡均busy+reserved。实际resource_check受阻，未执行GPU检查、准入或新启动。
- 新增Y74专用完整终态准备，20生命周期/source/资源/原收据/完整epoch/finite checkpoint及仅原mapping归档守卫通过；实际只读核原源码/YAML/原receipt/freeze及原声明目录20261031_185748，不加载checkpoint，不确认尚未结束的30轮，不归档或停止训练。
- 保存当前handoff及七段记忆，原17项完整冻结FIRST/cost等待匹配资源；无正式指标更新。

## 5. Decisions Made
- 两项独立准备Y77/Y78已在，优先推进Y77实际native检查，当前真实资源被三原训练占用；保持固定30，不提前腾卡，不重复已完成准备或为占额度新增副本。
- Y74真实自然退出后，先核专用plan.dependencies/原receipt/freeze，再完整核验并canonical显式GPU0归档；新native检查仍需fresh idle/no compute/no reservation，PASS后同轮原完整freeze/新准入/安全派发。
- 保留原科学goal字节/数据协议/seed1999/fixed30/禁用项/既有功耗及Y57专用暂停；不触碰server6或其他项目，不改频率。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
