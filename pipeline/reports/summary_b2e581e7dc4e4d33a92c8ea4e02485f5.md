# Research Pipeline Report - AdaRES_project
快照时间：2026-11-07T12:50:40.250943+00:00；生成报告时快照年龄：646 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-07 20:50:39
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
- 可用 GPU：未做实时占用确认，不能据此分配。
- 监控摘要（begin 已执行，不再重复探测）：
```text
================================================== 2026-11-07 20:50 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y90_query_peer_affinity_res] Best=0.5038 16/30 no_improve=1/30 gpu=0 ? 序列: 0.5008, 0.4987, 0.4996, 0.5038, 0.5013

--- server-2 (192.168.1.163) ---
  [adares_y89_stage_evidence_norm_res] Best=0.5037 27/30 no_improve=3/30 gpu=0 ? 序列: 0.5001, 0.5037, 0.5009, 0.4980, 0.5002

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=1
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y89 | configs_song/model_paper5/adares_y89_stage_evidence_norm_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y90 | configs_song/model_paper5/adares_y90_query_peer_affinity_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 4；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 7、待资源 0、数值诊断待解 1、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 144：complete=16 / rejected=79 / closed=12 / blocked=35 / training=2。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 67（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve paused Y57 and its saved model/optimizer/scheduler. Do not resume or restart it under the generic GPU1 resource grant; wait for explicit Y57 continuation. Preserve current Y65/Y66/Y67. | 2026-10-22T00:26:29.471010+08:00 |
| AdaRES-Y67 | Preserve original complete fixed30 products, signed receipt and full frozen GPU1/390W FIRST/evaluation/cost. Wait for explicit user resource reauthorization; do not use another card or restart training. | 2026-11-04T06:58:39.676817+00:00 |
| AdaRES-Y70 | Preserve original complete fixed30 products, signed receipt and full frozen GPU1/390W FIRST/evaluation/cost. Wait for explicit user resource reauthorization; do not use another card or restart training. | 2026-11-04T06:58:39.677616+00:00 |
| AdaRES-Y73 | Preserve original complete fixed30 products, signed receipt and full frozen GPU1/390W FIRST/evaluation/cost. Wait for explicit user resource reauthorization; do not use another card or restart training. | 2026-11-04T06:58:39.678374+00:00 |
| AdaRES-Y75 | Preserve original complete fixed30 products, signed receipt and full frozen GPU1/390W FIRST/evaluation/cost. Wait for explicit user resource reauthorization; do not use another card or restart training. | 2026-11-04T06:58:39.679091+00:00 |
| AdaRES-Y78 | Preserve original complete fixed30 products, signed receipt and full frozen GPU1/390W FIRST/evaluation/cost. Wait for explicit user resource reauthorization; do not use another card or restart training. | 2026-11-04T06:58:39.679826+00:00 |
| AdaRES-Y81 | Preserve original complete fixed30 products, signed receipt and full frozen GPU1/390W FIRST/evaluation/cost. Wait for explicit user resource reauthorization; do not use another card or restart training. | 2026-11-04T06:58:39.680927+00:00 |

## 4. This Tick's Actions
- 本轮快照生成于2026-11-07T12:50:40Z。核验实际会话归属、禁用的后台调度、原科学goal及最新GPU1暂停授权。正式begin先查询进展并尝试派发已有合格队列，无新增终态、就绪待跑或运行故障。唯一到期项仍为单独用户暂停的Y57，不构成恢复授权。
- 顺序核验Y89和Y90实际源文件、原生YAML、收据、部署标记、既有功耗及递增原生Train/Val计数。Y89原PID60578在server-2 GPU0/280W推进至Train27/30、4403/4686；Y90原PID2442962在server-1 GPU0/350W推进至Train17/30、126/4686。未重启、迁移、同步或修改原训练。
- 只读核验Y91/Y92各六项冻结准备引用及Y89/Y90精确终态计划SHA256，均匹配。两项候选仍无GPU原生结果或准入收据；两项原训练尚未结束，未加载checkpoint或实际归档。不重放已完成CPU、版本、绑定、计划或生命周期测试，不新增第三个准备候选。
- 收尾前顺序采集两台授权服务器新鲜正式占用证据并绑定候选及tick_attempt。两张GPU0均被原训练占用和预留，无合法原生检查或派发空槽；外部GPU1 PID3701未动。备份并更新七节记忆及报告说明，postflight及释放结果以正式finish收据为准。

## 5. Decisions Made
- Y91先于Y92，等待对应原占用者自然固定30轮结束、精确完整checkpoint核验与规范归档，以及匹配授权GPU0新鲜独占空闲且无预留证据，再执行原生YAML/TRAIN两步检查，通过后方可准入派发。资源已满且已有两个准备候选，不凑额度或重放准备。
- 保留Y89/Y90原精确终态计划、科学goal及全部原收据。日志结束不等于完整checkpoint核验；HEAD来源若推进，必须新增经过测试的显式绑定，不修改或绕过旧签名帮助程序。本轮新增候选、准入、启动、归档、GPU原生任务、CPU科学及操作测试重放均为0。
- server-2 GPU1全部使用与Y57实验分别继续用户暂停，原GPU1后评及成本不迁移、不换设备或协议。Y86原数值控制挂起及负证据保留，不原样重试或放宽守卫。正式冻结FIRST最佳仍为Y36 0.510725438294、目标0.5150；原完整9508表达/141场景、重复、场景bootstrap、子组伤害、全指标与同卡成本协议不变，无正式后评、成本或晋升结论。
- 未使用server-6、跨项目、改变功耗或调度。当前仅正常训练及原资源等待，无新增需用户处理事项。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
