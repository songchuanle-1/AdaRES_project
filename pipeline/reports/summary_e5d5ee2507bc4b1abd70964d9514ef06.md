# Research Pipeline Report - AdaRES_project
快照时间：2026-11-06T04:51:24.290764+00:00；生成报告时快照年龄：325 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-06 12:51:23
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
================================================== 2026-11-06 12:51 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y88_mask_query_quadratic_res] Best=0.4802 9/30 no_improve=0/30 gpu=0 ? 序列: 0.4522, 0.4550, 0.4664, 0.4647, 0.4802

--- server-2 (192.168.1.163) ---
  [adares_y87_self_language_product_res] Best=0.4970 14/30 no_improve=0/30 gpu=0 ? 序列: 0.4955, 0.4911, 0.4912, 0.4918, 0.4970

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=1
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y87 | configs_song/model_paper5/adares_y87_self_language_product_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y88 | configs_song/model_paper5/adares_y88_mask_query_quadratic_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 4；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 7、待资源 0、数值诊断待解 1、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 143：complete=16 / rejected=79 / closed=12 / blocked=34 / training=2。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 65（含修订或重试，不代表成功率）。
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
- 核验实际CODEX_THREAD_ID等于AdaRES所属会话，foreground和experiment_first有效、Pipeline-AdaRES_project为Disabled；保持最新GPU1全用途暂停及Y57单独暂停，仅server1GPU0/350W和server2GPU0/280W。
- begin已查询进展、检查目标并尝试既有合格队列派发；无新终态、无就绪waiting、无连接或部署故障。未重复运行preflight或已归档Y81/Y82/Y83/Y84/Y85的终态helper。
- 本轮原生进度核验：Y87在server2GPU0/PID26281，Train14/30、579/4686步；Y88在server1GPU0/PID3527278，Train9/30、2493/4686步。均通过实际源码、YAML、原收据、部署marker、既有功耗及两次原生计数递增核验；不是只看running登记或PID。
- 定向核对Y89/Y90/Y91的原保存版本、CPU证据及两套GPU0原生计划的哈希引用；三者均仍未执行GPU原生、未准入，不计为就绪waiting。未重放已经完成的CPU/计划检查，不按本轮时钟改写其科学结果。
- canonical物理占用证据确认server1GPU0仍由Y88、server2GPU0仍由Y87占用且有原运行预留，均无可执行原生检查的独占空槽；外部GPU1 PID86754未动。收尾再刷新同一token的新标签证据。
- 本轮只更新真实训练及资源观测，没有新的模型、CPU或GPU科学实验；新候选0、准入0、启动0、归档0、正式FIRST更新0。保持Y89、Y90、Y91顺序等待原属任务完整fixed30 checkpoint核验/归档及实际授权GPU0空闲，不挤占现有任务。

## 5. Decisions Made
- 沿用既有资源等待安排，未新增路线或重复变体，不为凑额度复制候选，不建立科学无收益结论或新增全局冷却。三套已保存候选不能用CPU结果代签原生GPU检查或准入。
- 正式冻结FIRST仍为Y36=0.510725438294，目标0.5150；本轮监控训练best约Y87=0.4970、Y88=0.4802，仅为训练期记录，不替代完整冻结评测、成本或晋升。
- 继续保留29个fixed30模型的原全量后评与同设备成本协议；GPU1后评不迁移、不替代，Y57单独暂停，Y86数值实现挂起不重放或弱化守卫。科学goal原字节、原运行收据、原数据/私有环境/映射/seed1999固定30轮和调度保持。
- 既有运行正常、资源等待状态未实质改变且无新的用户动作，按静默心跳收尾；完整六段报告和当前资源证据留档。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
