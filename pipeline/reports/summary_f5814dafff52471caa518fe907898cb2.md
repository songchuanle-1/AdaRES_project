# Research Pipeline Report - AdaRES_project
快照时间：2026-11-10T00:50:57.594134+00:00；生成报告时快照年龄：101 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-10 08:50:56
================================================================================
  Target: AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515.

## Phase 1: Metric Target
  Target: mIoU ≥ 0.5150
  [✗] Champion: AdaRES-Y91 — mIoU=0.5114 (gap to target: +0.0036)
       Epochs: ?, State: done_complete
  Phase 1: NOT YET — continue exploring ✗

## Baseline
  Baseline: AdaRES-B1 — mIoU=0.5020
  Champion vs Baseline: γ = +0.0094

## Phase 2: Story Verification
  (等待 Phase 1 达标后启动)

## Overall
  Phase 1: champion=0.5114, target=0.5150, gap=+0.0036
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
================================================== 2026-11-10 08:50 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y94_query_geometry_bias_res] Best=0.5076 25/30 no_improve=3/30 gpu=0 ? 序列: 0.5071, 0.5076, 0.5060, 0.5046, 0.5034

--- server-2 (192.168.1.163) ---
  [adares_y95_query_set_mean_res] Best=0.5001 16/30 no_improve=0/30 gpu=0 ? 序列: 0.4940, 0.4937, 0.4909, 0.4923, 0.5001

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=1
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y95 | configs_song/model_paper5/adares_y95_query_set_mean_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y94 | configs_song/model_paper5/adares_y94_query_geometry_bias_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 4；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 7、待资源 0、数值诊断待解 2、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 151：complete=16 / rejected=79 / closed=12 / blocked=42 / training=2。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 7；收据文件 72（含修订或重试，不代表成功率）。
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
- 仅AdaRES_project，核验实际owner、Disabled后台调度、正式前台/experiment_first及AGENTS/goal/SERVERS/最新GPU1暂停后begin；先查进展并派发已有合格任务，ready waiting为0。
- 运行：AdaRES-Y94 Train26/30 step89/4686；AdaRES-Y95 Train16/30 step2761/4686。原PID/source/YAML/签名收据、私有runtime、350W/280W和原生计数递增核验成立；无新终态或操作故障，未停止、重启或迁移。
- canonical只读物理GPU/compute PIDs和canonical reservation确认两张授权GPU0均busy+reserved，无独占空闲卡。保留collector真实resource_check，未执行GPU任务或将资源复查当科学PASS。
- 保留已有Y96/Y98/Y99准备，优先当前HEADY95/当前owner已绑定的Y96；Y94/Y95专用精确终态计划、所有旧CPU/版本/原生程序体和Y97原始/修订1负证据保持。没有重复科学或绑定测试、新增占位候选。
- 本轮新科学ID/CPU科学尝试/GPU原生/准入/启动/归档/正式评测均为0。精确备份七节记忆，保存真实资源和训练证据后官方report/finish，以实际收据和租约释放为准。

## 5. Decisions Made
- 无新科学决策或用户行动项。原训练占用全部授权GPU0时保持已有候选优先，不抢占、不凑额度设计/重测或伪造waiting；健康推进和不可行动的资源等待保持静默。
- 只有匹配原运行自然fixed30结束、完整稳定有限last/best/model/optimizer/scheduler和Train/Val1..30精确核验/canonical归档后，现场独占空闲未保留的授权GPU0才能执行run_y96_bound_native_y95.py。Y94原目录20261109_111838、Y95原目录20261109_081912，日志结束不是完整checkpoint核验；不重放旧已归档终态。实际native通过再完成另一在跑任务的新测试终态绑定、设备匹配完整原freeze/receipt/准入/精确source/安全派发/递增计数。
- 全部GPU1使用及原GPU1冻结后评/成本继续暂停，Y57独立暂停；不迁移替换资源、不用server6、不跨项目、不改调度或功耗、不干预外部任务。保留科学goal字节、旧签名收据、原数据/私有runtime/部署映射/seed1999固定30轮及完整FIRST/有害子组/原设备成本协议。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
