# Research Pipeline Report - AdaRES_project
快照时间：2026-11-06T12:48:17.217126+00:00；生成报告时快照年龄：319 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-06 20:48:16
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
================================================== 2026-11-06 20:48 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y88_mask_query_quadratic_res] Best=0.5024 18/30 no_improve=2/30 gpu=0 ? 序列: 0.4987, 0.5011, 0.5024, 0.4931, 0.4972

--- server-2 (192.168.1.163) ---
  [adares_y87_self_language_product_res] Best=0.5030 25/30 no_improve=6/30 gpu=0 ? 序列: 0.5010, 0.5016, 0.5013, 0.5025, 0.5025

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
- 核验实际CODEX_THREAD_ID属于AdaRES前台会话，后台调度Disabled，experiment_first启用。goal原字节和GPU1暂停授权哈希不变；仅server1GPU0既有350W、server2GPU0固定280W，GPU1全用途及Y57实验各自保持用户暂停。
- begin完成本轮进展、目标和既有合格队列派发检查；无新终态、就绪waiting或运行故障。实际原生观察Y87为server2GPU0/PID26281，Train25/30、677/4686步；Y88为server1GPU0/PID3527278，Train18/30、3936/4686步。原源码、YAML、收据、marker、功耗和短间隔计数递增均核验通过。
- 补读到期候选分页，8项为Y57/Y67/Y70/Y73/Y75/Y78/Y81/Y86。原暂停、原GPU1后评和Y86数值实现失败无新解除条件；未因到期恢复GPU1/Y57、迁移后评或重放不变的失败检查。
- 定向检查最近可推进的Y89：独立版本5f1c4bde4b682fabd84fdc84a4341ba2c845fe71、CPU证据和两套原生GPU0计划保持，实际GPU原生检查仍待执行且未准入。Y90/Y91及Y87/Y88原终态计划继续保留，不重放CPU、版本或计划准备。
- canonical物理占用检查确认两张授权GPU0均仍有原compute PID及预留，没有独占空槽。server1GPU0利用率51%、显存22760MiB；server2GPU0利用率53%、显存22734MiB。外部GPU1 PID86754未动；final标签证据按本轮token绑定候选和真实resource_check tick_attempt。
- 本轮只产生真实训练和资源观测；新科学候选0、准入0、启动0、归档0、checkpoint加载0。七段memory已精确备份后更新，没有改goal、模型、旧签名证据、全局冷却或调度。

## 5. Decisions Made
- 保留正常推进的Y87/Y88，仍未完成fixed30。自然结束后用原计划完整核验稳定有限last30/best/model/optimizer/scheduler、全Train/Val1至30及原源码/YAML/收据/freeze；不得把日志结束或文件存在当作完整checkpoint验证。
- 已有Y89/Y90/Y91三套CPU准备，下一步均需要原生GPU0独占资源，本轮不凑数新增候选或重放准备。原属运行精确终态核验归档并确认匹配授权GPU0实时空闲无预留后，优先执行Y89保存的原生YAML两步检查，通过后同轮依真实证据准入和派发。生产HEAD变化需新增加法版本绑定，不改旧签名助手，不组合机制。
- 正式冻结FIRST仍Y36=0.510725438294、W2=0.510107995189、B1=0.502221764356，目标0.5150；本轮没有正式FIRST、成本或晋升。训练指标和CPU检查不替代正式效果。
- 29个fixed30模型仍保留原全量冻结后评和同设备成本协议。GPU1原后评继续挂起，不迁移或替换资源；Y57独立暂停，Y86原负证据及数值守卫保持。原数据、私有环境、路径映射、seed1999、固定30轮、功耗和调度均不变。
- 当前只是健康训练推进及既有资源等待，无新完成、失败、实质研究结果或需用户处理事项，静默收尾。下轮仍先观察和派发现有合格工作。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
