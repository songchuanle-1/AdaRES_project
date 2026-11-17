# Research Pipeline Report - AdaRES_project
快照时间：2026-11-06T08:43:42.001495+00:00；生成报告时快照年龄：345 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-06 16:43:41
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
================================================== 2026-11-06 16:43 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y88_mask_query_quadratic_res] Best=0.4968 13/30 no_improve=0/30 gpu=0 ? 序列: 0.4802, 0.4828, 0.4945, 0.4936, 0.4968

--- server-2 (192.168.1.163) ---
  [adares_y87_self_language_product_res] Best=0.5030 19/30 no_improve=0/30 gpu=0 ? 序列: 0.5013, 0.4960, 0.4965, 0.4989, 0.5030

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
- 核验实际会话归属、foreground/experiment_first、Disabled后台调度和最新资源授权；仅AdaRES，server1GPU0既有350W、server2GPU0固定280W，GPU1全用途和Y57实验各自保持用户暂停。
- begin已查进展、目标并尝试既有合格队列派发，无新终态、就绪waiting或运行故障。原Y82/Y83/Y81等已归档状态按项目真实记录保留，未重启或重放终态helper。
- 实际原生进度核验：Y87仍为server2GPU0/PID26281，Train19/30、2486/4686步；Y88仍为server1GPU0/PID3527278，Train14/30、105/4686步。原源码/YAML/收据/部署marker/功耗和两次计数递增通过，均不是仅凭PID或running登记。
- canonical物理占用检查确认两张授权GPU0仍有原任务compute PID及预留，没有可运行原生检查的独占空槽；外部GPU1 PID86754未动。收尾前刷新同一token的新标签证据。
- 定向核对Y89/Y90/Y91的原版本、CPU证据和两套GPU0原生计划，以及Y87/Y88上轮已保存的终态计划哈希引用。全部保持；未重放模型/CPU/原生/终态准备检查，未加载checkpoint、执行归档或把CPU计划算作准入。
- 到期列表含Y57、Y67、Y70、Y73、Y75、Y78、Y81及Y86；最新用户暂停没有改变，到期不代表恢复GPU1/Y57授权，也不允许重放Y86原数值失败或弱化其守卫。保持既有挂起安排，不重复请求同一确认。
- 本轮只更新真实训练和资源观测，无新科学候选、模型改动、CPU/GPU科学实验、准入、启动、归档或正式FIRST结果。现有三套候选继续等待资源，不凑数复制新变体，不改全局冷却或调度。

## 5. Decisions Made
- 沿用原运行和资源等待安排。Y87/Y88自然fixed30结束后才可按原计划执行完整稳定有限last30/best/model/optimizer/scheduler及全Train/Val1至30、源码/YAML/收据/freeze核验；日志结束或文件存在本身不是完整checkpoint校验。
- Y89、Y90先于Y91；只有原属运行精确终态核验并归档、匹配授权GPU0实际独占空闲，才执行保存的原生两步检查、根据真实结果准入和派发。生产HEAD变化须新增显式加法版本绑定，旧签名助手和计划不改写。
- 正式冻结FIRST仍Y36=0.510725438294，目标0.5150。begin监控训练best约Y87=0.5030、Y88=0.4968，只是训练期记录，不替代正式评测、成本或晋升。
- 保留29个fixed30模型原全量后评及原设备成本协议，GPU1原后评不迁移、Y57单独暂停、Y86原失败证据保持；科学goal原字节、原收据、原数据/私有环境/映射/seed1999固定30轮/功耗/调度全部不变。运行健康且无实质新结果或用户动作，静默心跳收尾。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
