# Research Pipeline Report - AdaRES_project
快照时间：2026-11-08T04:50:19.095220+00:00；生成报告时快照年龄：783 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-08 12:50:18
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
================================================== 2026-11-08 12:50 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y92_mask_feature_quadratic_res] Best=0.4474 5/30 no_improve=0/30 gpu=0 ? 序列: 0.3044, 0.3823, 0.4240, 0.4474

--- server-2 (192.168.1.163) ---
  [adares_y91_hidden_confidence_gate_res] Best=0.5028 19/30 no_improve=0/30 gpu=0 ? 序列: 0.5007, 0.4977, 0.4988, 0.4993, 0.5028

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=1
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y91 | configs_song/model_paper5/adares_y91_hidden_confidence_gate_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y92 | configs_song/model_paper5/adares_y92_mask_feature_quadratic_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 4；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 7、待资源 0、数值诊断待解 1、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 146：complete=16 / rejected=79 / closed=12 / blocked=37 / training=2。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 69（含修订或重试，不代表成功率）。
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
- 仅处理AdaRES_project。实际CODEX_THREAD_ID与前台owner一致，后台Pipeline-AdaRES_project仍Disabled；已读取当前AGENTS、goal、SERVERS、closure及GPU1暂停授权，科学goal和暂停记录哈希保持。begin已执行进展/目标/合格队列派发检查，ready waiting为0，无新终态或操作故障。
- 原Y91/server2 GPU0/PID83827从Train19/30的3851/4686推进至Val19/30的338/363；原Y92/server1 GPU0/PID1305468从Train5/30的2557推进至3434/4686。当前y92_deployment实际核验原source、YAML、receipt、runtime、marker、280W/350W和递增原生计数，未中断或重启训练。末次观察仅证明继续训练，不是完整checkpoint或正式FIRST。
- 优先复查已有独立Y93与Y94：两项已完成CPU和版本/原生计划准备，均尚未执行GPU原生任务或准入；Y93优先。当前原Y91/Y92未到fixed30终态，完整精确checkpoint核验归档及匹配GPU0空槽条件均未满足。保留原工具绑定，不重放旧CPU、版本、准备或归档，不按额度强加第三项候选。
- 末次canonical gpu_occupation分别实测两台授权服务器：server1 GPU0由原Y92 PID1305468占用且保留，server2 GPU0由原Y91 PID83827占用且保留，均无空闲授权GPU。server2 GPU1只读库存显示外部PID3701，未使用或处置。资源证据已由正式collector归入对应候选并更新真实resource_check tick_attempt，未冒充科学实验或准入。
- 本轮新增候选0、CPU科学测试0、GPU原生任务0、准入0、训练启动0、归档0；无新科学或操作失败。七节短记忆先备份再更新，历史失败/负结果/原运行收据保持。按原300秒时效收集末次原生进度与物理占用后执行官方report/finish，不预先宣称postflight成功。

## 5. Decisions Made
- 两张授权GPU0继续各自原独立实验；待匹配当前owner自然fixed30并完整精确checkpoint核验归档后，再以实时独占空闲及当前有效绑定执行Y93原生两步检查，通过后同轮推进完整冻结协议、准入、安全同步、派发及递增步数核验。Y94排在Y93之后。
- Y91未来精确终态入口仍为finalize_y91_fixed30_y92.py、原目录20261107_102124；Y92为finalize_y92_fixed30.py、原目录20261108_090202。本轮两者均未到终态，不加载或归档checkpoint，不用日志进度替代完整校验。任何后续HEAD/owner变更须先新增经测试绑定，不改旧hash-bound工具。
- server2 GPU1全部AdaRES使用及原GPU1冻结后评/成本继续挂起，Y57另行暂停；不迁移或替换资源。仅server1 GPU0既有350W与server2 GPU0固定280W，一卡一实验；禁止server6，不改调度、功率、goal、gates、数据或恢复协议。无新正式FIRST、成本、晋升或需用户处理事项；健康运行及资源等待保持静默。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
