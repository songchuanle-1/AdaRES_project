# Research Pipeline Report - AdaRES_project
快照时间：2026-11-07T14:53:20.090417+00:00；生成报告时快照年龄：2779 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-07 22:53:19
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
================================================== 2026-11-07 22:53 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y90_query_peer_affinity_res] Best=0.5038 19/30 no_improve=4/30 gpu=0 ? 序列: 0.5038, 0.5013, 0.4988, 0.4984, 0.5027

--- server-2 (192.168.1.163) ---
  [adares_y89_stage_evidence_norm_res] Best=0.5037 30/30 no_improve=6/30 gpu=0 ? 序列: 0.4980, 0.5002, 0.4995, 0.4973, 0.5004

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
- 本轮快照生成于2026-11-07T14:53:20Z。实际会话归属匹配，后台调度禁用；核验原科学goal、最新GPU1暂停授权、原环境及数据映射。正式begin先查询进展并尝试派发已有合格队列；本轮开始无就绪待跑或运行故障，七项到期事项仍为既有GPU1授权或Y86数值控制挂起，不构成恢复许可。
- 先核验Y89进入Train30但仍在原PID60578运行、Y90原PID2442962继续Train19。GPU占用期间补齐Y91的操作帮助程序，检查发现生成的一个反例误用正确Y91 ID作为应拒绝ID。原程序及生成记录保留；新增v2仅将该反例改为Y92，并核验实际所选原生负载。13项操作守卫首次实际执行通过，未改变模型、原生任务、生产守卫或科学控制。
- Y89自然结束后，使用原精确目录20261106_122357及既有终态流程检查原PID和同配置进程消失、完整Train/Val1..30、源文件/YAML/收据/冻结协议，以及稳定且有限的last30/best23/model/optimizer555/scheduler30-23。全部通过并规范归档，原checkpoint留远端；训练best 0.5036926865577698不是正式FIRST结果，没有重训或协议替换。
- 确认server-2 GPU0无进程、无预留、280W且独占空闲后，原Y91绑定原生计划真实执行一次。原YAML/TRAIN两步loss为4.615510940551758、6.3108367919921875，峰值预留6727663616字节；14组原hidden/gate/final-linear/evidence消费者、70次原MLP层调用、原有及新增参数和输入梯度有限，两次更新通过。两批不证明训练效果或全数据显存上限。
- 在选择新版本前，为仍运行的原Y90新增显式Y91来源绑定。原PID2442962、目录20261107_071428、原收据/冻结协议及server-1 GPU0/350W保持；原帮助程序和计划未改。新增25项生命周期与6项来源/运行环境/真实Y91原生证据守卫首次通过，没有加载Y90 checkpoint、停止或归档Y90。
- 仅选择Y91已测七个新增科学文件，保留无关索引及工作区；22项真实原生准入检查通过，保存完整原后评冻结协议和准入收据，当轮入队并仅向server-2安全同步派发。新PID83827的原生Train1/30计数递增至336/4686，原Y90仍在server-1推进至Train19/30、4615/4686；server-1原部署标记保持Y90，没有同步或打断其训练。
- 本轮真实新增准入1、训练启动1、精确原运行归档1、GPU原生任务1，新增稳定科学候选和CPU科学重放均为0。保存完整归档/原生/准入/冻结/来源/派发/递增进度收尾证据。Y92原CPU和原生准备均保留且GPU未执行；只更新其当前占用者及下一步，明确需新增同字节科学内容在Y91上的来源/运行环境/占用者绑定，不直接使用旧HEAD/旧占用者帮助程序。
- 收尾前顺序采集两台授权服务器新鲜正式占用证据并绑定候选及tick_attempt。当前两张GPU0分别由原Y90与新Y91占用预留，外部GPU1保持未动。备份并更新七节记忆及正式报告；postflight和释放结果以finish收据为准。

## 5. Decisions Made
- Y89完整固定30轮核验归档，不代表科学晋升。32个已完成fixed30模型的原完整冻结FIRST/成本协议保持，正式冻结最佳仍为Y36 0.510725438294、目标0.5150。原9508表达/141场景、候选及Y36/W2/B1各两次、FIRST主结果、场景bootstrap、TRAIN切分、子组伤害、全指标和同卡成本均未缩减或替换。
- Y91真实原生检查通过后已完成准入和验证启动；两个授权GPU0再次占满。本轮已有原运行归档及一个新准入，不再新增候选凑额度。Y92继续等待合法资源，同时保留精确原科学文件和原生内容；下轮先做必要的新来源绑定，只有匹配当前Y90或Y91自然full30、完整checkpoint核验/归档及新鲜独占空闲GPU0后才运行原生检查。
- 未来只使用新增finalize_y90_fixed30_y91.py及新绑定完成原Y90精确终态核验，旧帮助程序及原计划保留但不得在新HEAD下绕过守卫。已归档Y89/Y88/Y87等不再重复核验或归档；Y91后续checkpoint目录须来自实际原生保存声明，不能从启动时间推断。
- server-2 GPU1全部使用和Y57实验分别继续用户暂停，原GPU1后评及成本不迁移、不换设备或协议。Y86原实现层数值控制挂起及全部负证据保持，未原样重试或放宽守卫。未使用server-6、跨项目、改变功耗或调度；原goal字节和所有旧运行收据保持，无新增需用户处理事项。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
