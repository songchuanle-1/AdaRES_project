# Research Pipeline Report - AdaRES_project
快照时间：2026-11-07T08:46:56.922516+00:00；生成报告时快照年龄：937 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-07 16:46:56
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
================================================== 2026-11-07 16:46 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y90_query_peer_affinity_res] Best=0.5008 12/30 no_improve=0/30 gpu=0 ? 序列: 0.4661, 0.4787, 0.4904, 0.4920, 0.5008

--- server-2 (192.168.1.163) ---
  [adares_y89_stage_evidence_norm_res] Best=0.5017 22/30 no_improve=2/30 gpu=0 ? 序列: 0.4931, 0.4918, 0.5017, 0.4999, 0.4994

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
- 本轮快照生成于2026-11-07T08:46:56Z。核验实际会话归属、禁用的后台调度、原科学goal及最新GPU1暂停授权。正式begin先查询进展并尝试派发已有合格队列，无新增终态、就绪待跑或运行故障。唯一到期项Y57仍为单独用户暂停，不构成恢复授权。
- 先后两次顺序核验Y89与Y90原训练源文件、原生YAML、收据、部署标记、既有功耗及递增原生计数。最终Y89原PID60578在server-2 GPU0/280W推进至Train22/30、2311/4686；Y90原PID2442962在server-1 GPU0/350W推进至Train12/30、1672/4686。未重启、迁移、同步或修改两个原训练。
- 补齐Y90精确终态核验准备：从原日志实际save声明、原PID/cwd/config及源文件/YAML/标记中只读发现目录20261107_071428，不以启动时间推断目录。使用AST限定替换已有server-1生命周期模板，保留原Y88/Y89帮助程序、全部原模型、收据、冻结协议、队列、索引和goal。
- Y90终态准备25项生命周期测试首次通过；新增6项精确来源、运行环境、路径映射、任务身份、部署标记、未来HEAD及只读边界测试亦首次通过。保存哈希绑定计划和独立检查记录。未加载checkpoint、未执行完整终态核验、未归档、未使用GPU运行新任务；31项均为操作安全检查，不是模型效果或原生准入证据。
- 只读核验Y91/Y92各六项冻结准备引用SHA256均匹配，二者仍无GPU原生结果或准入收据。收尾前顺序采集两台服务器新鲜正式占用证据并绑定候选及tick_attempt，确认授权GPU0均被原训练占用及预留，外部GPU1 PID3701未动。备份并更新七节记忆及本轮报告；postflight及释放结果以正式finish收据为准。

## 5. Decisions Made
- Y90未来仅在原运行自然固定30轮结束、原PID及同配置进程全部消失后，按精确原目录、稳定且有限的last30/best/model/optimizer/scheduler、原生Train/Val1..30、源文件/YAML/收据/冻结协议核验完成再规范归档。资源保持原server-1 GPU0/350W。当前计划要求HEAD为Y90；若Y91等新增来源先推进HEAD，必须新增经过测试的显式绑定，不能修改或绕过旧签名帮助程序。
- Y91先于Y92，继续等待匹配当前占用者自然fixed30完成、完整checkpoint核验和归档，以及相应授权GPU0新鲜独占空闲且无预留证据，才执行原生YAML/TRAIN两步检查；通过后方可准入及派发。本轮无新科学候选、准入、训练启动、实际归档或模型科学测试，不为凑额新增第三个候选，不重放已完成CPU/版本/绑定/计划检查。
- 正式冻结FIRST最佳仍为Y36 0.510725438294，目标0.5150。完整原9508表达/141场景、候选及Y36/W2/B1各两次、FIRST主结果、场景bootstrap、子组伤害、全指标及同卡成本协议保持。本轮无正式后评、成本或晋升结论。
- server-2 GPU1全部使用与Y57实验分别继续用户暂停，原GPU1后评及成本不迁移、不换设备或协议。Y86原数值控制挂起及全部负证据保持；已归档产物不重复加载或归档。未使用server-6、跨项目、改变功耗或调度。本轮新增成果仅为Y90终态核验准备，无需用户新增处理。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
