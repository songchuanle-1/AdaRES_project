# Research Pipeline Report - AdaRES_project
快照时间：2026-10-19T04:31:18.662269+00:00；生成报告时快照年龄：1862 秒；模式：steady

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-19 12:31:18
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
| server-2 | 0 | 未知 | 未知 | 未知 |
| server-2 | 1 | 未知 | 未知 | 未知 |
- 可用 GPU：未做实时占用确认，不能据此分配。
- 监控摘要（begin 已执行，不再重复探测）：
```text
================================================== 2026-10-19 12:31 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [不可达]

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2:
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| 无 | 未知 | 未知 | 未知 | 未知 | 未知 | 未知 | 未知 | 未知 | 快照内无运行任务 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 0（运行 0 / 认领 0 / 等待 0）；目标 2；准入预算 2。
- 研究储备 2 / 2；储备准备预算 0，不增加训练并发或启动授权。
- 候选累计记录 107：complete=14 / rejected=79 / closed=12 / blocked=2。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 3；收据文件 30（含修订或重试，不代表成功率）。
- 研究等待：储备已满且仅有未到复查时间的外部阻塞；复查条件时间 2026-10-19T13:21:55.539122+08:00，不代表已解锁或获得启动授权。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y52 | Human explicitly authorizes only the seven exact tested Y52 files, or supplies their exact commit; then revalidate hashes/current goal and six gates, issue bound receipt and use canonical dispatcher with fresh idle checks. Do not deploy diagnostic W2 weights. | 2026-10-19T13:21:55.539122+08:00 |
| AdaRES-Y54 | Explicit authorization ONLY for the seven exact Y54 files in tested_bundle, or supply their exact commit. Then revalidate current goal/source/gates/commit and use existing dispatcher with live idle checks and normal Y50 terminal reconciliation. Do not extend Y50 authorization, commit other files, or deploy diagnostic weights. | 2026-10-19T13:22:24.348599+08:00 |

## 4. This Tick's Actions
- 本次为用户直接回复继续后的手动续做，确认上下文为Y52/Y54各七个已测文件的限定本地提交。正式begin未启用无变化自动释放，以在有效租约内处理已确认动作；12:31:18上海快照仍为steady/due0，monitor/goal各一次且均0，原始快照、模式、pending绑定和13:31:18 deadline未修改。
- 对两份冻结清单的14个真实文件及各自全部绑定证据逐项复核哈希一致。三个隔离Git保护测试最终全部通过：变更字节先拒绝、保留其他暂存/工作区改动、错误租约及重放拒绝；首次夹具缺目录失败已修复，不计作科学门禁。
- 已留档本会话继续消息01a15264-f969-74c2-9640-2edb5406a06e的限定许可，仅执行各七文件本地提交。Y52提交967faadc10a52b5ea7080ca796b71d33eafd0f97，Y54提交8943c8c460501b200a8dea958964579ffea65f76，各只有七项新增、Git字节与已测字节一致、无关暂存项未变；未推送Git，未修改模型、目标或调度。
- 在当前共同部署HEAD8943c8c460501b200a8dea958964579ffea65f76下，复核六项既有有范围限制的实测证据、固定30/seed1999/releasedSPFormer/128query及预注册评估。Y52初始辅助路径零活动与原indicator前提否定保留，固定已训父模型梯度不算Y52收益；Y54有限原生缺陷/有效性及恢复/资源控制也不算训练收益。未重跑原生科学门禁，不放宽准入。
- 12:55:47记录两项准入并加入server-2规范waiting队列：Y52偏好GPU0，Y54偏好GPU1，不代表已占用、预留或空闲。当前回执为pipeline/admissions/AdaRES-Y52_v2.json及AdaRES-Y54.json。Y52评估记录复制模板时名称误为Y54，训练前另存v2仅更正候选标签并更新回执引用，旧计划/旧回执/负面证据保留，其余协议、比较对象和判定准则不变。
- 实际canonical预检标server-2不可达；随后SSH内网192.168.1.163和已保存公网别名222.200.181.37均连接超时。prepared_candidates_connectivity_e4e47d5857d44da683b1d9fed03bfe66.json记录真实观测，服务器身份/实时GPU占用/远端字节和训练进度未知，未部署或启动，未操作server-6或其他项目。研究阻塞已从提交许可改为真实external_resource连接问题，不能写成GPU不足。
- 已显式通过两项closure.admission与完整closure.validate_work。报告0-3段保留12:31初始事实表，新增waiting两项仅在本段按实际发生时间说明。收据用量只覆盖本次begin至receipt本地窗口，具体数值当前未知；不含更早读取、晚到记录和最终回复。缓存属于输入、推理属于输出，不重复相加，不作为账单；实际产出为上述提交、准入、保护测试与记录，不是新指标或科学收益。

## 5. Decisions Made
- 继续同一Y52/Y54，不另造变体或重开历史路线；已测七文件许可和本地提交已完成，不再反复索要同批确认。提交许可不自行等同科学PASS或新训练许可，仍遵循既有研究/准入授权与原六门禁。
- 两项已规范准入但仅waiting，候选研究阶段为blocked/external_resource。连接恢复后沿用同一规范队列，由dispatcher重新核实远端提交、授权GPU当前占用与独占租约，并验证递增原生训练步数；此前不能称启动成功，也不能盲重启或使用诊断权重。
- 保留真实连接超时和旧缺许可/全部负面证据。retry_after均为2026-10-19T13:55:47.112333+08:00，是真实资源复查条件，不是调度承诺；队列、资源、goal或证据变化可提前唤醒，不硬撑deadline、不重复无信息增益探针。
- 以规范report/finish收尾并独立核实收据、context提交及租约释放；只有明确成功后才称本轮完成，失败则保留证据规范abort。释放后不写项目，不改调度或后台，不增加任务。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
