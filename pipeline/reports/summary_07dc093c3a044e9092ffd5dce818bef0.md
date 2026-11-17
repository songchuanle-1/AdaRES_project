# Research Pipeline Report - AdaRES_project
快照时间：2026-10-19T01:18:02.543856+00:00；生成报告时快照年龄：455 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-19 09:18:02
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
================================================== 2026-10-19 09:18 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  无训练进程

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
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y52 | Human explicitly authorizes only the seven exact tested Y52 files, or supplies their exact commit; then revalidate hashes/current goal and six gates, issue bound receipt and use canonical dispatcher with fresh idle checks. Do not deploy diagnostic W2 weights. | 2026-10-19T07:55:11.701865+08:00 |
| AdaRES-Y54 | Explicit authorization ONLY for the seven exact Y54 files in tested_bundle, or supply their exact commit. Then revalidate current goal/source/gates/commit and use existing dispatcher with live idle checks and normal Y50 terminal reconciliation. Do not extend Y50 authorization, commit other files, or deploy diagnostic weights. | 2026-10-19T08:51:34.477821+08:00 |

## 4. This Tick's Actions
- 快照为上海2026-10-19 09:18:02；正式begin只执行一次monitor和goal，均返回0。训练队列active/running/claiming/waiting均0，准入预算2、储备2/2、准备预算0，绑定due仅Y52/Y54。本轮不从旧GPU快照推断空闲。
- 按用户要求修复共享流水线：前台冷却复用closure.pending排除历史closed；储备满且仅有未到期已证实外部阻塞时，首次event校验后转无变化steady；到期、授权/证据/目标/资源变化继续唤醒，未知阻塞与可执行工作不被抑制。紧凑视图及报告显示具体人工动作，修正文档里一律不得提问的冲突。隔离回归121项全部通过，含真实closure事件收尾与自动steady、混合阻塞、错误字段和边界测试。
- Y52于09:21:55、Y54于09:22:24各完成一次真实到期提交依赖复查。限定提交授权仍缺，两组各7个已测文件字节不变且untracked/索引空，HEAD仍26ee1cec57c53ee2f94d6969d0bb637565b6ed59；证据分别为pipeline/analyses/y52_authorization_recheck_07dc093c3a044e9092ffd5dce818bef0.json与y54_authorization_recheck_07dc093c3a044e9092ffd5dce818bef0.json。保持blocked/human_authorization与inconclusive；不代表GPU短缺、准入PASS或科学进展。
- 真实产出：共享流程修复与文档、121项隔离回归、两项依赖复查证据、七段记忆、notes和六段报告。本轮没有新候选、模型修订、源码提交、准入或训练启动；未重跑原生科学门禁，未修改其他项目、目标、调度、closure门槛或历史失败记录。
- 收据用量窗口仅本轮begin至receipt写入，更早修复与回归、晚到日志及最终回复不在其中；本报告生成时具体用量未知，最终以收据为准。缓存属于输入、推理属于输出，不重复相加，不将用量当作账单。报告不预称postflight成功，最终以规范finish与独立释放核验为准。

## 5. Decisions Made
- 两项已准备候选仍等待各自精确已测七文件的本地提交确认或匹配commit；流程修复请求不替代该确认，不沿用Y50许可、不部署W2诊断权重、不凑新候选或重复科学测试。
- 保留原wake_paths；Y52复查条件时间为2026-10-19T13:21:55.539122+08:00，Y54为13:22:24.348599+08:00，是真实4小时依赖复查条件而不是调度承诺。真正授权/证据变化可提前唤醒；不硬撑deadline。
- 共享代码缺陷并非AdaRES专属，其他调用相同canonical前台入口且满足同一状态条件的项目也可能受影响并继承修复；本轮不操作或认定其他项目已发生同一故障。
- 具体授权动作一次性明确交接；本会话无歧义人类确认按项目契约留档，不要求固定口令或用户手写JSON，不自行授权。训练仍须当前goal/source/六门禁/commit/准入回执及实时GPU租约检查。
- 两项真实依赖复查结束后按外部阻塞契约收尾，调用正式report/finish并独立核实收据、context提交、冷却与租约释放。只有明确成功后称完成，否则保留证据并规范abort，释放后不写项目。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
