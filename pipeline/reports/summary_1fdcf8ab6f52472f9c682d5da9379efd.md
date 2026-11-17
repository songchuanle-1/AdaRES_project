# Research Pipeline Report - AdaRES_project
快照时间：2026-10-18T14:49:53.132724+00:00；生成报告时快照年龄：404 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-18 22:49:52
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
================================================== 2026-10-18 22:49 : AdaRES_project ==================================================

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

## 4. This Tick's Actions
- 上海2026-10-18 22:49:53快照，design；monitor/goal各一次且均0，active/running/claiming/waiting均0，无新终态。due1仅Y52、blocked_not_due1为Y54，储备2/2、准备预算0、准入预算2。第0-3段只代表快照；GPU实时占用未知，不推断空闲，不重复preflight。
- 22:51:58真实复查到期Y52的限定提交许可、七文件和索引：许可入口仍不存在，七个冻结已测文件哈希逐一一致且均untracked，索引空，HEAD26ee1cec57c53ee2f94d6969d0bb637565b6ed59。证据pipeline/analyses/y52_authorization_recheck_1fdcf8ab6f52472f9c682d5da9379efd.json，SHAe2929603f04e5b71c0bca15429ea00435570eab75c8a7f425329cadef77a481d；22:52:58独立复核七文件、证据及context/pending未变。它是到期依赖检查，不是科学诊断、原生门禁PASS或GPU不足。
- Y52保持blocked/human_authorization、inconclusive、模型修订0及原wake_paths；仅追加实查证据、retry至2026-10-19T02:51:58.850032+08:00及带实证external_resource_blocked design_cycle。其他候选不变。Y54许可只引用2026-10-18 18:51:45缓存、当前变化未知；本轮22:49绑定清单未到期，墙钟跨22:51 retry不扩展清单、不改helper/snapshot或重做preflight。
- 本轮实际产出仅一次授权/源码/索引依赖实查、一份哈希证据、原候选恢复条件更新、七段记忆和六段报告。无新科学诊断/候选/模型修改/修订/Git提交/准入/训练启动，无新训练指标。历史CSV Y36 0.5110、B1 0.5020，提升+0.0090、距目标0.515差0.0040；累计107项不是本轮实验数。原indicator质量回归前提已反证，剩余辅助score MSE仅有限原生证据，训练收益未知。
- 收据用量仅begin上海22:49:52至实际receipt本地可见窗口，当前计数未知，不含更早读取/晚到记录/最终回复；缓存已含input、推理已含output，不重复相加、不当账单。拟规范finish并独立核实context提交及原租约释放，不预写PASS；失败只处理具体问题，无法修复则保留证据abort并核实释放，释放后不补写。

## 5. Decisions Made
- 续做稳定Y52，准备预算0且两项储备已满，不为填卡凑候选、不宣称路线穷尽；不重复已完成原生/资源门禁，保留Y53/Y55反证、Y50实现拒绝和所有历史closed。
- 用户此前提问不是提交同意；不自授、不扩大Y50许可、不自动git add/commit、不写Y52/Y54授权文件。须人类明确授权该候选七个确切已测文件或提供匹配commit，再验当前goal/源/六门禁/commit及准入回执，经canonical派发实查授权GPU和训练步数；GPU许可不是空闲或Git许可，不干扰外来作业。
- Y52按真实检查四小时后或真实wake变化重试，跨日时间为2026-10-19T02:51:58.850032+08:00；Y54保持2026-10-18T22:51:45.741814+08:00，由后续新绑定快照确认due再续做。retry不是调度承诺，不等待凑截止或重复催问。
- 复用两项question，不新增同义通知/决策。共享closed冷却缺陷仍未修复，本轮不重跑回归、不改共享运行时、历史、目标门槛或调度。真实到期依赖证据支持本轮收尾，不代表科学成功、有效冷却或训练启动；不跨项目、不建任务、不恢复后台。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
