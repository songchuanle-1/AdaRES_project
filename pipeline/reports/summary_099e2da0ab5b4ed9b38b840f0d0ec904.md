# Research Pipeline Report - AdaRES_project
快照时间：2026-10-18T10:50:26.665686+00:00；生成报告时快照年龄：284 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-18 18:50:26
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
================================================== 2026-10-18 18:50 : AdaRES_project ==================================================

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
- 快照为上海2026-10-18 18:50:26，design；begin各执行一次monitor/goal且返回0，active/running/claiming/waiting均0，无新终态。due1仅Y54，blocked_not_due1为Y52；储备2/2、准备预算0、准入预算2。第0-3段仅代表该快照，GPU实时占用未知，不推断空闲，未重复preflight。
- 上海18:51:45对绑定清单中真正到期的Y54执行一次既有依赖复查：限定提交许可入口仍不存在，七个冻结已测文件逐一哈希一致且全部untracked，索引空，HEAD26ee1cec57c53ee2f94d6969d0bb637565b6ed59。证据pipeline/analyses/y54_authorization_recheck_099e2da0ab5b4ed9b38b840f0d0ec904.json，SHA5ca2841b74d3ec9ba8189ea71bcc5e8258aaa549e336aea812cf9d65e7a62111；上海18:52:20独立复核七文件及证据，context/pending未变。此为真实到期授权依赖检查，不是科学诊断、原生门禁PASS或GPU不足。
- Y54保持blocked/human_authorization、failure_scope=inconclusive、模型修订0及原wake_paths；助手仅追加实查证据、retry至2026-10-18T22:51:45.741814+08:00和带证据的external_resource_blocked design_cycle。Y52和其他候选不变；Y52限定许可缺失只引用本日17:53:06缓存，当前授权变化未知，不重复未到期复查。不重跑已完成原生/边界/资源对照或共享冷却回归。
- 本轮实际产出为一次授权/源码/索引依赖实查、一份哈希证据、原候选恢复条件更新、七段记忆与六段报告。新科学诊断/候选/模型修改/修订/Git提交/准入/训练启动均0，没有新训练指标。历史CSV最佳Y36 0.5110、B1 0.5020，提升+0.0090，距目标0.515差0.0040；Y54同权重诊断不能冒充训练收益，累计107项不是本轮实验数。
- 收据用量仅覆盖begin至实际receipt的本地可见窗口，不含更早读取、晚到记录或最终回复；当前计数待收据/未知。缓存已含input、推理已含output，不重复相加，不当账单。拟以规范finish验证并分别核实context提交及原租约释放，尚未预写成功；失败只处理具体问题，不能修复则保留证据abort并核实释放，释放后不补写。

## 5. Decisions Made
- 续做稳定Y54而不扩大储备。两项储备已有真实外部授权依赖，准备预算0，不为填满卡凑候选、不宣称独立路线穷尽；保留Y52原indicator前提反证、Y53/Y55反证、Y50实现拒绝与所有历史closed。
- 用户此前询问授权不是明确同意；不自授、不扩大Y50许可，不自动git add/commit、不写Y52/Y54授权文件。该候选须人类明确授权七个确切已测文件或提供匹配commit，再验当前goal/源/六门禁/commit及准入回执，最后由canonical派发核实实时授权GPU及训练步数。既有GPU0/1授权不是空闲或Git许可。
- 保持Y52 retry2026-10-18T21:53:06.777851+08:00；Y54按本轮真实检查四小时后重试或由真实wake证据变化唤醒。retry不是调度承诺，不重复催问、不等待凑截止、不重复preflight。
- 复用20261017-145336_question和20261017-132812_question，不新增同义催问或通知。共享closed冷却缺陷仍为既有故障；本轮不重跑回归、不改共享运行时/目标/门槛/调度或历史记录。到期依赖证据支持本轮收尾，不代表冷却已修复、科学成功或训练启动；不跨项目、不恢复后台。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
