# Research Pipeline Report - AdaRES_project
快照时间：2026-10-18T05:52:38.337790+00:00；生成报告时快照年龄：555 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-18 13:52:37
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
================================================== 2026-10-18 13:52 : AdaRES_project ==================================================

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
- 快照为上海2026-10-18 13:52:38，design；begin各执行一次monitor/goal且返回0，active/running/claiming/waiting均0，无新终态。本轮绑定due仅Y54，无bootstrap要求；第0-3段保留快照事实，未重复preflight。
- 上海13:55:03用冻结助手对到期Y54执行真实四小时提交依赖复查：限定授权文件仍不存在，七个已测文件字节一致且全部untracked，索引条目0，HEAD仍26ee1cec57c53ee2f94d6969d0bb637565b6ed59。新证据pipeline/analyses/y54_authorization_recheck_e8a66aa6de7b4a049335a1f318f7ef0e.json，SHA256 147dc825b8c41c7d61a8f251bbf35cfc02d16d6a172d961e08ee0dba49879c44。
- 上海13:56:19独立复核新证据token/ID、七文件、保留包/原生审查/助手哈希、账本绑定及原context/pending完整性；Y52状态/retry未变，冻结助手验证其他候选完整摘要不变。仅Y54新增依赖证据/retry及design_cycle，账本SHA256 3071b2f614906b8531c0631d95cda41afbbcd21a4eb755ca58319f4fc800619e；共享foreground/closure仍为既有哈希，未修共享冷却。
- 本轮新增一份到期依赖证据，不是科学实验或新门禁PASS；新科学候选0、科学诊断0、模型修订0、Git提交0、准入0、训练启动0。未重跑原生/资源门禁或额外GPU诊断，GPU实时占用未知；Y54训练收益、新颖性、全局Pareto仍未知。历史最佳Y36 0.5110、B1 0.5020，提升+0.0090，距目标0.515差0.0040，无新训练指标；累计107项不是本轮产出。
- 储备2/2、准备预算0；Y52未到期不重查，其授权缺失仅引用本日12:54:10缓存，当前变化未知。收据用量仅覆盖本轮begin上海13:52:37至实际receipt的本地可见日志窗口，不含更早读取、晚到记录或最终回复；当前计数未知。实际产出为一份Y54依赖证据及相关账本/记忆/报告，缓存已含input、reasoning已含output，不重复相加、不作账单。尚未finish，结果以命令、收据及独立释放核验为准，释放后不补写。

## 5. Decisions Made
- Y54保持blocked/human_authorization、failure_scope=inconclusive、实现修订0；基于真实复查设retry2026-10-18T17:55:03.572119+08:00，保留wake_paths。仅接收七个确切已测文件的限定提交许可或匹配commit，之后仍须当前goal/源字节/六门禁/commit及实时授权GPU核验，不外延Y50许可、不自授或自动提交。用户此前询问授权并非同意，不重复催问。
- Y52保持原状态、证据及retry2026-10-18T16:54:10.457479+08:00，不重复未到期依赖检查。其余候选、历史拒绝/closed、队列/CSV及冻结训练协议不变；不把既有原生可执行性、W2诊断权重、梯度或CPU边界fixture当作训练收益，也不凑新候选。
- 保留已报告的shared closed冷却故障与原失败证据，不改共享脚本、12条历史closed、目标门槛或调度。真实到期依赖进展可尝试一次finish，不表示冷却已修复、科学机制有效或目标已达成；不宣称全局Pareto或新颖性获证。
- 本快照其余候选动作均有明确外部依赖或未到期，按external_resource_blocked真实处置收尾，不等待凑deadline。使用新Y54依赖证据执行一次canonical finish；仅命令成功、收据FINALIZED/PASS/context已提交且原租约及pending均消失后称本轮完成。若失败保存实际错误并abort核实释放，不反复无进展finish或放宽准入。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
