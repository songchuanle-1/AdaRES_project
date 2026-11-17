# Research Pipeline Report - AdaRES_project
快照时间：2026-10-18T00:50:15.313239+00:00；生成报告时快照年龄：524 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-18 08:50:14
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
================================================== 2026-10-18 08:50 : AdaRES_project ==================================================

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
- 快照为上海2026-10-18 08:50:15，design；begin各执行一次monitor/goal且返回0，active/running/claiming/waiting均0，无新终态。第0-3段保留预检时点事实；未重复preflight，实际线程与原token/context/pending绑定已核实。
- 上海08:52:11对绑定快照唯一到期项Y54执行冻结助手的真实提交依赖复查：限定授权文件仍不存在，七个已测文件字节一致且全部untracked，索引条目0，HEAD仍26ee1cec57c53ee2f94d6969d0bb637565b6ed59。新证据pipeline/analyses/y54_authorization_recheck_35f79dd56ae94899aee4f2bf7ce34480.json，SHA256 2ef13860376adf6723647535c9397dbba10258bb902ca47dca9cc3352da8d45b。
- 上海08:53:26独立核实新证据token/ID、七文件哈希、账本绑定和原context/pending完整性，Y52状态及retry未变；冻结助手验证其他候选完整摘要不变。仅Y54新增依赖证据/retry及本轮design_cycle，账本SHA256 a10ba52d1d94728ef82fdc950b2393167ff4c9e5d68af07dda13ded5098a9c06。
- 本轮新增一份到期依赖证据，不是新科学测试或门禁PASS；新科学候选0、科学诊断0、模型修订0、Git提交0、准入0、训练启动0。未重跑原生/资源门禁或额外GPU/远程诊断，GPU实时占用未知；Y54训练收益、新颖性及全局Pareto仍未知，既有梯度、固定W2权重或CPU边界证据不当作训练收益。
- 储备仍2/2、准备预算0；Y52在本轮未到期，不重复复查或改其retry。其授权缺失仅引用本日07:56:56缓存，当前Y52授权变化未知。历史最佳Y36 0.5110、基线B1 0.5020，提升+0.0090，距目标0.515还差0.0040；没有新训练指标，累计107候选记录不是本轮产出。
- 收据用量只覆盖本轮begin上海08:50:14至实际receipt生成的本地可见日志窗口，不含更早读取、晚到记录或最终回复；当前收据计数未知。实际产出为一份Y54依赖证据与相关账本/记忆/报告更新；缓存已含input、reasoning已含output，不重复相加、不作账单。尚未finish，执行结果以命令、收据和独立租约释放核验为准，释放后不补写。

## 5. Decisions Made
- Y54保持blocked/human_authorization、failure_scope=inconclusive及实现修订0；基于实际复查设retry为2026-10-18T12:52:11.767091+08:00，保留原wake_paths。只接收七个确切已测文件的限定提交许可或匹配commit，之后仍须当前goal/源字节/六门禁/commit和实时授权GPU检查；不外延Y50许可、不自授、不自动Git提交或部署诊断权重。
- Y52保留本日07:56:56证据及2026-10-18T11:56:56.902556+08:00重试条件，其他候选、历史拒绝/closed、队列、CSV和冻结训练协议不变。无额外储备预算，不凑新候选或重复已完成的原生控制来制造进展。
- 共享closed阶段冷却缺陷仍仅保留既有回归/失败证据与已报告问题；共享foreground/closure源码哈希本轮核实未变，不修共享运行时、不改12条历史closed或调度。到期依赖证据可尝试finish，不表示冷却缺陷已修复或科学目标达成。
- 本快照其余候选动作均有明确外部依赖或未到重试，不等待凑deadline。按external_resource_blocked契约以真实Y54依赖证据执行一次canonical finish；仅命令成功、收据FINALIZED/PASS/context已提交且原租约和pending均消失后称本轮完成。若失败保留实际错误并abort核实释放，不反复无进展finish或放宽准入。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
