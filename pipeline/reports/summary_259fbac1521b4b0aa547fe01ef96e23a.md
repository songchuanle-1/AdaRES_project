# Research Pipeline Report - AdaRES_project
快照时间：2026-10-17T08:48:54.615489+00:00；生成报告时快照年龄：411 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-17 16:48:54
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
================================================== 2026-10-17 16:48 : AdaRES_project ==================================================

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
- 本轮快照2026-10-17 16:48:54上海时间，design；已核实当前归属、foreground模式、goal、server-2 GPU0/1单实验单GPU权限及旧后台Disabled。begin仅执行一次monitor/goal，返回码均0；项目运行/认领/排队均0，无新终态。GPU实时利用率、显存及外来作业未知，不把白名单或无本项目训练推断为空闲。
- 紧凑视图唯一到期ID为Y52，原retry_after=2026-10-17T16:42:22.893657+08:00；按ID仅加载所需阻塞字段及两项证据引用，不倾倒113项引用。Y54未到期，不读取其授权文件、不重跑门禁；储备保持2/2，准备预算0，不创建第三个原型。
- 16:53:10上海时间对Y52执行一次真实限定提交依赖复查：对应pipeline/authorizations/AdaRES-Y52_commit.json仍不存在；原已测bundle和native review哈希有效，七个模型/配置文件均与冻结字节相同、全部为untracked，Git索引无这七文件，当前HEAD=26ee1cec57c53ee2f94d6969d0bb637565b6ed59。没有新的人类提交许可或指定匹配commit，不能沿用Y50旧许可。
- 复查结果保存为pipeline/analyses/y52_authorization_recheck_259fbac1521b4b0aa547fe01ef96e23a.json，SHA256 dbfebf68db236c0298dc74c6badd019bd79d6452138e6b6aaec0a9257fb73f68。Y52仍blocked/human_authorization，追加真实证据和复查历史，retry_after按实际检查时间后四小时设为2026-10-17T20:53:10.188457+08:00，wake_paths保持可提前唤醒。该时间是复查阈值，不是已核实的调度承诺。
- 其它106个候选的结构摘要前后完全一致，历史closed及Y54未改变。没有重新执行科学诊断、原生前后向、资源门禁或GPU探针；没有模型/配置修改、实现修订、Git提交、准入或训练启动。复查不是新的门禁PASS、方法收益或GPU资源阻塞。
- 已通过规范入口记录20261017-165355_other.md。当前共享冷却脚本哈希仍a78da0060b280300359adc58a20f6c325cabb487313492112e8fc0bf4535bee6，已报告closed终态筛选故障未修复，未重跑其回归、未改共享程序/调度/历史或扩大授权。
- 科学状态无变化：历史CSV最佳Y36 mIoU 0.5110、基线B1 0.5020、目标0.515、差0.0040；本轮无新训练或评估指标，Y52/Y54训练收益未知。完整六段审计和七节记忆在租约内留档，当前尚未执行postflight，不能预称完成。
- 收据用量只覆盖本轮begin至收据前的本地可见响应窗口，实际token在生成收据前未知；不包括更早阅读、迟到记录或最终答复，不是账单，缓存已含input、推理已含output，不重复相加。实际产出为一个已有候选的到期授权/源码/Git依赖复查与证据更新；新科学候选0、科学测试0、训练启动0。

## 5. Decisions Made
- 保持原Y52同一稳定ID和冻结七文件，不自动git add/commit、不写admission或派发；未来需明确限定提交许可或相应已测commit，之后再核对最新goal/source/六门禁/commit回执及实时GPU，经现有gated dispatcher启动并验证原生步数。不得部署诊断W2权重。
- 本轮的真实进展是到期依赖复查，不是新的科学证据。按external_resource_blocked记录恢复条件，attempted_ids只含实际复查的Y52，所有其它候选和未到期Y54保留；不制造新机制、不凑变体或重复门禁、不宣称路线穷尽或deadline接近。
- 本轮有新到期复查证据，与前两轮完全无候选变化的状态不同，可执行一次规范finish校验；只有明确PASS/COMPLETE及独立核实租约/pending消失才称执行完成。若不通过则保存精确错误并abort，不改门槛或重复无效finish。
- 共享冷却问题仍按既有请求等待限定修复许可；成功收尾也不代表它已修复，更不代表方法有效。候选提交依赖没有实质解除，不重复催问或通知同一阻塞；释放后不写项目。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
