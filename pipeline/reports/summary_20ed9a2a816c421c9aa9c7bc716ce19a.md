# Research Pipeline Report - AdaRES_project
快照时间：2026-10-17T09:49:13.771183+00:00；生成报告时快照年龄：485 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-17 17:49:13
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
================================================== 2026-10-17 17:49 : AdaRES_project ==================================================

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
- 本轮快照2026-10-17 17:49:13上海时间，design；当前归属、foreground、goal、仅server-2 GPU0/1单实验单卡授权及后台Disabled均已核实。begin只执行一次monitor/goal，返回码均0；项目运行/认领/排队均0，无新终态。GPU实时利用率、显存及外来作业未知，不据此分配或称空闲。
- 唯一到期ID为Y54，原retry_after=2026-10-17T17:26:21.740407+08:00；按ID仅读取所需阻塞字段与bundle/native review引用。Y52未到期，不读取其许可文件或重跑门禁；有界储备保持2/2、准备预算0，不建立第三个模型候选。
- 17:53:58上海时间完成Y54一次真实限定提交依赖复查：pipeline/authorizations/AdaRES-Y54_commit.json仍不存在；原冻结bundle/native review绑定正确，七个模型/配置文件字节均与已测哈希一致、全为untracked，Git索引无该七文件，当前HEAD=26ee1cec57c53ee2f94d6969d0bb637565b6ed59，未形成Y54已测文件提交。没有新增人类许可或指定匹配commit，不能扩展Y50旧许可。
- 观测保存为pipeline/analyses/y54_authorization_recheck_20ed9a2a816c421c9aa9c7bc716ce19a.json，SHA256 5adc2acc26f83246d2207bb168edf38252d224493408ebbfd196fbcc9ed7e13c。保持原Y54 blocked/human_authorization，追加复查证据/历史，将retry_after设为实际复查后四小时2026-10-17T21:53:58.735844+08:00，wake_paths保持。该时间是复查阈值，不是已核实的调度承诺。
- 复查使用项目内可复用helper，既有冻结脚本不改；其它106个候选的结构摘要前后一致，未到期Y52及历史closed/rejected不变。没有原生前后向、资源、GPU探针、模型/配置修改、实现修订、Git提交、准入或训练启动。不是新门禁PASS、方法收益或GPU不足。
- 已通过规范入口记录20261017-175436_other.md。共享前台源码仍为a78da0060b280300359adc58a20f6c325cabb487313492112e8fc0bf4535bee6，既有closed冷却故障未修复，未重跑回归或修改共享程序/调度/目标/历史。Y50已于13:50自然终态归档，旧行动字段提及的正常核对不代表仍在等待训练结束。
- 科学状态无变化：历史CSV最佳Y36 mIoU 0.5110、基线B1 0.5020、目标0.515，差0.0040；本轮无新训练或评估指标，Y52/Y54训练收益未知。完整六段审计与七节记忆在租约内留档；当前postflight尚未执行，不能预称完成。
- 收据用量仅本轮begin至收据前的本地可见响应窗口，实际token在收据生成前未知；不含更早阅读、迟到记录与最终答复，不是账单。缓存已含input、推理已含output，不重复相加。实际产出为一个已有候选的到期授权/已测字节/Git依赖复查及证据更新；新科学候选0、科学测试0、训练启动0。

## 5. Decisions Made
- 保持同一Y54和原七文件，不自动git add/commit、不写admission或派发。以后需明确仅这七文件的提交许可或真实匹配commit，再核验最新goal、源、六门禁、commit回执及实时GPU，经现有gated dispatcher并观察原生步数。诊断W2权重不部署，Y50旧许可不扩大。
- 本轮进展是到期提交依赖的真实观测，而非新的科学证据。以external_resource_blocked、attempted_ids=[AdaRES-Y54]和当前证据留档；其余候选不变，不虚构新机制、等价测试、GPU资源阻塞、路线穷尽或deadline接近。
- 存在本轮真实到期复查证据，可执行一次规范finish；只有PASS/COMPLETE及独立核实租约/pending消失才称执行完成。若失败则保留精确错误并abort，不改门槛或重复无效校验。成功收尾不代表方法有效、许可解除或共享冷却缺陷修复。
- 共享脚本修复与候选提交许可仍为不同授权，不重复催问既有请求。当前授权阻塞未实质变化，无新增训练指标，会话保持静默；释放后不写项目。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
