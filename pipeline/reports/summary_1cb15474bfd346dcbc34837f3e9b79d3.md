# Research Pipeline Report - AdaRES_project
快照时间：2026-10-16T05:23:44.003004+00:00；生成报告时快照年龄：2888 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-16 13:23:43
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
================================================== 2026-10-16 13:23 : AdaRES_project ==================================================

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
- 研究储备 1 / 2；储备准备预算 1，不增加训练并发或启动授权。
- 候选累计记录 101：complete=14 / rejected=74 / closed=12 / validation=1。
- 否决范围：unclassified=72 / implementation=2；实现修订累计 3；收据文件 29（含修订或重试，不代表成功率）。

## 4. This Tick's Actions
- 同一design租约续做；TRAIN24捕获传输超时后恢复同一195267813字节文件并校验SHA，无GPU重捕获。新增24场景96表达的Y49原判据见证0，原32表达0保留；6314子图无循环回退，不执行无依据修订。
- 本轮唯一新储备Y50：完整128查询等面积/等Gram固定对照32种子，实际W2指示器误差0，新统计可区分；冻结原TRAIN8及新增TRAIN24共128表达均非恒定。不是上游全模型缺信息或训练收益证明。
- 独立包父字节/剥离AST/初始化/RNG/最终调用控制通过；native TRAIN8/32缓存对应精确重复且错配差非零，新增4张量梯度通过；released初始化实际tools.train两步、全4张量更新、模型/AdamW/PolyLR精确恢复。
- 成对推理中位父/Y50 0.090738/0.090892秒；压力0.697397/0.703661秒，预留峰值14.080/14.352GiB。仅测得可运行性，非Pareto优势。
- 两项测试程序兼容失败及原始日志保留，模型/判据不改；压力原始模板Y46字样/修订号已另文澄清，不重写历史结果。
- 14:05:14+08实时核验：GPU0外来PID19723/100%/24102MiB保留；GPU1无PID/0%/1MiB。正式准入0、训练启动0。收尾状态PENDING，以finish和独立租约释放核验为准；科学状态Y49实现rejected、Y50 blocked/human_authorization。
- 收据用量及产出待finish后读取；仅begin至收据本地窗口，不含早先读档/最终回复，缓存属于输入、推理属于输出，不重复相加、不当账单；缺项未知。

## 5. Decisions Made
- Y49当前实现因实际高成本及未补齐诊断拒绝，failure_scope=implementation；不普遍证伪全局机制，不重开既有拒绝候选。
- Y50科学准备验收通过但仍未提交；canonical契约禁止自动git add/commit，旧HEAD不含untracked候选，不能形成真实代码绑定receipt。待明确授权限定提交已测Y50字节，再复核目标/源/证据并走canonical准入派发。
- 本轮按真实提交授权外部阻塞提前收尾，不声称无信息增益/路线穷尽或GPU不足，不耗尽deadline、不弱化门槛。保留4小时重查与授权唤醒条件，不修改调度。
- Y50无训练后指标，现有最好指标没有新提升证据；后续沿原冻结FIRST配对/场景聚类/大小目标伤害与资源方案评估。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
