# Research Pipeline Report - AdaRES_project
快照时间：2026-10-02T07:48:16.157997+00:00；生成报告时快照年龄：167 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-02 15:48:15
================================================================================
  Target: AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515.

## Phase 1: Metric Target
  Target: mIoU ≥ 0.5150
  [✗] Champion: AdaRES-W2 — mIoU=0.5100 (gap to target: +0.0050)
       Epochs: 30/30, State: done_complete
  Phase 1: NOT YET — continue exploring ✗

## Baseline
  Baseline: AdaRES-B1 — mIoU=0.5020
  Champion vs Baseline: γ = +0.0080

## Phase 2: Story Verification
  (等待 Phase 1 达标后启动)

## Overall
  Phase 1: champion=0.5100, target=0.5150, gap=+0.0050
================================================================================
```

## 1. GPU Overview
- 仅列授权卡；未采集利用率、显存或进程时标为未知，不推断空闲。
| Server | GPU | Util | Memory | Process |
|---|---|---|---|---|
| server-2 | 1 | 未知 | 未知 | 未知 |
- 可用 GPU：未做实时占用确认，不能据此分配。
- 监控摘要（begin 已执行，不再重复探测）：
```text
================================================== 2026-10-02 15:48 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y18_point_measure_res] Best=0.5076 23/30 no_improve=1/30 gpu=1 ? 序列: 0.4971, 0.4981, 0.5065, 0.5076, 0.5037

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y18 | configs_song/model_paper5/adares_y18_point_measure_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 1；准入预算 0。

## 4. This Tick's Actions
- 已读取canonical前台规则及最新execution_mode/goal/SERVERS/closure_policy/项目AGENTS，实际任务ID与绑定一致。遵循最新用户授权：仅GPU1，Y23暂停等待Y18完整30轮结束后在GPU1续训，GPU0保留不可使用；Windows后台Disabled，未更改调度或操作其他项目。
- 快照北京时间2026-10-02 15:48:16，模式event，原因research_artifact_changed/research_work_changed；活跃1/1、运行1、等待0、准入预算0、到期候选0、未到期blocked1，无新增终态。begin已运行monitor/goal各一次，未重复preflight。
- 15:49:14至15:49:19复用动态token有界脚本，经密钥BatchMode SSH核实server-2主机dl01-MS-7C60、Y18原PID7449/config/GPU1/独有日志，Train23/30步数523→533/4686持续递增。GPU1设备显存22720MiB、利用率28%，非候选隔离峰值；未查询或使用GPU0。
- 两次采集各只读至多24000字节日志尾，并记录尾部hash，未见Traceback/OOM仅限该窗口，不称全日志无故障。新证据pipeline/analyses/y18_progress_cfe21a220bb643588b3be90679a9cb0d.json SHA256=42a3c2024a460931740f4534ddb47d7736a461e573f42ec6639af8d323c33f01及原始日志已保存，追加Y18稳定ID并更新七段短记忆。
- 科学状态Y18training/Y23用户依赖blocked。Y18中途best从0.5070更新为0.5076，预检23轮/no_improve1、latest0.5037，仍低于W2历史0.5100与目标0.5150，不能据此宣称终态有效性/Pareto或修改协议。Acc@0.25/Acc@0.5/完整延迟/隔离峰值本轮未独立采集，记未知。
- Y23原pause/epoch4 checkpoint/AdamW/PolyLR/冻结评估证据及未到期blocked状态不变，依赖尚未满足，不重复guard/检查点加载、不改genericwaiting、不重启。未新增候选、准入、训练或科学拒绝。
- 执行完成以finish实际postflight和独立锁释放为准，报告生成时未完成。实际产出1份新Y18双观测证据、1份原始日志、1个候选证据更新、七段记忆及六段报告。收据用量仅本地begin至收据部分窗口，不含前置阅读/晚到记录/最终答复；生成前用量未知，缓存属于输入、推理属于输出，不重复相加或当账单。

## 5. Decisions Made
- 无新科学决策；继续Y18原GPU1固定30轮训练，不因中途改善改变冻结协议、不提前停止以满足依赖、不使用GPU0或新增填槽变体。
- 等Y18正常30轮done_complete事件出现，优先检查原终态checkpoint与GPU1真实空闲，以该轮token执行已有guardedresume，同一Y23从epoch5至30恢复模型/优化器/调度器；原生步数核验后才称续训已启动。
- 本轮只有正常中途进度，无需用户介入。未制造无信息增益/外部阻塞收尾、不改历史design_cycle；保留自动化，精确下次调度未知，以已保存自动化为准。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
