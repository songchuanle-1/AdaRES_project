# Research Pipeline Report - AdaRES_project
快照时间：2026-10-27T16:48:07.646347+00:00；生成报告时快照年龄：323 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-28 00:48:07
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
| server-1 | 0 | 未知 | 未知 | 未知 |
| server-2 | 0 | 未知 | 未知 | 未知 |
- 可用 GPU：未做实时占用确认，不能据此分配。
- 监控摘要（begin 已执行，不再重复探测）：
```text
================================================== 2026-10-28 00:48 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y63_peer_bilinear_res] Best=0.5074 27/30 no_improve=5/30 gpu=0 ? 序列: 0.5034, 0.5034, 0.5000, 0.5003, 0.5026

--- server-2 (192.168.1.163) ---
  无训练进程

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y63 | configs_song/model_paper5/adares_y63_peer_bilinear_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 4；准入预算 2。
- 研究储备 0 / 3；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 117：complete=16 / rejected=79 / closed=12 / blocked=9 / training=1。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 41（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve the paused original Y57 and saved model/optimizer/scheduler checkpoint. Do not queue a from-scratch restart or ask repeatedly to resume. Await explicit user instruction; keep fixed30/seed1999 and do not preempt the continuing Y56 on GPU0. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 上海10月28日00:48:29至00:48:34核实Y63仍由server-1 GPU0原PID3071037训练，第27/30轮2600到2608/4686递增；六源码/原生YAML确定性适配/marker82f/独占均通过。证据pipeline/checks/y63_server1_progress_67b0bd14ecf4457ea74a11ce44533fe9.json，SHAb2c36b7a4268bcc4fa5ed0e28f90b01c88fe3ed4eddcfd7374fac0a716c8439f。
- 实现Y65原生检查准备脚本：probe_y65_native_smoke.py、prepare_y65_native_smoke.py、test_y65_native_smoke_plan.py，Python3.8 AST编译通过。实际执行python -B scripts/prepare_y65_native_smoke.py --token 67b0bd14ecf4457ea74a11ce44533fe9；8项CPU守卫测试通过，包括缺少第二步/多漏调用/首步非零/无实际次步响应/非有限或假布尔证据/反馈绕过下游/错GPU拒绝。日志pipeline/checks/y65_native_plan_guards_67b0bd14ecf4457ea74a11ce44533fe9.log，SHA84b3958b823126666d163c6348eec65ec7a77b3f6eb54bd84146eca19d1323d4。
- Y65原CPU测试七文件逐字节与旧manifest、payload、测试结果一致，按已有项目授权仅提交这七文件，commit9706a8c36bead4459668ab66a3d1abc39344ef8b、父e278。当前goal/原训练入口验证通过，真实拒绝旧goal/篡改源码；Y64当前冻结plan仍有效，Y63原准入仍有效。无push、生产sync、队列或running变更。
- 保存Y65未执行原生计划pipeline/checks/y65_native_67b0bd14ecf4457ea74a11ce44533fe9_plan.json，SHA42023c049476fdda20f88b167a644f61282217916b5756349cc6f6f127aee5ee；version同前缀_version.json，SHA2ddb793b466ecc969f60ae00b7f15d86bd8660acaf2f94231d29c84040231df0。计划要求同YAML/真实TRAIN/tools.train两步、六feedback/七indicator、独立logistic公式、输出被下一coarse attention实用、首步零锚点/次步非零响应、有限非零方向梯度与输入梯度/峰显存。native_smoke_executed=false/admitted=false；CPU守卫不是GPU训练或效果证据。
- 本轮canonical initial/final资源证明：server-2 GPU0外来PID13813、final98%/24138MiB，server-1 GPU0为Y63 PID3071037、100%/22756MiB。final证据分别pipeline/checks/gpu_occupation_server-2_final_67b0bd14ecf4457ea74a11ce44533fe9.json(SHAb6c314546d4b76d6fd363635921456df06acf3917135035f5d2a0695b9336cd9)和server-1同前缀(SHAe607c1b0a4d1297b784b8486ef4988c87238b21439e7bbd3886b6d2e429352f6)。另次只读server-2 hostname/97%/24138MiB/85°C/280W，两服无授权空槽。
- 更新七标题memory并保留Y57完整恢复边界、六项冻结后评与历史负结果；本报告等待canonical finish核验，不预称收尾通过。

## 5. Decisions Made
- 优先推进已有Y64/Y65，真实GPU占用阻塞原生检查时完成不占GPU的下一步；本轮不创建重复新变体或为了额度重放旧CPU测试。
- Y63继续固定30轮原训练；不因近期验证波动调epoch/selector/温度/阈值/损失/特征，不迁移或挤占现有任务，不恢复Y57/GPU1，不用server-6、不跨项目、不改调度。
- 授权空槽出现后先推进Y64。server-1方案须先核Y63真实终态并canonical归档后activate，再重核GPU0物理独占空闲；server-2方案保持280W。随后Y65按已保存当前版本执行原生两步，实际通过后冻结原全量后评协议、准入入队并当轮尝试启动。未经实际原生检查不把准备记录冒充待跑或训练结果。
- 无新增用户待办，现有项目操作授权继续有效；当前仍不能据训练日志或CPU测试声称创新有效、精度提升或目标达成。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
