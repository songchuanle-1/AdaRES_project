# Research Pipeline Report - AdaRES_project
快照时间：2026-11-07T22:54:40.743430+00:00；生成报告时快照年龄：480 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-08 06:54:40
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
================================================== 2026-11-08 06:54 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y90_query_peer_affinity_res] Best=0.5038 28/30 no_improve=13/30 gpu=0 ? 序列: 0.5031, 0.5013, 0.5007, 0.5017, 0.4978

--- server-2 (192.168.1.163) ---
  [adares_y91_hidden_confidence_gate_res] Best=0.4942 11/30 no_improve=0/30 gpu=0 ? 序列: 0.4657, 0.4703, 0.4741, 0.4886, 0.4942

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=1
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y91 | configs_song/model_paper5/adares_y91_hidden_confidence_gate_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y90 | configs_song/model_paper5/adares_y90_query_peer_affinity_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 4；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 7、待资源 0、数值诊断待解 1、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 145：complete=16 / rejected=79 / closed=12 / blocked=36 / training=2。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 68（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve paused Y57 and its saved model/optimizer/scheduler. Do not resume or restart it under the generic GPU1 resource grant; wait for explicit Y57 continuation. Preserve current Y65/Y66/Y67. | 2026-10-22T00:26:29.471010+08:00 |
| AdaRES-Y67 | Preserve original complete fixed30 products, signed receipt and full frozen GPU1/390W FIRST/evaluation/cost. Wait for explicit user resource reauthorization; do not use another card or restart training. | 2026-11-04T06:58:39.676817+00:00 |
| AdaRES-Y70 | Preserve original complete fixed30 products, signed receipt and full frozen GPU1/390W FIRST/evaluation/cost. Wait for explicit user resource reauthorization; do not use another card or restart training. | 2026-11-04T06:58:39.677616+00:00 |
| AdaRES-Y73 | Preserve original complete fixed30 products, signed receipt and full frozen GPU1/390W FIRST/evaluation/cost. Wait for explicit user resource reauthorization; do not use another card or restart training. | 2026-11-04T06:58:39.678374+00:00 |
| AdaRES-Y75 | Preserve original complete fixed30 products, signed receipt and full frozen GPU1/390W FIRST/evaluation/cost. Wait for explicit user resource reauthorization; do not use another card or restart training. | 2026-11-04T06:58:39.679091+00:00 |
| AdaRES-Y78 | Preserve original complete fixed30 products, signed receipt and full frozen GPU1/390W FIRST/evaluation/cost. Wait for explicit user resource reauthorization; do not use another card or restart training. | 2026-11-04T06:58:39.679826+00:00 |
| AdaRES-Y81 | Preserve original complete fixed30 products, signed receipt and full frozen GPU1/390W FIRST/evaluation/cost. Wait for explicit user resource reauthorization; do not use another card or restart training. | 2026-11-04T06:58:39.680927+00:00 |

## 4. This Tick's Actions
- 仅处理 AdaRES_project；实际 CODEX_THREAD_ID 与前台 owner 一致，后台 Pipeline-AdaRES_project 仍为 Disabled。begin 已查进展并尝试派发现有合格队列；本轮快照 ready waiting 为0，未出现可派发合格任务。
- 真实原生进度核验：Y90 初次观察为 Train28/30、3486/4686；Y91 为 Train11/30、2606/4686。两者原 PID、来源链、实际运行文件、原生 YAML、350W/280W 与递增原生步数均已确认；收尾再以 final 标签刷新，精确计数保留在当轮生产观察证据，不把运行登记或 PID 单独当作健康证据。
- 两台授权 GPU0 均由原训练占用；收尾顺序收集 canonical 只读物理占用并保留其实际 resource_check/tick_attempt。没有停止、重启、迁移、抢占、checkpoint加载、提前归档或GPU原生测试。
- 本轮只做状态及资源核验，未新增候选、CPU科学运行、GPU原生任务、操作准备、准入、入队、训练启动或归档；已有 Y92/Y93 实现和证据保持，不重放已完成的CPU、原生计划、操作守卫或终态helper，不把合成CPU或有限梯度当效果结论。
- 已备份并更新七节短记忆，保留历史快照链和原运行收据；完整六节报告由官方 renderer 生成。科学目标字节、数据协议、原独立运行环境、精确路径映射、seed1999/fixed30及全部原冻结FIRST/成本要求保持，未产生新的正式效果或晋升结论。

## 5. Decisions Made
- 沿用既定顺序，Y92先于Y93。Y90仍在原server1 GPU0训练，尚不满足Y92真实原生检查前置；等待其自然fixed30、原last/best/model/optimizer/scheduler稳定有限及TrainVal/source/YAML/receipt/freeze完整精确核验归档，再以实时独占空闲350W GPU0执行保留的bound native计划。日志结束不替代完整checkpoint验证。
- Y92真实native通过后，用实际证据为仍在跑的Y91新建并测试支持附加Y92的精确终态绑定，再选择、准入、canonical派发并核验递增步数；旧签名helper不得修改。Y93如遇HEAD/owner改变须先建立新的同字节运行绑定，不擅自执行旧绑定。
- server2 GPU1全部AdaRES使用及其原冻结后评/成本继续用户暂停，Y57实验另行暂停；不迁移或替换历史资源。仅server1 GPU0既有350W与server2 GPU0固定280W，各卡一个独立实验；禁止server6，不改功率或调度。阻塞条件未变，无新用户处理事项。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
