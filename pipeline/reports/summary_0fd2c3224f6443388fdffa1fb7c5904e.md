# Research Pipeline Report - AdaRES_project
快照时间：2026-11-01T06:44:30.825720+00:00；生成报告时快照年龄：2317 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-01 14:44:30
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
| server-2 | 1 | 未知 | 未知 | 未知 |
- 可用 GPU：未做实时占用确认，不能据此分配。
- 监控摘要（begin 已执行，不再重复探测）：
```text
================================================== 2026-11-01 14:44 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y72_visual_mask_refinement_res] Best=0.5004 25/30 no_improve=10/30 gpu=0 ? 序列: 0.5003, 0.4981, 0.4922, 0.4912, 0.4938

--- server-2 (192.168.1.163) ---
  [adares_y74_language_update_gate_res] Best=0.4898 11/30 no_improve=0/30 gpu=0 ? 序列: 0.4699, 0.4720, 0.4821, 0.4893, 0.4898
  [adares_y73_initial_query_anchor_res] Best=0.4974 30/30 no_improve=8/30 gpu=1 ? 序列: 0.4871, 0.4922, 0.4907, 0.4945, 0.4956

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=2
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y73 | configs_song/model_paper5/adares_y73_initial_query_anchor_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y74 | configs_song/model_paper5/adares_y74_language_update_gate_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y72 | configs_song/model_paper5/adares_y72_visual_mask_refinement_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 3（运行 3 / 认领 0 / 等待 0）；目标 5；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 128：complete=16 / rejected=79 / closed=12 / blocked=18 / training=3。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 52（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve paused Y57 and its saved model/optimizer/scheduler. Do not resume or restart it under the generic GPU1 resource grant; wait for explicit Y57 continuation. Preserve current Y65/Y66/Y67. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 先核实际owner/foreground配置/Disabled旧后台，查询原运行并尝试已有派发。Y73本轮初查仍Train30，继续保留原训练；同步准备Y75 GPU1 exact资源/版本/准入/派发helper及六守卫、Y72 server1自然收尾20守卫、Y76 server1备用原生计划与8守卫和实际CPU runtime/data/callback检查。
- Y73真实自然退出且完整Val30后，才按原准备依赖/receipt/freeze核验完整Train/Val1..30、稳定finite last30/best/model/optimizer544/scheduler30和21、完整原日志/source/YAML，并canonical done_complete归档。训练best0.4974086/epoch21只是训练监控，原完整FIRST/cost仍待复评。
- fresh GPU1独占idle/unreserved390W后实际执行Y75原生YAML/tools.train/TRAIN两步，loss4.61551094/6.30928421、单次真实query-language初始化与初始prediction消费/7共享头、有限梯度和真实更新通过；14准入负守卫、原完整Y64训练/评估内容核等、准入和七A文件安全选择/部署后canonical派发。PID65454的原生步数实际递增。
- Y72 PID311445和Y74 PID49891的原source/YAML/receipt/功耗及native增长核验；新observer只接受明确87019/c60/7da/3c/9bf和精确新增包。Y57独立暂停保持，科学goal/原数据/seed1999/fixed30/禁用项和调度不变。
- Y76已在server1 verified private runtime实际构建原模型/加载发布backbone/精确映射TRAIN162738点8表达并执行两合成CPU DEC optimizer步骤和原回调；不是native sparse GPU任务PASS，备用计划仍inactive。两台当前canonical occupation确认三授权卡busy且预约，Y76 GPU任务和16项原完整后评/cost等待。
- 保存原记忆精确备份、七段记忆、完整依赖/准备/运行/终态证据和当前handoff；本轮新增候选0、准入1、启动1、终态归档1。

## 5. Decisions Made
- Y75实际native PASS即本轮生成原完整freeze/收据并入队启动，不把执行资格当效益；Y73训练best不晋升或代替正式FIRST。保留所有原远端checkpoint和日志，不重训或重放已完成操作。
- 当前三卡已被Y72/Y74/Y75占用。下一较近空槽Y72需真实自然fixed30退出、原完整核验和canonical归档；fresh server1GPU0 idle/no compute/no reservation350W后才激活/运行Y76已有CPU核验方案，actual native PASS后再原完整freeze/准入/精确source契约/门控启动。
- server1GPU0及server2GPU0/1一卡一实验，既有350/280/390W不变；保留Y57原epoch27模型/优化器/调度器并等待单独明确续训请求，不用server6、不跨项目或改调度。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
