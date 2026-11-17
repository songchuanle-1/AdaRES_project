# Research Pipeline Report - AdaRES_project
快照时间：2026-11-09T02:47:48.765325+00:00；生成报告时快照年龄：2148 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-09 10:47:48
================================================================================
  Target: AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515.

## Phase 1: Metric Target
  Target: mIoU ≥ 0.5150
  [✗] Champion: AdaRES-Y91 — mIoU=0.5114 (gap to target: +0.0036)
       Epochs: ?, State: done_complete
  Phase 1: NOT YET — continue exploring ✗

## Baseline
  Baseline: AdaRES-B1 — mIoU=0.5020
  Champion vs Baseline: γ = +0.0094

## Phase 2: Story Verification
  (等待 Phase 1 达标后启动)

## Overall
  Phase 1: champion=0.5114, target=0.5150, gap=+0.0036
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
================================================== 2026-11-09 10:47 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  无训练进程
  [adares_y92_mask_feature_quadratic_res] 进程未见 (deathwatch 1/3) — 暂不处置

--- server-2 (192.168.1.163) ---
  [adares_y93_evidence_hidden_skip_res] Best=0.5015 17/30 no_improve=1/30 gpu=0 ? 序列: 0.4959, 0.5007, 0.4949, 0.5015, 0.5015

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=1
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y93 | configs_song/model_paper5/adares_y93_evidence_hidden_skip_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y92 | configs_song/model_paper5/adares_y92_mask_feature_quadratic_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 4；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 7、待资源 0、数值诊断待解 2、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 150：complete=16 / rejected=79 / closed=12 / blocked=41 / training=2。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 7；收据文件 70（含修订或重试，不代表成功率）。
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
- 仅处理AdaRES_project；实际会话归属、Disabled后台调度、正式skill/experiment_first、AGENTS/goal/SERVERS和GPU1暂停核验，沿用本轮原租约/快照/deadline，先查进展派发现有任务。
- 完成：Y92自然结束后只读精确核验原Train/Val1..30与稳定有限last30/best20/model/optimizer544/scheduler30-20，原目录20261108_090202，canonical归档一次，原产物及收据保留。训练best0.5092514753341675不是正式FIRST，也不宣布科学收益。
- 新准备并启动：Y94原生YAML/真实TRAIN两步loss/backward/optimizer和原始位置/六层self-attention/硬mask路径实际通过，loss为7.886719703674316和8.23161506652832。完成10项操作、19项准入检查及Y93原运行29项生命周期+8项真实native加法绑定检查。仅选择已测七文件/保留其他index与working changes，原设备完整冻结协议/收据/入队/canonical安全同步派发均完成，原生训练步数增长已确认，不以PID代替启动。
- 运行：AdaRES-Y94 Train1/30，步438/4686；AdaRES-Y93 Train18/30，步3556/4686。Y94为server1GPU0/原350W、Y93为server2GPU0/280W，Y93原PID47380/source/YAML/收据/私有环境/数据和检查点目录不变，未加载或归档Y93检查点。两卡canonical只读占用均busy+reserved，保持GPU1外部任务不动。
- 保留Y95/Y96/Y98已测CPU/源码/native计划，不重放历史CPU/version/plan或已归档终态；当前HEAD已为Y94，后续执行前必须新建并测试同字节版本/当前owner绑定。Y97原始及修订1数值失败保持，无重试、阈值放宽或新修订。Y94精确终态位置/生命周期准备仍是后续动作。
- 本轮新科学ID0、科学CPU尝试0、真实GPU原生1、准入1、训练启动1、精确归档1。完整9508表达/141场景/candidate+Y36-W2-B1各两次/FIRST主报告/场景bootstrap10000-seed1999/TRAIN1487-3325/全部有害子组与指标/原设备cost协议保持。已精确备份七节记忆，保留collector实际resource_check，再正式report/finish，不预称收尾PASS。

## 5. Decisions Made
- 优先完成已有Y94的真实native到本轮入队和启动，不为凑额度新增或复制机制。后续观察Y93/Y94；Y93自然30轮仅使用新finalize_y93_fixed30_y94.py精确终态链。Y94先准备原始声明保存目录的新终态链，不从启动时间猜目录，不加载live checkpoint。
- Y92完整原server1GPU0/350W冻结后评等待Y94；原GPU1冻结后评继续挂起，不迁移或替代资源。科学goal字节、原收据、原数据/私有runtime/mappings、固定30轮seed1999、禁用项、专用恢复要求、功耗和调度不变。server2GPU1全部使用和Y57实验分别暂停，禁止server6/跨项目/外部任务干预，无新用户操作需求。
- 审计文字澄清：不可变advance_y94准入decision模板继承了Y92静态mask二次读出一句，并非实际执行机制；源代码、独立YAML、native路径和签名receipt均是Y94有界query位置self-attention偏置。原文和哈希保留，本条与独立decision明确纠正，不伪改历史证据。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
