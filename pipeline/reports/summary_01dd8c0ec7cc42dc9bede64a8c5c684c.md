# Research Pipeline Report - AdaRES_project
快照时间：2026-11-13T23:43:15.678316+00:00；生成报告时快照年龄：624 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：4
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-14 07:43:15
================================================================================
  Target: AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515.

## Phase 1: Metric Target
  Target: mIoU ≥ 0.5150
  [✓] Champion: AdaRES-Y100 — mIoU=0.5164 (target met, exceeded by +0.0014)
       Epochs: ?, State: done_complete
  Phase 1: PHASE1_REACHED ✓

## Baseline
  Baseline: AdaRES-B1 — mIoU=0.5020
  Champion vs Baseline: γ = +0.0144

## Phase 2: Story Verification
  (champion 已达标，检查论文故事是否闭环)

  [✓] Motivation (为什么需要这个方法): covered
  [✓] Method (我们的方法是什么): covered
  [?] Gap (champion vs baseline 提升): covered (γ=+0.0144)
  [✗] Ablation (核心设计是否必要): missing
  [✗] Stability (多seed稳定性): missing
  [✗] Generalization (跨setting泛化): missing
  [✗] Limitation (失败案例分析): missing

  Story: INCOMPLETE ✗ (2/7 elements covered)
  Phase 2: 需要补充消融/seed实验

  Suggested next steps:
    - 设计消融实验验证 champion 核心设计选择
    - 跑 champion 的多 seed 稳定性验证

## Overall
  PHASE1_REACHED — champion 达标，Phase 2 进行中
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
================================================== 2026-11-14 07:43 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  无训练进程

--- server-2 (192.168.1.163) ---
  [adares_y102_geometry_mask_skip_res] 1/30 尚无验证 - gpu=0

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=1
  server-1:
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y102 | configs_song/model_paper5/adares_y102_geometry_mask_skip_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 4；准入预算 0。
- 研究储备 1 / 0；储备准备预算 0，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 7、待资源 0、数值诊断待解 2、未分类阻塞 0。
- 非终态候选与排队项合计 1；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 155：complete=19 / rejected=80 / closed=12 / blocked=42 / validation=1 / training=1。
- 否决范围：unclassified=72 / implementation=5 / hypothesis=3；实现修订累计 7；收据文件 79（含修订或重试，不代表成功率）。
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
- 完成：Y92原四方法各两遍完整9508表达式/141场景、修正的原配对10000/1999/重复噪声、首次原冻结串行/原生压力成本及源码和原检查点后审计均已核验；原训练、预测、终态检查没有重跑。
- 科学负结果：正式FIRST0.5093426233；大目标相对Y36差值-0.0130957302，95%CI[-0.0252263420,-0.0004629638]，触发原冻结伤害判据。仅拒绝固定30轮/best20/seed1999本次实现，不宣称整个机制证伪；整体配对CI均跨零，未达0.515，不晋升、不重开。
- 成本：参数+65536，TRAIN8串行中位82.49ms vsW2 87.95ms，大场景205.42ms vs200.21ms；原生压力峰值保留13.44727GiB vs13.40039GiB。仅冻结样本/顺序描述，不证明全数据或Pareto；原权重/源码不变。
- 运行：已归档Y90原26源码文件/原YAML/私有环境导入和11项操作身份边界通过，在自己原server1GPU0/350W首次启动四组全量后评，已实测原生计数递增；无新训练、准入、迁移或重启。
- Y102本轮原PID4747/server2GPU0/280W仍Train1step7，原观察时间和现场保留；不重启、不重复此前拒绝的栈附加或索取管理员许可。

## 5. Decisions Made
- 保持本轮零新创建/准入/新训练和Phase2边界；Y92按原大目标判据保留实现范围负结果。Y90完整原配对/成本/后审计后再处置；Y92/Y94拒绝ID不得自动重开。
- 本轮16项到期：Y92负结果完成、Y90实际后评、其他12项原GPU0工作等待真实占用；Y57单独暂停和Y97原数值失败继续保留。GPU1六项原冻结后评仍暂停且不迁移，Y86失败不重跑；不跨项目、不用server6、不改goal/数据/收据/源码/功耗/调度/上限或外部进程。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
