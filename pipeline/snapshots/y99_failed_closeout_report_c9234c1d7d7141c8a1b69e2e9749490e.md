# Research Pipeline Report - AdaRES_project
快照时间：2026-11-13T17:52:20.502964+00:00；生成报告时快照年龄：625 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：4
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-14 01:52:19
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
================================================== 2026-11-14 01:52 : AdaRES_project ==================================================

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
- 候选累计记录 155：complete=17 / rejected=79 / closed=12 / blocked=45 / validation=1 / training=1。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 7；收据文件 79（含修订或重试，不代表成功率）。
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
- 完成：原Y99四组各两遍完整9508表达式/141场景、原修正配对10000/1999/重复噪声结果已归集核验；首次原设备冻结串行成本、原生内存压力成本和源码/原检查点后审计完成，原训练/终态/预测没有重跑。
- 科学结果：FIRST mIoU0.5109262481；同设备Y36 0.5107215774、W2 0.5101072505、B1 0.5022221993。整体配对区间均跨零，预设尺寸子组无伤害判据触发，未达0.515；记录complete_inconclusive，不宣称增益、等效或晋升。
- 成本：参数+8192；TRAIN8串行中位84.97ms vs83.87ms，大场景205.03ms vs205.38ms，原生压力峰值保留13.40430GiB vs13.40039GiB；仅冻结样本/顺序的描述成本，不构成全数据或Pareto结论。
- 新准备：原已归档Y96完成26文件身份、原YAML/私有环境类导入及11项操作边界；无GPU或权重加载，未启动后评或新训练。
- 运行：Y102原PID4747/server2GPU0/280W仍Train1step7；保存真实停滞日志和现场，不重启、不重复附加此前拒绝的栈或索取管理员许可。官方占用显示server1GPU0空闲，已真实释放16项旧GPU0资源挂起到validation；可执行后评保留待续，不伪称资源忙。
- 收尾操作修复：原冻结resource_condition含note说明，初次本地整对象比较误报失败；原脚本/诊断/失败记录均保留，改为逐项核验候选/服务器/GPU0/350W，说明原样保留。复用已完成停滞诊断，没有重跑科学测试或更改官方校验器。

## 5. Decisions Made
- 按本轮experiment_first真实有结果的有界尝试完成；本轮零新训练/准入/创建预算保持。已完成Y99/Y101不重测，下一原设备后评从准备好的Y96继续；不以训练best代签正式FIRST或绕过Phase2。
- GPU1六项原冻结后评仍用户暂停且不迁移，Y57单独暂停；Y97原数值失败不重跑；保留原goal字节、运行收据、原数据、专用恢复规则及独立环境，不跨项目、不用server6、不改功耗/调度/外部进程。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
