# Research Pipeline Report - AdaRES_project
快照时间：2026-11-11T11:52:10.592567+00:00；生成报告时快照年龄：973 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-11 19:52:09
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
================================================== 2026-11-11 19:51 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y99_spatial_mask_skip_res] Best=0.3620 3/30 no_improve=0/30 gpu=0 ? 序列: 0.3346, 0.3620

--- server-2 (192.168.1.163) ---
  [adares_y98_appearance_mask_skip_res] Best=0.4994 28/30 no_improve=7/30 gpu=0 ? 序列: 0.4926, 0.4903, 0.4921, 0.4916, 0.4893

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=1
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y98 | configs_song/model_paper5/adares_y98_appearance_mask_skip_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y99 | configs_song/model_paper5/adares_y99_spatial_mask_skip_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 4；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 7、待资源 0、数值诊断待解 2、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 151：complete=16 / rejected=79 / closed=12 / blocked=42 / training=2。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 7；收据文件 76（含修订或重试，不代表成功率）。
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
- 运行：AdaRES-Y99 Train3/30 step2188/4686；AdaRES-Y98 Train29/30 step1246/4686。两张授权GPU0均按原进程/源码/原生YAML/私有环境/350W280W训练，步数递增。无新物理终态，保留上轮Y96完整核验归档，不重放。
- 完成准备：Y99实际原目录20261111_180638只读发现并绑定，30项终态保护检查通过；未加载活跃权重、未归档、不停止训练。
- 新准备：Y100独立W2完整1024通道外观绕过原外观MLP到原sum_norm/视觉记忆，保留完整投影和全部输入。首次CPU原生YAML/完整模型/发布backbone/TRAIN加载通过，162738点8表达；标量float32/64、置换/局部性/符号/无RNG、七阶段基线与输入梯度对齐、因果旁路恢复及两次有限synthetic decoder更新通过。CPU结果不是GPU原生检查或效果证明。
- 保存七个实际测试科学文件到隔离版本，生产HEAD/主索引/原运行不变。10项实际证据和修正后8项运行计划检查通过；原单项旧字段元数据失败日志及原稿保留。两套原生GPU程序体不变，无科学重跑或放宽控制。

## 5. Decisions Made
- 两张授权GPU0均实测busy/reserved，Y100原生检查等待匹配原Y98或Y99自然固定30轮、完整精确有限检查点核验和规范归档，再验空闲独占资源。原生检查通过后同轮冻结完整原设备后评/成本协议、准入并尝试派发；当前不入队、不声称GPU PASS或效果。
- 仅AdaRES；server2GPU1全部使用与Y57单独暂停保持，不迁移冻结后评/成本、不重启受保护实验、不使用server6、不改数据/功耗/调度、不干预外部任务。新科学ID1/CPU科学尝试1/GPU原生0/准入0/训练0/归档0/FIRST0/成本0；无新增用户处理事项。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
