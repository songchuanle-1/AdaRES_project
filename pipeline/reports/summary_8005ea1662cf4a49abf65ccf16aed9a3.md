# Research Pipeline Report - AdaRES_project
快照时间：2026-11-11T17:47:27.363193+00:00；生成报告时快照年龄：737 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-12 01:47:14
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
================================================== 2026-11-12 01:46 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y99_spatial_mask_skip_res] Best=0.4893 10/30 no_improve=0/30 gpu=0 ? 序列: 0.4567, 0.4654, 0.4704, 0.4750, 0.4893

--- server-2 (192.168.1.163) ---
  [adares_y100_appearance_input_skip_res] Best=0.4407 5/30 no_improve=0/30 gpu=0 ? 序列: 0.3379, 0.3895, 0.4338, 0.4407

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=1
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y100 | configs_song/model_paper5/adares_y100_appearance_input_skip_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y99 | configs_song/model_paper5/adares_y99_spatial_mask_skip_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 4；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 7、待资源 0、数值诊断待解 2、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 153：complete=16 / rejected=79 / closed=12 / blocked=44 / training=2。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 7；收据文件 77（含修订或重试，不代表成功率）。
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
- 运行：AdaRES-Y99 Train10/30 step2170/4686；AdaRES-Y100 Train5/30 step4413/4686。原进程/源码/原生YAML/精确数据/私有环境/350W280W保持，原生步数递增；无新终态或派发故障，既有合格队列先检查，未重放历史终态工具。
- 新准备：Y102独立W2将全部原场景三维坐标相对原点质量场景中心的偏移，经零初始化线性连接加入原静态掩码，保留原mask MLP、完整输入、原批处理与七次读出。首次原生YAML/完整CPU模型/发布backbone/原TRAIN加载通过，162738点8表达；标量float32/64/centroid/投影/置换/平移/质量缩放/填充NaN/空与零mass/场景隔离/坐标质量梯度、七阶段基线与输入梯度对齐、因果mask/初始scores精确/旁路恢复及两次有限synthetic decoder更新通过。CPU不是GPU原生检查或效果证明。
- 七个实测科学文件保存隔离版本，生产HEAD/主索引/原运行/既有Y101及全部负证据保留；10项实际证据和8项运行计划检查通过。Y101准备不重跑，仍排在Y102之前。
- 本轮新科学ID1/CPU科学尝试1/GPU原生0/准入0/新训练0/归档0/FIRST0/成本0。未加载任何活跃权重。

## 5. Decisions Made
- 两张授权GPU0均实测busy/reserved。先Y101后Y102；首次原生检查均须匹配原Y99/Y100自然固定30轮、完整精确有限原检查点核验及规范归档，再验独占空闲GPU0。改变生产HEAD/原owner须新增同字节实测绑定，不静默复用旧parent；原生通过后同轮完整原设备协议/原运行兼容保护/准入/安全派发/步数核验。
- 仅AdaRES；server2GPU1全部使用和Y57单独暂停保持，不迁移后评/成本、不重启受保护实验、不用server6、不改数据/功耗/调度、不干预外部任务。无新增用户处理事项。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
