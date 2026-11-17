# Research Pipeline Report - AdaRES_project
快照时间：2026-11-12T13:48:17.917560+00:00；生成报告时快照年龄：1228 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-12 21:48:17
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
================================================== 2026-11-12 21:48 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y101_query_ffn_skip_res] Best=0.3872 3/30 no_improve=0/30 gpu=0 ? 序列: 0.3235, 0.3872

--- server-2 (192.168.1.163) ---
  [adares_y100_appearance_input_skip_res] Best=0.5164 30/30 no_improve=6/30 gpu=0 ? 序列: 0.5119, 0.5116, 0.5132, 0.5120, 0.5142

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
| AdaRES-Y101 | configs_song/model_paper5/adares_y101_query_ffn_skip_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 4；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 7、待资源 0、数值诊断待解 2、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 154：complete=16 / rejected=79 / closed=12 / blocked=45 / training=2。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 7；收据文件 78（含修订或重试，不代表成功率）。
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
- 运行：AdaRES-Y101 Train3/30 step2519/4686；AdaRES-Y100 Train30/30 step3759/4686。原PID/源码/原生YAML/精确数据/私有环境/350W280W与原生步数递增已验证；两张授权GPU0物理busy/reserved，未停止或挤占原训练。
- 新准备：Y101真实原保存目录20261112_200238只读发现，30项生命周期守卫通过；仅准备未来精确终态工具，未加载活跃权重或归档。Y99的上一轮完整归档保留，不重做。
- 新准备：独立W2变体Y103保留完整空间投影MLP并增加全32维线性旁路到原视觉记忆；首次CPU原生YAML/完整模型/原TRAIN数据加载、标量/梯度/W2锚定/因果旁路/两步合成解码器更新通过，7个测试文件隔离保存，10项证据和8项运行守卫通过。操作字段名未同步导致的原失败记录保留，v2按实际源码/owner/收据/索引补齐绑定，未重跑科学检查。
- Y102优先，Y102/Y103首次GPU原生检查均等待匹配原训练自然30轮、完整精确有限检查点与规范归档，再验独占空闲授权GPU0。CPU通过不等于原生或收益。新科学ID1、首次CPU实现检查1、科学重跑0、原生GPU检查0、准入0、新训练0、归档0、FIRST0、成本0。

## 5. Decisions Made
- 按本轮快照补齐两项有界待跑准备，不为凑额度重跑旧证据。资源仍忙，未执行GPU原生检查或准入；下一步Y102先于Y103，原生通过后同轮按完整原设备协议继续准入与门控派发。
- 仅AdaRES；server2GPU1全部使用和Y57单独暂停保持，不迁移冻结后评/成本、不重启受保护实验、不用server6、不改数据/功耗/调度、不干预外部任务。无新增用户处理事项。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
