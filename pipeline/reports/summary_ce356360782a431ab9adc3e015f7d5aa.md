# Research Pipeline Report - AdaRES_project
快照时间：2026-11-14T15:47:21.157492+00:00；生成报告时快照年龄：2066 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：4
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-14 23:47:20
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
================================================== 2026-11-14 23:47 : AdaRES_project ==================================================

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
- 研究储备 0 / 0；储备准备预算 0，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 7、待资源 0、数值诊断待解 2、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 155：complete=19 / rejected=82 / closed=12 / blocked=41 / training=1。
- 否决范围：unclassified=72 / implementation=7 / hypothesis=3；实现修订累计 7；收据文件 79（含修订或重试，不代表成功率）。
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
- 完成：本轮原server1GPU0释放后，复用Y88两遍原候选预测，仅补Y36/W2/B1各两遍。原观察器确认实际计数递增；用户暂停请求处理期间核验原工作已自然完成，现无本项目server1 GPU进程；八遍9508/141原预测、原日志/配对结果及精确暂停快照只读归档保留，没有训练重启或候选预测重跑。
- 完成：遵循用户最新原话“你暂停server1的实验”，已登记拥有者授权、将server1 allowed_gpus设为空并挂起12项未完成工作，覆盖旧server1授权与旧心跳文字；原AGENTS/SERVERS/待办已精确备份，科学goal及全部原准入收据、源码/权重、queue/running/实验账本保持。
- 完成：使用官方automation_update更新既有AdaRES提示词，并读取实际保存配置核对；原频率、启停、绑定会话和通知偏好不变。共享参考仅AdaRES代码块更新，其他项目字节保持。
- 运行保留：原server2GPU0 Y102 PID4747仍为原身份/源码6文件/YAML/私有环境，计数{'phase': 'Train', 'epoch': 1, 'total': 30, 'step': 7, 'count': 4686}；未重启或迁移。server2GPU1和Y57原单独暂停保持，未使用server6或干预其他项目。

## 5. Decisions Made
- server1所有GPU训练/派发/原生检查/冻结后评及成本暂停，恢复必须由拥有者再次明确授权；资源空闲、计时器、旧收据或旧心跳不构成恢复授权，不迁移到另一张卡。Y88原八遍后评已完成，旧“补基线”入口不应再次运行；原已准备成本入口未执行，完整正式比较/成本/后审计仍待后续合规处理，不宣称目标达成或自动提升。
- 本轮未新增科学ID、准入、训练或原生smoke，Y90已完成拒绝不重开，Y86/Y97原数值负证据及专用恢复要求均保留。不可变begin快照/deadline不修改；按最新暂停配置完成官方收尾，若旧快照校验不能接受中途资源变更，则保留证据并显式abort本轮，绝不恢复server1或削弱检查以通过。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
