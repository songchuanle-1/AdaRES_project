# Research Pipeline Report - AdaRES_project
快照时间：2026-11-08T18:48:02.294197+00:00；生成报告时快照年龄：2349 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-09 02:48:01
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
================================================== 2026-11-09 02:47 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y92_mask_feature_quadratic_res] Best=0.5093 21/30 no_improve=0/30 gpu=0 ? 序列: 0.4971, 0.5032, 0.5032, 0.4985, 0.5093

--- server-2 (192.168.1.163) ---
  [adares_y93_evidence_hidden_skip_res] Best=0.4724 7/30 no_improve=0/30 gpu=0 ? 序列: 0.3745, 0.4249, 0.4466, 0.4591, 0.4724

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
- 储备只计可推进候选及已排队项；另有待人工 7、待资源 0、数值诊断待解 1、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 148：complete=16 / rejected=79 / closed=12 / blocked=39 / training=2。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 70（含修订或重试，不代表成功率）。
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
- 仅处理AdaRES_project；已核验实际owner会话、Disabled后台调度、正式skill/experiment_first、AGENTS/goal/SERVERS及最新GPU1暂停记录。begin先查进展并尝试已有合格派发；初始ready waiting为0，无新终态。
- 运行：AdaRES-Y92 Train22/30，步2852/4686；AdaRES-Y93 Train8/30，步3934/4686；原PID/source/YAML/receipt/私有runtime、350W/280W及原生步数递增核实，canonical物理占用确认两GPU0 busy+reserved。GPU1及外部进程未干扰。
- 新实现Y97独立全token语言输入二次投影，保留原投影及同一lang_norm；原YAML、完整CPU模型和发布骨干/TRAIN初始化后，小组件批次置换torch.equal守卫首次实际执行失败，未生成CPU通过结果。
- 保存原失败程序和raw日志；单独CPU小组件诊断重复相同未改守卫，float32最大差1.1920928955078125e-7、64/70逐位相等，float64最大差2.220446049250313e-16、69/70逐位相等。虽处于原独立标量128eps容差内，但只作诊断证据，不能替代逐位守卫PASS。
- Y97按实现数值问题挂起，不放宽检查、不修改原模型/YAML、不进行修复或科学修订、不重命名重开。后续完整decoder锚点、共享native回调、两步反向均未执行；未留版、未生成native计划、未GPU检查、未准入/同步/派发。后创建的contract/tests/retainer源码均未执行，不冒充18项守卫通过。
- 较早Y94/Y95/Y96完整CPU准备/独立版本/native计划保持，不重复科学检查；原Y92/Y93终态计划只读核对哈希，未加载checkpoint或执行归档，已归档实验不重放。
- 本轮新增科学候选1、CPU候选检查1失败、CPU小组件诊断1、GPU原生0、准入0、启动0、归档0。七节记忆先精确备份，连续刷新当前训练及canonical物理占用后正式report/finish；保留collector实际resource_check，不预先宣称收尾结果。

## 5. Decisions Made
- Y97是implementation数值守卫失败，不是训练无效、缺用户授权或假设被反证。保留全部原证据与原检查；未来仅同一ID下事前登记、证据支持且预算内的局部实现修订可重测原完整控制，不凭计时重跑或改容差。
- 原Y92/Y93继续。Y94、Y95、Y96按原优先顺序等待匹配原owner自然30轮/完整有限checkpoint精确核验归档及实时独占空闲GPU0，然后真实native通过才按原freeze/admission/安全同步和递增counter推进。Y97挂起不阻断其他工作。
- 完整原FIRST/同设备cost、数据/私有环境/mappings/seed1999-fixed30/有害子组/禁用/功耗/调度/恢复要求、科学goal字节和原receipt保持。GPU1全部使用与原GPU1后评/cost继续暂停，Y57另行暂停，不迁移替代，禁止server6；无需新用户操作。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
