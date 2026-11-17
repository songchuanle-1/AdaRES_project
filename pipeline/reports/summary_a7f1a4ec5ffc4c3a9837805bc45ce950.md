# Research Pipeline Report - AdaRES_project
快照时间：2026-11-08T14:50:48.798029+00:00；生成报告时快照年龄：2151 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-11-08 22:50:48
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
================================================== 2026-11-08 22:50 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y92_mask_feature_quadratic_res] Best=0.4971 17/30 no_improve=0/30 gpu=0 ? 序列: 0.4868, 0.4943, 0.4962, 0.4957, 0.4971

--- server-2 (192.168.1.163) ---
  [adares_y93_evidence_hidden_skip_res] Best=0.3290 2/30 no_improve=0/30 gpu=0 ?

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
- 候选累计记录 146：complete=16 / rejected=79 / closed=12 / blocked=37 / training=2。
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
- 仅处理AdaRES_project；本轮所属会话/lease、Disabled后台调度、正式skill/experiment_first、AGENTS/goal/SERVERS/暂停资源记录已核验。begin已先查询进展并尝试已有合格派发，ready waiting为0。
- 运行：AdaRES-Y92 Val17/30，步227/363；AdaRES-Y93 Train3/30，步3221/4686。原PID、source、YAML、receipt、私有runtime、350W/280W及原生步数递增通过，canonical物理占用确认两GPU0 busy+reserved。没有使用GPU1或干扰外部任务。
- 完成：Y93原生保存声明只读核实实际目录20261108_082521，新建精确终态工具及29项生命周期守卫通过；绑定原PID47380/log/source/YAML/receipt/freeze，不按启动时间猜目录，未加载checkpoint、停止训练或执行归档。旧hash-bound工具保持。
- 新准备：独立Y95在原self/language和中加入全128查询mean投影旁路，原本地self注意力/SCG/gates保持。首次实际原YAML/完整CPU模型/发布骨干/TRAIN162738点8表达、float32/64独立标量/置换/复制/改同伴保持焦点/零均值/七阶段W2及输入梯度/干预和精确旁路、两合成更新通过；CPU不是稀疏GPU原生任务或收益证据。
- Y95只保留七个实际测试新增文件，私有index分支直接基于原生已测Y93；生产HEAD/index/无关worktree/队列和较早Y94保持。10项真实证据守卫与8项runtime/goal/source/owner负例守卫全部通过，无跳过，保存两授权GPU0原生计划及精确runner。GPU任务未执行、未准入、未同步或派发。
- 本轮新增候选1、CPU候选实测1、GPU原生任务0、准入0、启动0、归档0。原9508表达/141场景完整FIRST/原同设备cost/重复对照/scene bootstrap/有害子组/data/seed1999/fixed30/gates/现有功耗/调度和专用恢复均保持，未声称收益或达标。
- 记忆已先逐字节备份再压缩为七节短记录，完整历史与负证据沿snapshot链保留。收尾连续刷新实际训练及canonical物理占用并保存真实collector resource_check，随后正式report/finish；没有预先声称postflight通过。

## 5. Decisions Made
- Y94先于Y95；真实原生两步、原full freeze/admission、安全同步和递增counter仍是训练前必需步骤。当前两卡均busy，CPU不代替准入，不挤占现有或外部任务。
- 未来Y92仅用finalize_y92_fixed30_y93.py和原20261108_090202；Y93用本轮finalize_y93_fixed30.py/29守卫和原20261108_082521。自然fixed30完整checkpoint核验后方可归档；任何新增HEAD/owner需新的已测绑定。已归档Y91和更早实验不重放。
- server2 GPU1全部使用与原GPU1后评/cost继续用户暂停，不迁移替代；Y57单独暂停，各自恢复需用户明确授权。禁止server6，不改变科学goal字节、原receipt、data/runtime/mappings、功率或调度；无需新增用户处理。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
