# Research Pipeline Report - AdaRES_project
快照时间：2026-10-28T00:47:04.533826+00:00；生成报告时快照年龄：480 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-28 08:47:04
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
- 可用 GPU：未做实时占用确认，不能据此分配。
- 监控摘要（begin 已执行，不再重复探测）：
```text
================================================== 2026-10-28 08:46 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  [adares_y64_directed_containment_res] Best=0.3788 3/30 no_improve=0/30 gpu=0 ? 序列: 0.3288, 0.3788

--- server-2 (192.168.1.163) ---
  无训练进程

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1
  server-1: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y64 | configs_song/model_paper5/adares_y64_directed_containment_res.yaml | server-1 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 4；准入预算 2。
- 研究储备 0 / 3；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 117：complete=16 / rejected=79 / closed=12 / blocked=9 / training=1。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 42（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve the paused original Y57 and saved model/optimizer/scheduler checkpoint. Do not queue a from-scratch restart or ask repeatedly to resume. Await explicit user instruction; keep fixed30/seed1999 and do not preempt the continuing Y56 on GPU0. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 本轮实际会话/项目owner、旧调度Disabled和前3cc FINALIZED/PASS/context_committed/released已核。canonical begin/design，无新终态或操作故障。Y64/server-1 GPU0原PID2752219在上海08:47:14至08:47:19实际第3/30轮755到763递增，精确cwd/config、六源码、原生YAML路径映射、marker9706、准入和GPU独占通过。证据pipeline/checks/y64_server1_progress_fdc1493e8ea342beba65516c0948a04d.json，SHA34070c4a55c1d3d5c7963497eed3881dd5c1e9345f743d259f9dfe56e3aa8023。
- 待跑不足且两GPU忙，针对性扫描feedback/refinement/query-update/attention-gate等历史描述，新增稳定候选AdaRES-Y66同伴状态反馈。原W2 decoder六次细化前q加上soft point-mass IoU去自边/行归一peer query均值乘零初始化W[256,256]；+65536参数/总164695173。保留原gate、七共享confidence损失、native YAML、发布SPFormer、seed1999/fixed30；区别于Y65标量confidence方向反馈，未重开Y25/Y57等旧路线或声称新颖性/效果。
- 实际执行prepare_y66_peer_state_feedback.py prepare、run_experiment_first_probe.py y66_cpu ... --seconds 600（未指定GPU）和record。上海08:50:45至08:51:07真实CPU完成原生YAML/完整模型/发布骨干/真实TRAIN collate162738点8表达，float32/64零输出与输入梯度/RNG锚点、独立标量IoU/context公式、query置换/padding/质量缩放/超点拆分/singleton/零质量/全padding/错对应/极端值控制、七阶段输出/输入梯度锚点、六反馈七头调用、完整合成DEC后向及非零投影梯度、非零反馈改变后续mask和绕过分支负对照，全通过。CUDA_initialized=false，native_task_backward/native_sparse_forward/native_smoke_passed/admission均false；合成DEC后向不等于真实稀疏训练criterion/optimizer。
- Y66 CPU结果pipeline/analyses/y66_cpu_fdc1493e8ea342beba65516c0948a04d.json(SHA6021feaefb6fb9f4be92c8a6c3ca5bd3915c280ac3b9b8d9857538bb99123944)及原日志、冻结program和精确七文件manifest已保存，manifest pipeline/checks/y66_cpu_version_fdc1493e8ea342beba65516c0948a04d.json(SHA5af9ba3c31d6043339902e7d4a8539f77530801d0999a0194b1c39281f212a27)。stage blocked/exclusive_gpu_slot/server-2；七文件未提交，无native plan/准入/启动。
- 另核Y64准入与运行、Y57暂停、HEAD9706和空暂存区保持，无生产同步或队列/running变更。canonical final occupation：server-2 PID13813/98%/24138MiB，证据SHA316072d312f7c2cf92b38dc4ffc34e7ebabefdf0f3710a0ad394e225d6eb3fed并绑定Y65/Y66等；server-1为Y64 PID2752219/74%/22750MiB且预留，证据SHAb4dcf710b77304b345988e46fe5acbfc8f35f4cc10237ef4ce90f5ce92707a43绑定Y62/Y63。另只读hostname及server-2 280W。两服无空槽，保存本轮resource_check/blocked。
- 更新七标题memory并保留Y65既有版本/计划、Y57专用恢复、七项原冻结后评和全部负历史。新增模型备选1项；无新commit/push/同步/准入/入队/训练启动、硬件或调度改变。报告待canonical finish，不预称收尾通过。

## 5. Decisions Made
- Y64继续同一固定30轮原训练，不因正常进度重复启动或依据初期验证改结构。当前备选不足时只补一个有明确结构差异、可证伪且完成真实CPU检查的Y66，不为额度制造第二个重复变体。
- Y66使用完整peer状态向量投影，Y65使用标量confidence乘方向；共享递归插入位置不等于相同实现。现有自注意力可能隐式表达同类信息，CPU结果仅证明实现/锚点/梯度可行，不证明新信息、文献创新或精度上涨。
- 当前GPU物理占用使原生稀疏两步检查不可执行，不挤占Y64或未登记任务、不恢复Y57/GPU1。空槽出现先推进Y65已保存的当前版本计划，再为Y66保存相关版本/原生计划并核当前HEAD，执行原生YAML/TRAIN两步后才冻结后评/准入/当轮派发。
- 不跨项目、不使用server-6、不改调度或原数据/seed/fixed30/原损失/selector/阈值等协议，旧拒绝与暂停保留。Y66 CPU准备是实质新进展，通知用户；无新增授权或用户待办。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
