# Research Pipeline Report - AdaRES_project
快照时间：2026-10-16T08:17:47.331319+00:00；生成报告时快照年龄：401 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-16 16:17:46
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
| server-2 | 0 | 未知 | 未知 | 未知 |
| server-2 | 1 | 未知 | 未知 | 未知 |
- 可用 GPU：未做实时占用确认，不能据此分配。
- 监控摘要（begin 已执行，不再重复探测）：
```text
================================================== 2026-10-16 16:17 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y50_joint_incidence_res] Best=0.3657 3/30 no_improve=0/30 gpu=1 ? 序列: 0.3115, 0.3657

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y50 | configs_song/model_paper5/adares_y50_joint_incidence_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 2；准入预算 1。
- 研究储备 1 / 2；储备准备预算 1，不增加训练并发或启动授权。
- 候选累计记录 104：complete=14 / rejected=76 / closed=12 / training=1 / blocked=1。
- 否决范围：unclassified=72 / implementation=3 / hypothesis=1；实现修订累计 3；收据文件 30（含修订或重试，不代表成功率）。

## 4. This Tick's Actions
- 执行状态INCOMPLETE，准备按契约abort；科学状态Y50 training、同一Y52 blocked/exclusive_gpu_slot，未作新的科学PASS。项目/当前任务身份与调度Disabled已核实。
- 先读新紧凑快照，再按稳定ID续做Y52。原型七文件和既有21条证据哈希复核未变；CPU准备已完成，不重复相同控制，不为储备预算造候选。Y52原retry17:10:31+08保持不变。
- 客户端2026-10-16 16:20:31.30至16:20:31.79+08只读核验：GPU0外来PID19723/100%/23326MiB；GPU1本项目Y50 PID31539/85%/22694MiB；没有真实空闲独占卡，不运行GPU诊断、不触碰外来作业。
- Y50同一进程Train[3/30]402→403/4686，无本次有限尾部崩溃标记。begin监控序列mIoU .3115→.3657；stdout最新约.366、Acc@.25约.470/Acc@.5约.406、spmIoU约.337；是早期验证，不是项目最佳或已证明提升。延迟与峰值显存未知。
- 上一轮abort已释放，本轮不是恢复旧租约。已知检查器SHA未变；程序只读执行本轮同一queue-opening拒绝谓词仍为真，虽然实时两卡分别被外来与本项目占用。本轮不重复必失败finish，也不改阻塞类型、目标、调度或检查器；保留证据并abort。
- 实际产出：新候选0、新科学测试0、模型改动0、准入0、启动0；新增占用/训练进度证据1，Y52科学阶段不变。收据用量及窗口待abort后只读核实，当前未知；缓存和推理不重复相加，不作账单解释。

## 5. Decisions Made
- 无新科学路线或准入决策，保持同一Y52冻结实现与原恢复条件；只有真实空闲授权卡且成功取得独占租约，才续做原生匹配/反向与资源验收。
- 不反复推送已知阻塞故障或常规训练进度提醒；保存完整六段事实报告。Y50训练继续，等待实际结果或资源变化；下一真实调度时间未知。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
