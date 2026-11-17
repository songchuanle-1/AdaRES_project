# Research Pipeline Report - AdaRES_project
快照时间：2026-10-07T20:48:12.723844+00:00；生成报告时快照年龄：161 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-08 04:48:12
================================================================================
  Target: AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515.

## Phase 1: Metric Target
  Target: mIoU ≥ 0.5150
  [✗] Champion: AdaRES-W2 — mIoU=0.5100 (gap to target: +0.0050)
       Epochs: 30/30, State: done_complete
  Phase 1: NOT YET — continue exploring ✗

## Baseline
  Baseline: AdaRES-B1 — mIoU=0.5020
  Champion vs Baseline: γ = +0.0080

## Phase 2: Story Verification
  (等待 Phase 1 达标后启动)

## Overall
  Phase 1: champion=0.5100, target=0.5150, gap=+0.0050
================================================================================
```

## 1. GPU Overview
- 仅列授权卡；未采集利用率、显存或进程时标为未知，不推断空闲。
| Server | GPU | Util | Memory | Process |
|---|---|---|---|---|
| server-2 | 1 | 未知 | 未知 | 未知 |
- 可用 GPU：未做实时占用确认，不能据此分配。
- 监控摘要（begin 已执行，不再重复探测）：
```text
================================================== 2026-10-08 04:48 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y31_mask_shape_res] Best=0.4991 15/30 no_improve=2/30 gpu=1 ? 序列: 0.4935, 0.4881, 0.4991, 0.4945, 0.4990

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y31 | configs_song/model_paper5/adares_y31_mask_shape_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 1；准入预算 0。

## 4. This Tick's Actions
- 本轮仅操作 AdaRES_project；实际任务 ID 与配置所有者一致，Windows Pipeline-AdaRES_project 保持 Disabled，closure_policy 已启用，仅 server-2 GPU1 授权。begin --auto-steady 执行一次后先读紧凑视图，模式为 event，原因仅为 research_artifact_changed，监控和目标检查均返回 0，未重复 preflight。
- 快照时间为 2026-10-08 04:48:12.723844+08（UTC 为 2026-10-07 20:48:12.723844），真实截止为 2026-10-08 05:48:12.723844+08，未延长。复用第 0-3 段事实表：规范监控摘要显示 Y31 正常运行至 15/30，best mIoU=0.4991，最新验证值 0.4990，no_improve=2/30，近期序列为 0.4935、0.4881、0.4991、0.4945、0.4990。表格未提供的结构化指标保留未知，训练日志不替代冻结终态有效性结论。
- 授权容量和目标均为 1，运行 1，认领和等待均为 0，准入预算 0，到期候选 0，新终态 0。仅更新七段运行记忆及中文动作，research_work.json 的 SHA256 保持 58cde459b2ec04dc9a80d8515e8da270524f41e06e47ba34b9a35071d97f4f6e；没有新增候选、科研证据引用、候选阶段变化、准入或训练启动。
- 没有额外采集 GPU 占用、PID 或原生步骤，当前利用率、显存及计算进程均未知；历史占用不代替实时观测。GPU0 继续保留给用户，未查询或使用。未重启 Y31、重测已关闭方向或 Y30 冻结评测，未操作其他项目、调度或后台。
- 报告撰写时执行状态待 report/finish 验证，科学状态为 training、最终有效性未知；只有 finish 明确 PASS/COMPLETE 且独立确认所属锁释放才称本轮完成。当前收据用量窗口及统计撰写时未知；本地 begin 至收据可见响应窗口排除更早读取、晚到记录与最终回复，缓存包含在输入、推理包含在输出中，不重复相加，不是账单。实际产出为运行记忆和完整六段报告，不把正常训练进度计作新增科研证据。

## 5. Decisions Made
- 没有新的科学决策或需要用户处理的故障。Y31 继续原冻结 30 epoch 方案；连续两次验证未改善不构成提前停止、调参、改协议、放宽门槛或重复准入理由。
- W2 历史 champion=0.5100，ScanRefer 目标 mIoU=0.5150 仍未达到，Phase 2 未获准。Y31 的必要性、有效性、创新性、训练种子不确定性及训练后 Pareto 仍未知；保留 Y30 原实现的冻结中等尺寸组伤害与 rejected 结论，不重测、不改写、不泛化为方向禁令。
- 授权容量已满且无到期候选，本轮没有额外需要执行的研究动作；不凑变体，不制造资源阻塞、设计耗尽、冷却或临近截止理由。后续依既有流程监控并执行终态冻结评测。下一触发以已保存自动化为准，精确调度时间本轮未读取。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
