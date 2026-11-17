# Research Pipeline Report - AdaRES_project
快照时间：2026-10-05T21:48:56.905030+00:00；生成报告时快照年龄：110 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-06 05:48:56
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
================================================== 2026-10-06 05:48 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y28_mask_readback_res] Best=0.5024 15/30 no_improve=0/30 gpu=1 ? 序列: 0.4927, 0.4966, 0.4998, 0.4995, 0.5024

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y28 | configs_song/model_paper5/adares_y28_mask_readback_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 1；准入预算 0。

## 4. This Tick's Actions
- 本轮仅对AdaRES_project核对当前归属、冻结协议和资源授权。实际CODEX_THREAD_ID与execution_mode.json的前台目标一致，Windows Pipeline-AdaRES_project禁用，closure_policy显式启用，server-2仅GPU1授权；未修改配置、调度或其他项目。begin仅执行一次并先读精简模型视图；快照2026-10-05 21:48:56.905030 UTC，真实截止22:48:56.905030 UTC。
- 复用begin已完成的规范监控和目标检查：容量1、活跃1、运行1、认领0、等待0、准入预算0、到期候选0、新终态0。唯一事件原因为research_artifact_changed，无本轮列出的终态、崩溃或启动验证故障；未重复preflight、倾倒完整快照/历史、重查关闭方向或额外探测GPU。
- 规范监控显示Y28当前第15/30轮，日志Best mIoU=0.5024，no_improve=0/30，最近序列0.4927、0.4966、0.4998、0.4995、0.5024。轮次标记不证明已完成15轮或固定30轮结束；科学状态training，最终效果、必要性和训练后Pareto尚未知。实时利用率、显存、计算PID、Acc@50及spmIoU未采集，保持未知；不把历史占用当作当前值，不查询或使用GPU0。
- 只更新七段记忆的三个运行摘要和本报告动作/决策；研究账本SHA256仍为e90e6e986791852c903619200d1619a7a180ee026567347c1677545345603e4a。无新增候选、新证据引用、候选阶段变化或训练启动，无新增科学结论。到期可执行候选为0、准入目标容量已满，继续原训练，不追加准入、凑变体或无信息增益测试。
- 写入时report、finish及当前收据待执行，不预称COMPLETED。当前用量窗口和计数未知，最终以finish收据与独立释放核实为准；窗口仅覆盖本地begin至记录时段，排除前置读取、迟到记录和最终回复。缓存包含在输入、推理包含在输出，不重复相加、不当作账单；实际产出为运行核对与记忆/报告维护，并非新增科学成果。

## 5. Decisions Made
- 本轮没有新科学决策。保持Y28同一已准入实验的原固定30轮/seed1999/min_epochs30/patience30和冻结评估计划；不重启、不提前停止、不追加准入，不改变阈值、温度、损失权重、特征、子群或epoch。
- ScanRefer mIoU目标0.515仍未达到，历史W2冠军0.5100、差距0.0050，Phase1未达标、Phase2未开始。早期日志不用于冠军晋升、与历史B1的有效性结论或因果收益判断；训练结束后按冻结Y28/W2/B1各两遍全量9508表达/141场景、整场景bootstrap和训练后资源协议执行。保留全部既有不利对照、原始故障和证据，不重放旧Y23恢复。
- 规范报告后仅finish本token，必须确认postflight PASS/COMPLETE并独立核实真实headless租约目录已释放；若不能诚实通过则保留证据并以原token明确abort，不修改截止或弱化检查。未操作其他项目、调度、后台或GPU0；下一次调度时间未读取，未知。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
