# Research Pipeline Report - AdaRES_project
快照时间：2026-10-08T15:48:04.996313+00:00；生成报告时快照年龄：169 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-08 23:48:04
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
================================================== 2026-10-08 23:47 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y33_token_attention_res] metric streams not aligned; defer early-stop check
  [adares_y33_token_attention_res] 8/30 尚无验证 - gpu=1

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y33 | configs_song/model_paper5/adares_y33_token_attention_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 1；准入预算 0。

## 4. This Tick's Actions
- 仅执行AdaRES_project；核实实际任务归属、项目调度Disabled、closure_policy启用及前轮锁已释放。begin仅一次并先读紧凑视图；快照北京时间2026-10-08 23:48:04.996313，event，真实截止2026-10-09 00:48:04.996313，未延长。monitor/goal均0，容量1、活跃1、运行1、等待0、预算0、到期0、新终态0。
- 本轮监控出现metric streams not aligned并延后early-stop检查，随后摘要8/30与尚无验证。只读定位规范监控源码666-674：保护分支不使用未对齐stdout/logger副本并返回未知指标；并不证明历史没有验证，也不证明训练崩溃。对齐异常的具体原因和是否恢复仍未知，未改监控器或绕过保护。
- 因本轮指标可用性变化，调用现有租约绑定只读进度工具一次，未重复preflight/monitor或执行占卡诊断。本地观察23:48:59.367764至23:49:05.454660核实原PID64797、cwd、配置、源码eaf33016和GPU1参数，Train9/30步数99增至110/4686；GPU1设备显存22800MiB、利用率75%，不是独立候选峰值。两段日志尾部各最多24000字节，仅这些尾部未见Traceback/OOM/RuntimeError，不宣称全日志无故障，未使用远端钟推断本地时间。
- 新证据pipeline/analyses/y33_progress_d97ad1047d7142189e6ef0efcabadbbe.json，SHA256 b45cfacad45c074a60b26a5930c76504f0b7772b80aec799a94375570d945872，原始.log保留。只更新同一Y33的next_action及该证据引用、七段短记忆和中文报告；科学阶段仍training，效果、必要性、创新性与训练后成本优势未知。0-3表仍严格为preflight事实，晚于快照的GPU/进度数据只写本段。22:48最佳0.4730为历史早期舍入值，本轮对齐指标未知。
- GPU0未查询或使用；未新增候选/准入、重启训练、重放Y23恢复或改动冻结设计、模型、目标、门槛、调度及其他项目。写入时本轮执行状态仍待finish；前轮d49770b9已COMPLETE、FINALIZED/PASS且独立验证锁消失。
- 本轮收据在写入时未生成，窗口、用量及产出计数未知；finish后核对收据与释放，禁止释放后写入。局部用量仅本地begin至收据，排除早读、晚到记录及最终答复；缓存已含输入、推理已含输出，不重复相加或当账单。实际产出为一次有针对性的原生运行进度证据、同候选操作记录、记忆和六段报告，不当作训练效果证据。

## 5. Decisions Made
- 原生步数确实递增，保持同一Y33训练，不将指标对齐保护误判为crash或用历史0.4730冒充当前有效指标。对齐原因/恢复未知，留给下一轮规范监控重新观察，当前不重复preflight、绕过保护或更改固定30协议。
- 无新科学准入决策。授权活跃容量已达目标，预算0、到期0；维持seed1999/原生128queries/标准单次argmax及全部冻结全量评估、整场景bootstrap、训练来源分组与配对成本方案。目标0.515和W2历史champion0.5100不变。
- 当前无需用户操作，原训练仍在推进；不发主动提醒。完成真实postflight并独立核实释放后安静收尾，tick完成不等于指标对齐已修复或科学成功。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
