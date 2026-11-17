# Research Pipeline Report - AdaRES_project
快照时间：2026-10-02T06:32:36.710374+00:00；生成报告时快照年龄：878 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-02 14:32:36
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
| server-2 | 0 | 未知 | 未知 | 未知 |
| server-2 | 1 | 未知 | 未知 | 未知 |
- 可用 GPU：未做实时占用确认，不能据此分配。
- 监控摘要（begin 已执行，不再重复探测）：
```text
================================================== 2026-10-02 14:32 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y23_valid_token_support_res] Best=0.4257 4/30 no_improve=0/30 gpu=0 ? 序列: 0.3327, 0.3713, 0.4257
  [adares_y18_point_measure_res] Best=0.5070 21/30 no_improve=7/30 gpu=1 ? 序列: 0.5048, 0.5043, 0.4971, 0.4981, 0.5065

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=2
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y18 | configs_song/model_paper5/adares_y18_point_measure_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |
| AdaRES-Y23 | configs_song/model_paper5/adares_y23_valid_token_support_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 2；准入预算 0。

## 4. This Tick's Actions
- 本轮为用户手动暂停/条件续训操作；begin快照2026-10-02 14:32:36+08，原状态Y18/Y23均running、2/2。后续实时变化另列，不改写快照事实表。
- 用户明确要求先暂停Y23释放GPU0，等Y18完成后在其GPU1 resume同一Y23。14:36:44至14:36:47核实server-2主机/原PID125903/config/GPU0后SIGTERM，进程消失，GPU0仅1MiB/0%且无compute进程；Y18PID7449命令未变、仍占GPU1。
- Y23暂停时Train5/30第256步；只保存到epoch4last.pth，955396040字节，SHA256=57aafa1f85f7214cf85c91ce525e373716d3a9dcf1b78164e626a4246dd9e685。CPU读取验证model/optimizer/scheduler/metaepoch4/best_miou0.45115045，从第5轮续训至固定30轮；未保存的256步重跑，没有完整loader/RNG轨迹等价保证。
- 在项目写锁下同步queuepaused/dispatch_hold、移除runningGPU0映射、实验history与resume_checkpoint/dependency。SERVERSallowed_gpus仅[1]，项目AGENTS记录用户覆盖指令，防止后续自动填GPU0或误用genericwaiting从头训练；目标与科学准入未改。
- 已有canonicalAdaRESlauncher不支持resume；新增项目内foreground-only守卫helper，检查绑定租约/项目写锁、Y18done_complete与原日志epoch30检查点、GPU1真实空闲无compute、checkpoint/sourcehash和重复进程，才传--resume启动。--check-only实际返回blocked_dependency（Y18running），不启动、不等待后台、不创建调度。真实续训/恢复图与进展验证尚未执行。
- Y18跟进原生Train21/30步数2104→2113/4686；新鲜进展/检查点证据加入Y18原候选。Y23改blocked/user_resume_dependency，记录具体unblock条件与retry时间，保留全部原始日志、冻结评估和历史负证据，不称rejected/crashed/complete。
- 科学状态Y18training/Y23用户依赖blocked，最终有效性未知；执行完成以finish实际返回和锁释放为准。收据只含begin到收据的本地部分用量窗口，前置阅读/晚到/最终答复不含，缓存属于输入、推理属于输出，不重复相加或视为账单；尚未生成时用量未知。
- 手动decision已通过canonical log_decision记录20261002-144602_manual.md。首次假定Git安装路径Program Files失败，核实实际Program Files(x86)后成功；一次本地AST探针取值AttributeError修正后本地及内嵌远端语法PASS。这些是工具探针故障，不是训练崩溃；未运行续训launch分支。恢复helper最终SHA256=811695561e8ae9dd1b75fec6244ccde70cc9436f63414a147ae0bcf827aba228，支持Y18终态从队列移除后读取experiment生命周期。

## 5. Decisions Made
- 遵循用户最新授权：GPU0腾空并从本项目白名单移除，直到用户归还。Y18完整30轮结束后仅在GPU1续训同一Y23，恢复AdamW/PolyLR状态，不从releasedpretrain重启、不重置epoch数。
- 不把Y23改genericwaiting、不启动新变体或占GPU0测试，不终止Y18以提前满足依赖。下一次foreground事件看到Y18终态时先执行guardedresume与原生进展核验；未满足依赖继续等待，不重复GPU探针。
- 未修改调度、目标、历史admission/design_cycle；暂停是资源安排，不是依据中期指标的科学拒绝。恢复未实际发生，不能称完成续训。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
