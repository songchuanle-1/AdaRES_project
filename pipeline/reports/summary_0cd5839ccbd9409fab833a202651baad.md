# Research Pipeline Report - AdaRES_project
快照时间：2026-10-24T22:44:22.391310+00:00；生成报告时快照年龄：95 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-25 06:44:21
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
- 可用 GPU：未做实时占用确认，不能据此分配。
- 监控摘要（begin 已执行，不再重复探测）：
```text
================================================== 2026-10-25 06:44 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y61_peer_value_gradient_res] Best=0.4984 17/30 no_improve=3/30 gpu=0 ? 序列: 0.4914, 0.4984, 0.4960, 0.4914, 0.4964

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1 running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y61 | configs_song/model_paper5/adares_y61_peer_value_gradient_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 3；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 115：complete=16 / rejected=79 / closed=12 / blocked=7 / training=1。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 39（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve the paused original Y57 and saved model/optimizer/scheduler checkpoint. Do not queue a from-scratch restart or ask repeatedly to resume. Await explicit user instruction; keep fixed30/seed1999 and do not preempt the continuing Y56 on GPU0. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 本轮核验实际会话与前台owner一致、后台调度Disabled；begin完成监控、目标检查和既有任务派发检查，waiting为0。继续使用server-2仅GPU0、一卡一实验与280W，保护Y57用户暂停和现有训练。
- 上海06:44:48实核Y61原PID78290及GPU0归属，Train17/30步数2924至2948/4686递增，82°C、22720MiB、280W；采样未见热降频、Xid/fallen-off或尾部错误。证据pipeline/checks/ef_gpu0_runtime_0cd5839ccbd9409fab833a202651baad.json，SHA256 8e56b334a78a6a2b85ad0d828a3cf4469169a116a375761ccbe73a2c69e6aae0。GPU1仍有外部计算，不能声称整机单卡隔离或硬件根因已解决。
- 生命周期资源检查实核Y56/Y58/Y59/Y60的完整终态和原冻结后评文件，GPU0仍由Y61占用，83°C、280W、22720MiB，无独占空槽。证据pipeline/checks/gpu0_lifecycle_v3_0cd5839ccbd9409fab833a202651baad.json，SHA256 3ae6dbcd67ca34d86fd0b6888ec3739ef16ef03a68f76b20c24997be79c1c0a9。已记录Y56/resource_check/blocked并将证据归入四项候选，保留原协议；本轮未重放归档或执行全量后评。
- 保留Y62、Y63已有的两套独立CPU准备、已测代码/YAML及当前HEAD82fbd9f绑定的原生两步计划；本轮没有重做CPU/守卫检查、新增变体、提交、准入、同步或启动。v3选择器不含Y62/Y63，不将本次四项冻结检查写成这两项原生检查通过。已更新七段记忆；完整六段报告由canonical report生成，postflight以后续finish与收据为准。

## 5. Decisions Made
- Y61继续在GPU0独占训练并保持280W，Y57继续用户暂停。begin训练best 0.4984仅是过程记录，不代表冻结后评或晋升。本轮没有新故障或需用户处理的事项。
- Y62、Y63的原生两步检查仍等独占GPU0；空槽后按candidate.native_smoke_plan精确读取前10ecd轮计划，Y62使用rebound版本，先核当前版本/goal/280W/真实占用，真实原生PASS才准入、入队及尝试既有派发。四项完成实验沿用各自冻结后评，不重训。
- 已有两项CPU准备候选等待GPU，当前真实资源检查确认继续阻塞，按实验优先规则收尾；不为填额度制造额外变体或重复合成测试。无push、生产同步、GPU1/server-6、硬件设置、跨项目或调度操作；Y61终态仍须其匹配核验入口，不使用Y60专用finalizer。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
