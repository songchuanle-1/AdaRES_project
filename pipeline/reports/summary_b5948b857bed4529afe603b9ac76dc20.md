# Research Pipeline Report - AdaRES_project
快照时间：2026-10-06T08:48:12.659143+00:00；生成报告时快照年龄：1425 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-06 16:48:12
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
================================================== 2026-10-06 16:48 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  无训练进程
  [adares_y28_mask_readback_res] 进程未见 (deathwatch 1/3) — 暂不处置

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
- 本轮仅AdaRES_project；当前任务ID与owner匹配，closure启用，后台Pipeline-AdaRES_project保持Disabled；begin只执行一次。快照2026-10-06 16:48:12.659143+08，event/budget0/due0，表0-3仍为该preflight快照而非结束实时状态。
- 16:51:38.301597至16:51:41.654523核验原PID19745及同配置进程均不存在；原日志Val30/30及last.pth metaepoch30已保存，best.pth metaepoch21/0.505526483。CPU读取model/optimizer/scheduler键与完整哈希，不重启、不伪造canonical终态；24KB尾部无Traceback/OOM仅是有限范围证据。
- 16:52:19至17:04:00完成冻结Y28/W2/B1各2次全量9508表达/141场景native评测，首遍mIoU0.505520490/0.510107995/0.502223014；Y28 Acc25/Acc50/spmIoU0.607593605/0.556583929/0.469977198，128queries/1selected，目标0.515未达。
- 场景聚类bootstrap10000/seed1999与TRAIN尺寸切点1487/3325未变：Y28对W2总体delta-0.004587505，95%CI[-0.012384596,0.003363911]跨0；小目标2816表达/124场景delta-0.017564302，CI[-0.032367651,-0.003092544]触发冻结实质伤害。对B1总体CI跨0，中目标正向CI[0.000726688,0.029626897]完整保留。
- 训练后成本测量为17:04:19.325057至17:05:24.455193+08。冻结TRAIN8加728/6780、warm2/read3：Y28/W2中位0.076628631/0.079588681秒，大场景0.195548557/0.188331055秒；最大reserved均1.7578125GiB，参数164892293/164629637。成本方向混合，无Pareto优势证明；p95描述性、串行硬件混杂与单seed局限保留。
- 17:05:36.711179至17:05:38.361124最终只读审计：last/best完整哈希未变，原训练与同配置进程为空；GPU1设备1MiB/util0/无compute PID。此前和此后未查询或使用GPU0；空闲快照不等于预约或额外准入。
- 新增候选身份适配测量helpers，未改模型源码、goal、SERVERS、冻结设计或任何历史证据；4个helper语法检查通过、成本AST仅身份替换后与既有流程相同、复用修正比较函数、5个算术/边界/身份测试通过。已保存所有原始日志、哈希证据、结果分析与决策，七段记忆将在finish前同步。
- 科学状态REJECTED（此次Y28实现的小目标实质伤害），canonical生命周期仍为preflight running/deathwatch1/3待常规确认；本轮执行状态PENDING_POSTFLIGHT，finish成功与锁释放前不称COMPLETED。当前收据用量窗口及统计未知，finish后只在可见报告补实际收据，不在释放后改项目；缓存包含在输入、reasoning包含在输出，用量不是账单。

## 5. Decisions Made
- 拒绝此次Y28实现，不晋升champion、不放宽目标或gate、不对验证集调阈值/温度/权重/特征/epoch，不将训练前门槛通过误当成训练后有效性。
- 保留总体不确定性、中目标对B1正向结果、逐表达重复误差和混合延迟；退化原因、机制必要性、创新性和训练seed不确定性仍未知，不泛化为所有mask-content-readback路线禁令。
- 不强行修改队列或deathwatch，不重复monitor，不启动新训练或后台。待常规canonical终态确认及下一轮真实capacity-gap后继续新的独立候选诊断；不改调度，不造no_information_gain冷却或资源阻塞。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
