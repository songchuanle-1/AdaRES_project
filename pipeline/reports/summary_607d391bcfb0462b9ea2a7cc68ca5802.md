# Research Pipeline Report - AdaRES_project
快照时间：2026-09-28T04:26:31.622154+00:00；生成报告时快照年龄：2655 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-09-28 12:26:31
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
================================================== 2026-09-28 12:26 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  无训练进程

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2:
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| 无 | 未知 | 未知 | 未知 | 未知 | 未知 | 未知 | 未知 | 未知 | 快照内无运行任务 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 0（运行 0 / 认领 0 / 等待 0）；目标 2；准入预算 2。

## 4. This Tick's Actions
- 本轮 design 快照为 2026-09-28 12:26:31 北京时间，截止 13:26:31。Y1 对 W2/B4 的 train64、val32 按场景、表达式、scan、ann 严格配对；W2 选中 IoU 较低而 oracle 接近，四组双次推理均无不一致。
- Y2 在原生数据提取目标点占比并以 train64 冻结 Q1/Q3，val32 不调界；训练场景均值弱化小目标差距，场景内错位对照也出现相似组间差异，因果解释不成立。
- Y3 在 server-2 GPU0 进行真实点数、等权点数与场景内旋转点数的原生固定权重对照；train64 真实对等权的明显胜负为 5/5，val32 为 0/0，双次推理无差异。
- Y4 首次原生测试因把场景级实例标签错当表达式级而失败，未生成结果；保留 GT 支持代理结果，并在检查原生 collate 后另测实例 ID 子集。val32 明显选错表达的 oracle 目标内覆盖 263/455，已选 query 132/455；train64 修正后有一条重复不一致。完整 nearest-match 训练标签未重建。
- Y1-Y4 的预注册、脚本、逐表达式 JSON 和分析摘要均已计算 SHA-256 记入 research_work.json；本轮未修改训练队列、目标或准入门槛，也未启动 30 epoch 训练。

## 5. Decisions Made
- Y1 的冻结 W2 排序替换、Y2 的目标大小因果解释、Y3 的强点数身份机制及 Y4 的 oracle 标签遗漏假说均按实际反证拒绝；保留 Y1 的诊断发现，不把拒绝扩展成所有未来排序路线无效。
- 六项新模型训练准入未通过，两个空闲名额不自动填充。以本轮独立对照的负结果建立最多两小时研究冷却，15:08 北京时间后或有实质变化时才继续新机制验证；不调 ScanRefer val 参数。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
