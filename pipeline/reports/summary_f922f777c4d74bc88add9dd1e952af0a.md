# Research Pipeline Report - AdaRES_project
快照时间：2026-09-30T12:41:14.699009+00:00；生成报告时快照年龄：3981 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-09-30 20:41:14
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
================================================== 2026-09-30 20:41 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y15_query_completion_res] 1/30 尚无验证 - gpu=0

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y15 | configs_song/model_paper5/adares_y15_query_completion_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 2；准入预算 1。

## 4. This Tick's Actions
- 本轮快照20:41:14+08，design，原deadline21:41:14；只操作AdaRES_project。完成Y17物理独立包、训练专用配对数据集及单位权重JS，不更改验证/推理和Y15运行任务。
- 510组采样单测；实际分词15129对、30258条表达、7649/9372配对chunk，36665条表达保持。四组原生全反向和14个受影响张量JS梯度/优化器更新通过，正式main两步通过；旧W2/Y15入口保持。
- 105万点压力测试14.082GiB/.911-.923秒，无OOM；train8推理种子/选择/二值翻转0，概率噪声约1e-6。恢复前模型/AdamW/PolyLR精确，下一步权重差5.87e-5保留，不宣称逐位一致。
- 六项准入完成，候选代码提交90c7519；门控推送被CRLF/LF源文件哈希差异拒绝。恢复脚本先遇Windows命令长度，修复传输后遇远端默认Python3.5JSONbytes异常；没有成功同步或正式启动Y17。故障证据已留存。
- 21:42:38本地观察：Y15 GPU0 PID18796，Train[2/30]3920到3921/4686，显存15788MiB/86%；GPU1 1MiB/0%。早期bestmIoU0.0760非完整30epoch结果。队列1运行1等待，远端源标记仍2bb1c。
- 执行状态待finish验证；科学状态Y17已准入/deployment待修复，训练状态未启动。收据用量窗口及实际产出待final读取，缺项未知，用量不是账单。

## 5. Decisions Made
- 保留Y15固定30epoch；Y17只允许server-2 GPU1单卡，不调参、不绕过源哈希或准入门控、不恢复后台调度。
- 到实际deadline停止新增测试，保留可执行恢复步骤：两处远端helper均改用项目Python3.8，独立安全测试、严格旧新哈希的换行规范化、仅候选提交并刷新收据，再门控启动及原生递增步数确认。
- JS较小、初始化generic smoothing、配对重组混杂和nativeargmax差异仍未排除；收益需正式训练和完全相同重组的pair-off对照验证，本轮不宣称新颖性或mIoU提升。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
