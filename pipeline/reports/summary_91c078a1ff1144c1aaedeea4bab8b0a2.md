# Research Pipeline Report - AdaRES_project
快照时间：2026-09-30T14:48:15.601296+00:00；生成报告时快照年龄：1344 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-09-30 22:48:15
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
================================================== 2026-09-30 22:48 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y15_query_completion_res] Best=0.1239 3/30 no_improve=0/30 gpu=0 ? 序列: 0.0760, 0.1239

  >>> 队列推送: [Y17 target-equivalent paired training] on server-2 gpu1
  [同步阻止] AdaRES additive sync failed: Refusing mismatched existing remote file: ipdn/model_paper5/adares_y17_target_equivalent_pairing/backbone.py

  >>> 代码同步失败，取消本次推送

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=1 waiting=1
    待推: [AdaRES-Y17] configs_song/model_paper5/adares_y17_target_equivalent_pairing_res.yaml gpu=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y15 | configs_song/model_paper5/adares_y15_query_completion_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 1 | AdaRES-Y17 | configs_song/model_paper5/adares_y17_target_equivalent_pairing_res.yaml | waiting | - | server-2 |
- 活跃 2（运行 1 / 认领 0 / 等待 1）；目标 2；准入预算 0。

## 4. This Tick's Actions
- 本轮快照2026-09-30 22:48:15+08，event，deadline23:48:15；归属及server-2GPU0/1授权已核实，Windows旧后台保持Disabled。继续已准入Y17部署，没有新候选准入。
- 修复远端默认Python3.5/JSONbytes/Windows命令长度：两处helper用项目Python3.8、明确UTF8、stdin传payload。5项隔离安全测试通过；15文件哈希核对、5处仅CRLF转LF，未降低保护。
- 随后门控推送暴露MSYS把远端POSIX路径转成本地路径，再暴露关闭转换后的本地/d脚本路径。分别修复并实际验证argv/文件存在/bash语法；最终helper-only提交42ffd58，模型/YAML/nativeobjective均不变。
- 门控同步验证16个AdaRES文件及train.py旧新哈希，源标记匹配42ffd589bfb5c27fa18d735866ddefa3a543e2ee；23:04:19启动Y17 GPU1 PID109632，queue/experiment launch_verified=true。23:06:54独立capture确认Train[1/30]313->314->315/4686且日志尾无Traceback。
- 23:06:54 Y15GPU0PID18796继续Train[4/30]780->781->782/4686；GPU0/1为15928/17030MiB、47/72%。Y15早期bestmIoU0.1239非完整30epoch结果。两卡各一个任务，Y17尚无val结果。
- 本轮执行待finish验证；科学状态Y17training，只证明启动与原生进展，不证明收益/新颖性。收据窗口用量/实际产出在final读取；未知不记0，缓存计入输入非账单。

## 5. Decisions Made
- Y17部署故障已恢复，保留稳定ID及历史失败证据，正式单卡训练固定30epoch/seed1999，不调阈值温度权重或验证集特征。
- 两授权卡均已占用，停止本轮部署工作，不追加容量填充或未准入对照；后续监测进度/故障/终态并使用冻结的完整ScanRefer/W2/B1、场景聚类/噪声/实质伤害/Pareto评估。
- 初始化generic smoothing、微小JS、重组上下文混杂及nativeargmax差异仍在；因果收益需要相同重组pair-off对照。保持每小时:47前台自动化，不恢复Windows后台。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
