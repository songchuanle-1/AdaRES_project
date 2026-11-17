# Research Pipeline Report - AdaRES_project
快照时间：2026-10-16T12:50:50.417452+00:00；生成报告时快照年龄：755 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-16 20:50:49
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
================================================== 2026-10-16 20:50 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y50_joint_incidence_res] Best=0.4823 9/30 no_improve=0/30 gpu=1 ? 序列: 0.4456, 0.4593, 0.4649, 0.4719, 0.4823

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
- 本轮执行状态INCOMPLETE，待规范abort并独立核实释放；科学状态Y50原进程仍在，同一冻结Y52仍待独占GPU。仅AdaRES项目，目标、执行归属和禁用后台已核对，未改门禁、调度、训练协议或其他项目。
- 本轮快照20:50:50上海：目标mIoU .5150，历史CSV最佳Y36 .5110/B1 .5020；Y50日志监控过程最佳 .4823、9/30，低于历史基线，不能当作已完成配对收益或项目最佳。Acc/点级取整验证与精确mIoU分开记录；延迟和峰值显存未知。
- 资源回查未到期时守卫确实跳过，未查询资源或人为等待。客户端20:58:33.515772至20:58:38.398877+08到期唯一一次实测：GPU0外来PID19723/99%/23978MiB，GPU1本项目Y50 PID31539/71%/22700MiB；没有空闲授权卡，未获得GPU诊断租约。回查条件21:58:38.398877+08来自实际观察，不是自动化调度承诺。
- 原Y50进程命令核对一致，但本轮两次独立训练读数均Train[9/30]4686/4686，未见短窗步数增长。20:59:26.3537461至20:59:29.5846147+08额外有限日志核对显示Val[9/30]363/363、日志字节未增加，未见有限尾部崩溃标记。保留原训练，不能把PID或GPU利用率当作当前训练进度正常的证明，也不凭两秒末尾快照宣布卡死。
- 只读核验已有冻结TRAIN捕获的字段：仅含coordinates/graph_edges/masks/masses/query/tokens，没有实际FPS/sample_inds或记录匹配索引；冻结Criterion明确使用这些索引构造监督。因此旧全查询mask回放不能代替真实匹配后频率，不补造索引、不重跑旧CPU构造/组件/双场景控制。本轮新科学测试0、模型改动0、实现修订0、新候选0、准入0、新启动0。
- 已知检查器字节未改变，当前不可变快照上的queue-opening拒绝谓词仍为真；精确旧失败为CLOSURE_INCOMPLETE: Blocked GPU candidate has a queue opening。外来GPU0占用不能通过伪造本项目队列或改阻塞类型绕过；不重放必失败finish，保存真实证据后规范abort。
- 复用begin的六段事实表，0至3段仍是preflight快照；后续观察只记本段。收据用量仅begin到receipt本地可见窗口，目前尚未生成、实际用量未知；缓存包含在输入、推理包含在输出，不重复相加、不当账单。实际产出为一次资源实测、有限日志状态与缓存输入缺口审计，没有训练效果结论；释放后不补写报告。

## 5. Decisions Made
- 保持同一稳定Y52及全部冻结/负证据，阻塞范围仍为inconclusive；只在授权卡真实空闲且独占诊断租约取得后续做稀疏TRAIN匹配、前后向、共享query梯度、资源与完整恢复。未执行项不是PASS，旧缓存不冒充新原生捕获。
- 保持Y50 released SPFormer/seed1999/固定30/128queries/singleargmax，不移动、重启、停训或用验证调参；外来GPU0作业不操作。短窗末尾进度未增先如实保留未知，不当作已经确认健康或训练故障。
- 已有科学阻塞和收尾不一致无新可行动变化；本轮不称完成，不新增审批或更改通知策略。实际下次自动化时间未知；当前租约按契约释放后仅只读核实。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
