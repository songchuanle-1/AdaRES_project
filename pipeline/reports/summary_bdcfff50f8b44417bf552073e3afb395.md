# Research Pipeline Report - AdaRES_project
快照时间：2026-10-21T06:43:58.535032+00:00；生成报告时快照年龄：1439 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-21 14:43:58
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
================================================== 2026-10-21 14:43 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y56_unique_competitors_res] Best=0.5040 18/30 no_improve=1/30 gpu=0 ? 序列: 0.5040, 0.5000
  [adares_y57_peer_second_moment_res] Best=0.5010 20/30 no_improve=0/30 gpu=1 ? 序列: 0.4980, 0.5010

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=2
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y56 | configs_song/model_paper5/adares_y56_unique_competitors_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y57 | configs_song/model_paper5/adares_y57_peer_second_moment_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 4；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 0、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 112：complete=16 / rejected=79 / closed=12 / training=2 / blocked=3。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 35（含修订或重试，不代表成功率）。

## 4. This Tick's Actions
- 实际CODEX_THREAD_ID核验归属，后台Pipeline-AdaRES_project保持Disabled；14:43:58快照experiment_first启用，预算2，仅server-2 GPU0/1，无跨项目或改调度。
- 14:49实核原Y56 PID13738/Train18 step3361至3442、Y57 PID16317/Train20 step3917至3996；均命令/cwd匹配、原生步数增长且无尾部报错。授权卡无独占空槽，三项已实现候选尚无可执行GPU smoke。
- 当前续训片段读取Y56验证.5038473010/.4996519685，Y57 .4984684885/.5009573102；这些是训练日志验证，不是冻结双遍完整评估或方法有效性结论。Y57片段best低于恢复前.5017439，不覆盖旧best。
- Y60保持七个源码/YAML及提交6ac772a0c3eb4b5e598d25ce00c0ab7f203ce1e3、既有原生两步计划不变。新增完整原生DEC图CPU fixture集成：父W2生产源LF实核、共同初始state/七阶段masks/scores/indis逐位相同，各七头实际执行，全部有限后向且最终loss不向六兄弟头传梯度。共享头与私有梯度和max差1.862645149e-9，仅最终私有头与共享头梯度范数差.126597是fixture描述，不证明共享有害。
- 集成结果pipeline/analyses/y60_decoder_cpu_bdcfff50f8b44417bf552073e3afb395.json及.log；native_data_loaded/native_task_backward/native_smoke_passed均false。CPU合成fixture不能代替稀疏骨干、真实TRAIN criterion、optimizer两步或GPU显存。tick_attempt=Y60 implementation_test/passed已绑定新证据；未新增候选、修订、提交、准入、队列或训练启动，不强凑额度。
- GPU0在14:49为91C且热降频。用户在所属会话明确批准仅GPU0临时降至280W，已记录哈希绑定操作授权。首次sudo免密被拒，逐卡确认仍390W并完整留档；核验受保护账号引用后，仅在内存认证，实际sudo命令成功和逐卡回读确认权限，不假定SSH凭据自带管理权限，不输出/保存密码。
- 15:01:53确认仅GPU0运行时390W降至280W，GPU1仍390W；hostname/boot/GPU UUID/原Y56进程归属检查通过。原Y56实验和候选追加runtime_hardware_history；无训练重启、模型/数据/优化器/seed1999/fixed30、风扇、持久启动配置或调度变更。失败/成功证据分别为pipeline/checks/gpu0_runtime_power_bdcfff50f8b44417bf552073e3afb395.json、gpu0_runtime_power_authenticated_bdcfff50f8b44417bf552073e3afb395.json。
- 15:04:25降功耗后实核原PID13738/Y56 Train19 step0至66，原PID16317/Y57 Train21 step643至741，均增长无报错。GPU0 280W/89C/风扇85%/util88%/22682of24576MiB，GPU1 390W/68C/util68%/17098of24576MiB。GPU0热降频仍Active，内核读取成功且无新Xid/fallen-off；短期温度读数不是稳态改善或永久修复。证据pipeline/checks/gpu0_post_power_verify_bdcfff50f8b44417bf552073e3afb395.json。

## 5. Decisions Made
- 保持原两项独立单GPU续训，三项候选等待真实独占原生检查，不挤占、共跑或重启。下一空卡优先Y58/Y59再Y60；重核当前租约、精确版本/goal和实际资源，真实原生两步通过即同轮入队和尝试既有门控派发。
- 已执行用户批准的GPU0临时280W，不重复索权或盲目重施390W前提脚本；GPU1保持原限制，不自动回升、进一步降功耗或改风扇。当前散热风险和掉卡根因仍未解决，后续成本比较须注明Y56硬件条件已改变。
- 原生GPU smoke尚未执行，不把CPU图集成和本轮校验完成当科学成功。两项新决策记录pipeline/decisions/20261021-150256_manual.md、20261021-150602_other.md；随后canonical finish，以真实收据为准。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
