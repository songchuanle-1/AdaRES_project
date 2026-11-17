# Research Pipeline Report - AdaRES_project
快照时间：2026-10-08T09:49:37.837464+00:00；生成报告时快照年龄：2627 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-08 17:49:37
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
================================================== 2026-10-08 17:49 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  无训练进程
  [adares_y31_mask_shape_res] 进程消失 ×3次确认 → 终态判定: done_complete
  [日报] 1 条事件 → D:/project\AdaRES_project\pipeline\monitoring\2026-10-08.md

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
- 活跃 0（运行 0 / 认领 0 / 等待 0）；目标 1；准入预算 1。

## 4. This Tick's Actions
- 本轮 DESIGN，快照为北京时间2026-10-08 17:49:37.837464，实际截止18:49:37.837464未改。begin仅一次、首先读取compact model view；未重复monitor/preflight。研究、准入及步数验证已执行，最终执行状态仍须以finish及租约释放结果为准。
- 规范monitor本轮确认Y31 done_complete。此前冻结完整评估第一遍mIoU 0.505514618，低于同次W2 0.509909353；场景区间跨0，协议已完成但效果不确定、未晋升冠军。没有重跑旧完整评估或强改deathwatch。
- 先创建稳定Y32 diagnosis并执行实际源码32例CPU干预。实际C1已有全query质量通路，非匹配query干预31/32例响应，W2不响应；搁置同一Y32实现。并非数值损失/梯度等价或训练伤害结论；保留首版float64/native-float32测量dtype故障和dtype-only修正，原始证据未覆盖。
- 独立Y33采用最终mask条件token注意力读回置信度。预注册后实际CPU32等FG/BG均值、不同token分布输出差0.052794486；原生TRAIN64/256中163条表达有选择差距，全部64场景错误token/mask控制有响应。配对TRAIN8/init4所有7个新增张量有限非零梯度，非indicator损失不变，重复和mask-detach前向一致。只证明可区分的机制，不证明效果或必要性。
- 完成真实tools.train两步/7张量实际更新/407项decoder梯度有限及模型、AdamW、PolyLR精确恢复。后续模型仍有1.525879e-5数值差异，不称逐位等价。初始W2配对损失+0.000569448、released-init -0.068547368及错误输入同样可学习结果全部保留；固定原生优化器临时调查没有选设置或部署诊断权重。
- 配对资源结果：新增328450参数约0.20%；压力训练reserved Y33/W2=13.677734375/13.40234375 GiB，推理中位时间93.264/89.863毫秒。成本增加，没有速度或训练后Pareto优势结论。六项真实门槛和mandatory smell审查、动机及冻结完整评估计划已存档。
- 仅提交Y33七个新候选文件，commit eaf330166aed361ffc2532a52e352b70cba4e2fd。按既有additive-marker策略逐字节核验Git/本地/远端，五文件CRLF到LF机械对齐且AST不变，保留原始测试并实际复跑CPU。并未声称origin push成功；无历史源文件、目标、门槛或冻结协议改动。
- 18:25:48规范门控派发器准入Y33并启动server-2 GPU1单卡固定30轮，PID64797、launch_verified=true。18:26:49至18:26:55本地采集独立确认第1/30轮steps133到143/4686，唯一日志train_adares_y33_token_attention_res_20261008_182513.log。GPU1设备16470MiB/56%是占用，不是候选独立峰值；只在两个24KB尾部无崩溃标记。以本地采集窗口为准，不用远端日志墙钟估算起止或耗时。Y33科学状态training、效果未知。
- 本轮准入1/目标1已满足，无额外候选准入或GPU0查询/使用；未操作其他项目、任务、自动化或后台。原119行记忆原样归档SHA959bb4d9，更新七段短记忆；上一design_cycle原样保留。保留只读错误路径/宽输出以及无写入的重复路径patch拒绝，不作为科学进展。
- 收据当前尚未生成，用量窗口/实际tokens/响应数/产出计数在写入时未知；finish后在任务报告补实际值。窗口只覆盖本地begin到receipt，排除更早读取、晚到记录和最终回复；缓存已包含在输入、推理已包含在输出，不重复累加、不作为账单。

## 5. Decisions Made
- 18:23:41 question：依据实际C1源码和32例干预搁置Y32同一实现，smell3/6；不是全质量监督家族禁令。见pipeline/decisions/20261008-182341_question.md。
- 18:23:53 next_batch：六项真实测试/源字节复验通过后仅准入Y33；18:25:48已规范派发、独立步数验证通过，admission_target_met。见pipeline/decisions/20261008-182353_next_batch.md。
- 继续监测SAME Y33固定30轮/seed1999/原生128query及标准argmax，终态按未改冻结计划完整9508表达/141场景、Y33/W2/B1各两遍、整场景bootstrap10000及训练派生sizecuts1487/3325/材料伤害规则/训练后资源比较。GPU0继续保留，W2历史冠军0.5100与目标0.515未改；效果、必要性、新颖性和训练后Pareto仍未知。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
