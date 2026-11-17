# Research Pipeline Report - AdaRES_project
快照时间：2026-10-05T09:49:26.450890+00:00；生成报告时快照年龄：1677 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-05 17:49:26
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
================================================== 2026-10-05 17:49 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  无训练进程
  [adares_y27_relative_peer_res] 进程未见 (deathwatch 1/3) — 暂不处置

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y27 | configs_song/model_paper5/adares_y27_relative_peer_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 1；准入预算 0。

## 4. This Tick's Actions
- 本轮仅操作AdaRES_project；实际CODEX_THREAD_ID与foreground owner一致，closure_policy启用，后台Pipeline-AdaRES_project保持Disabled。最新用户覆盖和SERVERS只授权server-2 GPU1，未查询或使用GPU0/server-6、未操作其他项目/调度/嵌套codex exec。canonical begin仅一次，先读紧凑视图；快照北京时间2026-10-05 17:49:26.450890/event截止18:49:26.450890，容量1/活跃1/运行1/等待0/预算0/到期0/新终态0。第0至3段复用preflight事实表，不冒充后续实时状态。
- 处理实际变化而非复用失效的运行进度采集器：preflight发现原进程消失/deathwatch1/3，17:51:57至17:52:01核实Y27原PID9901及同配置重复进程均不存在，原日志Val30/30 363/363、last metaepoch30、best metaepoch23、完整model/optimizer/scheduler和稳定CPU哈希。终态证据a45fb8b2，检查点best_miou=.5099911689758301；这证明固定30轮物理完成，不等于收益。canonical队列仍running1/3，未造假改队列或重复preflight。
- 在有效租约及每次GPU1实时空闲/独占检查下，17:55:10至18:07:19完成Y27/W2/B1各两遍原生全验证，每遍9508表达式/141场景；证据a386df85/cf67c989/24806191及原始日志保留。mIoU分别Y27=.509987856/.509987856、W2=.510107995/.510107995、B1=.502119335/.502221764。Y27 Acc25=.608960875/Acc50=.562999579/spmIoU=.475034731，128可用查询/1选中。重复最大表达式差0/0/.973895609，保留B1差异，不称全局确定性。
- 18:07:58完成冻结10000次seed1999整体场景bootstrap/身份配对/固定TRAIN切点1487/3325，证据4384f87f。Y27对W2差-.000120139，CI[-.008415036,.008403480]；对B1差+.007868521，CI[-.001015767,.017045974]；所有固定大小组区间均包含零，large对W2下界-.000028637仍为负，不四舍五入为正。冻结CI上界<0损害规则未触发；收益、损害均不确定，不证明等价或因果机制收益，目标.515未达。
- 18:07:57至18:09:01完成训练后最佳检查点的冻结TRAIN8+728/6780原生serial batch1/two warmups/3reads成本测试，证据234c0c2e，七个候选源码/配置哈希与准入源码一致。Y27/W2 train8中位耗时.078462590/.075131276s（约+4.4%），最大场景对.196855826/.192487053s（约+2.3%），对峰值allocated1.244914532/1.243689060GiB、reserved1.759765625/1.7578125GiB，参数164958597/164629637。三次读数及顺序效应属描述性，未证明训练后Pareto。18:10:06至18:10:10收尾审计f49c2a59确认GPU1=1MiB/util0/无computePID，原进程不存在、last/best哈希未变。
- 新增hash-pinned终态/完整评估/统计/成本/收尾测量入口，不改模型、训练或冻结方案；新统计入口5个合成算术/契约检查PASS，仅验证工具。一次owner.json读取意外展开work_before，未把历史缓存当新科学证据，之后只选字段；发现此前独立释放复核误用pipeline/.foreground.lockdir，本轮改查真实pipeline/headless/.foreground.lockdir，保留历史COMPLETE输出但不倒填正确路径的历史验证。限制与工具哈希记录在Y27_measurement_checks_a77da140.md。
- Y27科研记录推进complete，含义为固定训练/完整冻结比较协议完成，科学收益仍不确定。新增真实终态、六遍验证、聚类、成本及收尾证据，0新候选/0新准入/0新训练；终态评审、七段记忆和哈希绑定已更新。写报告时执行状态待postflight，收据窗口/实际用量未知，finish后读取真实收据并核实真实租约释放。收据仅为begin至收据的局部响应窗口，排除此前读取/迟到记录/最终回复；缓存包含于输入、推理包含于输出，不重复相加、不当账单。

## 5. Decisions Made
- 冻结研究协议完成但没有相对W2或B1的可靠收益/损害结论；目标未达，不升级champion，不以点估计接近W2证明等价，不将complete解释为方法有效。
- 保留小目标点估计下降、large区间负下界、B1重复异常、不利初始对照、原生噪声/恢复数值差异及实测成本回退。不挑large组、不换切点/特征、不验证集调节阈值/温度/损失权重/轮数、不重启或重跑筛出有利结果；不泛化为所有相对几何机制都失败。
- posteval时物理GPU1空闲不改变本轮preflight预算0或canonical deathwatch1/3。让既有监控正常确认终态；待后续合法容量/预算打开后继续证据驱动设计，不伪造队列、重复preflight或增加填充变体。不再用运行进度采集器检查已退出的Y27，不重复已完成冻结比较。
- 仅finish明确PASS/COMPLETE且真实headless锁消失才称本轮执行完成；若具体收尾失败则保留证据，原token下修复或abort并核实释放。下一调度时间未读取，未知；不改自动化/后台/其他项目，不改目标、门槛、截止或历史证据。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
