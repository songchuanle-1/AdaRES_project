# Research Pipeline Report - AdaRES_project
快照时间：2026-10-20T18:44:22.292294+00:00；生成报告时快照年龄：478 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-21 02:44:21
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
================================================== 2026-10-21 02:44 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y56_unique_competitors_res] Best=0.4503 5/30 no_improve=0/30 gpu=0 ? 序列: 0.3396, 0.3920, 0.4378, 0.4503
  [adares_y57_peer_second_moment_res] Best=0.4316 5/30 no_improve=0/30 gpu=1 ? 序列: 0.3258, 0.3825, 0.4176, 0.4316

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
- 候选累计记录 111：complete=16 / rejected=79 / closed=12 / training=2 / blocked=2。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 35（含修订或重试，不代表成功率）。

## 4. This Tick's Actions
- 本轮已核验实际owner、项目实验优先授权及server-2 GPU0/1单卡单实验边界；begin先监控、检查goal和尝试已有合格任务派发，无waiting任务。准入预算2不等于必须制造2项候选，先续做已实现且最近可入队的Y58/Y59。
- 实际执行python -B scripts/check_ef_execution.py --token f27d10d52a2b4fabadfd1c28afd46c65：两项各7个源/YAML文件的冻结payload、工作区、已测试Git commit a83449f、原生train源及current goal完全一致；注入过期goal和篡改ipdn源码的负控制均正确拒绝。这是完整性/执行准备检查，不是原生GPU smoke。
- 项目内检查脚本首次在结果落盘时因checks目录未建出现FileNotFoundError，未写候选/队列、未执行GPU后向；补上输出目录创建后重跑通过。初始源与错误记录保留，不作为候选科学失败。
- 随后移除检查脚本要求Y56/Y57原进程必须同时存在的固定前提，按当前running/claiming和实际GPU compute占用判资源；模拟已归档任务释放空卡可识别，模拟未确认running/claiming槽位不被抢占，两控制均通过。旧检查器原字节及hash绑定归档，未覆盖负面记录。模拟控制不代表真实资源已空闲。
- 表0-3对应上海02:44:22预检。最新实采上海2026-10-21 02:49:44：Y56原PID83638/GPU0第5/30轮step4049到4127，Y57原PID95947/GPU1第5/30轮step2968到3051；原cwd/config/实际compute PID与GPU UUID核实。本轮两次快照间仍增长，当前best mIoU分别.4503282606601715/.4315843880176544，各4次验证，非最终效果。
- 真实GPU0/1分别22606/17326MiB、利用率47%/61%，没有授权独占空槽；检查保留现有训练和队列槽位，不借剩余显存共跑。Y58/Y59两步native loss/backward仍未执行，无准入/新训练/新waiting，queue仍2running/0waiting/0claiming。
- 本轮没有改候选模型/YAML、没有新变体或Git提交、没有改调度/资源/goal/数据协议；完成当前resource_check真实证据与七段记忆/决策留档。报告尚待canonical finish，不预称其结果。

## 5. Decisions Made
- 2026-10-21 02:50 上海 other：优先检查最近可入队的Y58/Y59，不重复旧科学评审或常规授权；固化当前goal/测试提交/源码/YAML一致性，并修复项目检查器的固定双进程前提，详见pipeline/decisions/20261021-025057_other.md。
- 无授权空整卡时不运行真实GPU后向、不假签native_smoke或准入凭据；保留两个既有独立实现及原生YAML，current tick_attempt=resource_check/blocked，与当前新测证据绑定。不为填额度新增重复变体，不建立全局研究冷却。
- 资源或队列打开时，先核验当前租约/目标/精确版本与实际资源并协调running/claiming保留槽，再执行既有两步原生smoke，通过后当轮入队与尝试门控派发；无需原Y56/Y57同时仍存活，无需重复常规授权。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
