# Research Pipeline Report - AdaRES_project
快照时间：2026-10-27T22:49:09.210514+00:00；生成报告时快照年龄：634 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-28 06:49:08
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
| server-1 | 0 | 未知 | 未知 | 未知 |
| server-2 | 0 | 未知 | 未知 | 未知 |
- 可用 GPU：未做实时占用确认，不能据此分配。
- 监控摘要（begin 已执行，不再重复探测）：
```text
================================================== 2026-10-28 06:49 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  无训练进程

--- server-2 (192.168.1.163) ---
  无训练进程

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1
  server-1:
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| 无 | 未知 | 未知 | 未知 | 未知 | 未知 | 未知 | 未知 | 未知 | 队列无登记运行；远端进程须另核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 0（运行 0 / 认领 0 / 等待 0）；目标 4；准入预算 2。
- 研究储备 0 / 4；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 1、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 117：complete=16 / rejected=79 / closed=12 / blocked=10。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 41（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve the paused original Y57 and saved model/optimizer/scheduler checkpoint. Do not queue a from-scratch restart or ask repeatedly to resume. Await explicit user instruction; keep fixed30/seed1999 and do not preempt the continuing Y56 on GPU0. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 实际所属会话与项目owner、旧Windows调度Disabled及前afe4 FINALIZED/PASS/context_committed/released已核。本轮canonical begin/design快照初始running0/waiting0/ready0、准入预算2；new_terminal:Y63为上轮归档进入本轮快照，不重复归档。server-1 GPU0实际98MiB/0%/无compute/无预留，确认可执行Y64。
- 按既有冻结方案执行prepare_y64_server1_alternative.py activate和run_y64_server1_native_probe.py。e278版本的真实两步TRAIN loss/backward/optimizer通过，原结果pipeline/analyses/adares_y64_server1_smoke_3cc27c56dedf451bbf778724f18a93e0.json(SHA9f6fc1dc0961e2571570a8b60e619869f3c278ccd0c16baed5cc6a2a837f8b42)完整保留。
- 发现canonical launcher要求准入commit等于当前HEAD9706；新增bind_y64_current_launch_version.py，以不可变新文件证明七个Y64文件、原生YAML、goal及执行body与原测试字节相同，只改FROZEN的commit绑定。旧结果未改写或代签；按新plan重新真实两步。binding pipeline/checks/y64_current_launch_version_3cc27c56dedf451bbf778724f18a93e0_binding.json(SHA2e81e2bf1ac605d8e76c267d46f6d4efec0fb168782e5849ebe81530a40fb4bf)，plan同前缀(SHA81a59ef68af5c4290e2c16191d067b86f93a003919ca3a39ecf37df818bb20c1)。
- 当前9706原生结果pipeline/analyses/adares_y64_server1_smoke_3cc27c56dedf451bbf778724f18a93e0_attempt2.json(SHAc749e1c0465e4abae6264f920993879619227423b1c8693f805975e32a47c5a2)：loss7.886719703674316/8.231611251831055，峰6618611712B，两批4个不同TRAIN场景/16表达，540有限梯度张量、11头参数/每步7调用、True*14、包含权重非零梯度、独立有向context公式、首步零锚点及次步实际非零响应通过。未用val、未保存诊断权重；不能据此证明效果或全数据显存上界。
- 新增admit_dispatch_y64_server1.py及test_y64_server1_admission.py，13项实际CPU准入/空闲守卫通过；拒绝错ID/config/commit/goal/YAML、缺两步/非有限loss/未执行分支、val或保存诊断权重、非有限或零包含梯度、缺次步响应、错数据/显存预算、compute占用/高util或显存/错功耗等。冻结方案与原Y63逐项比较，除方法ID/commit外训练、全量评测、goal/资源授权/350W说明均一致。
- 签发pipeline/admissions/AdaRES-Y64.json(SHA2ef4f73f54f64015fe647e21ebd067815b401ed20469de2648b497c1f0b78f65)，当前native_smoke SHAaa6fe95c2c836c52bcafdb3c9d82d420692157256ad743ac416cf8a3098921d9，post_training_freeze SHAa685d25898ee73f0b93de42f96eea4a5015ae998868342ef370e1559a8132f73。保留原val9508/141、方法/Y36/W2/B1两次FIRST、scene bootstrap10000/seed1999、TRAIN cuts1487/3325、预定伤害及全部成本指标。依据原已验证精确数据部署记录并执行当前真实TRAIN检查，没有改数据。
- 以canonical _process_queue执行安全同步和启动，server-1 marker前移当前HEAD9706（仅新增已CPU测试的Y64/Y65七文件各一组，未改旧源码）；Y64 status=running、GPU0、PID2752219、launch_verified=true。dispatch证据pipeline/checks/server1_dispatch_3cc27c56dedf451bbf778724f18a93e0.json(SHA0ef512ba2ec6fd44aad9e916f40fd17a79fa7e3daeb98b450e7fd2bd28192e6b)。
- 新增observe_y64_server1.py并实际核实上海06:56:34至06:56:39原生第1/30轮73到82/4686增长；精确cwd/config、六源码、原生YAML确定性路径映射、marker9706、准入、GPU0独占通过。pipeline/checks/y64_server1_progress_3cc27c56dedf451bbf778724f18a93e0.json(SHA88ba37c0e7b22bec6399fdb9c136f3b11a541e1ba26aa49fdcd248e777b4e59a)，采样14%/16230MiB/58°C/350W，不代表长期峰值。
- Y64启动后Y62/Y63冻结后评重新按实际占用挂起；canonical final server-1证据SHA24024dc30ec932cb6fc54cd8c78e042b261c050bb0215a5819c7fbd70b909a6e，当前Y64 PID2752219/100%/16422MiB并预留；server-2 final SHA6700abf009bdd7a7ac1c705fa6b6ca1aa5cb742a68d8fcf79a109e65112812f3，仍未登记PID13813/100%/24138MiB，另只读确认280W。两服没有剩余授权空槽，Y65原生检查仍待跑，Y57保持暂停。
- 本轮新增准入和运行各1、独立模型变体0；无模型源码修改、新commit或push，无硬件/调度变更。更新七标题memory，报告等待canonical finish，不预称收尾通过。

## 5. Decisions Made
- 优先推进已具备CPU版本和原生计划的Y64，在本轮实际空闲授权GPU上完成检查、当前HEAD版本绑定、准入和验证启动，避免将执行阶段拆到以后。
- 当前HEAD与既有Y64代码版本不同属于启动契约要求；保留原native结果和全部旧证据，以字节等价验证和当前版本真实重跑处理，不放宽launcher、不改写旧结果，不以loss差异选版本。
- 保持同一Y64固定30轮/seed1999/发布SPFormer/原损失优化器和单GPU训练，初始化检查与原生步数只证明执行，不证明mIoU上涨。后续按完整冻结协议比较精度、伤害和成本。
- 物理GPU现已全部占用，剩余预算不要求凑满，不挤占Y64或server-2未登记进程、不恢复Y57/GPU1。Y65原CPU版本/原server-2计划保留，下一真实空槽再推进；七项已完成方法的冻结后评保留。
- 不使用server-6、不跨项目、不改调度、不重开拒绝或禁用路线，不追加重复操作审批。本轮有新验证启动，应通知；无新增用户处理事项。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
