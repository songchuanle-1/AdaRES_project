# Research Pipeline Report - AdaRES_project
快照时间：2026-10-25T13:39:34.049557+00:00；生成报告时快照年龄：1596 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-25 21:39:33
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
================================================== 2026-10-25 21:39 : AdaRES_project ==================================================

--- server-1 (192.168.1.175) ---
  无训练进程

--- server-2 (192.168.1.163) ---
  无训练进程

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: paused=1
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
- 候选累计记录 115：complete=16 / rejected=79 / closed=12 / blocked=8。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 39（含修订或重试，不代表成功率）。
- 待人工确认（列出动作不代表已授权）：
| ID | Required Confirmation | Retry After |
|---|---|---|
| AdaRES-Y57 | Preserve the paused original Y57 and saved model/optimizer/scheduler checkpoint. Do not queue a from-scratch restart or ask repeatedly to resume. Await explicit user instruction; keep fixed30/seed1999 and do not preempt the continuing Y56 on GPU0. | 2026-10-22T00:26:29.471010+08:00 |

## 4. This Tick's Actions
- 用户在本会话明确“继续推进训练  然后server1 的gpu也可以使用”。已核server-1=192.168.1.175/hostname dl02-MS-7B73、唯一GPU0独占空闲，并保存资源授权；AGENTS/SERVERS/goal仅更新资源范围，原数据/seed1999/固定30轮/损失优化器及禁用项不变。新goal SHA4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a，模型HEAD仍82fbd9fb1fc56da6afee5b145f59cfa533464707。
- server-1建立项目私有代码和Python3.8/PyTorch1.12.1+cu113运行副本。修复新系统对旧libtorch_cpu.so可执行栈标志的加载拒绝，只改私有副本PT_GNU_STACK的4字节，其他字节一致；补充原CUDA11.3共享库和项目进程LD路径，不修改原环境、CUDA核或系统。Y62首次缺libcusparse及Y63首次继承280W守卫的失败保留，后者另存计划绑定server-1实际350W，未改硬件或模型数值门槛。
- 完整原数据checksum检查发现旧server-1副本7项差异，其中3个场景.pth。独立data目录链接100766个已匹配只读文件，7项均从原server-2私有复制并逐个SHA256一致；train1201/val312 *_refer.pth文件名/大小集合逐split一致，共享旧数据未改。证据pipeline/checks/server1_bootstrap_63b27dd8efe04a0bab6dc0291cefdbf0.json、server1_data_verified_63b27dd8efe04a0bab6dc0291cefdbf0.json；通过canonical源码检查和安全同步，代码marker82f一致。运行适配6项控制/62项回归/库路径后3项专项通过，初次回归3项兼容失败已修复重测。
- Y62在21:53:34至21:54:33完成真实原生YAML/TRAIN/tools.train两步，loss7.886719703674316/8.23161506652832，峰6606028800B；Y63在21:55:41至21:56:24两步loss7.886719703674316/8.118072509765625，峰6627000320B。每项540梯度张量有限、11头参数、每步七阶段调用、相应新权重非零梯度及第二步响应实核；两份真实结果分别拒绝旧goal/NaN/未执行分支，CPU证据未冒充原生检查。
- 两项在本轮预算2内签发admissions/AdaRES-Y62.json与AdaRES-Y63.json，固定后评FIRST/全量9508表达141场景/scene bootstrap10000/seed1999及预定伤害规则已保存。canonical门控将Y62部署到server-1 GPU0，PID2540376，launch_verified=true；22:00:28至22:00:33第1/30轮240→249/4686增长，六文件和原生YAML精确路径适配实核，尾部无错误。GPU67°C/16920MiB/350W是一次采样。Y63 waiting同卡，未同时占卡；尚无训练验证主指标。
- Y57仍用户暂停、原epoch27恢复要求不变；server-2 GPU0外来PID13813仍占用、280W不变，GPU1外来任务不操作；五项固定30完成实验只复核终态/冻结文件，不重训或从训练best晋升。当前资源证据gpu0_lifecycle_v4_63b27dd8efe04a0bab6dc0291cefdbf0.json。通过应用工具更新现有heartbeat的资源提示为server-1 GPU0+server-2 GPU0，核对原ACTIVE、2小时频率、:47和所属会话不变。
- 完整六段报告复用begin的21:39:34事实表；上述21:53至22:00变化以实际观察时间单列。已知收尾校验仍把空server-2队列当作GPU空槽，与外来实时占用冲突。只读核验五项拒绝谓词仍为真，completion audit保存两份有效准入、预算2和训练证据；本轮不重放已知失败finish，不改该谓词/blocker类别/快照、不造队列。规范report后abort/INCOMPLETE并独立核释放，已启动训练继续运行。

## 5. Decisions Made
- 按本会话新增资源授权继续训练，不重复请求常规操作许可；server-1仅GPU0、一卡一独立实验，server-2仅GPU0且280W，Y57/GPU1暂停和server-6禁用保留。资源路径适配只允许原数据/检查点根路径的确定性替换，不能改变训练参数。
- 本轮已完成两项真实检查和两项准入预算，Y62先运行、Y63排队接续同卡。后续只监控实际PID/原生步数并依既有门控派发，不凭PID称正常、不重复smoke/准入/启动；Y63等待期间保持其82f版本有效，不随意推进HEAD。
- 原生两步只证明当前数据加载、有限后向、目标路径与初始化资源可运行；不同batch loss不是收敛曲线，训练效益与全数据成本仍未知。Y62/Y63以及五项旧完成实验均依各自冻结后评，不验证集选参或重开旧拒绝。
- 共享运行适配改变仅为授权server-1的身份/路径/库支持，科学门槛和exclusive_gpu_slot校验未放宽。由于既有收尾谓词不接受真实外来占用，按foreground规则保留证据后abort/INCOMPLETE并核释放；这不撤销已签发准入或停止已验证Y62训练。需后续具体维护该旧校验不兼容问题。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
