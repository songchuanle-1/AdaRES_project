# Research Pipeline Report - AdaRES_project
快照时间：2026-09-30T11:30:36.350827+00:00；生成报告时快照年龄：3749 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-09-30 19:30:36
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
================================================== 2026-09-30 19:30 : AdaRES_project ==================================================

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
- 本轮设计快照2026-09-30 19:30:36+08，真实deadline20:30:36；研究测试均在deadline前结束，最后持久化与收尾在其后。0-3段是preflight事实，不冒充启动后的状态。
- 完成Y15源码独立性审计24Python/2configs及63旧证据哈希；提交候选代码2ccc9051和项目归档CRLF修复2bb1c107。保留同步失败证据，不改全局Git配置或已测试模型字节。
- Y15六门准入收据通过哈希校验；canonical gated pipeline_monitor于19:50确认server-2 GPU0训练PID18796启动，单卡固定30epochs，task min_epochs/patience30，globalqueue政策不变。
- 启动后本轮实际状态：running1/claiming0/waiting0；Y15原生Train[1/30]102->1084->2407->3201/4686，唯一日志train_adares_y15_query_completion_res_20260930_194938.log，queue launch_verified=true。20:31核验GPU0利用率100%/17908MiB，GPU1 0%/1MiB且诊断租约已释放。没有新验证mIoU。
- Y16 CPU全部562训练场景/7875目标/36665表达式；702穷举公式单测通过，超点oracle上限0.999584，主要全局空间分辨率瓶颈不支持，拒绝该实现而非所有点级优化。冻结Y15训练尺寸分组1487/3325点，无验证集选值。
- Y17 train16同目标真实描述对照已执行：同目标/重复/异目标maskIoU0.910843/1/0.093917，sameJS0.003079，重复JS6.24e-15。四个双场景原生初始化梯度与W2 checkpoint诊断均执行；初始化置信度3/4批次略冲突，训练checkpoint4/4同向，保留全部结果。
- Y17全量配对采样审计保留36665表达式和每场景原chunk规模；30258可配对/6407单例；JS解析推导与double64自动微分误差约1e-18，self-gradient微小舍入未冒充精确零。已保存独立训练实现设计，但物理包、native sampler、完整optimizer/backward/resources未执行，科学状态design未准入。
- Y15/Y16/Y17共88条候选证据全部哈希一致；memory47行；没有更改其他项目/goal/调度，没有新任务或嵌套codex exec。收据用量窗口以begin到收据为限，缺项未知；实际产出为1次准入和verified训练启动、1条反证筛除、1条新路线的重复/梯度/采样与公式验证，不能将token统计当账单。finish结果由命令和锁释放核验确认，本文件不预先声明完成。

## 5. Decisions Made
- 保持Y15单卡30epoch训练，不重启、不重置optimizer、不扩展epoch；训练启动成功不等于方法有效或教师必要。
- Y16全局超点分辨率路线有反证，暂不训练；Y17在stable ID继续独立实施与六门验证，不把GT对照或未验证方案填入第二卡。
- 真实deadline已到，保留Y17具体可执行next_action；下一次已授权研究先做物理独立包/config/native sampler、固定objective审查与pair-off/wrong对照，再做完整原生梯度更新/大场景资源/推理校验。下一调度时刻未读取，未知。
- 决策日志pipeline/decisions/20260930-202630_route_change.md；研究记录Y15_launch_verified、Y16_ceiling_review和Y17_progress_review_20260930.md。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
