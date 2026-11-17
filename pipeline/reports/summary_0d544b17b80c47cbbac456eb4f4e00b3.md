# Research Pipeline Report - AdaRES_project
快照时间：2026-10-14T22:49:39.554651+00:00；生成报告时快照年龄：153 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-15 06:49:39
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
================================================== 2026-10-15 06:49 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y45_metric_decomposition_res] Best=0.5077 25/30 no_improve=8/30 gpu=1 ? 序列: 0.5007, 0.5065, 0.5060, 0.5064, 0.5016

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y45 | configs_song/model_paper5/adares_y45_metric_decomposition_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 2；准入预算 1。

## 4. This Tick's Actions
- 本轮仅处理 AdaRES_project，沿用已核实任务 01a07e32-78bf-7670-925b-370911ea48db 的同一租约 0d544b17b80c47cbbac456eb4f4e00b3；未再次 begin 或重复 preflight。快照为 2026-10-15 06:49:39.554651+08，实际截止为 07:49:39.554651+08。项目冻结、closure、server2 GPU0/1 白名单及后台调度 Disabled 已核对。
- 2026-10-15 06:51:21.329839 至 06:51:26.608849+08 的新观测核实原 Y45 PID78001、工作目录、GPU1、配置、commit 及七个源文件哈希；原生训练在第25/30轮由1433增至1444/4686步。preflight best mIoU为0.5077、no-improve为8/30，均非训练终态结论。本轮没有重启训练或改动模型、优化器、调度器、种子和固定30轮协议。
- 同次实时采集：server2 GPU0 为外部 PID19723，24206/24576MiB、利用率100%；GPU1 为 Y45 PID78001，22754/24576MiB、利用率79%。没有可用独占CUDA卡；两项现存作业均保留。以上是06:51:26观测，不替换第0至3段的preflight快照或未知值。
- 按稳定ID续查 Y46、Y48，核实两份原生诊断脚本及冻结清单哈希未变。Y46仍待同一TRAIN8原生完整损失/反向与配对成本测试，之后才可审查真实训练更新、released-init对照及独立实现；CPU代理不算原生门槛。Y48仍待同一TRAIN8联合捕获掩码和原生匹配索引；实际非空混合标签组数未知，未拼接不同捕获，六项准入门槛仍待完成。
- 实际产出：一份实时训练/资源JSON及原始日志、一份候选资源阻塞处置JSON；Y45追加新观测引用，Y46/Y48各追加观测与处置引用并设置真实重试时间2026-10-15 07:21:26.611181+08。新增科学CPU/原生诊断0、候选0、准入0、新训练启动0；未重复已经完成的CPU或源码对照。实时证据 pipeline/analyses/y45_continuation_resource_0d544b17b80c47cbbac456eb4f4e00b3.json，SHA256 c52837b854899cd4e5351d8dc0deb6ea93465d009a4dd4587873c854bab55369；处置 pipeline/analyses/candidate_resource_retry_0d544b17b80c47cbbac456eb4f4e00b3.json。
- 科学状态：Y45 training，Y46/Y48 blocked，目标未达；本轮执行状态尚待finish验证，不能提前称COMPLETED。收据统计仅覆盖本地begin至写收据的可见日志窗口，不含初始读取、最终回复及迟到记录；此时收据用量与机器统计产出尚未知。缓存包含于输入、推理包含于输出，不重复相加，也不是账单。最终以finish输出、收据及独立租约消失核验为准。

## 5. Decisions Made
- 依据本轮真实的两卡占用，以external_resource_blocked收尾；保持稳定Y46/Y48及未执行的冻结测试，恢复条件为授权GPU真实空闲且获得独占诊断租约。没有宣称全部研究路线穷尽，也不硬等截止或通过凑变体填满第二个名额。
- 继续原Y45固定30轮训练，不以中途0.5077判定失败或提升方法；W2保留科学比较基线，账面Y36的0.5110不等于科学晋升。Y43/Y44的关机续训及终态结论、Y47的反证拒绝均保持，不重启或改写历史。
- 仍只有一个实际运行实验，不等于仅有一个研究候选：另有Y46/Y48两项独立稳定候选被原生验证资源阻塞，均未获准入。保留外部作业，禁止占用忙卡、放宽门槛、跨项目或跨白名单操作。未改变或恢复调度，下一次已保存触发时间未知。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
