# Research Pipeline Report - AdaRES_project
快照时间：2026-10-04T11:52:28.095787+00:00；生成报告时快照年龄：176 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-04 19:52:27
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
================================================== 2026-10-04 19:52 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y27_relative_peer_res] 1/30 尚无验证 - gpu=1

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
- 本轮event快照为北京时间2026-10-04 19:52:28.095787；begin只执行一次，活跃1/运行1/等待0/容量1/目标1/准入预算0，到期候选0、新终态0。表0-3复用脚本事实，不将未知实时GPU读数解释为空闲。
- 复用已绑定哈希的只读捕获脚本b590ad3f，在19:52:59.924031至19:53:05.442399独立核实同一Y27/PID9901、配置、日志、源标记359a89d和GPU1参数。Train1/30由2564递增至2574/4686；两个最多24KB尾日志无Traceback/OOM/RuntimeError，仅代表这两段日志。GPU1设备19844MiB、利用率51%，不是孤立候选峰值显存。
- 真实证据保存至pipeline/analyses/y27_progress_b8e5374216a74f5c9cf2fdd5ac15478a.json，SHA256=1123f145160913f272b591b1b7a3b1f66bcee9286e098894699a9a648f753932，原始日志同时保留。只补Y27证据及下一动作、更新七段记忆，不改模型、准入、历史结果、冻结方案或其他候选。
- 当前没有到期可执行候选且唯一授权GPU1被同一已准入训练占用；不重复已闭合工程测试、不新增训练或凑变体、不查询GPU0、不操作其他项目或调度、不恢复后台。训练递增不是效果改善；最佳mIoU、Acc@50、spmIoU及训练后收益/伤害/Pareto仍未知。
- 第4段实际产出为1份新的原生递增证据及对应持久状态更新，新增设计/候选/准入/部署均为0。报告写入时本轮收据尚未生成，实际用量窗口与数值未知；finish后以真实收据为准。缓存包含在输入、推理包含在输出，不能重复相加或把用量当账单。
- 报告写入时执行状态尚未完成postflight；只有finish返回CLOSURE_POSTFLIGHT_PASS/COMPLETE并独立核实锁目录消失才称本轮完成。前一轮11c9ed2a已成功完成且19:37:36锁不存在，不复用或重复finish旧租约。

## 5. Decisions Made
- 保持Y27科学状态training，等待同一固定30轮训练自然推进；效果、伤害、创新性和训练后轻量化Pareto未知，已测成本增长及不利对照保留。目标0.515、seed1999、释放SPFormer骨干和冻结全量评估不变。
- 唯一卡已占用、准入预算0且无到期动作，本轮不启动新设计准入；实际终态或故障后再执行已冻结的9508表达/141场景、Y27/W2/B1各两遍、10000场景bootstrap和TRAIN尺寸分组评估，不改门槛、权重、温度、轮数、子集。
- GPU0仍由用户保留，server-6禁用；Y26保持结构资格rejected、Y25保持已完成且科学拒绝，不改名重开。进程或启动状态变化时先精确核实再决定恢复，不能盲目重启。
- 正常训练进展无需用户动作，最终通知保持安静；本轮以成功收尾及独立释放核验为准，失败保留证据并abort原token。下一次调度时间未知，不猜测。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
