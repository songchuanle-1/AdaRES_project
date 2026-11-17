# Research Pipeline Report - AdaRES_project
快照时间：2026-10-07T12:49:15.896140+00:00；生成报告时快照年龄：159 秒；模式：event

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-07 20:49:15
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
================================================== 2026-10-07 20:49 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y31_mask_shape_res] Best=0.3735 3/30 no_improve=0/30 gpu=1 ? 序列: 0.3326, 0.3735

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y31 | configs_song/model_paper5/adares_y31_mask_shape_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 1；准入预算 0。

## 4. This Tick's Actions
- 仅AdaRES_project。规范SKILL前台规则及foreground契约、最新execution_mode/goal/SERVERS/closure_policy/项目指令均已核对，实际CODEX_THREAD_ID匹配owner，Pipeline-AdaRES_project为Disabled、GPU0继续用户保留、GPU1-only白名单不变。begin/preflight仅一次，先读compact view；快照2026-10-07 20:49:15.896140+08，event，真实截止21:49:15.896140+08未改动。
- 快照capacity1/active1/running1/waiting0/budget0/due0/newterminal0；event理由为前轮research_artifact_changed/research_work_changed。规范监控已给出同一Y31第3/30轮、Best=0.3735、no_improve=0/30，验证序列0.3326、0.3735。这是正常早期训练记录，不是训练结束或冻结全量比较，不能据此宣布效果、必要性、Pareto或达标。
- 第0-3段保留本轮脚本事实表；运行表未填的Epoch和指标仍未知，规范监控日志中已观察的3/30与0.3735在本动作段注明，不静默改写事实表。GPU当前利用率/显存/compute PID未额外采集，保持未知；前轮19:49的17422MiB/53%仅为历史观测，不能当当前占用。本轮没有重复preflight或健康进度双尾部探测。
- 目标仍W2历史mIoU0.5100/ScanRefer目标0.5150/差距0.0050，Phase1未达、Phase2未授权。研究ledger SHA256 58cde459b2ec04dc9a80d8515e8da270524f41e06e47ba34b9a35071d97f4f6e保持不变；无新增候选、证据引用、阶段推进、准入、终态、重启或科学决策。只更新119行七段memory的当前观测和本轮动作记录，所有冻结方案/准入/不利对照/Y30科学拒绝证据保留。
- 科学状态Y31仍training，效果/正确形状信号必要性/新颖性/训练后Pareto/训练种子不确定性未知。GPU0未查询或使用，未重放旧Y23恢复流程，未操作其他项目、调度、新任务、后台或嵌套codex exec。没有编造路线耗尽、资源冷却、截止到期或新的独立测试。
- 本条及report写入时finish结果和本轮收据用量窗口/计数尚未知；finish成功、context提交且独立核实匹配租约释放才称完成。收据只统计begin至receipt本地部分窗口，排除前读、晚记录及最终回复，缓存为输入的一部分、reasoning为输出的一部分，不重复相加或当账单。前轮d629be56已FINALIZED/PASS并19:52:47核实释放，不冒充本轮完成。

## 5. Decisions Made
- 无新增科学决策。维持同一Y31固定30轮/seed1999、原冻结全量评估/场景bootstrap/规模组实质伤害及训练后成本计划；健康早期验证增长不触发调参、提前结束、重复门禁或额外候选。
- 当前容量已满、准入预算0、无到期候选且无新的终态/故障，继续由既有监控检查真实进度异常或终态；不把缺失的GPU实时测量当空闲许可，不占用GPU0。
- 正常持续训练无需用户行动，保持安静，完整六段中文报告保存在本轮artifact。规范report后仅调用一次finish，若失败按具体证据修复或abort核实释放，不二次begin、不预称完成或留下租约。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
