# Research Pipeline Report - AdaRES_project
快照时间：2026-10-11T07:49:44.135856+00:00；生成报告时快照年龄：3008 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-11 15:49:43
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
================================================== 2026-10-11 15:49 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y39_point_distribution_res] Best=0.5078 26/30 no_improve=2/30 gpu=1 ? 序列: 0.5061, 0.5036, 0.5078, 0.5021, 0.5045
  [adares_y37_pair_distribution_res] 进程消失 ×3次确认 → 终态判定: done_complete
  [日报] 1 条事件 → D:/project\AdaRES_project\pipeline\monitoring\2026-10-11.md

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y39 | configs_song/model_paper5/adares_y39_point_distribution_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 2；准入预算 1。

## 4. This Tick's Actions
- 范围仅 AdaRES_project；核实当前任务 01a07e32-78bf-7670-925b-370911ea48db、服务器身份、GPU0/GPU1 授权、冻结目标及闭包政策；begin 一次、compact view 优先、未重复 preflight。0–3 段保留 2026-10-11 15:49:44.135856+08 初始事实表，不冒充后续状态。
- Y37 原进程消失且 last.pth=30 轮、best.pth=19 轮；对冻结 best checkpoint 完成 Y37/W2/B1 各 9508 表达、141 场景、两次评测，FIRST 为主。首次子进程 import 失败未产生评测结果，原日志保留；修复继承导入路径后原科学评测协议不变。
- Y37 FIRST mIoU=0.505061，W2=0.510108，B1=0.502119；相对 W2 差 -0.005047、场景聚类 95% CI=[-0.012978,0.002910]，相对 B1 差 +0.002942、CI=[-0.004362,0.010245]。六个子组区间均跨零，未触发预设伤害规则，但不证明无伤害或等效。表达级重复最大差 Y37=1.0/W2=0.9763，平均值接近不代表确定性。
- Y37 冻结训练后成本已测：参数 164893831、最大固定训练样本对预留显存 14.7676 GiB；较 W2 时间/显存增加，不宣称 Pareto 优越。六段终态审查已保存 pipeline/analyses/AdaRES-Y37_frozen_review_5e172bd4.md，科学状态 complete_inconclusive，稳定候选阶段 complete。
- Y40 先登记稳定 diagnosis 再测试；32 个匹配控制中 W2 响应为零、已有 Y35 对词分布变化响应最小 0.004687，实测已有实现覆盖拟议路径，故 rejected；不把该结论扩大为所有语言证据无效。
- Y41 独立检验最终掩码边界概率流与能量读出；所有有效超点最近邻并列平均，无半径/邻居数/子集搜索，只增加最终 confidence 分支。CPU、原生匹配控制/反传、物理 tools.train 更新与恢复、资源、代码独立性、冻结约束六项实际证据通过。
- Y41 前两项 CPU 控制失败记录保留；非空硬控制与全部原始 32 有限值控制重新实测，未放宽 W2 严格零或既定容差。初始平均损失变差、2/8 正确输入场景损失升高、错误控制同样可学习等负面结果均保留，不宣称必要性或训练获益。
- Y41 新增 4 张量/67074 参数；预设资源准入下最大固定训练样本对预留显存 13.7773 GiB，较 W2 的时间和显存略增。整模块在同一 Python 3.8 下去除新增分支后与实际远端 W2 AST 相同、配置结构差异为空；未声称本地/远端 W2 原始字节相同。
- 仅提交 7 个已测试 Y41 文件，commit=79ce2de9582c369422facb27c3cbe85559bf227c；原始 Git blob/本地/远端字节核对通过。正式准入收据 pipeline/admissions/AdaRES-Y41.json、next_batch 决策及规范派发均已保存，未 origin push、未修改其他脏文件。
- 2026-10-11 16:34:12.257494–16:34:17.494734+08 独立启动验证：Y41 原生 PID116906/GPU0/1-of-30 轮，步数 462→471/4686，配置/cwd/source hashes 核实；GPU0 设备 17246 MiB、82% 利用率仅为当时占用。Y39 原 PID102267/GPU1、队列记录及源文件保持。终态后活动实验为 2，Y41 稳定阶段 training，效力未知。
- 七段研究记忆及旧记忆逐字归档已保存。本轮执行状态仍为待 postflight，finish/收据/锁释放未发生时不称 COMPLETED；科学状态为 Y37 complete_inconclusive、Y40 rejected、Y39/Y41 training。当前用量窗口与产出计数待收据核实，缺失记未知，缓存不重复相加、不作账单。

## 5. Decisions Made
- 目标 0.515 未达，冻结科学比较基准仍为 W2；自动 CSV Y36=0.5110 不是新的科学冠军晋升。Phase2 未获批准。Y37 不重启、不重调、不替换冠军。
- Y40 因真实匹配控制证据拒绝，不更名凑变体；Y41 已真实六门准入并在授权 GPU0 启动固定 30 轮、seed1999 训练，诊断权重不部署，原生 128/单 argmax/损失及训练协议不变。
- 两卡目标已满足，结束本轮新增诊断与准入，不为耗时制造实验、冷却或资源阻塞。继续保留 GPU1 Y39；未来只在真实终态/故障事件后完成冻结全量评测、子组伤害、重复噪声及训练后成本审查。
- 所有失败、负面结果和旧证据保留；没有修改其他项目、调度、goal、准入规则或历史科学证据，没有新任务、嵌套 codex 或恢复后台。下一调度时间未知；实际 finish 与锁释放结果将在前台最终报告给出。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
