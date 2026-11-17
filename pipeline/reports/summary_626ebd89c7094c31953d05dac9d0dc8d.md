# Research Pipeline Report - AdaRES_project
快照时间：2026-10-07T10:49:38.007872+00:00；生成报告时快照年龄：2063 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-07 18:49:37
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
================================================== 2026-10-07 18:49 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  无训练进程
  [adares_y30_query_trajectory_res] 进程消失 ×3次确认 → 终态判定: done_complete
  [日报] 1 条事件 → D:/project\AdaRES_project\pipeline\monitoring\2026-10-07.md

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
- 本轮只操作 AdaRES_project。已核实实际线程与前台 owner 一致、closure_policy 启用、Windows 后台调度 Disabled、GPU1-only 白名单及用户 GPU0 保留；begin/preflight 仅执行一次。快照为 2026-10-07 18:49:38.007872+08，模式 design，截止 19:49:38.007872+08 未改动。第0-3段保持该快照事实，不冒充后续实时状态。
- 规范监控将 Y30 进程消失第3次确认判为 done_complete，释放队列容量。此前固定30轮和全量两次冻结评估均已完成，本轮没有重跑：Y30 mIoU 0.504229394/0.504314270；原生中等规模组相对 W2 的差值 -0.017392276，场景 bootstrap 95%CI [-0.029706029,-0.004614514]，同一 Y30 科学状态仍 rejected，不代表禁掉整个时间信息研究方向。
- 先创建稳定 AdaRES-Y31 diagnosis 候选和哈希绑定预注册，再做实际测试。机制为预测掩码支持、点数加权的超点质心中心二阶矩完整6项，经形状读出加入最终原生置信度；物理独立 W2 副本，128查询/一次标准argmax、不变原生掩码/采样器/损失。它省略超点内部方差并保留坐标轴方向，不声称精确原始点协方差、旋转不变性或上游没有隐式几何。
- 实际 CPU32 测试区分同质心同质量不同形状，差异1.600000024；重复0，置换/平移误差不超过5.96046e-8。预测盲 TRAIN64场景/256表达诊断有163个 oracle gap，全部选中与oracle形状不同，场景平均gap 0.154148389；仅支持可执行假设，GT不作为候选输入或样本筛选依据。
- 实际 native TRAIN8/初始化4对照：6个新增张量原生梯度有限且非零，最小7.096558e-7，掩码梯度路径差异最小2.685579e-8；重复、禁用、掩码及非indicator损失差异0，联合查询/超点置换0，平移不超过7.451e-8。错配/均匀形状会改变置信度，但尚未证明正确形状的必要性。
- 实际 tools.train 两个优化步：6个新增张量更新、406个decoder梯度张量有限；续训前模型/AdamW/PolyLR恢复精确，下一步loss差异0、模型最大误差1.811329e-5、optimizer最大9.447795e-7保留。4个缓存原生推理场景禁用候选路径与W2掩码/indicator/采样索引一致；没有完整loader/RNG或逐位轨迹等价承诺。
- 保留初始不利证据：启用相对禁用平均loss差为 +0.001815051（W2权重）/+0.030405641（初始化）。固定临时新增6张量两步调查中，正确/错配/均匀模式都能降低loss，均匀有时更好；调查权重没有部署，不把可学习性当必要性、效果或消除初始损害。
- 实际匹配串行资源测量：W2/Y31原生batch2压力中位0.912364116/0.921358014秒、reserved13.400390625/13.505859375GiB；原生batch1推理中位0.090743456/0.090666337秒、p95 0.219791822/0.216859546秒、reserved均1.7578125GiB。新增133634参数/6张量，总164763271，共享初始化精确。两预热三读取仅描述性，不能宣称提速、显著性或训练后Pareto。资源JSON末尾继承trajectory一词是标签错误，审查已说明实际执行Y31 shape_readout，原始产物未改写。
- 冻结方案在准入前已绑定 SHA256 62f264ec31794535ea209e9e0ef9081fe02de49b7fc217ca921936a39646df62：30轮/seed1999/发布backbone；全9508表达141场景，Y31/W2/B1各独立两次，整场景bootstrap10000、训练点数分组切点1487/3325、同实现组别实质损害及训练后成本检查。效果、形状必要性、新颖性和训练种子不确定性仍未知。
- 候选仅7文件提交5f2489de78196e8d89c004e452a665c0946f2a94；origin push失败（origin不是仓库），未伪称推送或添加origin。按既有 additive-marker 规范部署成功。首次严格git/local/remote原始字节核验因5文件CRLF/LF差异在写准入前失败；租约保护的仅Y31字节规范化与git blob/AST等价核验、实际CPU重测后重试成功，未放宽哈希或科学门槛，也未覆盖历史测试。
- 六项实测准入门及独立性/目标约束审查通过，收据 pipeline/admissions/AdaRES-Y31.json SHA256 3a02606cd6436ca9d4450ffdd257af9208091faa86cf96b913050d91c9214d10。19:14:04+08规范派发到server-2 GPU1，PID66677，配置adares_y31_mask_shape_res，launch_verified=true/native training_progress；没有手工重启或重复准入。
- 独立只读核验窗口（本地）19:18:47.497171至19:18:53.693723+08：确切PID/配置/cwd/source marker一致，Train1/30步589增至599/4686。GPU1设备占用16930MiB/49%，不是候选隔离峰值；两个24KB尾部无crash，不代表全日志。原始远端时间保留，不替代本地观察时间。证据 pipeline/analyses/y31_launch_626ebd89c7094c31953d05dac9d0dc8d.json SHA256 c70f7b3da88b5f61e87b3b5b1e3051a5575a5763bfb1d216dad218d266399182；GPU0未查询或使用。
- 持久化Y31 training阶段、各实测/修复/准入/启动产物及实际脚本哈希、design_cycle admission_target_met；旧design_cycle完整保留，七段memory保持少于120行。当前收据用量窗口/计数及本轮执行完成状态在此写入时未知，须以finish返回和独立核实锁释放为准；收据只统计begin至receipt的本地部分窗口，缓存属于输入、reasoning属于输出，不重复相加、不当账单。

## 5. Decisions Made
- 本轮准入目标1已达到且原生步数增长已核验，停止额外候选或GPU诊断；这是 admission_target_met，不是无信息增益、超时或路线耗尽。
- Y31按同一冻结30轮配置继续训练，实际效果未知；只有出现真实进度异常/终态时才核对确切PID、配置、日志、源码及续接冻结全量评估/训练后成本，不能据梯度、初始化资源或旧champion宣布有效。
- Y30保持同一实现科学 rejected/规范 done_complete，保留所有原始组别伤害、重复差异和成本证据，不改评估门槛或历史结果。Y31初始不利证据、错配/均匀对照和续训数值噪声保留。
- GPU0继续用户保留，server-2只用GPU1，不操作其他项目/调度、不新建任务、不嵌套codex exec、不恢复后台；不重放早已处理的Y23恢复流程。目标、阶段授权、资源白名单及真实截止未放宽。
- report保留preflight事实表并记录后续启动，不预写postflight成功；finish成功且匹配租约确实释放才称COMPLETED，否则保留证据、具体故障并abort核实释放，不二次begin或用无变化重试掩盖失败。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
