# Research Pipeline Report - AdaRES_project
快照时间：2026-10-09T09:47:55.705233+00:00；生成报告时快照年龄：3058 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-09 17:47:55
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
================================================== 2026-10-09 17:47 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y35_language_evidence_res] Best=0.4961 12/30 no_improve=0/30 gpu=0 ? 序列: 0.4782, 0.4737, 0.4908, 0.4955, 0.4961
  [adares_y33_token_attention_res] 进程消失 ×3次确认 → 终态判定: done_complete
  [日报] 1 条事件 → D:/project\AdaRES_project\pipeline\monitoring\2026-10-09.md

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=1
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y35 | configs_song/model_paper5/adares_y35_language_evidence_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | running |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 1（运行 1 / 认领 0 / 等待 0）；目标 2；准入预算 1。

## 4. This Tick's Actions
- 执行范围仅 AdaRES_project；17:47:55+08 快照、design 模式、原始 18:47:55 截止不变。begin 已执行一次 monitor/goal，未重复 preflight；closure_policy 启用、调度禁用、server2 授权 [0,1]、归属与 GPU0 用户授权均已核实。
- Y33 固定30轮完成；9508表达/141场景的 Y33/W2/B1 各重复两次、首轮为主结果，mIoU=.505137821/.510007498/.502111562。整体差异区间跨零，不能声称无害或等效；小目标对 W2 差值 -.024630811，场景 bootstrap CI[-.041461090,-.008034531]，触发预注册伤害规则。重复最大表达差1.0、训练后配对成本、原始权重/源码保全审计均留存。
- 先创建稳定 Y36 diagnosis，再按同一 ID 持续实际测试：CPU32 等边际不同邻接、原生TRAIN64/256表达、8张量有限非零梯度、物理训练2步及模型/AdamW/PolyLR恢复。掩码/辅助损失/重复/关闭新路径与 W2 对照均一致；续步模型差1.52588e-5保留，不宣称逐位轨迹等价。
- Y36 初始损失上升 +.000390172（W2输入）/+.074906707（释放初始化）未掩盖。固定两步新8张量调查中，正确/错掩码/错误邻接都能学习，部分个例恶化；这不是几何必要性或泛化收益。诊断权重未部署，原失败夹具/KeyError及仅夹具、字段名修复证据保持不变。
- 资源实测：新增460034参数；压力训练中位 .94730/.91725 秒、保留显存14.0371/13.4023 GiB；推理中位 .09760/.09492 秒。只有可执行性证据，成本增加且训练后轻量Pareto仍未知。真实六门槛和六项smell审查完成，固定评估/伤害/重复噪声/训练后成本计划未放宽。
- 仅七个新候选文件提交 6725f9bd6a8f9b47addae6a4b8336d004f73f1ae；Git/本地/远端字节一致，提交版CPU实测18:32:01通过。18:33:37规范门控在GPU1启动Y36；独立18:33:44至18:33:50捕获 PID100018 原生Train1/30步19至29，非仅PID。Y35原PID108331/GPU0/配置/源码保留，后续实际2运行/0认领/0等待；这不是17:47快照表的改写。
- 科学状态：Y33 rejected；Y36 training；目标未达。七段记忆与完整0–5报告已写入，执行状态仍待 finish。收据生成后在聊天补充实际用量窗口和候选/证据/启动产出；缺失记未知，不累加缓存或当作账单。下一次调度时间未知。

## 5. Decisions Made
- 基于原冻结小目标伤害关闭同一 Y33 实现，W2 保持 champion；不恢复、不换名重跑、不把整体跨零解释为无害。
- Y36 通过可执行性/可证伪性准入而非有效性证明；保留初始退化、错误对照可学习、资源增加等风险，部署未使用临时诊断权重。固定30/seed1999/native128/标准argmax及目标.515不变。
- 本轮准入目标1已满足，保留GPU0 Y35和GPU1 Y36原生训练，等待真实终态或故障事件。训练后按冻结9508/141/双重复首轮主结果/场景区间/子组伤害/轻量Pareto裁决；不新建任务、不嵌套执行、不恢复后台、不操作其他项目或调度。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
