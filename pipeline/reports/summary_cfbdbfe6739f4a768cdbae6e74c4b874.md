# Research Pipeline Report - AdaRES_project
快照时间：2026-10-19T06:11:27.133430+00:00；生成报告时快照年龄：909 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-19 14:11:26
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
================================================== 2026-10-19 14:11 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [不可达]

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=1 waiting=1
    待推: [AdaRES-Y54] configs_song/model_paper5/adares_y54_fine_support_res.yaml gpu=None
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y52 | configs_song/model_paper5/adares_y52_point_target_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 1 | AdaRES-Y54 | configs_song/model_paper5/adares_y54_fine_support_res.yaml | waiting | - | server-2 |
- 活跃 2（运行 1 / 认领 0 / 等待 1）；目标 2；准入预算 0。
- 研究储备 1 / 2；储备准备预算 1，不增加训练并发或启动授权。
- 候选累计记录 107：complete=14 / rejected=79 / closed=12 / training=1 / blocked=1。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 3；收据文件 33（含修订或重试，不代表成功率）。

## 4. This Tick's Actions
- 仅AdaRES_project。已核实当前实际任务owner、closure启用、server-2 GPU0/1单实验单GPU授权、旧后台Disabled且开始前无遗留租约。规范begin --auto-steady在2026-10-19 14:11:27上海生成design/reserve_only快照：active2/running1/wait1、准入预算0、储备1/2及准备预算1，due仅Y54。原context/pending/mode/deadline未修改。
- 本轮monitor/goal各一次均返回0，但monitor实际标server-2不可达。未重复预检或额外SSH，未碰远端进程、GPU、挂载或其他项目。Y52登记running/研究training，当前进程/进度/源/GPU状态未知，不能称当前训练正常或已经停止。缓存14:01:33第1/30轮、步数223到228仅是上轮最后确认数据；14:08任务已告知SSH再次超时。
- 按唯一绑定ID续做Y54：本地规范admission及当前goal/commit/既有六门禁绑定证据完整性复核通过；不重跑原生门禁、不重建回执或重放许可。y54_wait_recheck_cfbdbfe6739f4a768cdbae6e74c4b874.json，SHA f38ff1fe471a2bf412acd064e892291eac02a81fe97fac599da73d09947696e6，归档本轮monitor原文/时间和真实能力检查。只把Y54资源阻塞明确为当前不可达，旧卡占用标superseded而非已释放；retry为2026-10-19T15:11:27.133430+08:00，恢复条件为先对账Y52原配置/PID/日志及当前递增步数，再按实时授权卡空闲/源/租约守卫续做同一Y54。
- 按储备预算进行了父loss/matching/IPDN源审计、有限相关旧ID及最新保留决策原文核对。本地实际检查的配置解释器无Torch，有NumPy/SciPy；AdaRES/ipdn只镜像model_paper5，无共享数据集/整理实现，无法将源审计直接当实际数据行对账或原生前后向。第一次尝试的ipdn/data路径不存在已如实确认；未安装库或把仿真冒充原生。没有形成可执行且区分了历史路线的独立新候选，准备预算1本轮未使用，不以文档占位、调参变体或重开Y2/Y4/Y23/Y53/Y55填充。
- 新证据是部署依赖和能力检查，不是机制反证、模型收益或科学门禁。107候选记录不变；本轮新候选0、科学测试0、模型修订0、准入0、远端写入0、训练启动0。Y52/Y54提交许可均已解决，queue/running/CSV/回执/源模型/目标/授权及调度不改；显式closure.validate_work通过。
- 科学状态保持不确定：Y52原indicator前提否定、released-init辅助零活动及有限父模型梯度限制保留；Y54有限原生有效性/反向/资源证据不代表训练收益。历史CSV最佳Y36 mIoU0.5110、B1 0.5020、相差+0.0090，目标0.5150仍差0.0040，本轮没有新的训练或验证指标。
- 复用14:11脚本0-3段事实表，仅补4-5段动作/决策并刷新七段记忆。收据用量仅本轮14:11:27 begin至receipt本地窗口，当前具体用量待收据/未知，不含更早读取、晚到记录和最终回复；缓存属输入、推理属输出，不重复相加或当账单。实际产出是一项Y54真实依赖证据、一项决策及能力检查/notes/记忆/完整六段报告，不计作上轮部署/启动或历史回归成果。

## 5. Decisions Made
- 本轮按有证据external_resource_blocked收尾，不伪造no_information_gain、独立路线耗尽或到达deadline。储备准备预算是上限，不是凑候选的任务；当前已完成绑定Y54的真实依赖检查，缺少能判别独立机制的真实执行证据不作PASS或训练准入。
- 保留同一Y52已启动记录和Y54已准入waiting；当前网络不可达时不重启、迁移、叠加训练或按缓存分配GPU。恢复连接后先核实原Y52实际进度，再由原规范队列按实时空闲授权GPU0/1派发Y54；不能把GPU1未枚举说成外人占用、坏卡或已恢复。
- 不重复索要已完成的本地提交/实验确认，不改固定30/seed1999、评估方案、目标、门槛、资源或调度，不新建任务/后台。retry只是资源复查条件，不是未核实的调度承诺；真实网络/队列/证据变化可以唤醒。
- 规范report/finish后独立核实收据、context提交及租约释放，只有成功才称本轮检查完成，科研结果仍未知。若失败保留证据并按契约abort；释放后不再写项目。当前与用户已知的SSH超时/等待一致，无新增需用户处理事项，沿原通知策略保持安静。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
