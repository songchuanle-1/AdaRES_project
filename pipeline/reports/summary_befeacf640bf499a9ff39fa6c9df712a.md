# Research Pipeline Report - AdaRES_project
快照时间：2026-10-21T12:21:44.596279+00:00；生成报告时快照年龄：750 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-21 20:21:44
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
================================================== 2026-10-21 20:21 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y56_unique_competitors_res] Best=0.5110 25/30 no_improve=0/30 gpu=0 ? 序列: 0.5040, 0.5050, 0.5080, 0.5070, 0.5110
  [adares_y57_peer_second_moment_res] Best=0.5030 28/30 no_improve=7/30 gpu=1 ? 序列: 0.5020, 0.5000, 0.5020, 0.5020, 0.5020

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=2
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y56 | configs_song/model_paper5/adares_y56_unique_competitors_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y57 | configs_song/model_paper5/adares_y57_peer_second_moment_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 4；准入预算 2。
- 研究储备 0 / 2；储备准备预算 2，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 0、待资源 0、数值诊断待解 0、未分类阻塞 0。
- 非终态候选与排队项合计 0；总上限 4。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 112：complete=16 / rejected=79 / closed=12 / training=2 / blocked=3。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 6；收据文件 35（含修订或重试，不代表成功率）。

## 4. This Tick's Actions
- 本轮处理用户明确要求：y57断开吧，先试用单卡GPU0跑。20:21的冻结preflight表仍记录变更前两项running和双卡资源，不倒改快照；以下为之后实测的手动干预。实际所属CODEX会话已核验，后台调度保持Disabled。
- 20:26:27上海，核验server-2 hostname/boot/GPU UUID、两项原进程的完整命令/cwd/starttime及Y57归属后，仅向Y57 PID16317及已核验子进程发送SIGTERM。原Y56 PID13738未重启或迁移，GPU0既有280W保持。GPU1确认无compute进程、显存1MiB。没有驱动解绑、GPU reset或物理断电。
- 暂停前CPU完整加载并哈希保留Y57最新完整last.pth和best.pth：epoch27、543项optimizer状态、scheduler27、best mIoU 0.5028846859931946。原远端日志保存并拉取匹配本地副本。暂停时Train28/30 step2773/4686的未保存部分及RNG不承诺恢复。证据pipeline/checks/y57_user_pause_befeacf640bf499a9ff39fa6c9df712a.json。
- Y57队列及实验状态改为paused并设置dispatch_hold、resume_requires_user_authorization；research候选保留为blocked/human_authorization/from training，running映射仅移除GPU1。未来另获恢复指令才从第28轮保持原模型/优化器/调度器/seed1999/fixed30续训；未登记科学拒绝、崩溃或训练完成。
- 按最新明确用户指令将SERVERS.yaml allowed_gpus限制为[0]，AGENTS.md增加临时GPU0单卡条款，禁止GPU1自动补位、诊断或恢复；不改变自动化或任务调度。操作授权pipeline/authorizations/GPU0_only_trial_befeacf640bf499a9ff39fa6c9df712a.json及manual决策20261021-202908已留档。
- 新增项目内scripts/check_prepared_resources_gpu0_trial.py，保留旧hash绑定工具不改。20:28:07实测仅allowed GPU0可计入资源，闲置GPU1被排除；Y58/Y59/Y60当前各7文件、提交、原生YAML、冻结payload及goal均一致，但GPU0占用，未执行原生两步smoke。当前resource_check/blocked证据同时归入三项候选，tick_attempt绑定Y58，未把CPU结果或空闲GPU1代作准入。
- 同次20:28实核Y56原PID/命令/cwd及第25/30轮，训练步4680到4686后进入验证；续训日志best 0.5109891295433044，仅训练日志观测，不是冻结全量评估或晋升。GPU0=280W/81C/22684MiB，GPU1=390W/44C/1MiB无compute，当次无热降频或新Xid；证据pipeline/checks/prepared_resources_befeacf640bf499a9ff39fa6c9df712a.json。

## 5. Decisions Made
- 当前只让server-2 GPU0上的原Y56继续，保留既有280W限制；Y57及GPU1等待新的用户恢复指令，不自动补位、不迁移或从预训练重开。旧heartbeat双卡表述被最新用户单卡指令及项目条款收窄。
- 无新实验、模型或YAML修改、Git提交、生产代码同步、准入、入队或训练启动；保留当前goal、禁用项、数据边界、seed/固定30轮及全部历史失败/恢复证据。挂起Y57不阻断其他合法单卡准备，但不能挤占Y56。
- 硬件故障根因仍未定。暂停后温度采样伴随Y56训练转验证，不能把降温全归因于停GPU1，不能宣称散热/掉卡已解决；未改变GPU1功耗、风扇、驱动或系统配置，也未读取私密凭据。
- 后续空槽检查使用GPU0 trial入口，需当前租约和实时空闲GPU0；Y58/Y59再Y60均须以原生YAML真实两步检查通过后才准入/门控派发。按本轮实际一次资源检查结果收尾，不凑新候选或全局冷却。当前报告不预称postflight通过，随后执行canonical finish并核验context提交和租约释放。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
