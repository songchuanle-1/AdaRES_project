# Research Pipeline Report - AdaRES_project
快照时间：2026-10-19T08:55:55.975308+00:00；生成报告时快照年龄：2382 秒；模式：design

## 0. Goal Status
- 目标：mIoU / 0.515；数据集：ScanRefer
- 目标检查返回码：0
- Champion / Gap / Phase：以下为本轮目标检查原始结果，缺项为未知。
```text
================================================================================
  Goal Check Report — 2026-10-19 16:55:55
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
================================================== 2026-10-19 16:55 : AdaRES_project ==================================================

--- server-2 (192.168.1.163) ---
  [adares_y54_fine_support_res] Best=0.3393 2/30 no_improve=0/30 gpu=0 ?
  [adares_y52_point_target_res] 1/30 尚无验证 - gpu=1

--- server-6 (100.64.3.28) ---
  [白名单] SERVERS.yaml allowed_gpus=[] ，本项目流水线跳过该机

--- 队列: AdaRES_project ---
  server-2: running=2
```

## 2. Running Experiments
| ID | Config | Server | GPU | Epoch | Best mIoU | Acc@50 | spmIOU | No-Impr | Status |
|---|---|---|---|---|---|---|---|---|---|
| AdaRES-Y52 | configs_song/model_paper5/adares_y52_point_target_res.yaml | server-2 | 1 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |
| AdaRES-Y54 | configs_song/model_paper5/adares_y54_fine_support_res.yaml | server-2 | 0 | 未知 | 未知 | 未知 | 未知 | 未知 | 登记运行；实时进度须核实 |

## 3. Queue Status
| Position | ID | Config | Status | Depends On | Server |
|---|---|---|---|---|---|
| 未知 | 无 | 未知 | 无排队 | 未知 | 未知 |
- 活跃 2（运行 2 / 认领 0 / 等待 0）；目标 2；准入预算 0。
- 研究储备 1 / 2；储备准备预算 1，不增加训练并发或启动授权。
- 储备只计可推进候选及已排队项；另有待人工 0、待资源 0、未分类阻塞 0。
- 非终态候选与排队项合计 1；总上限 8。阻塞不授权重开旧路线；其他合法独立工作按剩余预算推进。
- 候选累计记录 108：complete=14 / rejected=79 / closed=12 / training=2 / diagnosis=1。
- 否决范围：unclassified=72 / implementation=4 / hypothesis=3；实现修订累计 3；收据文件 33（含修订或重试，不代表成功率）。

## 4. This Tick's Actions
- 仅AdaRES_project，本轮上海16:55:55快照，design/reserve_only；准入预算0/新增储备预算1/绑定due Y56。规范begin的monitor/goal各一次，0-3保留脚本事实表；以下17:24/17:31/17:33为后验，不重复预检、不改调度/其他项目。
- 先核实67MB既有原生capture哈希5716585e...与源程序，发现可直接CPU回放真实TRAIN最终指标头输入；预注册固定8scene/32表达/每条128query、原query/softmask精确复制及单独native零logit硬mask等价计数，未重新挑样本或扫阈值。原采集源代码加载W2路径，但未当时记录checkpoint blob SHA；当前head严格加载e340728...不追认旧采集权重字节绑定。
- 首次回放在JSON输出时因NumPy float32 checkpoint元数据失败，原冻结program/prereg/execution保留。随后只新增metadata-v2副本和预注册，单行元数据标量转换不改输入、head、测试或界；上海17:24:22至17:24:29成功运行，CUDA隐藏、CPU单线程、checkpoint前后hash/参数state不变。不是模型修订或训练失败。
- 实际缓存每条query_unique=128、softmask_unique=128，32表达共4096query位置的joint_duplicate_excess为0；hard-mask等价额外3072不是query/softmask复制。保持唯一原型集合的真实复制转移反事实适用0条，未强造阳性。32条实际输入repeat0/排列最大1.9073486328125e-6在预设界内；4个相关合成fixture/dtype case中trained-head主logit差float32=.00399639830/float64=.00865047191且阴性在误差界内。不能将该合成敏感性当任务缺陷、mIoU提升或新颖性。
- 同一Y56结论保持inconclusive：固定单场景最终层样本未观察到精确复制前提，暂不实现/准入，也不外推总体或全层不存在。已冻结原8scene/32表达/4nativebatch2/全部7层捕获程序，源/config/checkpoint绑定及BN缓冲恢复；Python3.8解析/compile、缺失/非法GPU入口拒绝测试通过，未执行GPU或原生反向/资源门禁。native_prereg与program同token留档。
- 上海17:31:47至17:31:53原PID双采样：Y52 GPU1/PID69064 Train[2/30]501至511/4686；Y54 GPU0/PID47332 Train[3/30]1791至1801/4686。两卡均真实占用（22710/22716MiB及各自compute PID）、精确cwd/config、有界tail无异常；waiting0，未改任何训练进程。证据pipeline/analyses/y56_native_occupation_222ed2a6e80c43f6b1de2d050be97fa0.json。
- 17:33只读抽取精确当前stdout尝试，舍入3位小数的mIoU序列Y52 [.335]、Y54 [.339,.376]；不是终态或冻结重评估，早期值不说明方法优劣。历史CSV冠军Y36 .5110/B1 .5020、目标.5150差.0040保持，不混用精确旧比较或W2 checkpoint meta。
- 本轮产出为1项预注册真实缓存CPU诊断及保留原输出失败/v2格式修复、32表达回放、4个相关合成case、1份待执行GPU程序和入口测试、真实占用/进度与指标记录及决策/记忆/报告；新增候选0、模型修订0、新准入0、训练启动0、GPU诊断0。上轮125项工程回归不是本轮成果，108累计记录不是实验数或成功率。收据用量窗口仅owner.started_at至receipt本地可见记录，具体值待收据/未知，排除早读/晚到/最终回复；缓存含在input、推理含在output，不重复相加、不当账单。当前尚待finish，未预称PASS。

## 5. Decisions Made
- Y56暂不进入实现/准入：真实缓存未见精确复制，与trained-head合成阳性分开；hard-mask等价不是feature/softmask复制，不重开Y48或凑新ID。完整review见pipeline/analyses/y56_cache_review_222ed2a6e80c43f6b1de2d050be97fa0.json，决策见20261019-173154_other.md。
- 所有本轮非GPU工作已完成，Y56以真实exclusive_gpu_slot等待已冻结native batch2/all7层捕获；retry2026-10-19T18:31:53.002417+08:00只是复查条件。空闲后先核实新租约/占用/source/checkpoint/goal，再用现有受控runner执行；不停止两项原训练腾卡，不再声称缺挂载或缺训练授权。
- 按真实external_resource_blocked有界收尾，不称无信息增益/路线穷尽或等待耗尽deadline；剩余新增储备预算1不通过复制或调参变体填充。规范report/finish后独立核对收据及锁释放，科学有效性仍未确定。

Next tick: 以已保存自动化为准；本脚本未读取调度时间。
