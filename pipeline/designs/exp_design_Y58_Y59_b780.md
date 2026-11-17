# Experiment Design - AdaRES_project

时间：2026-10-21，上海时间。状态：DRAFT，诊断储备；非已准入训练变体。

## 1. Motivation

项目动机原文：AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515.
本轮触发为满载储备准备，非容量缺口训练：running=2、claiming=0、waiting=0、target=2；candidate_budget=0、reserve_budget=2。
上游动机、完整目标与历史限制见 `pipeline/reviews/motivation_review_Y58_Y59_b780.md`。
两条路线回答置信度是否含不相关上下文的问题，不以加层或更改数值分布直接作为可训练贡献。

## 2. Current Best Results

| Experiment ID | Config | State | Best mIoU | Acc@50 | spmIOU | Epochs |
|---|---|---|---|---|---|---|
| Y36 | 历史冻结独立配置 | complete，历史 CSV | .5110 | 本轮未采集 | 本轮未采集 | 本轮未采集 |
| W2 | adares_w2_pointmass_competition_res | 历史全量 FIRST | .510107995189 | 本轮未采集 | 本轮未采集 | 已完成，详情保留旧记录 |
| Y56 | adares_y56_unique_competitors_res | 确认训练中 | .339575380087 | 本轮未采集 | 本轮未采集 | 2/30，验证仅第1轮 |
| Y57 | adares_y57_peer_second_moment_res | 确认训练中 | .325815230608 | 本轮未采集 | 本轮未采集 | 2/30，验证仅第1轮 |

后两项来自本轮 00:15:52 上海时间精确注册原生日志，不用于训练前选模型或修改超参数。

## 3. Analysis & Direction

### 3.1 哪些方向有效
已有 W2/Y36 完整评估超过历史 B1，但本轮未重做因果验证，不能外推到 Y58/Y59。

### 3.2 哪些方向失败或不确定
Y54 的既有有害子组证据保留；Y52 总体差异不确定。Y57 原数值失败及同 ID 连接修订分别保留。
本轮 Y58/Y59 的 CPU 行为不是训练成功或失败；Y58 稀疏任务关系、Y59 原生任务关系尚未知。

### 3.3 哪些尚未消融
两项储备目前没有独立模型实现，不进入消融或验证集搜索。正在运行的 Y56/Y57 不作任何热更。

### 3.4 结论
先用预注册因果对照测原生关系；若关系成立，再审查是否存在符合结构约束的实现，而非直接把模式/归一化开关作为创新入队。

## 4. Experiment Plan

| ID | Config Path | Purpose / Hypothesis | Expected Result | Priority | Depends On |
|---|---|---|---|---|---|
| AdaRES-Y58 | 无新 YAML；只读 W2 父配置 | 固定焦点 TRAIN 场景，切换两个预定同伴，原生 BN 训练/固定统计双分支 | 得到可重复的上游特征/置信度差异或有界负结果，不设收益门槛 | P1 | 授权独占空卡与冻结清单匹配 |
| AdaRES-Y59 | 无新 YAML；只读 W2 父配置 | 同批次/权重/种子，仅恢复冻结文本编码器 eval | 得到原生 loss/backward 与置信度对照，收益方向未知 | P0 | 授权独占空卡与冻结清单匹配 |

Y58 CPU：实际训练 BN 仿射参数、float32/64、重复/逐场景/固定统计/排列对照及实际头可分离性已测；真实数据 collate 焦点身份已测。
Y58 原生：0+1、0+2、0+1 重复，在两个 BN 分支共六臂；全部 dropout 关闭、每臂恢复缓冲区、固定种子。
按体素坐标对齐焦点骨干激活，并记录焦点采样 ID、置信度差、重复误差、loss 与峰值显存。采样不同时只能报槽位差，不能声称固定 query 因果对照；总 loss 含不同同伴目标，不判为焦点收益/伤害。
Y59 CPU：实际 W2 冻结 Roberta 和已训练投影、真实 TRAIN 4 表达/140 有效 token 已测；精确原 train 方法由 CPU 容器调用，不冒充完整稀疏模型。
Y59 原生：original、original_repeat、restored_frozen_mode、restored_repeat 四臂；保留 BN 训练、其他 dropout 关闭。
编码器进出钩子恢复随机数状态，避免 dropout 消耗改变下游随机路径；记录实际采样变化，不把槽位差偷换成固定查询差。
执行完整 native loss/backward，检查七层输出及所有指标参数有限非零任务梯度、冻结参数无梯度、参数版本不变，恢复缓冲区且不进行 optimizer step 或保存权重。
两份程序、prereg 和 static 证据均在 `pipeline/analyses/y58_native_*b780*`、`y59_native_*b780*`，由候选清单绑定具体 SHA256。
目前两项实际 GPU 测试均未运行，六个完整准入门槛仍 pending，不签发 admission receipt。
未来训练若设计成立：先独立模型/唯一配置、发布 SPFormer 初始化、固定30/seed1999/Q128/单 argmax；训练后报告 mIoU、Acc@.25/.5、活跃 query、时延、显存。
完整匹配对照、场景聚类 bootstrap/重复、目标大小等有害子组规则必须在训练前冻结；本轮尚未形成新训练协议，不得直接启动。

## 5. GPU Allocation Plan

| Server | GPU | Assignment | Config | Justification |
|---|---|---|---|---|
| server-2 | 0 | 当前 Y56，不干预 | adares_y56_unique_competitors_res | 原 PID83638，步数增长 |
| server-2 | 1 | 当前 Y57，不干预 | adares_y57_peer_second_moment_res | 原 PID95947，步数增长 |
| server-2 | 首张真正空闲的0或1 | 后续单次独占诊断，非排队任务 | 只读 W2 配置 | 先核实实时利用率/显存/全部 compute PID，再取独占租约 |

不访问 server-6 或其他项目，不改变并发策略、调度、资源白名单或现有训练。任何 GPU 打开都不是新训练准入授权。

## 6. Implementation Checklist

- [x] 两个稳定 ID、事前冻结诊断问题和对照，不复用历史拒绝 ID 的新名字。
- [x] 完成非 GPU 组件/实际语言和数据控制；保存结果、源、脚本和日志。
- [x] 冻结 source-bound 原生程序；Python3.8 解析/编译及缺 GPU 入口拒绝检查通过，未执行 CUDA。
- [x] 记录实际训练原进程和全部授权 GPU 的占用证据。
- [ ] 空卡后重验实际依赖、权重、数据、源码、当前租约并执行两项原生诊断。
- [ ] 按原生结果决定是否实现独立模型，不提前通过结构、资源或后向门槛。
- [ ] 如将来存在已测试候选，仅在具体授权覆盖时作精确本地提交；推送及其他文件不在本轮操作范围。
- [ ] 将来完整准入后才允许规范队列与门控部署。本轮不新增队列，不同步候选或提交 Git。

## 7. Risk Assessment

代码：当前仅项目内诊断与文档；原生程序静态通过不代表运行正确，首次 native 失败按实现/不确定范围记录，不自动否定机制。
资源：两张 3090 均被真实训练占用。脚本含24GiB峰值检查但不证明不会OOM；必须空卡独占、受时限保护，绝不借空闲显存共跑。
数据：冻结 TRAIN 场景及增广；组件合成行不算任务样本，场景是独立统计单位。数据集实源在远端 scannetv2_test.py，不能猜本地缺失文件。
因果：改变同伴会改变目标及采样；修复语言模式可能改变学习采样。记录并限制解读，固定统计也不保证数值误差为0。
收益：所有完整训练收益、泛化和人口层面的伤害结论均未知；模式契约问题可以存在而 dropout 仍有正则化作用。
回退：保留原始失败/重复及程序；任何新实现必须原稳定 ID 有界修订，不反复扫描、不重启当前训练、不降低门槛。

## 8. Smell Check

Y58 若仅替换 normalization，触发 smell:1/2/3/6；本轮只做统计依赖诊断，没有通过结构创新标准，不可据 CPU 信号直接入队。
Y59 若仅设置 eval，触发 smell:1/3/6；冻结模式契约是可复现工程问题，但没有证明新计算图或足够独立的研究贡献。
两条诊断保留是因为问题可证伪、真实组件证据及匹配对照齐备；并不等于通过新结构门槛。超参数不扫描、无验证集择优。
后续若没有合规结构或明确工程修复定位，应如实停止训练转化，不能为占满资源换名字或包装贡献。

Status: DRAFT - diagnostic reserve only.
