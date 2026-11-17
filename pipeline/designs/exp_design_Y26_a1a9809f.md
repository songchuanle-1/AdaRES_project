# Experiment Design - AdaRES_project

2026-10-04，北京时间。Y26稳定ID；本文件为工程实现设计，
测试结论见后续审查，不提前把资源/准入标成通过。

## 1. Motivation

项目动机：“AdaRES: improve a faithful standard-indicator 3D referring
expression segmentation baseline by learning query confidence from
metric-aligned evidence, targeting ScanRefer mIoU >= 0.515.”
用户手动要求继续设计；不是容量缺口tick。18:29快照active1/budget0，
实际Y25已训练结束，正常deathwatch2/3未释放canonical名额。
目标是验证语言条件化query中被丢弃的归一化/affine学习路径。
上游审查：`pipeline/reviews/motivation_review_Y26_a1a9809f.md`。

## 2. Current Best Results

| ID | 配置basename | 科学状态 | 冻结mIoU | Acc50 | spmIoU | 轮数 |
|---|---|---|---|---|---|---|
| W2 | adares_w2_pointmass_competition_res | champion、未达标 | .510108 | .563 | .474 | 30 |
| B1 | 标准指标器基线 | 完成 | .502122 | 见原冻结证据 | 见原冻结证据 | 30 |
| Y23 | adares_y23_valid_token_support_res | CI跨0、完成 | .507472 | .558162 | .470801 | 30 |
| Y25 | adares_y25_mask_feedback_res | 相对W2有害、拒绝 | .494117 | .551115 | .456580 | 30 |

W2表中Acc50/spmIoU为此前同协议约数，不冒充最新Y25比较捕获精确值。
来源：既有不可变Y23/Y25 terminal_review及本轮动机审查。

## 3. Analysis & Direction

W2领先B1但不足.5150；Y18/Y17/Y23没有可靠收益证明，Y25明显负向。
不扫描原机制或归一化位置。新观察为qla既有affine参数长期失活；
其因果位置独立于此前目标监督和mask feedback。诊断证明“非恒等且有
失活参数”而非“归一化必改善”。RMS略高于connected读数，不选择RMS。
初始loss增加已做缓存center/RMS分解；尺度变化风险保留。

## 4. Experiment Plan

| ID | Config | 假设/操作 | 判读 | 优先级 | 依赖 |
|---|---|---|---|---|---|
| AdaRES-Y26 | configs_song/model_paper5/adares_y26_qla_postnorm_res.yaml | 仅qla接回已有postnorm，TRAIN/EVAL一致 | 工程门与冻结30轮有效性分开 | P0 | 六门审查、budget释放 |

预注册：`Y26_postnorm_prereg_a1a9809f.md`。不改dropout0、损失权重、
学习率、温度、阈值、query数、特征、匹配或mask门。
实际W2 checkpoint用于诊断；未来生产训练从released SPFormer开始。
后训练固定9508表达式/141场景，Y26/W2/B1各2次；10000 whole-scene
bootstrap seed1999，TRAIN原尺寸界1487/3325；上界低于经验重复噪声负值
的分组delta判material harm，跨界为不确定。报告Acc25/50、spmIoU、
128queries/1selected、独立串行时延和峰值显存；不宣称三读可靠p95。

## 5. GPU Allocation Plan

| Server | GPU | 用途 | 依据 |
|---|---|---|---|
| server-2 | 1 | 当前诊断/两步smoke/资源测试 | 每次live idle核验及独占锁 |
| server-2 | 1 | 后续Y26正式训练仅在合法准入后 | 本轮budget0，尚未启动 |

GPU0用户保留，不查询/使用；server-6不使用。禁止其他调度/项目操作。

## 6. Implementation Checklist

- [x] 独立复制W2六个Python源文件及字节相同parent配置。
- [x] 仅`CrossAttentionLayer.postnorm`默认False、qla设True；
  计算已有norm一次并接回，其他注意力保持原返回值。
- [x] 结构AST/无跨variant import/本地远端七文件hash核验。
- [x] TRAIN64、trained8/init4完整反传、缓存center/RMS/affine分解。
- [ ] native train两步、checkpoint模型/AdamW/PolyLR恢复，结果待绑定。
- [ ] 原冻结大场景728/6780压力及串行TRAIN8推理资源，结果待绑定。
- [ ] 审查六门、scoped commit、预算释放后的admission receipt。
- [ ] canonical dispatch及真实native递增step；本轮不执行。

后续检查结果写独立审查/ledger，不改原预注册或历史结果。

## 7. Risk Assessment

新增可学习路径并非新增参数；改变原W2尺度/梯度，初始loss上涨，
可能损害指标器校准和最终mIoU。TRAIN/EVAL同时改变，不存在Y25同款
故意训练/推理门差异，但也不证明一般分布稳定。
保留所有不利TRAIN场景与数值resume误差；不从小cohort强行统计收益。
GPU1资源需实际压力测试，若不通过则调查/保留原失败，不能放宽准入。
本设计为独立实现验证，不是已批准训练或novelty主张。

Status: IMPLEMENTED (diagnostic source only; production admission pending)
