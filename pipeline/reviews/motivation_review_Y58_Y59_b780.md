# Motivation Review - AdaRES_project

时间：2026-10-21，上海时间。本稿汇总本轮先行预注册及实际 CPU 诊断；不是新增训练审批。

## 1. Goal

当前 `pipeline/goal.json` 完整内容：

```json
{
  "metric": {"name": "mIoU", "target": 0.515, "dataset": "ScanRefer"},
  "baseline_experiment_id": "AdaRES-B1",
  "description": "AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515.",
  "constraints": [
    "Use server-2 GPUs 0 and 1 only; never use server-6.",
    "All experiments launched after AdaRES-B3 must use exactly one GPU; run at most one independent experiment on GPU 0 and one on GPU 1.",
    "Do not design or queue a successor while AdaRES-B3 is running; after B3 reaches a terminal state, admit at most two structurally independent single-GPU candidates.",
    "Use a fixed 30-epoch protocol for the B2 and B4 extension. Resume both from their epoch-20 last checkpoints without resetting the optimizer or LR scheduler; do not tune the epoch count on validation.",
    "Train from the released SPFormer sparse-UNet backbone; the decoder is authored in this project.",
    "Report mIoU, Acc@0.25, Acc@0.5, active queries, latency, and peak memory.",
    "Do not tune thresholds, temperatures, loss weights, feature subsets, or the fixed 30-epoch protocol on ScanRefer validation."
  ],
  "banned": [
    "Do not use server-6.",
    "Do not report the released b005/CFNet checkpoints as AdaRES-B0 results.",
    "Do not scan the ranking margin or the occupancy loss weight on validation."
  ],
  "pipeline_policy": {
    "research_on_capacity_gap": true,
    "steady_ticks_before_design": 1,
    "fill_to_capacity": true,
    "design_buffer": 0,
    "max_candidates_per_design_tick": 2
  }
}
```

目标 .5150；历史 CSV 冠军 Y36 .5110，记录精度下差 .0040，仍为 Phase 1。
上轮冻结全量评估 Y36 FIRST=.510725438294，不替换 CSV 值或与早期训练验证混用。
本轮容量 2/2、准入预算 0、储备准备预算 2。B3 已非运行任务，仍仅准备两个稳定诊断 ID。

## 2. Story Status

| 假设 | 状态 | 证据 | 缺口 |
|---|---|---|---|
| 指标对齐的查询竞争有用 | 历史相关结果，非本轮因果证明 | W2/Y36 的既有冻结评估 | 新方法完整训练、匹配对照与不确定性 |
| Y56 精确去复制竞争、Y57 同伴二阶矩 | 两项实际训练中 | 本轮原 PID/配置、GPU 与原生步数复核 | fixed30 结果及冻结的全量复评 |
| Y58 跨场景归一化同伴依赖 | CPU 组件见证成立 | y58_cpu 与 y58_data 本轮结果 | 原生稀疏特征、采样和置信度是否发生可重复变化 |
| Y59 冻结编码器训练模式契约 | 实际编码器模式及语言特征行为已实测 | y59_cpu 本轮结果 | 原生任务置信度、损失及后向的作用；收益或伤害均未知 |

## 3. Experiment Assessment

定向使用已有紧凑记忆及与两条假设有关的台账，不重新展开全部历史。

| 实验 | mIoU | 与动机的关系 | 对故事线贡献 |
|---|---|---|---|
| B1，既有冻结全量 FIRST | .502221764356 | 忠实标准指标基线 | 历史对照，不是本轮复跑 |
| W2，既有冻结全量 FIRST | .510107995189 | 点质量竞争父模型 | 本轮诊断采用其已训练权重，不用于替代发布初始化准入 |
| Y36，既有冻结全量 FIRST | .510725438294 | 历史完整评估最佳 | 未达目标，不保证其他路线有效 |
| Y52 / Y54，既有冻结全量 FIRST | .509708428366 / .506342952986 | 旧结构路线 | 保留原不确定性和 Y54 大目标伤害结论，不改名重开 |
| Y56 / Y57，本轮第 1 轮原生验证 | .339575380087 / .325815230608 | 新结构训练早期观测 | 仍在第 2/30 轮，不与完成训练结果判优 |

后两项数值来自 2026-10-21 00:15:52 上海时间、精确注册原生日志谱系；其他任务指标本轮未重新采集。

## 4. Gap Analysis

Q1：查询置信度可能混入与目标无关的批次上下文或冻结语言随机性，但这些是否影响任务尚未测出。
Y58 只在合成特征上替换同伴，使用实际 W2 BN 仿射参数，float32/64 焦点变化最大 2.54591/2.66248；固定统计、逐场景与重复对照均为 0。
实际 W2 指标头在固定输入下跨样本可分离，误差 0。这把 CPU 见证限定在归一化组件，不能指称整个网络有害。
真实 TRAIN 数据场景 0+1、0+2、0+1 重复的焦点坐标、特征、标签和语言均逐位一致。
Y59 直接使用实际冻结 W2 Roberta、4 条 TRAIN 表达和已训练语言投影；外层精确 train 方法重启 37 个 dropout。
未固定随机数时有效 token 特征最大差 13.71283、投影最大差 1.336459；固定随机数重复误差 0。
仅恢复编码器 eval 可精确恢复构造期特征，父模型仍训练、参数未变且编码器无梯度。不是完整稀疏任务收益试验。

## 5. Design Rationale

Q2：Y58 原生试验只改变非焦点同伴，并以 BN 训练/固定统计分支及精确重复隔离依赖；不扫描归一化配置。
Y59 原生试验固定同一批次及种子，只恢复冻结编码器模式；其他 dropout 关闭、BN 模式不变，并隔离编码器随机数消耗对下游的影响。
Q3：这些诊断只决定是否值得实现独立候选，不推出正收益、无伤害或论文新颖性。无信号时保留负面及重复证据；采样改变或误差不可分离则记录 inconclusive。
两条因果问题独立：一条改变跨场景统计依赖，一条改变冻结语言训练模式。均非 Y56/Y57 的超参数扫描。
定向台账检索未发现同假设拒绝项；不是完整文献综述，不据此宣称新颖。Y23/Y41 的其他归一化记录不自动反证 Y58。

## 6. Next Batch

见 `pipeline/designs/exp_design_Y58_Y59_b780.md`。两项 GPU 程序已冻结且通过 Python3.8 解析、编译及缺失 GPU 时的入口拒绝检查，实际原生执行仍 pending。
server-2 GPU0/1 均由已授权训练占用；保留原进程，不挤占或停止。空卡后以新一轮实际租约重验源码、数据、权重及实时资源，再执行同 ID 诊断。
本轮无新模型包、YAML、本地 Git 提交、推送、准入、队列新增、训练启动或调度改动。
