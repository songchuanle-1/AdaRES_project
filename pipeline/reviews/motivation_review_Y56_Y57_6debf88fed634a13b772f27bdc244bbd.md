# Motivation Review - AdaRES_project
采集/审查：2026-10-20T21:24:56.414176+08:00。上游证据：`pipeline/analyses/design_and_terminal_review_6debf88fed634a13b772f27bdc244bbd.json`。本轮不是新训练。
## 1. Goal
```json
{
  "metric": {
    "name": "mIoU",
    "target": 0.515,
    "dataset": "ScanRefer"
  },
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
历史规范冠军Y36 .5110、目标.5150、差.0040；本轮冻结重评估另列，不混用监控分段best。
## 2. Story Status
| 假设 | 状态 | 证据 | 缺口 |
|---|---|---|---|
| Y52点质量目标 | fixed30和自身冻结评估完成，比较未全闭环 | 全9508表达/141场景双FIRST；W2/B1已各双次 | 新Y36双次及配对比较待补，不自动晋升 |
| Y54细支持有效性 | 同上 | 同协议，当前W2/B1比较已做 | Y36待补，不把单次下降当机制反证 |
| Y56复制数量干扰上下文 | 局部原生见证 | 40独立场景中一个场景初始层3记录 | 原生任务损失可达性及最终伤害未知 |
| Y57均值丢失peer散布 | 完整向量有观察信号，仍不确定 | 原头全部输入条件下固定LOSO | 原生可达干预/后向/资源及训练收益未知 |
## 3. Experiment Assessment
以下四方法均本轮FIRST全量结果；Y36尚未运行，不替代历史结果，单位保持原mIoU/Acc：
| 实验 | mIoU | Acc25 | Acc50 |
|---|---|---|---|
| Y52 | 0.50970843 | 0.610013 | 0.562053 |
| Y54 | 0.50634295 | 0.603176 | 0.556584 |
| W2 | 0.51010800 | 0.610749 | 0.563000 |
| B1 | 0.50222176 | 0.610433 | 0.552903 |
预注册伤害触发数Y52=0、Y54=0，逐组保留；
是否超过基线或冠军须看配对场景区间，不能用rounded best代替。
## 4. Gap Analysis
Y56初8场景无复制；新增32个不同场景有3个精确复制额外项，均为scene0029_00/layer0，
不是1120独立样本，也不是hardmask相等。固定focal0/唯一集/GT的实际复制转移使置信度变化
.0017069-.0044413；逆复制参考误差<=1.20e-7，重复0，排列<=6.26e-7。并未证明任务损失变化。
Y57置信度条件正结果不能代替全输入条件证据：散布标量增益-.00037594、区间跨0；
完整256通道增益.02372823、区间[-.00259745,.04683551]，对同容量错位对照
.02439002/[.00053385,.04511317]。scene0004_00和scene0005_00下降保留。
该向量补测在标量结果之后预注册，属于自适应探索，不是独立确认或条件独立性证明。
## 5. Design Rationale
同一Y56先验证稀少复制是否能通过实际任务损失产生梯度；不从局部置信度扰动直接上训练。
同一Y57保留原头全部输入并增加逐通道二阶矩，避免标量抹去方向；
结构固定于既有d_model256，不扫描特征、层、正则、阈值或超参数。新颖性尚未检索确认。
## 6. Next Batch
见`pipeline/designs/exp_design_Y56_Y57_6debf88fed634a13b772f27bdc244bbd.md`。两项均未正式实现、提交、准入或训练；不存在缺新GPU授权的当前阻塞。
