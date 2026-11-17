# Y25 Hard Mask Attention Support Preregistration

- 固定原W2训练数据前64个独立scan，原生增强和seed1999，单卡GPU1；不看验证集选择场景。
- 因果假设：上一层硬掩码注意力把匹配正query的真实目标点质量排除，下一层无法读取这些目标证据。训练期连续log(mask probability)先验可重开证据和梯度路径；不增加参数，推理继续原硬掩码。
- 主读数：每个匹配正query/每个实际用于下一层cross-attention的阶段，真实GT点质量被gate排除的比例；逐scan等权汇总，全体结果保留。完全排除/部分排除、原all-blocked fallback单独报告。
- 对照：同一原生mask/匹配/GT，原Boolean gate、all-valid支持、有限连续logsigmoid bias；二次seed-reset全模型捕获与缓存输入重放区分原生数值噪声。
- 判定：若没有目标排除或原生训练路径根本未使用该gate，则这一实现被否定；若排除存在，仅证明信息缺口，不能直接PASS。连续干预必须真实恢复目标支持、产生有限非零mask-to-attention梯度；all-valid/常量mask对照区分普通无条件平滑。
- 仍须原生模型backward/optimizersteps/save-resume、最大场景资源、物理独立、推理源码/同权重输出一致和目标约束六项门槛。不扫描温度、阈值、损失权重或特征。
- 准入前另冻结完整30轮/全量ScanRefer、W2/B1两遍、整场景bootstrap与尺寸伤害/成本比较。诊断不代表训练效果或新颖性。
