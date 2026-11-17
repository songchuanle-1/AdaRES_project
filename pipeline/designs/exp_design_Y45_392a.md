# Experiment Design - AdaRES-Y45

## 1. Motivation
AdaRES: improve a faithful standard-indicator 3D referring expression
segmentation baseline by learning query confidence from metric-aligned
evidence, targeting ScanRefer mIoU >= 0.515.
The SAME Y45 tests supervision that distinguishes FP and FN at equal IoU.
Native TRAIN target availability and controlled objective differences are
diagnosis evidence, not a scientific novelty or efficacy conclusion.

## 2. Current Best Results
W2 remains the scientific comparator. Rounded CSV Y36 .5110 is not a promotion.
Y43 FIRST .5083255, W2 .5101080, B1 .5022218. Y43 small-object harm review is
triggered by the frozen interval rule. Y44 frozen result is still pending.

## 3. Analysis And Direction
Matched binary labels, scalar IoU and historical listwise targets cannot
distinguish equal-IoU FP/FN partitions in the fixed witnesses. A categorical
target retains that distinction. This does not establish useful learnability.
Y29's rejected renamed ranking realization remains closed; no margin,
temperature, point-mass calibration or renamed scalar-ranking route is used.

## 4. Experiment Plan
Independent six-file W2-derived package and unique identical-semantic YAML.
Only final confidence receives a query/foreground/background point-mass
readout with inherited model width. Three output logits represent TP, FP, FN;
quality log odds zTP-logsumexp(zFP,zFN) add to the existing positive indicator
logit. No GT enters prediction. Masks used for aggregation and targets detach.
The new TRAIN loss is the categorical likelihood of literal point-confusion
fractions, averaged across expressions/queries, with no searched multiplier.
This factorization retains the parent matching objective, an explicit
complementary-evidence assumption, not an established calibration theorem.

Prospective controls: frozen TRAIN8 indices0,44,60,101,115,126,137,171 and
released-init first4; same masks/features/GT, repeat, cyclic wrong expression,
FP/FN collapse, padding/query/point permutations, disabled physical parent,
new-tensor finite nonzero gradients and optimizer updates. Initial adverse
losses are retained and investigated with fixed two new-only updates, not
discarded or tuned. Require true native source extraction and graph activity.
Native two-step optimizer/scheduler restore and shared-tensor identity remain
pending. Diagnostic updated weights are never deployed.

Post-training freeze, only after six actual gates: released SPFormer,
seed1999/fixed30/singleGPU; Y45/W2/B1 each full9508/141 twice, FIRST primary;
scene bootstrap10000seed1999/TRAIN size cuts1487/3325, meaningful subgroup
CI upper<0 versus either comparator triggers harm review. Intervals crossing
zero mean inconclusive. Report empirical repeats, object-count/official
unique-multiple limitations, active128/singleargmax and trained paired costs.

## 5. GPU Allocation Plan
Only server-2 GPUs0/1 are authorized. GPU0 PID19723 must be preserved.
GPU1 may run exclusive bounded tests only while live idle. No server6,
other-project operation, scheduler/background/new task or nested Codex.
This tick admission budget is one; missing gates do not authorize dispatch.

## 6. Implementation Checklist
- Diagnosis and actual historical objective controls recorded.
- Independent package and unique YAML pending.
- Native controls, all new gradients, physical-parent equality pending.
- Native optimizer/scheduler continuation and resource stress pending.
- Exact tested-byte commit, constraint review and six-gate receipt pending.
- Only canonical gated dispatch and increasing native steps prove launch.

## 7. Risk Assessment
Targets depend on current hard masks; changing masks makes supervision
nonstationary. The parent latent state may already encode these quantities.
Error decomposition may worsen small targets or conflict with matched labels.
No fixed task-effect cutoff is used as a pre-training gate. Added params<1%
of W2, native TRAIN pressure reserved<24GiB/noOOM/nonfinite; measure time/memory
regressions. No population, necessity, equivalence, novelty or no-harm claim.

## 8. Smell Check
Potential smell2: another MLP. Defense: the causal intervention is a new
point-confusion categorical supervision and metric-odds information path,
not normalization, a dimension scan, arbitrary scalar or reused ranking loss.
All three categories and inherited width are fixed; useful intervention,
native new gradients and real independence must still be demonstrated.
