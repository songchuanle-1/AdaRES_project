# Experiment Design - AdaRES-Y14 Executable Supplement

Created before new capture/fit: 2026-09-30 Asia/Shanghai.
Token: `0eff5f5d4e4847debba4208c85035a3b`.
This adds missing execution details to the original Y14 design; it does not
change Y12/Y13 results or Y14's frozen final acceptance rule.

## 1. Motivation

"AdaRES: improve a faithful standard-indicator 3D referring expression
segmentation baseline by learning query confidence from metric-aligned
evidence, targeting ScanRefer mIoU >= 0.515."
Question: can prediction-only mask correspondence predict when B4 confidence
is useful for W2 query selection?

## 2. Current Best Results

W2: full ScanRefer mIoU 0.5100, 30/30 epochs. B1: 0.5020.
Y13: diagnostic scene-paired gain +0.00521, p=0.11018, one selection
repeat mismatch. This is neither a full validation metric nor a passed gate.

## 3. Analysis And Derivation

For binary superpoint masks A and B, with point counts n_s:

I_ij = sum_s n_s A_is B_js
J_ij = I_ij / max(1, sum_s n_s A_is + sum_s n_s B_js - I_ij).

This equals binary point-mask IoU exactly; no sampling or mask threshold change
is permitted. Match by Hungarian maximization of J, using unmodified scores
and SciPy's returned order (no epsilon tie-breaking). All-empty unions map to 0.
Entropy is point-count-weighted Bernoulli entropy of sigmoid mask probabilities.
The numerical log clamp is 1e-7, not a tuned mask threshold.

Standardize each model's margins within each expression by population standard
deviation. A constant margin fails capture rather than receiving a tuned offset.
For matched index pi(i), use the original six features:

phi_i = [J_i, 1-J_i, abs(zW_i-zB_pi(i)), hW_i, hB_pi(i), 1].
g_i = sigmoid(w^T phi_i), c_i = zW_i + g_i zB_pi(i).

For y_i > y_j:
ell_ij = softplus(-(c_i-c_j)/0.25).
d ell_ij / d w =
-sigmoid(-(c_i-c_j)/0.25)/0.25 *
(zB_pi(i) g_i(1-g_i) phi_i - zB_pi(j) g_j(1-g_j) phi_j).

Average pairs within an expression, expressions within a scene, then scenes.
Exclude expressions with no strictly ordered pair from loss only; keep them in
selection evaluation. The original summed pairwise objective is thus assigned
explicit sampling weights; all three trained arms use the same weights.
Features J, 1-J and 1 are linearly dependent; retain the frozen features and
report effective rank rather than pretending six independent explanatory axes.

## 4. Experiment Plan

Capture first 64 train scenes and first 32 val scenes in the existing dataset
order, training=False, aug=False, with_elastic=False, lang_num_max=32.
Freeze W2/B4 checkpoints from the original design. Record scene IDs, ann IDs,
config/checkpoint SHA-256 and reject overlapping train/val scan IDs.
Target masks provide training IoU labels only, never inference features.

Sort distinct train scan IDs; assign fold = sorted_position modulo 4.
Train on 48 scenes, predict on 16 held-out scenes, separately for each fold.
Initialize w=0. Fixed 30 epochs, full-batch AdamW, lr=0.0001,
weight_decay=0.0005 (parent configuration), float32, seed=1999.
No scheduler, early stopping, best-epoch choice or hyperparameter selection.
Freeze all four final models; validation scores are their arithmetic mean.
All control arms receive the identical scene folds, initialization and budget.

| Arm | Assignment | Reliability features | Training |
|---|---|---|---|
| W2 | none | none | frozen |
| Y13 reference | Hungarian | none, g=1 | frozen |
| Y14 | Hungarian | original query features | train-only |
| State control | Hungarian | rotate feature rows by one query | same |
| Assignment control | rotate Hungarian assignment by one query | recompute corresponding features | same |

Train OOF improvements and pairwise loss are descriptive mechanism readouts,
not a new admission threshold. Evaluate val32 once, only after weights and
training evidence are saved. Preserve original final rule: positive scene
delta against W2 and state control, gated-vs-W2 two-sided sign p<0.05, zero
repeated-forward gated selection mismatches, no validation-derived parameters.
Assignment control is also reported, even if unfavorable. Never change the rule
after observing outcomes. No new full-model admission follows automatically.

For each scene, reuse one collated batch for two consecutive native forwards
per model without reseeding. Record native softmax selections, margin selections,
alignment assignment disagreement, feature/margin max differences and each
arm's selected-query disagreement. Synthetic repeat equality is not sufficient.
Set cuDNN deterministic=True, benchmark=False and TF32=False prospectively for
this capture; do not promise determinism for custom sparse CUDA operators.

## 5. GPU Allocation Plan

Only server-2 GPU0/1, at most one native capture per GPU. Acquire a nonblocking
GPU flock and check current utilization/memory inside it. Preserve foreign
processes. Run train and val captures serially on one GPU; val runs only after
train fitting, never in parallel with fitting to inspect validation early.
The offline gate optimizer runs on CPU and never updates W2/B4.

## 6. Implementation Checklist

- Standalone capture/fit diagnostic, checkpoint and config hashes.
- Weighted-mask equivalence, entropy equivalence, split isolation and gradient
  formula tests before native capture.
- Train64 OOF outputs, all controls, final weights, gradients and feature rank.
- One frozen val32 run, repeat diagnostics, accuracy@0.25/@0.5, query count,
  extraction latency, dual-model parameter count and GPU peak reserved memory.
- Full native training integration/resource Pareto comparison remain unpassed
  until separately implemented; cached CPU backward is not tools/train.py.

## 7. Risk Assessment

train64 labels are from a checkpoint's training set; OOF fits isolate the gate,
not checkpoint pretraining. val32 has already informed earlier research and
cannot certify unbiased final generalization or novelty. Full held-out
evaluation is required for later efficacy claims.
Noise floor is unquantified; do not invent a material-harm threshold.
State-control failure, effective-rank deficiency or repeat instability must
be reported. Dual-model cost is a risk, not automatically lightweight.

## 8. Smell Check

Signals 1/2 apply: six-parameter sigmoid gating. Y14 is a falsifiable diagnostic
with explicit reliability-destroying controls, not a paper-ready innovation or
a reason to spend a full 30-epoch backbone training budget. No coefficient scan.
Status: PREREGISTERED DIAGNOSTIC; full-model training not admitted.
