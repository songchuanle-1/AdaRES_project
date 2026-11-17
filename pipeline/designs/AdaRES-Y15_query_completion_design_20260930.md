# Y15 Design: Language-Conditioned Query Completion

Status: DESIGN, NOT ADMITTED. 2026-09-30 Asia/Shanghai.
Stable ID AdaRES-Y15; do not rename the already-tested candidate hypothesis.

## 1. Motivation

"AdaRES: improve a faithful standard-indicator 3D referring expression
segmentation baseline by learning query confidence from metric-aligned
evidence, targeting ScanRefer mIoU >= 0.515."
The train64 candidate-pool diagnosis shows a small, repeat-stable,
target-specific missing-mask signal. Projection diagnostics suggest part of
the signal is expressible in W2 features but absent from current queries.
Teacher necessity is not established: GT-only projection is slightly better
on the selected subgroup. This experiment tests candidate completion, not
another confidence reweighting or generic extra BCE.

## 2. Current Best Results

W2 full ScanRefer mIoU 0.5100, B1 0.5020, goal 0.5150.
Y14 frozen distinct-scan val32 rejected: -0.0027005 versus W2 and no
state-control separation. Y15 oracle/projection numbers are train diagnoses,
not model validation metrics.

## 3. Analysis And Derivation

Student mask basis F has N rows and dimension d=256, with point masses n_s.
Teacher projection q* solves weighted least squares in this basis; it is
training-only and is never available at inference.
Train a supplemental query from inputs already available to the student:

u = W_l mean(valid_language_tokens) + a
alpha_s = softmax_s((W_q u)^T W_k f_s / sqrt(d))
q_new = LN(u + W_o sum_s alpha_s W_v f_s)
x_new = F q_new.

Use one learned slot a and one cross-attention block, d=256, eight heads,
dropout=0 (parent settings). No learned reliability scalar.
Prediction appends one candidate and applies the existing shared indicator
and competition to all candidates. K -> K+1 is explicit, never silently
replace a query or compare to a smaller-pool control.
Exact decoder insertion point, mask-feature versus context-feature dimensions
and all auxiliary-loss indexing require native implementation audit first.

Teacher target discovery is restricted to training examples with
b4_oracle - w2_oracle > 0.05 (existing diagnostic material definition).
No validation-based filtering. Both models' oracle selection is detached.
Project teacher raw mask logits into the current detached student basis using
the frozen weighted least-squares solver; q* is also detached.
The new query's supervision is normalized point-weighted BCE to teacher mask
and the same native GT/matching losses used by all control arms.
Do not assign an arbitrary teacher-loss coefficient: before integration,
inspect native mask-loss reduction/weight and reuse that same branch budget.
If this cannot be matched without introducing a free coefficient, do not admit.

For BCE mask target t, dL/dq_new =
F^T D (sigmoid(Fq_new)-t) / sum_s n_s.
Thus missing candidate supervision reaches query generation without requiring
a second model at inference. This derivative proves connectivity, not benefit.
Regression to q* is diagnostic only; do not add a redundant norm-dependent
query MSE unless independently justified and preregistered.

## 4. Experiment Plan

| Arm | Query count | New query source | Training target |
|---|---|---|---|
| W2 reference | K | existing native | native GT |
| Y15 completion | K+1 | language-conditioned cross-attention | training-only teacher missing masks + matched native GT |
| GT-only completion | K+1 | identical block | GT-only matched mask branch |
| Language-shuffled completion | K+1 | identical block, cyclic different-object language | identical teacher targets |
| Capacity control | K+1 | same block, learned slot with language path disabled | identical teacher targets |

Same new-parameter count, optimizer, sampling, 30 epochs and data protocol for
all K+1 arms. Don't compare an extra-query effect only with K.
First execute train-only mechanism tests with shared frozen weights and native
loss: intended language interventions change q_new, all new tensors receive
finite nonzero gradients, positive-mask coverage and duplicate-query overlap
are reported versus each control. Include adverse examples, not just successes.
No tiny-cohort task-gain or significance requirement for prospective native
pretraining gates, but true no-op/reversed controls must be investigated.
Y14's old frozen acceptance and rejection are not changed by this policy.

Full post-training primary metric: official ScanRefer mIoU, full validation
split and official evaluator, W2/B1 comparators; report Acc@0.25/@0.5, spmIoU,
active candidates, same-hardware warmed single-pass latency and peak memory.
Cluster uncertainty by real scan_id, not dataset chunk. Predeclare object-size
and unique/multiple-reference subgroups from training metadata before validation.
Noise floor and material-harm boundary are currently unquantified: estimate
paired baseline evaluation repeats before freezing the final harm rule.
No thresholds, temperatures, epoch count or loss weights tuned on validation.
Novelty and the lightweight Pareto budget need separate substantiation; no
novelty or training-admission claim is made by this draft.

## 5. GPU Allocation Plan

server-2 GPUs 0/1 only, one task per GPU. First native integration on one live
checked, leased slot. Full dispatch uses the gated monitor after six receipts;
do not create a scheduler or manual background training bypass.

## 6. Implementation Checklist

- Independent package/config, no cross-variant model imports.
- Native initial query insertion and shared-indicator prediction equivalence.
- Training-only frozen teacher, no label or teacher access at inference.
- Matched GT-only/capacity/language control reduction and target bookkeeping.
- Every new tensor native backward, two real batches, memory/time comparison.
- Train-only teacher target manifests and additive source synchronization.
- Full evaluation/noise/harm plan and six-gate receipt before any launch.

## 7. Risks

Teacher candidate diagnosis is oracle-conditioned and cannot become an
inference selector. Projection gains are mixed (17 positive/6 negative).
GT-only may explain the entire benefit. Larger candidate count alone may
improve oracle coverage or confuse confidence. Teacher training doubles
backbone memory unless sequential/cached training targets are validated.
Changing student features can stale offline projections; compute targets using
the current detached basis, or explicitly test staleness before caching.
Two-scene native gradient probes are not population evidence.

## 8. Smell Check

This changes candidate generation and the information path from language to
mask queries, not numerical confidence mixing. Matched extra-capacity and
GT-only arms must separate these causal claims. Distillation itself is familiar;
the proposed combination's novelty remains unverified.
