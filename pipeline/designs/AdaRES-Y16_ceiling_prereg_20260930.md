# Y16 Train-Only Superpoint Ceiling Preregistration

Frozen before the new metadata outcomes, 2026-09-30 19:55 Asia/Shanghai.
Stable ID AdaRES-Y16, hypothesis superpoint-internal-point-residual.

## 1. Motivation

Goal: improve the faithful standard-indicator baseline using metric-aligned
evidence, ScanRefer mIoU >=0.515. Y15 adds a missing candidate; Y16 instead
asks whether constant-within-superpoint masks destroy useful point information.
This is NOT a boundary-based query selector, FPS budget change, score mixture,
or a reopening of H2/V1. No training admission is claimed.

## 2. Current Best Results

W2 best .5100, B1 .5020, fixed30 epochs. Y15 has verified native single-GPU
training on server-2 GPU0; no validation result yet.

## 3. Analysis And Direction

For target occupancy t_s and total points n_s in superpoint s, the best
binary superpoint-constant IoU is max_A sum_A t_s /
(T + sum_A(n_s-t_s)), T=sum_s t_s. An optimum includes a superpoint if
t_s > I*(n_s-t_s). Therefore sorting descending t_s/n_s and taking the
best prefix computes the exact ceiling; pure-background groups are excluded.
Pointwise oracle IoU is1 for any nonempty target. The difference measures
representation loss, not learnability or trained improvement.

Prospective residual structure: retain backbone point features h_i before
pooling and add q^T W(h_i-mean_{j in s(i)} h_j) to coarse mask logits.
The residual has zero superpoint mean exactly. A broadcast control has zero
residual; a within-superpoint permutation preserves feature counts/mean but
breaks alignment with point labels. This preserves query count and avoids
query-confidence reranking. Native reachability/gradients/resources remain
required; point-level cost and objective integration are not yet validated.

## 4. Experiment Plan

CPU raw-data pass on ALL actual training scans, no validation/model output.
Use native ScanRefer_train_new.json target object IDs and raw _refer.pth
instance labels/superpoints. Compute exact ceiling, majority-mask IoU,
mixed-superpoint target fraction and target count per distinct object tuple.
Record every missing/empty target and fail rather than silently filter it.
Aggregate both unique-target and expression-weighted summaries, by actual scan.
Unit-test the ceiling against exhaustive subsets for small count vectors.

In the same pass freeze Y15 size tertiles from expression-weighted raw train
target-point counts with linear quantiles at1/3 and2/3. Ties remain on the
lower side using <=cut1, <=cut2; no validation-selected cutpoints.
This is the previously registered raw train size definition, not an
augmented minibatch count or validation size distribution.

A nonzero ceiling deficit licenses further diagnosis ONLY, not admission:
next capture W2 prepool features on first eight distinct train scans and
test centered point-feature/target gradients against broadcast and shuffled
within-superpoint controls. If occupancy is pure, reject Y16 with evidence.
Otherwise assess learnability before building an independent physical variant.

## 5. GPU Allocation Plan

CPU metadata uses no GPU. Later bounded native diagnostic only on live-idle
server-2 GPU1 with exclusive flock; preserve Y15 GPU0. No server-6.

## 6. Implementation Checklist

Record annotation/point-file hashes; exact oracle brute-force unit checks;
frozen size cuts and missing-target audit; persist stable Y16 evidence.
No objective/temperature/threshold/epoch tuning and no new full training yet.

## 7. Risk Assessment

Small oracle loss may not matter near .5100. Mixed labels alone do not prove
features support refinement. A pointwise oracle is label-only and cannot be
used at inference. All-mask point logits may exceed24GiB; an eventual
bounded path must be justified without deleting native losses arbitrarily.
Published superpoint-text matching (3D-STMN arXiv2308.16632) and superpoint
feature refinement (Relation3D CVPR2025) already exist. Neither abstract
establishes the novelty of centered point-output residuals. Novelty unverified.

## 8. Smell Check

The proposed path changes spatial output resolution and uses information
discarded by pooling, not a numerical coefficient. Zero-mean is an algebraic
constraint, not a searched strength. No validation ranking or neighbor scan.
