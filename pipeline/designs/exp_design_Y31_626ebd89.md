# Experiment Design - AdaRES-Y31

## 1. Motivation

"AdaRES: improve a faithful standard-indicator 3D referring expression
segmentation baseline by learning query confidence from metric-aligned
evidence, targeting ScanRefer mIoU >= 0.515."
Canonical Y30 terminal frees capacity1; target unmet and Phase2 unapproved.
Test explicit mask-supported shape evidence, not a replacement selector.
Upstream review: pipeline/reviews/motivation_review_Y31_626ebd89.md.

## 2. Current Best Results

| ID | Config | State | Best mIoU | Acc50 | spmIoU | Epochs |
|---|---|---|---|---|---|---|
| W2 | adares_w2_pointmass_competition_res | done_complete | Historical0.5100 | Not reread | Not reread | 30/30 |
| B1 | Existing baseline | done_complete | Historical0.5020 | Not reread | Not reread | 30/30 |
| Y30 | adares_y30_query_trajectory_res | done_complete; scientific rejected | Full0.504229394/0.504314270 | Prior0.557530501 | Prior0.470150293 | 30/30 |

## 3. Analysis And Direction

Covariance differs from visual FG/BG content, peer-centroid relations and
ordered state histories. Preserve all prior negative/inconclusive results.
The actual W2 head has no direct spatial input; geometry upstream does exist.
No causal necessity, efficacy or novelty conclusion follows from this fact.

## 4. Experiment Plan

| ID | Config | Hypothesis | Expected Mechanism | Priority | Depends On |
|---|---|---|---|---|---|
| AdaRES-Y31 | configs_song/model_paper5/adares_y31_mask_shape_res.yaml | Mask-supported central spatial moments can inform confidence | Actual shape changes affect final confidence with native gradients | P0 | Six real gates |

Physical W2 copy. Let w_qs=sigmoid(mask_qs)*native_point_count_s, normalized
over valid superpoints. Compute central covariance of superpoint centroids,
divide by point-count-weighted scene squared radius, retain all six symmetric
entries (xx,yy,zz,xy,xz,yz). Accumulate in FP64, cast to original mask dtype.
This omits within-superpoint variance. No eigenvalue selection or feature scan.
Existing d_model256 defines shape6->256 GELU, concatenate with query256,
fusion512->256 GELU->2, final indicator residual only. Added133634 parameters,
six tensors. CPU fork_rng preserves shared initialization. No new loss, masks,
backbone, sampler, thresholds, auxiliary predictions or inference selector.

Freeze native diagnosis: first64 distinct TRAIN scenes, up to4 descriptions,
seed1999, unfiltered; oracle GT used only for diagnostic readout, never input.
Controls: disabled parent/repeat, joint query permutation, point permutation,
translation, shuffled-shape, detached masks with identical forward. Predeclared
mechanical tolerance1e-5 for cached invariants, not a task effect-size gate.
Retain every initial trained/init adverse loss and wrong-shape result; inspect
unexplained adverse mechanisms before admission. All six tensors must have
finite nonzero native-loss gradients and update in actual tools.train steps.

Post-training freeze BEFORE native outcomes/launch: fixed30; full validation
9508 expressions/141 scenes; Y31 and fresh W2/B1 each independently twice,
standard128 active queries/one selected. Primary mIoU; Acc25/Acc50/spmIoU.
Whole-scene paired bootstrap10000 seed1999; TRAIN-derived raw-point size cuts
1487/3325 unchanged. Meaningful size-subgroup material harm: upper95CI of
delta_vs_W2 below0 rejects same realization. Intervals crossing0 inconclusive,
not equivalence. Repeat differences empirical only, no invented noise constant.
Official unique/multiple split unavailable unless verified. One training seed
cannot establish training-seed uncertainty. Trained lightweight Pareto requires
parameter/latency/peak-memory reporting against W2, not just 24GB feasibility.
Serial native batch1 TRAIN8 and small/large fixed pair728/6780, two warmups,
three readings; separate native batch2 warmed AdamW memory stress. Small timing
sample descriptive; regressions must be retained. Do not adjust this plan on val.

## 5. GPU Allocation Plan

| Server | GPU | Assignment | Config | Justification |
|---|---|---|---|---|
| server-2 | 1 | Diagnostics, only after live idle check/exclusive lease | Y31 | Sole whitelist slot |
GPU0 remains reserved outside project. No other project/server/task/scheduler.
Snapshot running0 is not proof of live GPU idleness.

## 6. Implementation Checklist

- [x] Stable diagnosis candidate and prereg before broad review
- [x] Physical six-file copy/config, final-only residual, source byte bundle
- [x] Actual-class CPU32 discrimination/gradient/permutation/translation
- [ ] Prediction-blind native TRAIN diagnosis
- [ ] Native matched trained/init controls and backward/resume
- [ ] Resources, physical independence and goal constraints
- [ ] Tested candidate-only commit/push, admission receipt and gated dispatch
- [ ] Verified increasing native production steps

## 7. Risk Assessment

Geometry upstream may already encode the statistic. Shape can correlate with
size but does not directly encode IoU or semantic correctness. Superpoint
approximation and axis orientation are explicit limits. FP64 moment overhead
needs measurement. Training efficacy/necessity/novelty/Pareto unknown.
No original Y30 failure or adverse control is overwritten or renamed.

## 8. Smell Check

Fixed six entries follow the complete symmetric3D matrix, d_model follows
parent256, scale is scene-derived and clamps are numerical safeguards, not
scanned values. New mask-to-spatial-moment-to-query-confidence graph aggregates
geometry not previously supplied directly to this head. It is not only
elementwise/norm/scaling, not below5K, and not a changed value search.
Full trained benefit still must be falsified by the frozen comparison.

Status: DRAFT, native gates pending.
