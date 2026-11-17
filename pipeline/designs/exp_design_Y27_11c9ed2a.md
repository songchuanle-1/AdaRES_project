# Experiment Design: AdaRES-Y27
2026-10-04, owned tick11c9ed2af51a4ed9967596b52811d11d.

## 1. Motivation
Goal description: "AdaRES: improve a faithful standard-indicator 3D referring
expression segmentation baseline by learning query confidence from metric-aligned
evidence, targeting ScanRefer mIoU >= 0.515."
Upstream review: pipeline/reviews/motivation_review_Y27_11c9ed2a.md.
Snapshot18:57:31 active0/target1/budget1 after canonical Y25done_complete.
Role: test whether explicitly object-relative spatial evidence resolves
confidence selection ambiguity, not claim model has never seen geometry.

## 2. Current Best Results
| Experiment | Config | State | mIoU | Acc@50 | spmIoU | Epochs |
|---|---|---|---|---|---|---|
| W2 | adares_w2_pointmass_competition_res | done_complete | .510108 | .563000 historical full | .473719 historical full | 30 |
| B1 | standard indicator | done_complete | .502122 | .553008 historical full | .466784 historical full | 30 |
| Y25 | adares_y25_mask_feedback_res | done_complete, rejected | .494117 | .551115 | .456580 | 30 |
| Y27 | adares_y27_relative_peer_res | testing, untrained | Unknown | Unknown | Unknown | 0 |
W2/B1 auxiliary values explicitly from earlier frozen full comparisons, not
silently the same latest pass as displayed mIoU.

## 3. Analysis And Direction
Y25 whole-scene CI versus W2[-.023798195,-.008143030] rejects its realization.
Y26 mandatory smell review failed norm-only structure, not trained harm.
Y27 adds confidence-local centroid/query correspondence; it is not a renamed
context-removal/overlap-replacement/norm or augmentation variant.
TRAIN64/256 scene-mean oracle gap.154148389 with222relation/240other entity
annotations supports a real task-relevant information-path test, not efficacy.
Upstream query may already encode the relation, and wrong/zero geometry
controls must be retained even if they have more favorable initial loss.

## 4. Experiment Plan
| ID | Config Path | Hypothesis | Expected Mechanism | Priority | Depends On |
|---|---|---|---|---|---|
| AdaRES-Y27 | configs_song/model_paper5/adares_y27_relative_peer_res.yaml | Explicit ordered object-relative peers in confidence | Geometry correspondence changes confidence, native gradient active | P0 | No running dependency |
Physical W2 six-file copy, same config. Query masks define point-mass centroids.
3->256->256 position MLP plus bias-free q/k/v/output projections compute
diagonal-excluded all-peer attention with scale1/sqrt256. Result joins existing
overlap context, existing confidence head/output dimension unchanged.
328960 added parameters/8 tensors, no learnable scalar, temperature or scan.
Extra-path construction forks CPU RNG so shared initialization remains exact.
No change to qla, backbone, query pool, language encoder, matcher or losses.
No annotation/GT-derived inputs are introduced into inference.

## 5. GPU Allocation
| Server | GPU | Assignment | Config | Justification |
|---|---|---|---|---|
| server-2 | 1 | Exclusive current diagnostics; training only after admission | Y27 | Only authorized slot |
GPU0 reserved outside project; server-6 denied. Live guard before each GPU test
and production launch, no assumption that the preflight unknown GPU is idle.

## 6. Implementation Checklist
- [x] Physical independent package/config, unchanged other five source files.
- [x] No cross-variant imports; unique config basename/exact parent config.
- [x] TRAIN64 prediction-blind native diagnosis.
- [x] Native trained8/released-init4 controls/backward: all8 tensors active.
- [x] Native tools.train two optimizer steps/all8 updates.
- [x] Native model/AdamW/PolyLR save/resume exact before continuation.
- [x] Four-scene cached parent identity with peer disabled;128queries/1selected.
- [ ] Cached native centroid-to-mask gradient isolation and adverse-term review.
- [ ] Serial native stress and inference resources.
- [ ] Final six-gate plus smell review/scoped commit/receipt.
- [ ] Canonical whitelist launch and observed increasing native steps.

## 7. Risk Assessment
Native repeat outputnoise up to.000236511; cached repeat and whole-peer detach
forward0. Translation/joint permutation at most1.90735e-6, padding/singleton
exact0. Initial full losses change in both directions; means-.00200876trained/
-.00240481init are descriptive, not an admission gain threshold.
Resume lossdifference0 but next modelmax1.525879e-5/optimizer4.839239e-8;
no bitwise/whole-loader trajectory claim. No all-axis rotation invariance.
Pair geometry scales Q^2d and may increase resource cost; actual serial stress
required. Native batch1 prediction contract unchanged. Do not tune dimensions
or remove failed controls. Crash repair bounded/failed logs immutable.

## 8. Smell Check
Mandatory Phase1.6: PASS for structural eligibility, conditional on native gates.
Not "add a Linear/norm": new confidence input p(mask)*pointmass*3Dcoords defines
object centroid; learned ordered-pair position/features aggregate disjoint peers.
This changes information dependency, not only numerical distributions. Parent
decoder self-attention does not supply this explicit post-mask centroid edge.
Parameters>5K, new nonlinear spatial interaction; dimension-derived scale and
parent widths, no numeric optimization. Causal prediction has matched controls.
Not novelty proof or trained superiority; Y26's norm-only failure remains.

## 9. Frozen Post-Training Evaluation
Use preregistration Y27_relative_peer_prereg_11c9ed2a.md unchanged: released
SPFormer backbone, seed1999/fixed30, full9508expressions/141scene candidate/W2/B1
two passes, identity(scan_id,object_ids,ann_id), all metrics and measured costs.
Whole-scene bootstrap10000seed1999/95% CI; empirical repeat noise reported.
TRAIN target-size cuts1487/3325 unchanged; material subgroup harm if paired CI
upper bound<0 against W2, all/small/medium/large separately. Inadequate scenes/
interval crossing0 means inconclusive; no validation thresholds/features tuning.
Trained lightweight Pareto not proven by engineering feasibility; require
trained gain without demonstrated latency/peak-memory regression.
Status: TESTING. Admission and production training not yet executed at writing.
