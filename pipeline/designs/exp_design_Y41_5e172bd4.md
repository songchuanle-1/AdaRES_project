# Experiment Design - AdaRES_project
Status VALIDATED_FOR_ADMISSION, not efficacy, human approval or launch.
Upstream motivation pipeline/reviews/motivation_review_Y41_5e172bd4.md; prospective prereg and repairs retained.

## 1. Motivation
AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515.
Test whether final confidence benefits from boundary probability mass omitted
by normalized local messages. Begin active1/target2/budget1 after Y37 terminal.

## 2. Current Best Results
| ID | Config | State | mIoU | Acc50 | spmIoU | Epochs |
|---|---|---|---|---|---|---|
| Y37 | adares_y37_pair_distribution_res | done_complete | .505061022921 | .561001262095 | .468443808443 | 30 |
| W2 | adares_w2_pointmass_competition_res | comparator | .510107995189 | .562999579302 | .473719922153 | 30 |
| B1 | adares_b1_standard_res | baseline | .502119335285 | .552797644089 | .466681173553 | 30 |
Full FIRST captures and frozen terminal review are primary, not selected repeats.

## 3. Analysis & Direction
W2 remains comparator, not a causal mechanism proof. Y37 versus W2 and B1
intervals include zero, no promotion. Y40 proposed language realization is
already present in Y35. Y41 tests a distinct probability-edge information
path; absence of proven benefit remains explicit. Phase2 is not approved.

## 4. Experiment Plan
| ID | Config | Hypothesis | Expected result | Priority | Depends |
|---|---|---|---|---|---|
| AdaRES-Y41 | configs_song/model_paper5/adares_y41_boundary_probability_res.yaml | final boundary mass informs confidence | trained result may confirm or falsify; no promised gain | P0 | six gates |
Fixed native256-dimensional query plus3 scalars;259->256->2, four tensors.
Average exact nearest-distance ties; all valid SPs; no radius/feature subset,
new loss, threshold, temperature, margin, epoch or validation search.
Released SPFormer, fixed30/seed1999/original128/oneargmax/native optimizer.
Posttraining candidate/W2/B1 EACH9508/141 twice, FIRSTprimary; scene bootstrap
10000seed1999; TRAINcuts1487/3325; subgroupCIupper<0 versus either is harm.
Quantify empirical repeat noise and trained paired costs before promotion.
Official unique/multiple labels unknown if unavailable; target.515 unchanged.

## 5. GPU Allocation Plan
| Server | GPU | Assignment | Config | Justification |
|---|---|---|---|---|
| server2 | 1 | preserve Y39 | adares_y39_point_distribution_res | existing original job |
| server2 | 0 | proposed Y41 | adares_y41_boundary_probability_res | only fresh idle verified slot |
No server6 or foreign project operations. Canonical FIFO dispatcher limited to0.

## 6. Implementation Checklist
- [x] Physical independent six-file package/config, no cross-variant imports.
- [x] CPU32, nativeTRAIN8/init4 wrong controls, finite four-tensor gradients.
- [x] Actual tools.train two optimizer steps and model/AdamW/PolyLR restoration.
- [x] Disabled physical W2 masks/indicators/sampling identical on four scans.
- [x] Fixed initialized resource screen; source/checkpoint preservation.
- [ ] Commit exactly seven tested files, verify Git/local/remote bytes.
- [ ] Canonical six-gate receipt and scoped GPU0 dispatch; no blind restart.
- [ ] Independent increasing native training steps before training status.

## 7. Risk Assessment
Spatial graph/scalar information may correlate with existing learned features
or overfit. Initial mean loss rises and wrong controls also learn. Direct
edge squared differences avoid cancellation but nearest ties can be dense;
actual max-scene stress is reported, not a universal bound.
Restored state exact; next continuation model/optimizer differs numerically.
Resource summary {"parent": {"parameters": 164629637, "stress_train_median": 1.0375901013612747, "stress_train_reserved": 13.3984375, "inference_median": 0.09733216743916273, "inference_p95": 0.23892887774854896, "inference_reserved": 1.7578125}, "candidate": {"parameters": 164696711, "stress_train_median": 1.0673374366015196, "stress_train_reserved": 13.77734375, "inference_median": 0.09829028137028217, "inference_p95": 0.246000992693007, "inference_reserved": 1.7578125}}. Added67074<1% and peak<24GiB
are initialized feasibility only, not trained Pareto superiority.
All goal constraints/bans above are preserved. No GT at inference, schema,
data/schedule/loss changes, background scheduler, multiGPU or other project.

## 8. Smell Check
Not signal1:67074parameters and a new nearest-edge probability flow graph.
Potential signal2, 'add a Linear', is rebutted by the new nontrivial graph
descriptor, not an affine modification of existing identical inputs.
Not signal3: the information path changes; not just scaling or normalization.
No signals4/5/6: no numeric/configuration scan, banned-route rename or value
search. Width256 is inherited; exactly3 statistics are mechanistically defined.
Causal claim is falsifiable usefulness of explicit edge probability mass,
not 'a better coefficient'. Novelty/contribution and trained benefit unknown.
