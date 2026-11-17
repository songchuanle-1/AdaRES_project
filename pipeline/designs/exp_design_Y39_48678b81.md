# Experiment Design - AdaRES_project
Status: VALIDATION; no admission or launch is asserted here.
Tick48678b818b134bea9d9c3c3e1f3d3305, original deadline20:48:21+08 unchanged.

## 1. Motivation
AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515.
Upstream: pipeline/reviews/motivation_review_Y39_48678b81.md.
Stable diagnosis/realization frozen before controls; this document consolidates
those records, not a newly backdated freeze. Capacity2/active1/budget1 after
canonical Y36 terminal closure. Preserve original Y37/GPU0.

## 2. Current Best Results
| Experiment ID | Config | State | Best mIoU | Acc@50 | spmIOU | Epochs |
|---|---|---|---|---|---|---|
| W2 | adares_w2_pointmass_competition_res.yaml | retained frozen comparator | .510009849923 | not loaded | not loaded | 30 |
| B1 | not loaded here | frozen baseline | .502112417447 | not loaded | not loaded | 30 |
| Y36 | adares_y36_local_adjacency_res.yaml | done_complete/inconclusive | .510716624292 | not loaded | not loaded | 30 |
| Y37 | adares_y37_pair_distribution_res.yaml | preserved training | snapshot rounded .4926 | unknown | unknown | monitor10/30 |
These full results are reused from prior frozen reviews, not rerun today.

## 3. Analysis And Direction
### 3.1 Observed Positive Results
W2 improves on B1 in these fixed trained references. Y36's B1 interval is
positive but its W2 interval crosses0; not a necessity/superiority result.
### 3.2 Negative Or Unresolved Results
Actual Y30 history path already covers Y38. Native partition ceiling nearly1
disconfirms quantization as a major bottleneck. Sparse quantization is not an
arbitrary pretraining magnitude veto, but its limitation remains explicit.
### 3.3 Not Yet Verified
Whether within-SP feature heterogeneity generalizes as confidence evidence;
learned upstream means may already encode it. No Phase2 ablations authorized.
### 3.4 Selected Direction
Y39's explicit native point-feature distribution is separated on equal-mean
fixtures and matched native wrong-input controls; only falsifiable training
can decide benefit. It is not a renamed global geometry/history route.

## 4. Experiment Plan
| # | ID | Config Path | Purpose / Hypothesis | Expected Result | Priority | Depends On |
|---|---|---|---|---|---|---|
| 1 | AdaRES-Y39 | configs_song/model_paper5/adares_y39_point_distribution_res.yaml | final point-distribution confidence | reproducible mechanism response, then frozen trained quality/harm/cost; no promised gain | P0 | six gates |
Population centered variance across ALL32 raw backbone point features in each
native superpoint; log1p then shared32->256->256/GELU encoder. Detached final
soft masks and native point masses give FG/BG means. Final query, encoded
FG-BG, encoded(FG+BG)/2 ->768->256->2 indicator residual. Eight tensors,
271618 added parameters. No auxiliary head, mask output, loss weight,
query count, point input subset, inference argmax or native schedule changes.
Production from released SPFormer, seed1999, fixed30. Trained W2 weights
are diagnosis/comparison only, never production initialization.
Post-training frozen full validation9508/141, Y39/W2/B1 each two passes,
FIRST primary. Whole-scene10000 bootstrapseed1999 against both comparators.
TRAIN point-size cuts1487/3325; subgroup material-harm rule CIupper<0 against
either comparator. CIcross0 inconclusive, not equivalence/noharm. Measure
repeat noise and trained paired costs, no invented effect/noise threshold.
Target.515 and frozen harm/lightweight Pareto review required for promotion.

## 5. GPU Allocation Plan
| Server | GPU | Assignment | Config | Justification |
|---|---|---|---|---|
| server-2 | 0 | preserve original Y37 | adares_y37_pair_distribution_res.yaml | no restart, migration or source overwrite |
| server-2 | 1 | exclusive Y39 diagnostics; possible gated FIFO admission | adares_y39_point_distribution_res.yaml | fresh live idle check, no future-idle assumption |
Single GPU, one job per GPU; no server6 or other-project/scheduler operations.

## 6. Implementation Checklist
Own six-file physical package and parent-exact YAML created. Scoped additive
upload with all Y37 source hashes unchanged. CPU32, nativeTRAIN8/init4,
native tools.train2steps/all8updated/checkpoint restoration/physicalW2
disabled identity and fixed resource measurements completed.
Adverse native-loss investigation retained. Commit ONLY own sevenfiles,
git/local/remote exact-byte alignment and admission/dispatch/increasing
native launch steps still pending here. No unrelated dirty-file changes.

## 7. Risk Assessment
All32 variance may be an indirect appearance proxy, not uniquely sufficient
or necessary confidence evidence. Near-perfect partition ceilings are a
negative result and cannot establish expected trained gain.
Random new-head calibration initially worsens trained W2 loss. Correct and
wrong controls can all learn; mean correction is not causal generalization.
Point variance adds native backbone gradients although masks are detached
only in the new readout. Original losses and paths remain. CUDA scatter
accumulation and sparse continuation have numerical noise, not bitwise whole
trajectory reproducibility. Actual time/memory regressions retained.
Feasibility addedparams<1%/peakreserved<24GiB; trained Pareto still unknown.
No tuning, new data, GT inference, threshold/temperature/neighborhood search.

## 8. Smell Check
| Signal | Judgment |
|---|---|
| <5K/no new graph | no:271618 parameters plus raw-point centered aggregation |
| norm/linear/gate only | projections present but new per-SP distribution path before pooling |
| numeric-only variant | no: physical native input path and masked distribution aggregation |
| sweep | none; inherit native32/256, no dimension/feature/loss search |
| renamed banned route | not threshold/margin/radius/history/global covariance; explicit within-SP feature spread |
| equivalence to old head | W2 final state held equal in CPU fixtures while actual new readout changes |
These checks support executability only, not novelty or efficacy.
