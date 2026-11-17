# Experiment Result Analysis - AdaRES_project
Experiment AdaRES-Y35; canonical state done_complete; scientific state complete_inconclusive.
Observed 2026-10-10T12:12:19.847291+08:00; no postflight claim.

## 1. Basic Info
Original PID108331/server2GPU0/configadares_y35_language_evidence_res.
Begin confirmed canonicaldone_complete2026-10-10 11:48:56 local after3absences.
Bounded original process/checkpoint capture 2026-10-10T11:54:37.036644+08:00.
Start time/duration not reconstructed from directory names; unknown here.
Independent source b49b744dc2943a38ab4c3aeb788acc6a2cb90011 preserved.

## 2. Best Metrics
Frozen best checkpoint epoch17; metadata mIoU0.504700243473053.
All9508expressions/141scenes, first pass PRIMARY, never choose the better repeat.
| Method | mIoU | Acc25 | Acc50 | spmIoU | Repeated mIoU |
|---|---|---|---|---|---|
| Y35 | 0.504801213799 | 0.605279764409 | 0.559739167017 | 0.468181190538 | 0.504802890898 |
| W2 | 0.510105160439 | 0.610748843080 | 0.562999579302 | 0.473718529117 | 0.510007498366 |
| B1 | 0.502216334202 | 0.610433319310 | 0.552902818679 | 0.466773520613 | 0.502224107738 |
BestSHA6b30454c2b202538377056b0de960f3a7772cbe1d9996b6f86f6d7a3702b4527; lastSHAdaf9cd828bef52c7fb09ad7e36a41f6463eb66beb4850669fa047513901bc412, unchanged during read.

## 3. Convergence Summary
Last30; best17 in checkpoint metadata. Logrounded monitoring and ledgerrounded
values are not the frozen primary. Two repetitions quantify empirical noise
only; Y35 expression maxrepeat0.9763062000274658.
No whole-loader determinism or seed variance claim.

## 4. Comparison
Whole-scene paired10000 bootstrap seed1999; TRAINpointcuts1487/3325 unchanged.
| Reference | Group | Paired delta | Scene CI95 | Expressions/Scenes | Harm CIupper<0 |
|---|---|---|---|---|---|
| versus_w2 | all | -0.005303946640 | [-0.013835981878, 0.003056499187] | 9508/141 | False |
| versus_w2 | small | -0.017444125742 | [-0.035410643022, 0.000906456336] | 2816/124 | False |
| versus_w2 | medium | -0.005824597415 | [-0.020099308044, 0.009021013780] | 3304/136 | False |
| versus_w2 | large | 0.005294333914 | [-0.008900404502, 0.019454100073] | 3388/135 | False |
| versus_b1 | all | 0.002584879597 | [-0.006354111492, 0.011520956168] | 9508/141 | False |
| versus_b1 | small | 0.004849601243 | [-0.013543820410, 0.023755035670] | 2816/124 | False |
| versus_b1 | medium | 0.007166941315 | [-0.006872666838, 0.021659294173] | 3304/136 | False |
| versus_b1 | large | -0.003765943328 | [-0.017192762081, 0.009688043672] | 3388/135 | False |
CIcross0 inconclusive, never equivalence/noharm.
Trained paired resources:
```json
{
  "candidate": {
    "train8": {
      "median_seconds": 0.08331053797155619,
      "descriptive_p95_seconds": 0.13112627677619457,
      "peak_allocated_gib": 0.9619474411010742,
      "peak_reserved_gib": 1.236328125
    },
    "largest_pair": {
      "median_seconds": 0.22590549662709236,
      "descriptive_p95_seconds": 0.25752663565799594,
      "peak_allocated_gib": 1.244913101196289,
      "peak_reserved_gib": 1.759765625
    },
    "native_train_stress": {
      "median_seconds": 1.1192437149584293,
      "peak_allocated_gib": 12.854626655578613,
      "peak_reserved_gib": 13.5546875
    }
  },
  "parent": {
    "train8": {
      "median_seconds": 0.08211260940879583,
      "descriptive_p95_seconds": 0.1311020229011774,
      "peak_allocated_gib": 0.9607234001159668,
      "peak_reserved_gib": 1.234375
    },
    "largest_pair": {
      "median_seconds": 0.21829650457948446,
      "descriptive_p95_seconds": 0.2466625915840268,
      "peak_allocated_gib": 1.2436890602111816,
      "peak_reserved_gib": 1.7578125
    },
    "native_train_stress": {
      "median_seconds": 1.1357731837779284,
      "peak_allocated_gib": 12.850954532623291,
      "peak_reserved_gib": 13.400390625
    }
  }
}
```
Params {"candidate": 164958087, "parent": 164629637}.
Native128/oneargmax. TRAIN8plus frozen largestpair728/6780, serial2warmups/3reads,
separate memory-only train stress. Descriptive timings not reliable p95/
significance or trained lightweight Pareto dominance. Production checkpoints
unchanged by stress; post-audit confirms source/job/checkpoint preservation.

## 5. Conclusion
Target unmet; subgroup intervals do not establish material harm. Intervals crossing zero remain inconclusive, not proven equivalent or harmless.
Target.515 unmet; keep W2 champion. Close SAME realization with actual evidence,
no renamed rerun/restart/epoch selection or blanket ban on all word/geometry
routes. The initialization controls did not establish semantic necessity;
one trained seed cannot establish the population mechanism.
Y36 original PID100018/GPU1 preserved and its trained outcome remains unknown.

## 6. Crash Info
No new crash intervention. Bounded terminal tail noTraceback/noOOM is not
whole-log crash absence. First local terminal adapter failed because canonical
completed Y35 had been pruned from active queue; corrected to frozen original
identity without remote restart. No gate/protocol/goal change.

## Evidence
- pipeline/analyses/y35_terminal_e636ce91a12e45488093b07fa2fc092b.json SHA256 02c6c1d4802c7b903597078b1caa574fbd8180adb3e3797d58a034c4382d6b40
- pipeline/analyses/y35_y35_full_eval_e636ce91a12e45488093b07fa2fc092b.json SHA256 5d0074e76f0508d9d39ba1219118756cb16bd1974c0623e703d397229b29c994
- pipeline/analyses/y35_w2_full_eval_e636ce91a12e45488093b07fa2fc092b.json SHA256 eec9dd279a74a41e9c198752e43c7613b962bb4b0a3b555a02eb4b23f82fb650
- pipeline/analyses/y35_b1_full_eval_e636ce91a12e45488093b07fa2fc092b.json SHA256 4deacbd074a8628a25e1c7d4e9847023c48ca56a85ee937d982fd0d78653165a
- pipeline/analyses/y35_clustered_comparison_e636ce91a12e45488093b07fa2fc092b.json SHA256 184e1380544052fa1f37c954076cf12558f2d446eaaabe3469fc368b86c96014
- pipeline/analyses/y35_trained_cost_e636ce91a12e45488093b07fa2fc092b.json SHA256 4783cf20f493b9226f185fec23ee6f3d0065948c5beb545614a8c7c7a56934f7
- pipeline/analyses/y35_postaudit_e636ce91a12e45488093b07fa2fc092b.json SHA256 f6a80dbe06a075ee36ebc281d6b586e957a2464361f156db7af7ae4290b6ac9c
