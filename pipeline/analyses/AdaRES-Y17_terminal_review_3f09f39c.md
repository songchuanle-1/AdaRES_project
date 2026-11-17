# Experiment Result Analysis - AdaRES-Y17

## 1. Basic Info
- Config: configs_song/model_paper5/adares_y17_target_equivalent_pairing_res.yaml.
- Server-2 GPU1, originalPID109632 absent at20:50:06+08 and21:19:43+08.
- Fixed30epoch training completion supported by native finalval363/363,
  last checkpoint epoch30/model/optimizer/scheduler. Original OS exit code unknown.
- Queue still deathwatch1/3/running; do not equate lagging queue with a live run.
- Launch/final native log timestamps use a remote clock; wall-clock duration unknown.

## 2. Best Metrics
Independent fullval9508expressions/141scans, frozen best rule/epoch29.

| Metric | Y17 | W2 | B1 |
|---|---|---|---|
| mIoU | .50415748 | .51010516 | .50222422 |
| Acc25 | .60265040 | .61074884 | .61043332 |
| Acc50 | .55563736 | .56299958 | .55300799 |
| spmIoU | .46965912 | .47371853 | .46678394 |
| Available / selected queries | 128 / 1 | 128 / 1 | 128 / 1 |
| Parameters | 164629637 | 164629637 | 164430473 |
| First-chunk warm median seconds | .326822 | .328381 | .276783 |
| Peak allocated / reserved GiB | 4.04235 / 6.48633 | 4.04235 / 6.48633 | 4.04068 / 6.48438 |

Latency is five warm measurements on one identical chunk, not whole-val latency;
memory is standalone inference, not training peak. No latency superiority claim.

## 3. Convergence Summary
- Best checkpoint epoch29 meta.5041521192; final epoch30 native mIoU.5014972687.
- Full best-checkpoint recapture slightly differs from native best metadata;
  preserve both, no replacement of historical champion or goal records.
- Y17 two repeat passes identicalIoU. W2 differs4expressions/2scans,
  overall.0000976621/maxexpression.9763062; B1overall.0000078881/max.0399251.
- No post-hoc extra epochs, coefficient/threshold/temperature tuning.

## 4. Comparison
- Paired vsW2:-.00594768,95%scene-clusteredCI[-.01431070,.00209170].
- Paired vsB1:+.00193326,95%scene-clusteredCI[-.00709071,.01089971].
- All small/medium/large paired95%intervals crosszero for both references.
- Cuts1487/3325 frozen ontraining; no subgroup retuning.
- Goal.515 not reached. No confident efficacy gain or material-harm finding.
- Root official dataset metadata has no unique/multiple labels; not fabricated
  from object names or within-expression annotations.

## 5. Conclusion
- Training complete; scientific efficacy inconclusive and target notmet.
- No championpromotion, no Paretosuperiority, no causalpaired-loss benefit claim.
- Exact regrouped pair-off fullcontrol remains required for causal attribution;
  not launched as an independently admitted filler.
- Revisit only on explicit next scientific question/evidence; preserve stableID.
- ContinueY15 originalGPU0 and let canonicalmonitor reconcileY17deathwatch.
- Hashed raw captures, comparison, failures and tools referenced on stableY17.

## 6. Evaluation Faults
- FirstY17pass failed JSON persistence on NumPyfloat32 checkpointmeta.
- B1initialimport failed inheritedB3Criterion registrycollision.
- Fixed only standaloneevaluation metadata serialization/processlocalregistry;
  both repaired reruns completed and failures retained.
- No originaltraining/checkpoint/model package modifications.
