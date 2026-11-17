# Experiment Result Analysis - AdaRES_project
Experiment: AdaRES-Y28; scientific disposition: REJECTED.
Owned tick: b5948b857bed4529afe603b9ac76dc20; 2026-10-06, Asia/Shanghai.
Canonical experiment lifecycle at preflight remains running/deathwatch1/3.

## 1. Basic Info
- Config: configs_song/model_paper5/adares_y28_mask_readback_res.yaml.
- Server/GPU: server-2/GPU1 only; original PID19745 absent before/after capture.
- Canonical dispatch: 2026-10-05 20:23:04.828128+08:00.
- Actual finish time/duration: unknown; remote checkpoint directory timestamps
  are not substituted for verified local observations.
- At 2026-10-06 16:51:38.301597 to 16:51:41.654523+08:00 the original bounded
  log showed Val30/30 and last checkpoint metaepoch30 with model/optimizer/scheduler.
- Best checkpoint metaepoch21; meta best_miou0.5055264830589294.
  Metaepoch is reported literally, not used to infer a different log epoch.
- Source marker ce2dc75ee9e706da68c5768d9b3c855f33bda7fb; all seven candidate
  source/config hashes verified again during trained-cost measurement.
- No lifecycle override, counter forcing, restart, early stop or new admission.

## 2. Best Metrics
Frozen full9508expressions/141scenes, each model two native passes.
| Model/Pass | mIoU | Acc25 | Acc50 | spmIoU |
|---|---|---|---|---|
| Y28/1 | 0.505520490 | 0.607593605 | 0.556583929 | 0.469977198 |
| Y28/2 | 0.505523367 | 0.607593605 | 0.556583929 | 0.469982881 |
| W2/1 | 0.510107995 | 0.610748843 | 0.562999579 | 0.473719922 |
| W2/2 | 0.510006705 | 0.610643668 | 0.562894405 | 0.473630586 |
| B1/1 | 0.502223014 | 0.610433319 | 0.552902819 | 0.466783781 |
| B1/2 | 0.502121679 | 0.610328145 | 0.552797644 | 0.466686182 |
Queries128/selected1 throughout. Primary target0.515 not reached.

## 3. Convergence Summary
- Fixed30 training endpoint saved; best-checkpoint metadata21 retained.
- Final log tail reports rounded mIoU0.504/Acc500.554/Acc250.606/spmIoU0.470;
  these rounded values are not substituted for full checkpoint evaluation.
- Native no-improve at preflight was unavailable, not guessed from metadata.
- Mean repeat differences Y28/W2/B1: 0.000002877/0.000101291/0.000101336.
- Maximum expression differences: 0.026033640/0.933165193/0.973895609.
  These are distinct measurements, not universal repeat-noise bounds.

## 4. Comparison
Expression-weighted paired whole-scene bootstrap10000/seed1999/95%CI.
TRAIN size cuts1487/3325 fixed before validation.
| Reference/Group | Expressions/Scenes | Delta mIoU | 95%CI | Frozen Harm Upper<0 |
|---|---|---|---|---|
| W2/all | 9508/141 | -0.004587505 | [-0.012384596,0.003363911] | false |
| W2/small | 2816/124 | -0.017564302 | [-0.032367651,-0.003092544] | true |
| W2/medium | 3304/136 | 0.002492395 | [-0.010665121,0.016083549] | false |
| W2/large | 3388/135 | -0.000705961 | [-0.013308896,0.012637921] | false |
| B1/all | 9508/141 | 0.003297476 | [-0.004893723,0.011407563] | false |
| B1/small | 2816/124 | 0.004729425 | [-0.011790829,0.021211151] | false |
| B1/medium | 3304/136 | 0.015480391 | [0.000726688,0.029626897] | false |
| B1/large | 3388/135 | -0.009773576 | [-0.023550399,0.004383588] | false |
The positive medium-vs-B1 result is retained; it does not rescue small-vs-W2 harm.

Trained serial cost: first8 distinct TRAINscenes plus indices728/6780,
two warmups/three readings per scene, batch1.
| Model/Group | Median s | Descriptive p95 s | Peak Allocated GiB | Peak Reserved GiB |
|---|---|---|---|---|
| Y28/TRAIN8 | 0.076628631 | 0.118174517 | 0.961701870 | 1.234375 |
| W2/TRAIN8 | 0.079588681 | 0.116233934 | 0.960723400 | 1.234375 |
| Y28/large pair | 0.195548557 | 0.227908599 | 1.244667530 | 1.7578125 |
| W2/large pair | 0.188331055 | 0.222035129 | 1.243689060 | 1.7578125 |
Parameters164892293 vs164629637, addition262656. Mixed latency directions,
serial hardware confounding and descriptive p95 do not establish Pareto dominance.

## 5. Conclusion
- REJECT this Y28 realization under the preregistered small-group material-harm
  rule (upper95%CI<0 against W2). All-population comparisons remain inconclusive.
- No champion promotion, target/phase change, validation tuning, renamed retry
  or broader assertion that all mask-content-readback mechanisms are invalid.
- Mechanism necessity, cause of harm, novelty and training-seed uncertainty
  remain unknown. One seed/best checkpoint is not a causal ablation.
- Preserve all pretraining gates, mixed/adverse controls and old CPU raw faults.
- Full measurements ran 16:52:19 to17:05:24+08; final read-only audit
  17:05:36.711179 to17:05:38.361124+08 verified unchanged last/best hashes,
  no original/matching training process and GPU1 device1MiB/util0/no compute.
  This idle observation is neither a reservation nor training admission.
- Canonical monitor deathwatch1/3 is left untouched. Ordinary subsequent
  monitoring must finalize lifecycle; only a fresh capacity-gap tick may admit
  a newly tested independent candidate. GPU0 remains outside authorization.
- Helper syntax4/4 PASS, cost AST identical after candidate-identity substitution,
  corrected comparison object reused, and five arithmetic/identity tests PASS.
  These checks test measurement plumbing, not scientific efficacy.
- This analysis is pre-postflight; current tick completion awaits finish/receipt.

## 6. Crash Info
Not classified as crashed. No Traceback/OOM in the bounded24KB tail;
this is not a whole-log crash-absence proof. No crash repair or restart attempted.

## Evidence
The following files are in D:/project/AdaRES_project/pipeline/analyses and hash-bound in research_work.
- y28_terminal_b5948b857bed4529afe603b9ac76dc20.json: 3a1ad727893ac9d09d775cb626071f8cd6ee57e47dc2fcde2d871e760f875f64.
- y28_y28_full_eval_b5948b857bed4529afe603b9ac76dc20.json: 690de5d6608fa214222de9dac6ed78981cd966f445a6983cf6d8d6d0dd4cccfd.
- y28_w2_full_eval_b5948b857bed4529afe603b9ac76dc20.json: ee0649d7111474fed4d9aa3ead317b915bb7183efda39125cf3c1b6614adeb6f.
- y28_b1_full_eval_b5948b857bed4529afe603b9ac76dc20.json: 3a59c25b3566d971dea68708638a4a456010d551aaf3adb1550ad901b5dcdb06.
- y28_clustered_comparison_b5948b857bed4529afe603b9ac76dc20.json: 58a3a0944f3724b2a4cfab888862825dd1e27dc161b8de052045ee5ee9ece24e.
- y28_trained_cost_b5948b857bed4529afe603b9ac76dc20.json: e58958054b80bf4288ff7f259a2a8de9160b8922ec5a8819e7ca8e87b2e082e8.
- y28_posteval_close_b5948b857bed4529afe603b9ac76dc20.json: ca87f5017bd0da43d5cc8e4b79ec812d0c4e8cc13c73c9aa73db1dc6bda5612a.
