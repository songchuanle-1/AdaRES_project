# Experiment Result Analysis - AdaRES_project

Experiment AdaRES-Y33; canonical training done_complete, scientific realization
REJECTED for the prospectively frozen small-target harm rule.
Tick8fd5e469dee241da8f21da7c870c0d3f; measurements2026-10-09 local17:51-18:10.

## 1. Basic Info

- ID AdaRES-Y33; configs_song/model_paper5/adares_y33_token_attention_res.yaml.
- Original server2/GPU1/PID64797; codeeaf330166aed361ffc2532a52e352b70cba4e2fd.
- Launch2026-10-08 local18:25:48, original unique log20261008_182513 preserved.
- Actual training end timestamp/duration unknown; canonical confirmation
  2026-10-09 local17:47:55 after three absent-process monitors.
- CPU checkpoint capture17:51:50-17:51:57: last epoch30; best epoch20;
  exact metadata best.5051367878913879. Original/matching train PIDs absent.
- last SHAec6c4dd70133b5b55866d50b62a8e95a3d4fdd60b4664eb1d1829a8f637dce37;
  best SHA4a9fe581bc38c2de874f8110024b856abf70d8212034de7d8a418342aa5ff87c.
- Seven deployed source hashes unchanged under additive globalmarkerb49.
  Postaudit18:10:01-18:10:03 verified checkpoint hashes unchanged; GPU1
  memory1MiB/util0/no compute processes, not a future reservation.

## 2. Best Metrics

Unchanged best checkpoint20, native128/one selected argmax; first full
evaluation is primary, never select the better repeat.

| Metric | Y33 Primary | Y33 Repeat | W2 Primary | B1 Primary |
|---|---|---|---|---|
| mIoU | .505137821478 | .505147630805 | .510007498366 | .502111561750 |
| Acc25 | .602124526714 | .602124526714 | .610643668490 | .610328144720 |
| Acc50 | .556899453092 | .556899453092 | .562894404712 | .552797644089 |
| spmIoU | .470365716844 | .470395766725 | .473626083491 | .466667544708 |

All three configurations each evaluated twice on9508expressions/141scenes.
Y33/W2/B1 full captures SHA5c02a405/26a8453b/64671b6e; complete/full IDs and
high-precision values reside in the corresponding token-specific JSONs.

## 3. Convergence Summary

Fixed30 completed; best checkpoint epoch20, no training/epoch tuning.
The bounded final24KB log contains Val30/30 complete and last save, rounded
final mIoU.502/Acc50.554/Acc25.600/spmIoU.467. These rounded final-epoch
metrics are not the best checkpoint's full evaluation. Earlier rounded monitor
best.5050/.5051 cannot replace high-precision checkpoint/frozen metrics.
Y33 mean absolute repeat mIoU difference9.809327e-6; maximum expression
repeat IoU difference1.0. W2 mean9.814527e-5/max.933165; B1 mean1.097261e-5/
max.039925. Repeats reveal expression-level instability; global determinism,
equivalence, seed variance and a universal noise bound are not established.

## 4. Comparison

Whole-scene bootstrap10000/seed1999, expression-weighted paired delta,
original TRAINrawpoint cuts1487/3325. Corrected comparison SHA72b54acf.

| Group/Comparator | Paired mIoU Delta | Scene CI95 | Interpretation |
|---|---|---|---|
| All/W2 | -.004869676889 | [-.012949144799,.003261757643] | Inconclusive, crosses0 |
| All/B1 | +.003026259728 | [-.005549301464,.011331518625] | Inconclusive, crosses0 |
| Small/W2 | -.024630810599 | [-.041461090329,-.008034531270] | Frozen material harm |
| Small/B1 | -.002337083615 | [-.018328259057,.013173696190] | Inconclusive |

Small group2816expressions/124scenes, observed group repeat difference0;
harm rule CIupper<0 versus either comparator is met versus W2.
Medium/large intervals against both comparators cross0; do not call them
equivalent or harm-free. No subgroup/threshold tuning or better-repeat choice.

Trained paired cost SHA629ad512:164958087parameters vsW2164629637.
NativeTRAIN8 infer median.081474435/.078295105s, descriptivep95
.126438379/.122492731s, reserved1.236328125/1.234375GiB. Largest pair infer
median.196921087/.196747467s, same reserved1.7578125GiB. Separate native
train-stress median.941685621/.942809511s, reserved13.677734375/
13.400390625GiB. Serial order/small cohorts/transient hardware effects
remain; no speedup, equivalence, causal cost or Pareto dominance claim.

## 5. Conclusion

Target.515 unmet; retain W2 champion. Reject this fixed final-token-attention
realization for predeclared small-target harm, not every attention family.
Overall CIcross0 does not cancel a prospectively declared subgroup criterion.
No renamed rerun, checkpoint rewrite, validation tuning or blind restart.
Full protocol/paired trained cost/release audit are complete; scientific
benefit is not established. Y36 is a separately preregistered and tested
local-edge confidence hypothesis; all its actual admission gates remain required.

## 6. Crash Info

No crash asserted. The bounded final24KB capture has no traceback or OOM
marker; this is not whole-log crash absence. Actual ended native Val30/30,
checkpoint30 and absent processes substantiate canonical completion.
Evidence: y33_terminal SHA d7993b8e, postaudit333cb357, source/evaluation
protocol adapters are new files; original helpers and hashes retained.
