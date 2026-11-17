# AdaRES-G1 terminal analysis

## 1. Basic Info
- ID: AdaRES-G1; config `configs_song/model_paper5/adares_g1_metric_targets_res.yaml`.
- Server-2 GPU0; launched 2026-09-25 17:08:09, reconciled 2026-09-26 17:47:13 Asia/Shanghai.
- State: `done_complete`, 30/30 validations. The scheduled monitor, not this design tick, certified the terminal state.

## 2. Best Metrics
| Metric | Value | Epoch |
|---|---:|---|
| mIoU | 0.4980 | unavailable |
| Acc@0.25 | unavailable | - |
| Acc@0.5 | unavailable | - |
| spmIoU | unavailable | - |

The local terminal ledger contains rounded best mIoU only. The prior targeted last-validation log readout was mIoU 0.497, Acc@0.25 0.602, Acc@0.5 0.542, spmIoU 0.465; it is not the best-checkpoint metric. Active queries, full-run latency and peak memory are unmeasured. Pre-run paired batch reserved-memory measurements were 4.75 and 6.396 GiB, not full-run maxima.

## 3. Convergence Summary
- Best epoch and subsequent no-improve count are unavailable from the terminal ledger. The process completed 30/30 validations.
- The final validation was approximately 0.497 versus historical best 0.4980.

## 4. Comparison
- Versus B1 baseline 0.5020: -0.0040.
- Versus B4-E30 champion 0.5094: -0.0114.
- Versus target 0.5150: -0.0170.
- Paired fixed 16-scene audit (441 expressions) compared G1 best to B4-E30 best: top-1 mIoU 0.47072 versus 0.49186; oracle mask mIoU 0.83308 versus 0.87316. G1 top-1 improved >0.05 on 65 expressions and regressed >0.05 on 95; its oracle mask improved on 25 and regressed on 98. Only 4/241 G1 material-error oracles rank top three by confidence versus 20/240 B4. This small observational cohort does not establish population effects or isolate training causality.

## 5. Conclusion
Metric-aligned detached per-query mask-IoU targets did not improve B4 in this completed run. Both candidate mask quality and selected mask quality deteriorated in the paired fixed cohort. Close G1; do not relaunch or scan its target/weight variants on validation. B4-E30 remains champion; Phase 1 target is unmet.
