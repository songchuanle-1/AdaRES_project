# AdaRES Y7/Y8: set separation and competition diffusion

2026-09-29 Asia/Shanghai. User-requested foreground design tick,
snapshot 19:14 Asia/Shanghai, deadline 20:14 Asia/Shanghai. W2's 30-epoch
best checkpoint was unchanged. Both probes used the first 32 non-augmented
ScanRefer validation scenes (851 expressions), seed 1999, on server-2 GPU0.
The values below are fixed-cohort diagnostic point IoUs and rankings, not full
ScanRefer mIoU or estimates of a trained intervention.

| ID | Frozen question | Observed result | Disposition |
|---|---|---|---|
| Y7: native-positive set separation | For material errors whose oracle query is in the native positive set, does the true set have at least one positive outranking every native negative more often than an equal-cardinality cyclic-shift control? | Eligible 277 expressions over 32 scenes. True separation rate 0.2058 versus shifted 0.0550; scene-mean paired delta 0.1876 in favor of true, but only 20 scenes positive, 12 negative, giving a two-sided scene sign-test `p=0.2153`. Continuous margin was also favorable (true -0.3229 versus shifted -0.6414), but the predeclared primary rule required `p<0.05`. | Reject. The diagnostic direction is promising but does not establish control-specific set separation at the predeclared scene level. Do not implement a group-positive ranking loss from this evidence. |
| Y8: competition-graph diffusion | Does the fixed readout `s' = 0.5 s + 0.5 P s`, with `P` the row-normalized mask-overlap graph, improve selected-query IoU against raw W2 and a degree-preserving row-permutation control? | All 851 expressions: raw mean IoU 0.52963, diffused 0.51310, control 0.08589. Scene-mean diffused-minus-raw delta `-0.01678` (12 positive, 18 negative, 2 ties; `p=0.3616`). On the raw-material-error subset diffusion improved 0.23323 to 0.26207, but the preregistered primary cohort was all expressions and the overall condition failed. Identity and material-error repeat mismatches were both zero. | Reject. The apparent error-subset gain is a post-hoc subgroup result and does not rescue the frozen primary readout. |

Y7 had 843 raw row mismatches caused by low-order floating score fields, but
the frozen decision fields (material-error, oracle-positive, selected-positive,
true-separation) had zero repeated-forward mismatches. Y8 had zero identity and
material-error mismatches. This preserves the negative dispositions; it does
not justify changing either preregistered rule.

The two routes are independent: Y7 tests native positive-set score separation;
Y8 tests inference-only score propagation over predicted mask overlap. Neither
uses validation thresholds, feature scans, epoch tuning, or a renamed G1/C2
objective. No candidate passed diagnosis, no inference intervention is
admitted, and no training was queued or launched.

Resume only with new prediction-blind evidence for a distinct mechanism or a
new explicit research request. Do not scan the Y7 sign threshold or the Y8
diffusion coefficient on ScanRefer validation.
