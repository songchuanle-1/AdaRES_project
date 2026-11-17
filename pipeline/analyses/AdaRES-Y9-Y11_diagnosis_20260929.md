# AdaRES Y9/Y10/Y11: confidence, coverage, and scene-collision diagnoses

2026-09-29 Asia/Shanghai. User-requested foreground design tick, snapshot
21:22 Asia/Shanghai, deadline 22:22 Asia/Shanghai. W2's 30-epoch best
checkpoint was unchanged. All probes used the first 32 non-augmented ScanRefer
validation scenes (851 expressions), seed 1999, on server-2 GPU0. The values
below are fixed-cohort diagnostic point IoUs and rankings, not full ScanRefer
mIoU or estimates of a trained intervention.

| ID | Frozen question | Observed result | Disposition |
|---|---|---|---|
| Y9: decoder-layer confidence consensus | Does an equal-weight mean of seven decoder-layer indicator margins select better queries than the final layer and a layer-misaligned control? | All 851 expressions: final mean IoU 0.38231, consensus 0.38482, misaligned control 0.33485. Scene-mean consensus-minus-final delta was `+0.00205`, but only 9 scenes were positive, 5 negative, and 18 tied; two-sided scene sign test `p=0.42395`. Consensus-minus-control was `+0.05499` with `p=2.46e-7`. Repeated-forward identity mismatch was zero; peak reserved inference memory was 4.0059 GiB. | Reject. The control contrast is strong, but the predeclared comparison against the final layer fails the required `p<0.05` rule. Do not tune layer weights or temperatures. |
| Y10: complementary top-2 union | Does the fixed union of the top-ranked and second-ranked W2 mask probabilities improve IoU over top-1 and a same-rank random-second control? | All 851 expressions: top-1 mean IoU 0.52963, top-2 union 0.52891, random-second control 0.48641. Scene-mean union-minus-top1 delta was `+0.00193` (19 positive, 13 negative, `p=0.37709`). Union-minus-control was `+0.04481` with `p=2.46e-7`. Repeated-forward identity mismatch was zero; peak reserved inference memory was 4.0098 GiB. | Reject. The second query adds a small average gain, but not the predeclared scene-level significance relative to top-1. Do not scan `K`, union weights, or thresholds. |
| Y11: cross-expression instance collision | Within a scene, do W2-selected masks for expressions with disjoint target instances overlap more than under target-permutation controls, and would oracle selection reduce that overlap? | 32 scenes, 12,001 expression pairs, and 11,305 true-disjoint pairs. Selected-mask collision rate was 0.03661 versus permutation control 0.06777; scene paired delta `-0.03117` (0 positive, 30 negative, 2 ties; `p=1.86e-9`). Oracle collision rate was 0.00185, reducing overlap versus selected by `-0.03475` (`p=9.31e-10`). Repeated-forward identity mismatch was zero; peak reserved inference memory was 4.0059 GiB. | Reject. The oracle-reduction direction is supported, but the first frozen criterion is reversed: W2 collides substantially less than the permutation control. This disconfirms the assumed excess-collision deficit. Do not add an exclusivity loss from this premise. |

Y9, Y10, and Y11 use independent mechanisms: layer confidence aggregation,
rank-selected mask union, and cross-expression scene overlap. None passed its
complete predeclared rule, so no inference intervention is admitted, no native
backward gate was reached, and no training was queued or launched.

Y12 was preregistered before the deadline but not implemented or executed. Its
frozen candidate is a standardized W2+B4 confidence committee compared with
W2 and eight query-misaligned B4 controls. The next research window may execute
that preregistration as written; no committee result is claimed here.

Resume only after the evidence cooldown or on a new explicit research request.
Do not scan the Y9 layer weights, Y10 union threshold, or Y11 overlap statistic
on ScanRefer validation.
