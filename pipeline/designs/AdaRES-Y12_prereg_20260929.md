# AdaRES-Y12 preregistration: independent confidence committees

Date: 2026-09-29 Asia/Shanghai

## Frozen question

Do W2 and B4-E30 assign complementary query confidence such that a fixed
standardized committee selects better queries than W2 alone and than a
query-misaligned B4 control?

For model `m` and query `i`, let `z_i^m` be the positive-negative indicator
margin. Standardize within an expression:

`zhat_i^m = (z_i^m - mean_j z_j^m) / std_j(z_j^m)`.

The frozen committee score is `c_i = zhat_i^W2 + zhat_i^B4`; the selected query
is `argmax_i c_i`, and its W2 final mask is evaluated. No model weights or
temperatures are scanned.

## Matched control

For eight fixed permutations of B4 query indices, replace `zhat^B4` by its
permuted version before adding W2. This preserves each model's confidence
distribution and the selected mask model but destroys cross-model query
correspondence.

## Cohort and decision rule

- W2 and B4-E30 best checkpoints, unchanged.
- First 32 non-augmented ScanRefer validation scenes, seed 1999.
- Primary readout: W2 selected-query point IoU over all expressions.
- Pass only if all hold:
  1. scene-mean `IoU(committee) - IoU(W2)` is positive;
  2. scene-mean `IoU(committee) - IoU(misaligned B4 control)` is positive;
  3. the paired scene sign test on committee minus W2 has `p < 0.05`;
  4. repeated-forward query-selection identity mismatch is zero.

A failed rule rejects the independent-confidence-committee rationale. A pass
would justify a trained dual-head committee with disagreement regularization,
but would not itself admit training.

## Independence

Y12 uses two independently trained confidence heads; Y9 used decoder depths of
one W2 model. It is not a mask union, a query propagation graph, a native
positive-set diagnostic, or a scene-level collision check.
