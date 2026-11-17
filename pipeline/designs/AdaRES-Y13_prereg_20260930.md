# AdaRES-Y13 preregistration: mask-aligned cross-model confidence committee

Date: 2026-09-30 Asia/Shanghai

## Frozen question

Y12 assumed that equal query indices correspond to the same spatial hypothesis in
W2 and B4-E30. That assumption failed: on all 851 fixed val32 expressions, the
B4 top-query index evaluated with W2 masks achieved mean point IoU 0.33277,
whereas W2 selection achieved 0.52963. We therefore test whether cross-model
confidence is useful only after establishing an explicit mask-space query
correspondence.

For expression `e`, let `p_i^m in [0,1]^M` be the final mask probability of
query `i` in model `m`, and let `z_i^m` be its positive-minus-negative indicator
margin. Restrict masks to valid superpoints and define

`S_ij = |b(p_i^W2) ∩ b(p_j^B4)| / |b(p_i^W2) ∪ b(p_j^B4)|`,

where `b(p)=1[p>0.5]`. Let `pi` be the one-to-one assignment maximizing
`sum_i S_{i,pi(i)}` using the Hungarian algorithm. Standardize both score
vectors within the expression:

`zhat_i^m = (z_i^m - mean_j z_j^m) / std_j(z_j^m)`.

The frozen aligned committee score is

`c_i = zhat_i^W2 + zhat_{pi(i)}^B4`.

The selected query is `argmax_i c_i` and its W2 final mask is evaluated. No
alignment weight, temperature, or score coefficient is scanned.

## Matched controls

The primary control applies eight fixed cyclic offsets to the Hungarian
assignment, preserving one-to-one correspondence, both score distributions,
and the number of matched masks while destroying learned spatial correspondence.
An additional diagnostic compares the Hungarian assignment's mean matched-mask
IoU with the cyclic-offset assignments.

The raw-index committee from Y12 is retained only as historical evidence; it is
not rerun as a candidate.

## Cohort and decision rule

- W2 and B4-E30 best checkpoints, unchanged.
- First 32 non-augmented ScanRefer validation scenes, seed 1999.
- Primary readout: W2 selected-query point IoU over all expressions.
- Pass only if all hold:
  1. scene-mean `IoU(aligned committee) - IoU(W2)` is positive;
  2. scene-mean `IoU(aligned committee) - IoU(cyclic-aligned control)` is
     positive;
  3. the paired scene sign test on aligned committee minus W2 has `p < 0.05`;
  4. repeated-forward query-selection identity mismatch is zero.

A failed rule rejects mask-space cross-model confidence transfer. A pass would
justify a trained correspondence-aware committee with disagreement
regularization, but would not itself admit a 30-epoch model.

## Independence

Y13 is not a weighted Y12 rerun: Y12's query-index correspondence is replaced by
an explicit mask-space one-to-one assignment. It is not Y8 graph diffusion,
because scores move between models rather than through a within-model graph. It
is not Y10 union, because no second mask contributes spatial probability, and it
is not a Y1 ranking transfer because the learned W2 ranking is not replaced.
