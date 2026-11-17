# AdaRES-Y10 preregistration: complementary top-2 query coverage

Date: 2026-09-29 Asia/Shanghai

## Frozen question

Does the second-ranked W2 query add target coverage beyond the top-ranked
query, and does a fixed top-two union improve selected-query point IoU relative
to the top-one mask and a same-rank random-second control?

Let `p_i(m)` be the sigmoid mask probability of query `i` at superpoint `m`.
With `i1 = argmax s_i` and `i2 = second argmax s_i`, the frozen union is

`u(m) = 1 - (1 - p_i1(m))(1 - p_i2(m))`,

and the final binary mask is `1[u(m) > 0.5]`. The rank is determined by the
unchanged final indicator positive probability. No `K`, threshold, or fusion
weight may be scanned.

## Matched control

For each expression, eight fixed query permutations replace only the second
query with a random non-top-one query. The control also uses the same union
operator and threshold. The control readout is the mean IoU over permutations.
This preserves mask distributions and union computation while removing
second-rank specificity.

## Cohort and decision rule

- W2 30-epoch best checkpoint, unchanged.
- First 32 non-augmented ScanRefer validation scenes, seed 1999.
- Primary readout: selected-query point IoU over all expressions.
- Pass only if all hold:
  1. scene-mean `IoU(top2 union) - IoU(top1)` is positive;
  2. scene-mean `IoU(top2 union) - IoU(random-second control)` is positive;
  3. the paired scene sign test on top2 union minus top1 has `p < 0.05`;
  4. repeated-forward union-selection identity mismatch is zero.

A failed rule rejects the complementary-coverage route. A pass is only a
diagnosis; a trained set aggregator would still need native backward,
resource, independence, and constraint gates.

## Independence

This is not C1 expected-mask supervision: C1 takes a soft weighted average of
all query masks, whereas Y10 tests a hard union selected by rank. It is not D1
pairwise score comparison because the second query is allowed to contribute
spatial evidence even when its indicator score is lower. It is not Y9 because
it uses no auxiliary layer confidences.
