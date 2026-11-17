# AdaRES-Y9 preregistration: decoder-layer confidence consensus

Date: 2026-09-29 Asia/Shanghai

## Frozen question

Do indicator confidence rankings from different decoder depths contain
complementary evidence that a fixed layer-consensus selector can exploit,
relative to the final layer alone and a layer-misaligned control?

For query `i` and indicator layer `l`, let

`m_i^l = z_i^l[1] - z_i^l[0]`.

The frozen consensus score is the equal-weight arithmetic mean

`c_i = (1/L) sum_l m_i^l`.

The selected query is `argmax_i c_i`; its final-layer mask is evaluated.
No layer weights, temperature, or query count may be tuned.

## Matched control

For eight fixed random permutation sets, each layer's query indices are
permuted independently. This preserves each layer's multiset of confidence
values but destroys cross-layer query correspondence. The control is the mean
selected-query IoU over the eight misaligned layer permutations.

## Cohort and decision rule

- W2 30-epoch best checkpoint, unchanged.
- First 32 non-augmented ScanRefer validation scenes, seed 1999.
- Primary readout: selected-query point IoU over all expressions.
- Pass only if all hold:
  1. scene-mean `IoU(consensus) - IoU(final)` is positive;
  2. scene-mean `IoU(consensus) - IoU(misaligned control)` is positive;
  3. the paired scene sign test on consensus minus final has `p < 0.05`;
  4. repeated-forward query-selection identity mismatch is zero.

A failed rule rejects this route. A pass is only a diagnosis and still needs
native backward, resource, independence, and constraint gates for a trained
layer-consensus fusion head.

## Independence

Y9 does not use Y7 native positive labels; it uses only inference-visible
layer confidences. It is not Y8 graph diffusion because it never propagates
between queries. It is not C1/C2/D1 because it does not optimize a loss over
mask expectations or pairwise margins.
