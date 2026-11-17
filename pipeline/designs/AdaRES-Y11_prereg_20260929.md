# AdaRES-Y11 preregistration: cross-expression instance collision

Date: 2026-09-29 Asia/Shanghai

## Frozen question

Within a ScanRefer scene, do W2-selected masks for expressions with disjoint
target instances overlap more than the same overlap under a target-permutation
control, and does oracle query selection materially reduce that overlap?

This tests a scene-level deficit that single-expression scores cannot observe:
different expressions should normally select different object instances.

## Statistics

For scene `s` and two expressions `u,v`, let `M_u` and `M_v` be their binary
selected masks, and let `o_u,o_v` be their target-instance sets. Define

`overlap_uv = |M_u ∩ M_v| / |M_u ∪ M_v|`.

The true disjoint pair set is `o_u ∩ o_v = empty`. Eight fixed per-scene
permutations reassign target-instance sets to expressions; the control scene
mean is the permutation-weighted mean overlap among pairs classified disjoint
under those permutations. The oracle counterpart replaces each selected mask
by the query with maximum IoU to its target.

## Cohort and decision rule

- W2 30-epoch best checkpoint, unchanged.
- First 32 non-augmented ScanRefer validation scenes, seed 1999.
- Scene is the sampling unit.
- Pass only if both hold:
  1. true-disjoint selected-mask overlap exceeds the target-permutation control,
     with a two-sided scene sign test `p < 0.05`;
  2. oracle selection reduces true-disjoint overlap versus W2 selection, with a
     two-sided scene sign test `p < 0.05`.

A failed rule rejects the scene-level collision rationale. A pass is a diagnosis
only; a scene-exclusivity loss would still need native backward, resource,
independence, and constraint gates.

## Independence

Y11 is not an expression-level re-ranking signal. It is the first route that
uses relations between different expressions in the same scene. It does not
reuse Y7 positives, Y8 query-overlap propagation, Y9 layer consensus, or Y10
top-two union.
