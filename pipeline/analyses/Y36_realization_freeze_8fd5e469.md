# AdaRES-Y36 Fixed Realization

Tick 8fd5e469dee241da8f21da7c870c0d3f; before actual measurements.
The earlier diagnosis preregistration remains byte-identical.

Native input: final contextualized superpoint features, native superpoint
coordinates, soft masks, native point mass and final query. For each valid
superpoint use its exact nearest-distance non-self neighbors, averaging ties.
This is a single fixed nearest-neighbor realization, no radius/k search.
Compute edge messages from feature, mean neighbor feature and their difference
through Linear768-to256/GELU/Linear256-to256. Soft boundary weights are
p_i*(1-mean_neighbor_p)*point_mass_i; interior weights are
p_i*mean_neighbor_p*point_mass_i. Normalize each support independently with
the parent's numerical clamp1e-6. Fuse query/boundary/interior through
Linear768-to256/GELU/Linear256-to2 as final indicator residual only.

Eight new tensors; native dimension256; conservative inherited GELU and
two-layer width, not selected through validation. Parent initialization is
preserved by fork_rng after all parent modules. Fewer than two valid points
produce zero support summaries. Upstream spatial attention is unchanged.
Dense nearest-distance construction is potentially expensive: measure native
largest-pair resources before admission. No resource or quality PASS yet.

CPU freeze:32 seeds; repeat, support/query permutation, padded-value poison,
mask-detach forward, wrong-coordinate adjacency and gradient controls.
Equal global mask-coordinate moments plus equal feature-mask marginals with
different edge assignments must be distinguished; exact distance ties must
remain jointly permutation equivariant. Native TRAIN64 first distinct scenes,
up to4 native descriptions; matched W2/released-init TRAIN8/init4 controls,
physical2 optimizer steps/modelAdamWPolyLR restore and inherited stress cases
must precede admission. All results, including harms and weak gradients, retained.

Post-training comparison, if admitted: fixed30/seed1999/released backbone;
unchanged full ScanRefer9508 expressions/141 scenes; Y36/W2/B1 each twice,
first pass primary, whole-scene bootstrap10000 seed1999, TRAIN raw point
cuts1487/3325. Material subgroup harm: paired scene interval upper below0
against either W2 or B1. CI crossing0 is inconclusive, not equality/no harm.
Report mIoU/Acc25/Acc50/native128/one argmax/paired trained latency/peak memory.
Efficacy, novelty, necessity, seed uncertainty and trained Pareto unknown.
