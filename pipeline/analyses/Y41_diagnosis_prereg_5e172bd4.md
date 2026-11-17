# Y41: boundary probability mass, preregistered diagnosis

Scope: AdaRES_project only; token 5e172bd47ee3450dbee928bdecaede7f.

Hypothesis ID: final-mask-boundary-probability-mass.

The final confidence readout may lose spatial mask-boundary probability mass
when boundary and interior visual messages are normalized independently.
The source audit shows Y36 normalizes each pooled edge feature by its own total
weight. This motivates a test, not a claim that the trained decoder is blind.

Control: physical W2 and physical Y36, equal final queries, constant visual
features, unchanged coordinates, point mass and mask-value histogram. Compare
block and alternating mask assignments on a fixed nearest-neighbor graph.
Measure the actual existing readouts and the prospective probability descriptor.
Do not require validation gain, statistical significance or a numeric effect
size from this synthetic test. A descriptor that is invariant to this intended
intervention is a disconfirmation of the realization.

Prospective realization: final-indicator-only additive readout of the directed
nearest-neighbor FG-to-BG probability fraction, BG-to-FG fraction, and weighted
squared probability difference normalized by global mask variance. Average
exact nearest-distance ties. Use every valid superpoint, no selected feature
subset, radius, neighbor-count search or new loss. Use the native 256-dimensional
query plus these three scalars, Linear(259,256), GELU, Linear(256,2).

Required further tests if the CPU control separates: point/query permutation,
padding poison, repeatability, translation/rotation/uniform scale invariance,
mask-coordinate mispairing, finite nonzero gradients in all four new tensors;
prediction-blind TRAIN incidence; native loss backward/optimizer/resume;
physical package independence and matched resources. Shared parent tensors,
mask predictions, sampling and all non-indicator losses must be unchanged when
the added branch is disabled. A CPU gap is not a native gate PASS.

Freeze: 30 epochs, seed 1999, original 128 queries, single argmax, unchanged
loss weights and released SPFormer; no validation-driven search. Post-training
primary FIRST complete ScanRefer validation mIoU, 9508 expressions/141 scenes,
candidate/W2/B1 each twice; scene bootstrap 10000 seed 1999; TRAIN size cuts
1487/3325; subgroup material harm when paired CI upper < 0 against either
comparator. Quantify repeat noise and trained paired costs before efficacy or
Pareto claims. Official unique/multiple labels are unknown if unavailable.

This is independent of Y40's word/support information path. It is related to
Y36's geometry, and must be rejected if actual controls show the same scalar
information already survives that realization. It is not a claim of novelty,
semantic necessity, upstream blindness, benefit or harmlessness.
