# AdaRES-Y39 Prospective Diagnosis

Owning tick48678b818b134bea9d9c3c3e1f3d3305; before native outcomes.
Hypothesis ID: within-superpoint-feature-distribution-confidence.

The native mask broadcasts one decision over every point in a superpoint.
Superpoint mean features and point counts can be identical while the original
per-point feature distributions differ. Query confidence may lack evidence
about mixed/heterogeneous superpoints and mask representability.
This concerns within-superpoint information, not global support coordinates,
local superpoint adjacency, language attention or decoder mask history.

Matched CPU control: equal superpoint mean features, point counts, centroids,
query and masks, but differing within-superpoint feature variance. Use all
native backbone channels, no feature subset or numeric search. Independently
verify actual parent pooling and relevant existing readouts before claiming
an additional input path. Permutations must preserve the new descriptor.

First native diagnostic: first64 distinct TRAIN scenes, original raw point
ground truth/superpoint partition, fixed seed1999. Measure the exact maximum
IoU attainable by any union of whole superpoints using descending target
fractions and every prefix, including empty. Retain every scene/expression.
GT is diagnosis only, never an inference input. An independent small exhaustive
enumeration must verify this optimizer; shuffled labels and point permutations
are controls. Nonzero quantization gap is an information constraint, not proof
the proposed confidence model can estimate or overcome it.

If the diagnosis survives, freeze a predictor using original per-point
backbone feature moments and point-mass masked aggregation before controls.
Train/evaluate masks, standard indicator objective, native128/one argmax,
released SPFormer, fixed30/seed1999 and optimizer/scheduler remain unchanged.
All native backward, adverse-result investigation, resources, independence,
goal constraints and post-training W2/B1 clustered harm review are still
pending. No admission, scientific benefit, necessity or novelty claim.
