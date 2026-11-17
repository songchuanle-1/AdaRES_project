# Y41 nonempty pooling control diagnosis

The finite-logit original seed0 has W2 difference3.725290298461914e-8,
Y36 difference2.9802322387695312e-8, descriptor difference3.500000476837158
and new indicator difference0.11339519917964935. This records reduction
roundoff, not strict invariance PASS.

The first exact-binary alternating control has W2 difference0, but Y36
difference0.04223823547363281. On an alternating hard mask, the normalized
interior pooling denominator is zero and the existing code emits a zero
interior message. It is therefore not a matched equal-message control.
This positive existing-path response is retained, not relabeled as blind.

For a further arithmetic control, use the fixed hard block pattern
00001111 versus00110011, keeping both boundary and interior pools nonempty.
Keep the original32 finite cases and all backward/invariance tests. Require
the unchanged W2 exact0 and existing Y36 tolerance1e-6; do not loosen either.
This is a counterfactual control correction, not hyperparameter search or
selection of a training/validation value. Native masks remain finite.

The hypothesis concerns probability mass lost under nonempty normalized
pooling, not every topology signal or the empty-pool fallback. No necessity,
population deficit, training efficacy or novelty claim follows.
