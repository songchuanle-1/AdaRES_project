# AdaRES-Y31 Diagnosis Preregistration

Created during owned tick626ebd89c7094c31953d05dac9d0dc8d on 2026-10-07.
Stage: diagnosis. No gate, trained benefit or novelty is claimed here.

## Hypothesis

Mask-supported central second spatial moments may provide explicit geometric
shape evidence to the standard confidence head that its terminal query does
not directly receive. This differs from visual-content readback, relative
peer-centroid attention, and ordered decoder-state histories. Implicit geometry
can already exist upstream; its insufficiency and trained benefit are unknown.

## Matched Control

Use physical W2 parent and its original seed1999, fixed30, native128 queries,
standard one-query argmax, and unchanged masks/loss/sampling/backbone.
First64 distinct TRAIN scenes, first up to4 descriptions each, retained without
selection by outcome. Do not inspect validation for design or choose features.
Compute the six independent entries of the symmetric 3D covariance matrix,
with native point support or explicitly labeled superpoint approximation and
a scene-derived scale. Freeze the support definition before native outcomes.

Controls: disabled parent, repeated identical inputs, jointly permuted query
and mask rows, point permutation, translation, identical-shape and different-
shape equal-centroid/equal-mass supports, and shuffled covariance between queries.
Do not claim shape necessity from graph activity or from wrong-shape learnability.

## First Executable Test

Audit actual W2 source and test the covariance formula on deterministic CPU
equal-centroid/equal-mass point sets. Check whether the proposed statistic is
already exactly supplied to the confidence head, is a no-op, or distinguishes
support shape while respecting translation and point/query permutation.
Then collect prediction-blind TRAIN shape diversity with frozen native inputs.
Synthetic success alone is insufficient for diagnosis or admission.

## Gates And Limits

All six real gates remain pending: diagnosis, controls, native_backward,
resources, independence, constraints. Every new trainable tensor must receive
finite nonzero native-loss gradients. Before GPU work verify current GPU1
occupation and acquire an exclusive lease; GPU0 is excluded. Preserve original
Y30 negative evidence. No tuning thresholds, temperatures, feature subsets,
loss weights or epoch count; no new scheduler, other project or nested codex.
Novelty, learned necessity, full-validation efficacy and trained Pareto are unknown.
