# Y30 Numerical Fidelity Repair

Owned token86766fcd883d46c1a9d60563f4213c2e, 2026-10-06.
Recorded before the repaired source's native tests and any admission.

The original physical source32332300 was not admitted or trained. Its original
seven-mode controls v1/v3/v4 were not successful gate results. V4 quantified
superpoint-permutation output error1.4424324035644531e-5, above the unchanged
1e-5 test bound. Nonindicator loss differences were0 in v3/v4; an earlier
working-ledger attribution to the nonindicator assertion was mistaken.

The separate numerical diagnosis376f2dae retained all TRAIN8/init4 cases.
Float32 descriptor permutation errors reached9.5367431640625e-7; evaluating
the same sigmoid/weighted-area/intersection/log-ratio formula in float64 gave
descriptor errors at most1.7763568394002505e-15, and cast descriptors and
readout permutations were exactly0 for these cases. This is finite observed
numerical evidence, not a global determinism bound or scientific efficacy.

Repair ONLY the new path's mask statistics: double sigmoid and weighted
reductions, cast the final two descriptors back to query dtype before GRU.
Same point mass, padding, clamps, formula, stage order, width, parameters,
initialization, native losses, inference selector and all frozen protocols.
No thresholds, weights, temperatures, validation outcomes or scene selection
are changed. The original source is preserved as
pipeline/analyses/y30_dec_preprecision_86766fcd.py, and all raw failures remain.

Repaired code needs new source-byte identity, CPU32 contracts, strict native
TRAIN8/init4 controls, tools.train two-step/save/restore, shared initialization
and resources, and the same initial-adverse-loss investigation. Old physical
evidence cannot establish repaired-source admission. Native repeat/permutation
assertions remain unchanged. Keep adverse controls and cost regressions.
