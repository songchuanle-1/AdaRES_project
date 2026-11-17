# Y26 Adverse-Readout Follow-Up

Registered after TRAIN64 and native backward, before these follow-up outcomes.
Original preregistration is not rewritten.

Observed native total-loss shifts: trained8 mean+.07624048, released-init4
mean+.03564107. Connected/affine-detached full-model masks differ by up to
.00089788, similar to the repeat maximum.00089169, but this is not yet an
isolated repeat explanation.

Cache actual native decoder AND criterion inputs for the same first8 TRAIN
scenes/first4 released-init scenes. Run physical Y26 with postnorm off, on,
affine-detached, and on repeat under identical RNG/decoder buffers. Add
center-only (remove channel mean, no rescaling) to the already frozen RMS-only
control to decompose the adverse loss shift. These are diagnostic controls,
not selectable production variants. Preserve every term and every scene.

Expected connected versus affine-detached forward/shared-input gradients equal
within cached repeat noise; only affine parameter gradients differ. If not,
inspect the exact native computation rather than attributing to FPS. Report
whether centering or rescaling carries the observed loss increase. Neither
decomposition nor finite gradients predicts 30-epoch efficacy.
