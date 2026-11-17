# Y45 Native TRAIN Target Capture

- Stable candidate AdaRES-Y45; same metric-confusion-decomposition hypothesis.
- Prospective cohort: first eight distinct TRAIN scenes, indices
  0,44,60,101,115,126,137,171, seed1999, native TRAIN augmentation.
  No validation labels, predictive performance or observed outcomes choose it.
- Freeze native W2 best checkpoint and code; record final masks, binary
  matched indicator labels, point/superpoint incidence and existing logit
  gradients. No checkpoint update, new training or diagnostic weight deployment.
- Readout: literal point TP/FP/FN must match superpoint-weighted counts exactly
  at the native mask threshold. Query/point-order permutation must preserve
  mapped counts; cyclic same-scene expression swap is the frozen wrong control.
  Expressions with identical GT are retained as non-discriminating controls.
- Existing matched CE labels may contain many different confusion targets.
  That establishes only available supervision, not useful learnability,
  population benefit or a passed new-backward gate.
- Any future model needs an independent package and a preregistered intervention,
  matched control, native new-tensor gradients, resource and constraints review.
  No threshold, temperature, weight, feature or epoch search is permitted.
- The original scalar-confidence preregistration and its disconfirmation remain
  unchanged. Compare any proposed objective with original B2/G1/C1 and Y29
  implementations before claiming structural independence.
