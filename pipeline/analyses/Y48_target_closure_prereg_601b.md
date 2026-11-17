# Y48 Target Closure Preregistration

- Written under tick 601bcda714184133920945f8f067b382 before the new tests;
  its hash and machine registration time are bound before execution.
- Stable candidate: AdaRES-Y48; no new hypothesis or implementation revision.
- New joint native evidence: 8 TRAIN scenes/32 expressions, 171 nonempty
  duplicate groups, 31 with mixed native matching labels. Empty mixed groups0.
  Source: y48_joint_native_601b_601bcda714184133920945f8f067b382.json,
  SHA256 add87310d9f167148cfb1914bd94f7bd480d95ee15c080f8648b445f163c4762.
- Proposed intervention: in final indicator supervision ONLY, every nonempty
  identical hard-mask group inherits the OR of its original native positive
  labels. Keep empty groups, original positives, singletons, six auxiliary
  objectives, eos coefficient, num_boxes, selector, masks and inference intact.
  Equality uses the inherited hard-mask rule >0, not a searched tolerance.
- Call the original native indicator loss with only the derived index set
  changed. No new loss, multiplier, parameter, threshold or validation search.
- CPU32 fixtures: exact original/closed label bookkeeping, empty preservation,
  original-positive preservation, joint query permutation and cached repeats.
  Actual native loss gradients on fixed logits must change in the predicted
  direction only for newly positive queries. Fixtures are not native efficacy.
- Native: SAME TRAIN8 indices0,44,60,101,115,126,137,171, seed1999, trained W2
  diagnostic weights, full native loss/backward, parent and closed targets.
  Capture labels/masks from the same forward; inspect final-indicator gradients.
  Keep original source/checkpoint and remove hooks after testing.
- Harm/uncertainty: OR may overweight poor masks; a matching-label contradiction
  does not prove useful generalization. Reversal/inconsistent native behavior
  must be investigated. Released-init controls, real optimizer updates,
  independent production code, resources and all gates still precede admission.
