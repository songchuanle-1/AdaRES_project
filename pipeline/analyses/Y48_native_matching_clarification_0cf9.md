# Y48 Native Source Clarification Before Executing Controls

- The initial single-winner interpretation is disconfirmed by actual source:
  W2 Criterion.match_sampled_ins retains nearest-query assignments and also
  every query whose sampled instance label equals a referred object ID.
  Existing TRAIN summaries already include more than one positive query.
- Preserve the original preregistration. Do not describe this as Hungarian
  one-to-one matching, or change historical source/labels/evidence to fit it.
- The remaining conditional question is output-mask equivalence versus
  seed-based ownership: exact equal hard masks may come from different
  sampled instance labels. First execute the actual matching and indicator
  functions with explicit constructed seed metadata; these are CPU source
  controls, not proof that native TRAIN contains mixed-label duplicates.
- Inherit eos_coef from the actual Criterion/config (including the native
  default dtype of its weighting tensor). Compare literal gradients within
  prospective 8 * float64 epsilon * 128 * max(1, gradient magnitude).
  No changes to loss weights, mask signs, source, optimizer or production.
- Measure all32 frozen expressions' exact nonempty duplicate hard-mask
  incidence. Record empty groups separately. No approximate grouping.
- Cached Y46 data does not contain its corresponding native matching indices.
  Do not combine counts from another capture into inferred query labels.
  Actual mixed-label incidence remains unknown until a joint native capture.
- If the exact-equality intervention has no nonempty duplicate opportunity
  on this frozen cohort, report that cohort limitation as direct no-op
  evidence for this realization, not a universal statement about confidence.
