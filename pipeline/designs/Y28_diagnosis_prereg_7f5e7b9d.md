# AdaRES-Y28 Diagnosis Preregistration

Created under lease7f5e7b9d138646e1a4458341d88f7124 on2026-10-05.
Snapshot19:49:05.863150+08; real deadline20:49:05.863150+08.
This is a prospective diagnostic manifest, not admission or a positive result.

- Stable hypothesis: mask-semantic-readback-confidence. W2 query confidence may
  lack an explicit path from the FINAL predicted mask's pooled visual content
  to its language-conditioned query readout. A predicted-mask-weighted visual
  readback can distinguish similarly scored queries whose final mask content
  differs. Upstream queries already attend to visual/language features:
  absence of all semantic information is NOT presumed.
- First test: exact native source/AST audit of final-mask to confidence paths.
  Then FIRST64 distinct native TRAIN scenes, up to4 existing descriptions per
  chunk, seed1999/W2 best checkpoint, no validation or outcome-selected scenes.
  Cache identical query features, final mask probabilities, superpoint visual
  features and point masses. Report native selected-to-oracle point IoU gaps
  and predicted-mask pooled content versus rolled-mask/uniform pooling.
- Matched controls: same frozen native inputs; correct mask pool versus
  superpoint-roll mask pool, uniform scene pool, disabled path and repeat.
  Joint superpoint permutation must preserve pooled features; padding must
  contribute zero; no ground-truth feature input in the candidate.
- Graph question: does existing confidence ALREADY read the exact final
  predicted-mask pooled visual content? If yes, prove redundancy with actual
  source and cached native intervention before rejecting this realization.
  If not, quantify the task-relevant readout gap and test a new physical path.
- Expected mechanism: nonzero, reproducible mask-content-dependent confidence
  intervention, with native indicator-loss gradients through every new tensor.
  No pretraining effect-size or validation-benefit threshold.
- No changes to query count128, selection count1, native mask threshold,
  loss weights/temperature, released SPFormer backbone, seed or fixed30 epochs.
  A physical independent W2-copy package is required before any admission.
- Only server-2GPU1 is authorized; acquire live idle/exclusive guards for
  GPU tests. GPU0 is reserved and must not be queried or used.
- Current status: source audit and native capture PENDING. Scientific benefit,
  harm, novelty, training cost and lightweight Pareto efficacy UNKNOWN.
- After diagnosis, freeze six gates and a trained full9508/141 comparison with
  W2/B1, independent repeats, whole-scene bootstrap10000/seed1999, original
  TRAIN size cuts1487/3325 and subgroup harm rule before launch.
