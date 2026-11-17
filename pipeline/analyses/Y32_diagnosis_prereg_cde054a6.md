# AdaRES-Y32 Diagnosis Preregistration

- Created: 2026-10-08, owned tick cde054a6b0254e37a0c0d71febbcc497.
- This is a prospective diagnosis, not an admission or efficacy claim.

## Hypothesis

Native W2 final indicator classification labels geometrically matched queries
positive and the other queries negative. Queries outside that match may still
contain useful target masks. Supervising the relative confidence distribution
with exact augmented point-IoU for ALL active queries could convey a distinct
quality-ordering signal without changing inference, query count or parameters.
Whether this duplicates an older loss must be tested, not assumed.

## Frozen Intervention

For one expression, compute each query's detached hard-mask IoU on its current
augmented raw points. Ground truth is used ONLY to form a training target.
Normalize nonnegative IoUs by their sum and minimize cross-entropy against a
query softmax of the binary indicator logit difference. If every IoU is zero,
fall back to the normalized native matched-query positive set; with no positive
match, preserve the native no-target behavior. No temperature, margin, threshold
or loss-weight search. The native 0.5 mask binarization, loss_weight[4], fixed30
epochs, seed1999 and all other losses/auxiliary layers stay frozen.

This first realization replaces ONLY the final indicator term. It is not an
inference oracle, feature-subset search, matched-query calibration replacement,
or a retry of the exact Y29 matched-label objective.

## Matched Control And First Test

- Physical W2 parent; identical indicator logits, masks, point identities and
  matching indices for every paired readout.
- Cheapest first test: audit actual W2, G1, C1 and retained Y29 source on CPU.
  Execute their actual relevant loss methods where possible on 32 deterministic
  cases, compare losses/gradients with the preregistered formula and vary only
  nonmatched mask quality. Preserve no-target, zero-IoU and ties as explicit cases.
- A numerical/source identity with a closed realization is disconfirming
  evidence for independence, not permission to rename it.
- Then, if distinct, prediction-blind first64 distinct TRAIN scenes/up to4
  descriptions: quantify nonmatched positive-IoU incidence, native selection
  regret, and paired supervision gradient direction with repeats.
- Follow with same-ID native cached controls, backward, two optimizer steps,
  save/resume, resources, physical independence and structural smell review.
  Missing tests remain pending; adverse native behavior requires investigation.

## Outcomes And Bounds

No new training until all six real gates and unchanged goal constraints pass.
Only server-2 GPU1 may be used after live occupation and diagnostic-lease checks.
GPU0 is reserved. Native repeat noise is not assumed absent. Full trained
efficacy/necessity/novelty/Pareto are unknown. Existing negative realizations and
Y31's inconclusive completed protocol remain unchanged. Stop only under the
canonical closure contract and real tick deadline; no invented blocker or quota.
