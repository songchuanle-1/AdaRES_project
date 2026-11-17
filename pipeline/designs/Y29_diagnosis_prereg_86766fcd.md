# AdaRES-Y29 Preregistered Diagnosis

Recorded under owned tick86766fcd883d46c1a9d60563f4213c2e,
snapshot2026-10-06T10:51:41.101902+00:00, deadline11:51:41.101902Z.

- Stable hypothesis: confidence-selection-regret-coupling.
- Causal claim: independent absolute query-quality regression need not transmit
  the relative point-IoU regret of selecting one query over a competing query.
  A TRAIN-only coupled confidence objective could supply that missing relation
  without changing the inference selector, thresholds, query count or backbone.
- Matched control: physical W2 parent, identical native predictions/targets,
  model initialization, data and frozen30-epoch/seed1999 protocol. Candidate
  changes only the explicitly audited confidence-training graph.
- First discriminating test: extract actual W2 and historical ranking losses;
  compare their computation and gradients with a pairwise quality-regret
  objective on fixed CPU cases before selecting a realizable intervention.
  If already identical, reject this realization with measured identity.
- Prospective readouts: query-score gradient differences, target/query joint
  permutation equivariance, equal-quality tie behavior, repeat identity and
  native TRAIN confidence/point-IoU ranking disagreement on prediction-blind
  distinct scenes. A true tie must provide no directional ranking preference.
- Controls: parent absolute target regression, repeat, detached quality target,
  cyclic wrong-query correspondence. No validation-selected margin, temperature,
  loss weight, sample subset or target-size subset.
- Initial stage is diagnosis. No gate is passed by this document. Source audit,
  actual controls, native backward/optimizer/resume, resources, independence
  and constraints remain required before admission.
- Y28 frozen small-group harm and all older negative evidence are preserved.
  This is not a renamed mask-content readback module or a universal claim that
  confidence regression fails. Population efficacy and novelty are unknown.
