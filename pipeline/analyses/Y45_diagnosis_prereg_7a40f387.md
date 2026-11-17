# Y45 Prediction-Blind Diagnosis Preregistration

- Stable ID: AdaRES-Y45
- Hypothesis ID: metric-confusion-decomposition
- Scientific stage: diagnosis, not admitted, not efficacy.
- Goal: learn query confidence from evidence aligned with point-level IoU;
  preserve ScanRefer target 0.515 and every fixed protocol/resource constraint.
- Hypothesis: existing scalar-quality/indicator supervision may discard the
  separate true-positive, false-positive and false-negative point masses.
  An explicitly learned error decomposition could expose distinguishable
  quality evidence without changing mask generation, thresholds or selection.
  This is a hypothesis, not a claim that the decomposition improves ranking.
- Matched control: actual W2 quality and indicator targets, plus the original
  B2/G1/C1/Y29 target/ranking implementations where relevant. Hold native
  query masks, features, expression mapping and point/superpoint counts fixed.
- First test: identify exact scalar targets in actual source; build equal-IoU
  mask/truth witnesses with different FP/FN partitions, compare their actual
  target/gradient readouts, and check whether the proposed quantities reduce
  to an already tested objective. Preserve evidence of duplication/no-op.
- Native follow-up: capture prediction-blind TRAIN masks and point targets;
  compare correct vs expression-permuted TP/FP/FN targets and verify incidence,
  gradients, padding and exact point/superpoint alignment before implementation.
- No numeric searches, loss-weight tuning, validation component selection,
  higher-moment successor, renamed banned route or new training is authorized
  by this diagnosis record.
- Reject with executable duplication/no-op/disconfirming evidence; lack of
  native tests is pending, not a failed or passed gate.
- Post-training requirements, if admitted later: fixed30/seed1999/single GPU;
  frozen full9508/141 twice each against W2/B1, FIRST primary, scene-clustered
  uncertainty, predeclared subgroup harm and paired trained cost. No promotion
  from a synthetic witness, trained-log rounding or a small-scene screen.
