# AdaRES-Y30 Preregistered Diagnosis

Stable hypothesis: ordered-query-mask-trajectory-confidence.
Owned tick86766fcd883d46c1a9d60563f4213c2e; snapshot10:51:41.101902Z.

- Causal claim: the final confidence head receives a final query and its
  overlapping peers, but no explicit ordered query/mask evolution. Aligned
  decoder histories may supply reliability evidence lost by the terminal
  confidence readout. Upstream queries already encode history implicitly;
  this is not a claim of complete blindness or guaranteed improvement.
- Prediction-blind sample: first64 DISTINCT native TRAIN scenes, up to4
  descriptions per scene, seed1999, original W2 best checkpoint. All scenes
  are retained irrespective of target size or measured direction.
- Audit actual source: seven calls must retain query identity/order/shape;
  native final mask and point-IoU are unchanged. Record source and checkpoint
  hashes and require GPU1 live/exclusive guard. Never query/use GPU0.
- Native diagnosis: record final selection/oracle regret, same-query historical
  query RMS difference, consecutive point-mass soft-mask IoU and log-area
  ratio. Readouts use current augmented native superpoints/counts.
- Prospective implementation: physical W2 copy. A GRU consumes the ordered
  normalized query plus two mask-transition descriptors across native seven
  states, with hidden width inherited from d_model256 and a two-logit output.
  Add this explicit temporal readout only to FINAL confidence; parent auxiliary
  supervision, mask generation, sampling, backbone, native losses and native
  inference argmax remain unchanged. No numeric hyperparameter search.
- Matched controls: parent/disabled; cached repeat; joint query permutation;
  history-only cyclic wrong-query correspondence with final state fixed;
  earlier-state reverse order; seven repeats of final state. All controls
  share identical model weights/current inputs and are diagnostic only.
- Mechanism direction: explicit ordered history should change the intended
  temporal readout versus mismatched/no-history inputs. Every new tensor must
  be reached by native training with finite nonzero gradients and updates.
  Forward identity under detach but altered upstream gradients separates a
  genuine path from feature scaling. Initial task loss is descriptive; adverse
  native behavior requires investigation, not passing by weaker thresholds.
- Scientific uncertainty: Y9 fixed layer-score mean rejection and Y28 learned
  final mask-content small-group harm stay closed. Y30 is not either rule;
  actual source/control tests must substantiate that distinction. Generalization,
  temporal necessity, novelty and training-seed uncertainty are unknown.
- No admission from diagnosis. Freeze full post-training9508/141 two-pass
  W2/B1, whole-scene bootstrap10000seed1999, original TRAIN target-size
  cuts1487/3325/material subgroup harm and trained lightweight Pareto tests
  before launch. Fixed30epochs, seed1999, released SPFormer backbone,
  one authorized GPU1 only.
