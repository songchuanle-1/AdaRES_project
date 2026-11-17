# Experiment Design - AdaRES_project

Owned tick86766fcd883d46c1a9d60563f4213c2e, snapshot10:51:41.101902Z.
Prepared after the upstream motivation review and prediction-blind diagnosis,
before production training. All unchecked gates are genuinely pending.

## 1. Motivation

"AdaRES: improve a faithful standard-indicator 3D referring expression
segmentation baseline by learning query confidence from metric-aligned
evidence, targeting ScanRefer mIoU >= 0.515."

Capacity1/active0/budget1/due0. StableY29's realized regret objective was
rejected by32 actual B0 loss/gradient equality cases and a false source premise.
StableY30 tests a distinct explicit temporal confidence path. Not a promise
that the next training improves mIoU; not Phase2 ablation.

## 2. Current Best Results

| ID | Config | State | mIoU | Acc50 | spmIoU | Epochs |
|---|---|---|---|---|---|---|
| W2 | adares_w2_pointmass_competition_res | done_complete | .510107995 | .562999579 | .473719922 | 30 |
| B1 | adares_b1_standard_res | done_complete | .502223014 | .552902819 | .466783781 | 30 |
| Y28 | adares_y28_mask_readback_res | done_complete/rejected | .505520490 | .556583929 | .469977198 | 30 |
| Y27 | adares_y27_relative_peer_res | done_complete/inconclusive | .509987856 | unknown here | unknown here | 30 |

Historical checkpoint meta and frozen measurement values are not conflated.
Y28 material small-group harm and all older failures stay immutable.

## 3. Analysis And Direction

W2 confidence is metric-aware but explicit direct temporal inputs are absent.
Y30 captures aligned raw query states and mask transitions across seven native
stages with a GRU; emits two-logit residual into final confidence only.
Upstream implicit history exists; no global blindness claim. The TRAIN64/256
diagnosis shows163 oracle gaps with distinct histories, not future gain.

## 4. Experiment Plan

| ID | Config Path | Hypothesis | Expected Readout | Priority | Depends On |
|---|---|---|---|---|---|
| AdaRES-Y30 | configs_song/model_paper5/adares_y30_query_trajectory_res.yaml | Ordered history supplies final selection reliability evidence | Actual control-separated native path; trained benefit unknown | P0 | Six real gates |

- Physical W2 copy; six sourcefiles and exact parent YAML. Query generation,
  sampling, masks, backbone, losses, auxiliary predictions and argmax retained.
- Temporal inputs: normalized same-query states plus consecutive point-mass
  soft mask IoU and log-area ratio. GRU width256 inherited, seven native states,
  output2 logits;397314 new parameters/8 tensors, no numeric search.
- Current initialization uses CPU fork_rng so shared parent parameters should
  remain bitwise equal, to be verified on native full models.
- Prediction-blind diagnosis64 distinctTRAIN scenes/up to4 descriptions.
  Native controls8TRAIN+4released-init: identical cached decoder/criterion
  inputs, parent/disabled/repeat/mask-detach/wrong-history/reverse/terminal.
- CPU32 controls, finite active gradients for every new tensor; native two-step
  tools.train optimizer and model/AdamW/PolyLR restore plus continuation noise.
- Resource protocol: first8distinct TRAIN scenes plus fixed largest pair728/6780;
  serial native batch1 inference, two warmups and three readings, separate
  batch2 native train/optimizer stress. Report parameter/allocated/reserved/time
  regressions honestly. Descriptive p95 is not a reliable tail estimate.

## 5. GPU Allocation Plan

| Server | GPU | Assignment | Config | Justification |
|---|---|---|---|---|
| server2 | 1 | Exclusive diagnostics, then conditional canonical training | Y30 | Only authorized slot; live checks each acquisition |

GPU0 reserved, never queried/used. No server6/other-project/background task.
Production launch requires empty capacity, exact source/receipt/hash and
increasing native Train steps; PID existence alone is insufficient.

## 6. Implementation Checklist

- [x] Stable diagnosis and matched controls preregistered before their tests.
- [x] Native TRAIN64 information-path diagnosis executed, hashes in ledger.
- [x] Physically independent package/config, additive byte-verified deployment.
- [x] Actual class CPU32 control contracts.
- [ ] Native matched controls and all8 active new tensors.
- [ ] Native tools.train two optimizer steps/save/restore and continuation.
- [ ] Shared full-model initialization and measured resources.
- [ ] Scoped tested source commit and six-gate admission receipt.
- [ ] Canonical gated dispatch and increasing native training step evidence.

Unchecked boxes remain a prelaunch historical checklist, not later rewritten
proof. Subsequent signed review/receipt supplies actual gate results.

## 7. Risk Assessment And Frozen Evaluation

- Temporal trajectories could be redundant or promote unstable/tiny masks;
  a nonzero control effect proves pathway activity, not temporal necessity.
- All initial task-loss directions and wrong controls are reported; unexplained
  adverse results investigated without tuning or selecting favorable scenes.
- No trained efficacy, literature novelty or global determinism claim.
- Fixed30epochs/seed1999/released SPFormer backbone/unchanged AdamW and PolyLR,
  no validation threshold, temperature, loss weight, feature or epoch tuning.
- Freeze before launch: best native checkpoint after fixed30; full ScanRefer
  validation9508 expressions/141 scenes; independently repeat each Y30/W2/B1
  twice with identity checks and all raw per-expression values retained.
- Primary mIoU, plus Acc25/Acc50/spmIoU/128 active queries/one selected.
  Comparators are original W2 and B1 checkpoints, no oracle/fallback selector.
- Paired whole-scene bootstrap10000/seed1999. Existing raw TRAIN target-point
  tertile cuts1487/3325 frozen; validation groups small<=1487, medium<=3325,
  else large. Never regroup by validation outcomes.
- Predeclared material-harm rule: meaningful subgroup upper95% paired
  scene-cluster interval <0 versus W2 rejects this realization, even if the
  all-population interval crosses0. Intervals crossing0 are inconclusive,
  not evidence of equivalence. Retain B1 comparisons and all subgroup signs.
- Empirical repeats estimate observed noise only, not universal bounds.
  Inadequate independent scenes/seed uncertainty remain unknown.
- Trained serial resource protocol repeats the frozen TRAIN8+largestpair
  separate-model measurements. Lightweight Pareto requires supported task
  benefit without supported inference/resource harm; feasible24GB and small
  parameter fraction alone are not Pareto evidence. No promotion on gates.

## 8. Smell Check

New nonlinear ordered cross-stage interaction supplies a new information
aggregation path; not a standalone norm/scalar/sigmoid gate or coefficient
reconfiguration. Values derive from parent hidden width/native sequence and
physical mask-transition statistics, not a sweep. The causal claim is explicit
and wrong-history/reverse/no-history controls are preregistered. Older fixed
means and mask-content readback are not reopened. Literature novelty unknown.
