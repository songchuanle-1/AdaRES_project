# Experiment Design: AdaRES-Y43
Created during lease 0422d0d8ad4a4a0288d9e5d7e4e53f73 on 2026-10-13.
Status: IMPLEMENTATION, not admitted or human-approved.

## 1. Motivation
"AdaRES: improve a faithful standard-indicator 3D referring expression
segmentation baseline by learning query confidence from metric-aligned
evidence, targeting ScanRefer mIoU >= 0.515."

Snapshot 2026-10-13 01:49:06.538573+08:00: active1/target2/budget1;
new terminal Y41, Y42 original GPU1 training preserved. The stable Y43
diagnosis was registered before broad review. It asks whether asymmetric
point-feature distributions contain confidence evidence not represented by
the explicit first/second-order summaries. This is Phase1 research, not an
approved Phase2 ablation. It does not infer task benefit from synthetic gaps.

## 2. Current Best Results
| Experiment | Config | State | mIoU | Acc50 | spmIoU | Epochs |
|---|---|---|---|---|---|---|
| W2 | adares_w2_pointmass_competition_res.yaml | retained comparator | .51000985 historical frozen | not rechecked yet | not rechecked yet | 30 |
| Y39 | adares_y39_point_distribution_res.yaml | complete_inconclusive | .50770598 historical frozen | .56121161 | .47172302 | 30 |
| Y41 | adares_y41_boundary_probability_res.yaml | training complete; frozen review pending | .509053826 metadata, not frozen | unknown | unknown | 30; best21 |
| Y42 | adares_y42_point_covariance_res.yaml | original training | structured value unknown this tick | unknown | unknown | snapshot training |

Historical Y39-W2 scene CI crosses0; this is neither superiority nor equivalence.
Rounded CSV Y36 .5110 is not promotion. Current Y41/W2/B1 frozen results must
be reported separately, without changing Y43 based on validation performance.

## 3. Analysis And Direction
Confidence can use predicted-mask point-feature evidence while retaining
native query selection. Mean/variance/full covariance cannot identify all
distributional asymmetry: centered reflection keeps these statistics but
reverses third central moments. Actual existing extraction/readouts were
tested on32 fixed seeds: matched first/second-order and output gaps0,
third-stat gap>=4, numerical permutation floor1.67e-16.

The initial fixture permitted negative reflected features, so it is a
mathematical witness with a codomain limitation. A separately retained
confirmation requires both fixtures nonnegative and tests the actual new
independent branch. Native TRAIN8/init4 incidence and interventions remain
required. No claim of upstream blindness, useful semantic information,
necessity, generalization or novelty is made.

## 4. Experiment Plan
| ID | Config | Hypothesis | Mechanism Readout | Depends On |
|---|---|---|---|---|
| AdaRES-Y43 | configs_song/model_paper5/adares_y43_point_asymmetry_res.yaml | all32 within-SP third moments supply missing explicit distribution-asymmetry evidence | same mean/full covariance but different third statistic; final indicator response above empirical repeat/permutation floor | all six actual feasibility gates |

Complete all32 signed third central moments after the sparse backbone and
before SP mean pooling. Stable signed cube-root maps back to feature units;
machine epsilon bounds the derivative at zero, not a searched hyperparameter.
Reuse W2 d_model256, point-mass predicted-mask FG/BG aggregation and query
input. Add only to the final confidence output; detach mask weights.
Auxiliary indicators, masks, native128 queries/one argmax, loss, optimizer,
scheduler, seed1999 and fixed30 epochs remain unchanged.

Preregistered controls: nonnegative equal-mean/full-covariance reflection,
point/support/query permutations, repeat, padding poison, equal-third control,
zero-third finite derivative, wrong masks, rolled third summaries, reflected
third summaries, disabled physical-W2 equality. TRAIN first8 distinct scenes
and released-init first4 are selected without predictions; pressure pair
indices728/6780 has frozen scan IDs scene0054_00/scene0497_00.

Native loss reaches all8 new tensors with finite nonzero gradients; actual
tools.train two steps and model/AdamW/scheduler restoration are required.
Adverse initial losses prompt investigation with fixed two new-only native
steps on the same frozen cohort, not tuning or removal of negative results.
Wrong controls learning does not establish mechanism necessity.

Posttraining freeze: ScanRefer validation9508 expressions/141 scenes,
Y43/W2/B1 twice EACH, FIRST primary; all metrics, active queries, latency and
peak memory; whole-scene bootstrap10000 seed1999, TRAIN size cuts1487/3325.
Meaningful observed subgroup CI upper<0 versus EITHER comparator triggers
material-harm review; crossing0 is inconclusive, not no-harm. Object-count
strata are not official unique/multiple; unavailable metadata remains unknown.
Estimate empirical repeat noise and report single seed/checkpoint limitations.
Use trained paired serial TRAIN8 plus largest-pair cost, two warmups/three
readings. No trained Pareto claim from an untrained feasibility probe.

## 5. GPU Allocation Plan
| Server | GPU | Assignment | Condition |
|---|---|---|---|
| server-2 | 0 | Y41 frozen review, then Y43 diagnostics; training only after admission | live idle occupation check and exclusive owned diagnostic lease |
| server-2 | 1 | original Y42 training | preserve PID91215/config/source/checkpoints |

No server6, other project, scheduler edits, nested execution or new task.
Admission target is one candidate; multiple tests do not enlarge that budget.

## 6. Implementation Checklist
- [x] Stable diagnosis and first discriminating CPU test saved.
- [x] Independent W2-derived six-file model and unique YAML created locally.
- [x] All32 statistic; no channel subset, neighbors, validation search or arbitrary scalar.
- [ ] Nonnegative actual extraction/readout CPU confirmation.
- [ ] Remote byte identity/no cross-variant imports/whole-parent AST after stripping new path.
- [ ] Native TRAIN controls and adverse-loss investigation.
- [ ] Native two-step backward/restoration/disabled-parent equality.
- [ ] Paired native time/memory and parameter budget.
- [ ] Six-gate review; commit exactly seven tested files, bind goal/decision/evidence.
- [ ] Canonical GPU0-only admission/dispatch; verify increasing original training steps.

Unchecked items are pending, never automatically converted to PASS.

## 7. Risk Assessment
Moment magnitude and small-statistic derivatives may create numerical or
learning problems; zero, reflection and native finite-gradient controls are
required. Information may already be indirectly encoded upstream or not help
the task. Initial confidence may worsen loss; preserve all per-scene adverse
and wrong-control outcomes. Structural similarity to Y39/Y42 readout
aggregation is explicit, not disguised as novelty. Prospective budget:
added parameters<1% W2, native pressure reserved<24GiB, no OOM/nonfinite;
report time regression and check trained lightweight Pareto after completion.
Original runs and checkpoints remain immutable. Failures are kept under the
same ID and diagnosed; no renamed retries or lowered gates.

## 8. Smell Check
Potential smell2/3: repeated readout architecture and nonlinear statistic map.
The intervention is not a normalization/scaling sweep: two nonnegative
point distributions can have exactly identical means/full covariance and
different third moments. A new raw-point-to-confidence information path
changes a quantity that those explicit summaries cannot recover. Aggregation
uses all32 channels and inherited dimensions with no value search. Causal
hypothesis: missing distribution-asymmetry evidence may improve predicted-mask
confidence; controlled native/terminal tests can falsify it. This supports a
testable Phase1 candidate, not a publication-level contribution or efficacy.
