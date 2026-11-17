# Experiment Design: AdaRES-Y42

Status: FROZEN phase-1 training protocol; feasibility gates recorded separately,
not human approval, novelty or trained efficacy.
Tick: 062ae665ba824deba180a65efb4c44cf.
Upstream: pipeline/reviews/motivation_review_Y42_062ae665.md.

## 1. Motivation
Goal description: "AdaRES: improve a faithful standard-indicator 3D referring
expression segmentation baseline by learning query confidence from
metric-aligned evidence, targeting ScanRefer mIoU >= 0.515."

Preflight active1/target2/budget1. Y39 is canonical terminal and Y41 stays
on original GPU0. Y42 tests joint feature co-occurrence absent from equal
means and diagonal variances. This is a narrow readout information hypothesis,
not a claim of backbone blindness, new scientific novelty or generic benefit.

## 2. Current Best Results
| Experiment | State | mIoU | Acc50 | spmIoU | Epochs |
|---|---|---|---|---|---|
| Y36 | scientifically inconclusive | frozen .5107166, historical | unknown here | unknown here | 30 |
| Y37 | scientifically inconclusive | frozen .5050610, historical | .5610013 | .4684438 | 30 |
| Y39 | complete_inconclusive, full frozen review | FIRST .5077059832 | .5612116113 | .4717230174 | 30, best23 |
| Y41 | training | raw preflight .4697, not terminal efficacy | unknown | unknown | raw7/30 |

W2 is the scientific comparator; CSV Y36 .5110 is not promotion.

## 3. Analysis And Direction
No prior covariance-only intervention is assumed covered merely because Y39
uses diagonal point variance. CPU32 actual W2/Y39 matched outputs differ0
while covariance differs2. Mean, diagonal variance, masks, counts, queries
and geometry are held fixed; repeats/permutations differ0. Native TRAIN8/init4
controls respond to wrong masks, off-diagonal covariance and PSD-preserving
reflection with every marginal retained. All 8 new tensors have finite
nonzero gradients. This is a readout feasibility result, not semantic benefit.
Y39-W2 FIRST delta -.0023038668/scene CI[-.0113603903,.0063383101]:
inconclusive, not equivalent, no promotion. Single/multiple object-count
subgroups do not imply official unique/multiple metadata; the latter is unknown.

If trained Y42 remains uncertain or harms a preregistered group, retain W2
and close this same stable ID with evidence, not a renamed numeric variant.
Phase2 ablations are unapproved and not part of this proposal.

## 4. Experiment Plan
| ID | Config | Hypothesis | Expected Mechanism | Dependency |
|---|---|---|---|---|
| AdaRES-Y42 | configs_song/model_paper5/adares_y42_point_covariance_res.yaml | Complete point covariance carries joint evidence omitted by marginal variance | Matched covariance-only separation and nonzero native gradients; efficacy unknown | Six gates, then original fixed30 |

Every native32 channel pair is used once through the symmetric upper triangle,
528 entries including the diagonal. Stream covariance rows per superpoint,
encode signed log1p covariance with528->256->256, pool by detached final mask
point mass, and combine query/foreground-minus-background/midpoint through
768->256->2. Add only to final indicator. Parent paths/criterion unchanged.
All dimensions derive from32 native input/256 parent decoder, no subset,
temperature, scalar weight, neighbor setting or validation search.

Controls: actual W2/Y39 equal-marginal synthetic pair; native diagonal-only
and rolled off-diagonal covariance while preserving every diagonal; fixed
alternating-sign PSD reflection C->DCD frozen before native measurement; wrong
masks; independent repeats; point/support/query permutations; padded poison;
disabled physical W2 equality. TRAIN first8 distinct scenes/init4, native
two steps/resume, maxpair728/6780 are frozen before reading outcomes.

Training: released SPFormer, seed1999, exactly30 epochs, unchanged native
optimizer/scheduler/loss, single GPU,128 queries/one argmax. Diagnostic weights
are discarded, not used as a starting point.

Posttraining: full ScanRefer9508 expressions/141 scenes, Y42/W2/B1 each
twice, FIRST primary, scene bootstrap10000 seed1999, TRAIN cuts1487/3325.
Harm if any preregistered small/medium/large or single/multi group has a
strictly negative CI upper bound versus either W2 or B1, as in the prospective
resource/protocol freeze; crossing is inconclusive. Absent multi-object
population is insufficient data, not a passed no-harm gate.
Estimate repeat noise including per-expression differences; uniqueness
metadata unknown. No unobserved no-harm or target declaration.

## 5. GPU Allocation Plan
| Server | GPU | Assignment | Justification |
|---|---|---|---|
| server-2 | 0 | Keep original Y41 PID116906 | Authorized, ongoing fixed30 |
| server-2 | 1 | Exclusive diagnostics, then Y42 only if admitted | Live idle/lease check before each use |

No server6, other project, scheduler, new task, nested Codex or background
decision process. A slot opening does not grant admission.

## 6. Implementation Checklist
- Isolated six-Python-file package and unique YAML generated from W2.
- No cross-variant imports; same-interpreter complete dec/ipdn parent AST
  identity after removal of only the new path; structured config differences[].
- All new code formatted LF before native tests; old evidence preserved.
- Actual extraction/readout CPU32, native TRAIN8/init4 controls, physical
  tools.train two updates and exact model/optimizer/scheduler restore passed.
  Next-step numerical model1.5258789e-5/optimizer2.3625034e-8 retained.
  Paired resource measurements and final gate review recorded separately.
- Exact seven tested files only to be committed, then canonical additive
  marker sync and hash-bound six-gate receipt.
- Canonical dispatch and independently increasing native steps required.

## 7. Risk Assessment
Joint covariance costs more extraction operations and can consume additional
autograd memory. Freeze +params<1 percent and reserved<24GiB/no OOM/nonfinite,
report regressions rather than asserting initialized Pareto improvement.
Trained paired latency/memory plus metric are required for Pareto claims.
Native initial adverse loss+.0183003 remains. Fixed two new-only native AdamW
steps reduced correct released-init loss .0422233 but wrong masks/covariance
also learn similarly. Individual losses can rise. No evidence of necessity
or generalization; no diagnostic weight deployment or setting optimization.
Repeated evaluation can be unstable, and correct/wrong paths can both learn.
Single-seed/checkpoint/split and unknown uniqueness limit efficacy claims.
No crash retry or original-checkpoint overwrite is authorized here.

## 8. Smell Check
1. Added planned398594 parameters, but parameter count alone is not structure.
2. This is not an isolated Linear/norm/gate insertion: it creates point-level
   cross-channel aggregation not recoverable from the frozen marginal pair.
3. Signed log1p handles covariance signs; scaling is not the intervention.
4. No numeric sweep or deferred hyperparameter selection.
5. Same stable Y42; not a renamed Y39: native wrong controls preserve marginals.
6. Tested information co-occurrence is the causal hypothesis, not a better value.
7. New graph, native-derived dimensions, explicit matched controls satisfy
   the prospective structural criteria; six-gate admission receipt requires
   all completed hashed evidence, including paired resources.
