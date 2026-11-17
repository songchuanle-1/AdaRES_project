# Motivation Review: AdaRES-Y18

## 1. Goal
Goal: "AdaRES: improve a faithful standard-indicator 3D referring expression
segmentation baseline by learning query confidence from metric-aligned
evidence, targeting ScanRefer mIoU >= 0.515."
Latest goal.json read this tick: ScanRefer mIoU .515, W2 champion .5100, gap .0050.
All seven constraints and three bans remain binding: server-2 GPU0/1 only,
one task/card, B3 terminal prerequisite, fixed30epochs, released SPFormer,
complete metric/resource reporting and no validation-derived tuning.

## 2. Story Status
| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| Point-mass competition | Historical improvement | W2 .5100 | Causal training attribution unproved |
| Cross-description consistency | Training complete, inconclusive | Y17 .50415748 vs W2 .51010516, paired CI crosses0 | No confident benefit |
| Query completion | Training | OriginalY15PID18796 | Final efficacy unknown |
| Point-measure mask supervision | Train-only diagnosis | Y18 exact expansion and train64 gradients | Full training pending |

## 3. Experiment Assessment
| Experiment | mIoU | Relation To Goal | Contribution |
|---|---|---|---|
| W2 | Historical .5100 | Native indicator parent | Current comparator |
| Y17 | Fullbest .50415748 | Same-target mask objective | Does not prove gain; generic smoothing confound retained |
| Y15 | Unknown final | Candidate-pool expansion | Keep running frozen protocol |
| G1 | Historical .4980 | Score metric-IoU target | Closed; never relaunch its score-target mechanism |

## 4. Gap Analysis
W2 native matched mask BCE/Dice use equal superpoint measure while evaluation
uses original points. Native supervision masks come from superpoint instance
labels. The intervention is mask optimization, not the closed G1 score-IoU
target, a point-mass score transformation, or Y16 representational ceiling.
Representational ceiling .999584 does not rule out optimization mismatch.
No validation outputs were used to tune candidate, coefficients or sample set.

## 5. Design Rationale
Train64 first pass: exact point-loss gradient alignment .914328 versus native
.778066; second pass .914329/.778047;60scene improvements/4declines both passes,
12adverse expressions retained. All256expressions exhibit nonzero gradient
separation. Mass-only .918234 is slightly higher; occupancy-only .774834 and
equal-size .776060 separate the major mass pathway from smoothing alone.
These local derivatives cannot forecast learned mIoU. No claim of novelty,
confidence-learning necessity, population efficacy, or universal noise bound.

## 6. Next Batch
Stable AdaRES-Y18 physical point-measure package; prospective gates and
post-training rules in exp_design_Y18_20261001.md. Diagnose native loss,
backward, stress, resume and unchanged inference before receipt/dispatch.
No causal-control duplicates admitted merely to occupy the free card.
