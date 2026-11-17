# Motivation Review - AdaRES_project

2026-09-28 11:31 Asia/Shanghai

## 1. Goal
The `goal.json` description asks for metric-aligned query confidence on
ScanRefer, mIoU >= 0.515. Constraints: server-2 GPU0/1 only, one GPU per
task, released SPFormer sparse-UNet backbone and project decoder, no
validation tuning of thresholds, temperatures, loss weights, feature subsets
or the fixed 30-epoch protocol. B1 baseline 0.5020; W2 champion 0.5100;
gap 0.0050.

## 2. Story Status
| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| Query competition captures useful metric evidence | Tentative | W2 best 0.5100 vs B4-E30 0.5094 | The +0.0006 difference is not a controlled significance claim. |
| Correctly matched competitor context is necessary | Not supported by frozen test | X1 zero context 0.76579 vs native 0.76736; rolled context 0.77347 on train64 | One repeat mismatch; train64 is not validation. |
| Directional containment remedies selection | Not supported by frozen test | X2 containment 0.76608 vs native 0.76733 on train64 | Native retraining has not been justified. |

## 3. Experiment Assessment
| Experiment | mIoU | Relation to motivation | Contribution |
|---|---:|---|---|
| AdaRES-B1 | 0.5020 | Faithful standard indicator | Baseline. |
| AdaRES-B4-E30 | 0.5094 | Superpoint-uniform query competition | Reference, continued optimizer from epoch 20. |
| AdaRES-W2 | 0.5100 | Point-count-weighted query competition | Slight best observed result, 30/30; not proof of causal benefit. |
| X1 / X2 fixed-cohort probes | Not validation mIoU | Test necessity and direction of W2 evidence | Both direct replacements fail their matched diagnosis. |

## 4. Gap Analysis
W2's error oracle often lies outside its top three (367/466 material
selection errors on the diagnostic cohort). A mere score rescale or a frozen
edge substitution is not an independently justified corrective mechanism.
The X1 rolled-context anomaly and one repeated-forward mismatch need causal
isolation before a new context method could be proposed.

## 5. Design Rationale
X1 tests whether learned competition context is genuinely needed; X2 tests
whether symmetric overlap loses containment direction. Different graph
properties, but neither has evidence for a training-ready successor.
The next useful route must specify inference-available information, an
identifiable deficit, a falsifiable matched control, and native backward
verification. No new experiment is designed merely to occupy a card.

## 6. Next Batch
None admitted. See the hashed X1/X2 diagnosis and research work ledger;
resume on genuinely distinct evidence, not a renamed score or loss scan.
