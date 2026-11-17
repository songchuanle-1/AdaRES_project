# Motivation Review - AdaRES_project

2026-09-25 17:01 Asia/Shanghai

## 1. Goal
AdaRES: improve a faithful standard-indicator 3D referring expression
segmentation baseline by learning query confidence from metric-aligned
evidence, targeting ScanRefer mIoU >= 0.515. B1 0.5020; B4-E30 0.5094;
gap 0.0056. Goal constraints and bans in `pipeline/goal.json` apply.

## 2. Story Status
| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| Competition-aware confidence | Partial | B4-E30 0.5094 | Still below target; no causal isolation |
| Full-query pair comparison | Closed | D1 0.5080 | Learned rival weights did not beat uniform subset control |
| Mask-centric reanchoring | Closed | F4 0.5060 | Proxy coverage did not generalize to mIoU |
| Indicator supervision aligns with mask quality | Diagnosis | 37/240 material errors with seed-positive winner and seed-negative higher-IoU query | Native training efficacy unknown |

## 3. Experiment Assessment
| Experiment | mIoU | Relationship to motivation | Contribution |
|---|---:|---|---|
| B1 | 0.5020 | Faithful standard indicator | Baseline |
| B4-E30 | 0.5094 | Overlap competition context | Champion |
| C1-E30 / C2-E30 | 0.5008 / 0.5014 | Additional ranking losses | No gain |
| D1 / F4 | 0.5080 / 0.5060 | Pair context / spatial retrieval | Closed |

## 4. Gap Analysis
B4's indicator target is seed/object membership, while final ScanRefer
selection is evaluated by the predicted mask against the referred target.
The fixed first 16 validation scenes show a mismatch but do not estimate
population frequency or establish that corrected supervision trains better.

## 5. Design Rationale
G1 directly replaces indicator target semantics with the detached predicted
mask IoU against training ground truth. Unlike C1/C2, it does not add a
listwise or margin loss atop the seed-label objective. A gain would support
metric-aligned supervision; a loss would bound the stability of this target.
The one-run fresh 30-epoch test compares to B4-E30 with a known continuation
protocol limitation. No independent second mechanism is yet gate-qualified.

## 6. Next Batch
G1 only if all six gates and receipt pass. The second empty slot remains
unfilled unless an independent evidence-backed route passes its own gates.
See `pipeline/designs/exp_design_20260925-170100.md`.
