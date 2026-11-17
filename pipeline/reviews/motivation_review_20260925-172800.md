# Motivation Review - AdaRES_project

2026-09-25 17:28 Asia/Shanghai

## 1. Goal
"AdaRES: improve a faithful standard-indicator 3D referring expression
segmentation baseline by learning query confidence from metric-aligned
evidence, targeting ScanRefer mIoU >= 0.515." B1 0.5020; B4-E30 0.5094;
gap 0.0056. All constraints and bans in `pipeline/goal.json` apply.

## 2. Story Status
| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| Competition-aware confidence | Partial | B4-E30 0.5094 | Below target |
| Indicator label alignment | G1 training | Fixed mismatch and native gates | Trained outcome pending |
| Late query access | H5 diagnosis | 215/240 error pairs outside each other's final four nearest | Causality untested |

## 3. Experiment Assessment
| ID | mIoU | Contribution |
|---|---:|---|
| B1 / B4-E30 | 0.5020 / 0.5094 | Baseline and champion |
| C1-E30 / C2-E30 | 0.5008 / 0.5014 | Auxiliary ranking losses underperformed |
| D1 / F4 | 0.5080 / 0.5060 | Pair context and spatial reanchoring closed |
| G1 | pending | Distinct indicator supervision route |

## 4. Gap Analysis
The original decoder restricts query self-attention from 128 nearest at
the first layer to four at the final layer. Most B4 winner/oracle errors
on the bounded subset are spatially disjoint and not mutual final neighbors.
This does not establish that a late global message will improve selection.

## 5. Design Rationale
H5 adds one all-query message at the last layer, leaving original local
attention intact. It tests late information access, independent of G1
label semantics. H1 sampler omission had zero available-but-dropped target
seeds; H2 fixed per-query shortlist reduced coverage; H3 standalone score
head underperformed; H4 relation cohort did not enrich errors. None of
those diagnostic routes is admitted. H5's native backward and matched
resource checks pass.

## 6. Next Batch
Admit H5 on server-2 GPU1 only after committed-byte sync and receipt.
See `pipeline/designs/exp_design_20260925-172800.md`.
