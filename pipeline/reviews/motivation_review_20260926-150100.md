# Motivation Review - AdaRES_project

2026-09-26 15:01 Asia/Shanghai

## 1. Goal
AdaRES seeks to improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515. B1 0.5020, B4-E30 champion 0.5094, gap 0.0056. Seven goal constraints and three bans remain in force.

## 2. Story Status
| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| Competition-aware confidence | Partial gain | B4-E30 0.5094 | Target not reached; continuation control is not a fresh matched run |
| Metric-aligned indicator targets | Training | G1 29/30, best 0.4983 at preflight | Terminal metric and error profile pending |
| Rank or area shortcut | Disconfirmed as stated | K1/K2 fixed paired native screens | No justified intervention |

## 3. Experiment Assessment
B1 is 0.5020, B2-E30 0.5058, B4-E30 0.5094, D1 0.5080, F4 0.5060, H5 0.5061; prior closed routes B3, B5, C1/C2 did not reach champion. J1/J2 and K1/K2 do not pass diagnosis. G1 is still running and is not judged by an interim best.

## 4. Gap Analysis
The fixed 16-scene oracle 0.87316 versus selected 0.49186 leaves confidence-ordering headroom, but simple rank, mask-area, overlap, pooled-language and earlier structural changes do not provide a validated mechanism for recovering it. Full-run efficiency metrics and matched-control noise are incomplete.

## 5. Design Rationale
K1 checked whether sampler ordering creates a head shortcut; K2 checked whether mask scale dominates quality mistakes. Their paired counts are weak or contradictory, while standalone controls are worse. Neither advances to a model variant; no validation feature or weight search is justified. A new route requires independent evidence, matched control, backward and resource gates.

## 6. Next Batch
No admission (budget one). Keep GPU1 vacant; resume on G1 terminal outcome or new specific evidence for a structurally independent confidence mechanism. See `pipeline/designs/exp_design_20260926-150100.md`.
