# Motivation Review - AdaRES_project

2026-09-26 21:09 Asia/Shanghai

## 1. Goal
AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515. B1 0.5020; B4-E30 0.5094; gap 0.0056. Goal constraints and bans checked: only server-2 GPUs 0/1, one independent task per GPU, released SPFormer sparse-UNet backbone/project decoder, fixed B2/B4 continuation, report mIoU/Acc@0.25/Acc@0.5/active queries/latency/peak memory, no validation tuning or banned server/checkpoint/weight scans.

## 2. Story Status
| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| Query competition confidence | Partial | B4-E30 0.5094 | Below Phase 1 target |
| Metric-IoU targets | Negative | G1 0.4980 | No weight/label scan |
| Token attention concentration | Weak/non-specific | Q1 oracle higher 260/388 train errors, concentration-only 0.08395 | Target-token control only 206/388 vs rotated 160/388 |
| Scene attention support | Negative | Q2 oracle higher 70/388 | Opposite confidence direction |

## 3. Experiment Assessment
B1 0.5020, B4-E30 0.5094, D1 0.5080, F4 0.5060, H5 0.5061, G1 0.4980. Q1/Q2 are diagnostic inference probes, not trained results. Earlier J1 pooled text-mask contrast and P1 topology failed their specific directions.

## 4. Gap Analysis
The 64-scene training cohort has selected mIoU 0.82256 and many missed high-IoU candidates. Neither raw token-attention concentration nor attention-mask agreement identifies the missed candidate with adequate specificity. Active queries and historical full-run latency/peak memory remain unmeasured.

## 5. Design Rationale
Q1 tested a token-level information path that B4 computes but does not explicitly pass to its confidence head; Q2 tested a distinct query-to-scene interaction. Their paired controls rule out admitting these mechanisms on present evidence. A future candidate needs a new inference-available, metric-aligned mechanism with frozen native paired controls and six hard gates.

## 6. Next Batch
No admission. See `pipeline/designs/exp_design_20260926-210900.md`.
