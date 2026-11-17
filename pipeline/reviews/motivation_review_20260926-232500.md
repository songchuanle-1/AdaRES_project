# Motivation Review - AdaRES_project

2026-09-26 23:25 Asia/Shanghai

## 1. Goal
AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515. B1 0.5020; B4-E30 0.5094; gap 0.0056. Goal constraints/bans checked: server-2 GPU0/1 only, one experiment per GPU, released backbone/project decoder, fixed B2/B4 continuation, full metric/resource reporting, no validation scans.

## 2. Story Status
| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| Query competition | Partial | B4-E30 0.5094 | Below target |
| Equivariant confidence | Negative | S1 averaging 0.80927 vs B4 0.82256 on 64 training scenes | No selection benefit |
| Local 2D support | Sensitivity, no admission | S2 inside/outside disruption and top-two 0.82007 vs B4 0.82256 | No control-beating selector |

## 3. Experiment Assessment
B1 0.5020, B4-E30 0.5094, D1 0.5080, F4 0.5060, H5 0.5061, G1 0.4980. S1/S2 are native read-only probes, not trained experiments.

## 4. Gap Analysis
Selected-versus-oracle headroom persists, but neither point-order consistency nor local 2D sensitivity supplies a reliable inference-time ranking. Active queries, full latency and training peak memory remain unmeasured for prior variants.

## 5. Design Rationale
S1 tested semantic-preserving representation stability against unchanged-repeat control. S2 tested local modality reliance against equal-count outside-mask and unchanged-repeat controls. Their frozen ranking interventions failed; further tuning of averaging, mask area, top-k or feature subsets is not justified.

## 6. Next Batch
No admission. See `pipeline/designs/exp_design_20260926-232500.md`; resume only on an independently justified metric-aligned confidence mechanism with all six hard gates.
