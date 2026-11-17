# Motivation Review - AdaRES_project

2026-09-26 23:57 Asia/Shanghai

## 1. Goal
AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515. B1 0.5020; B4-E30 0.5094; target gap 0.0056. Constraints/bans checked: server-2 GPU0/1 only, one task per GPU, released backbone/project decoder, fixed B2/B4 continuation, required metric/resource reporting, no validation scans.

## 2. Story Status
| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| Query competition | Partial | B4-E30 0.5094 | Target unmet |
| Competitor influence | Negative | T1 fixed readout 0.82133 vs B4 0.82256 on train64 | No selector benefit |
| Local 3D support | Sensitivity only | T2 top-two 0.81370 vs B4 0.82256 and outside 0.82307 | No control-beating selector |

## 3. Experiment Assessment
B1 0.5020, B4-E30 0.5094, D1 0.5080, F4 0.5060, H5 0.5061, G1 0.4980. T1/T2 are native read-only probes, not trained experiments or validation evidence.

## 4. Gap Analysis
Selected-versus-oracle headroom remains, but neither competition counterfactual nor localized 3D reliance improves frozen selection. Active queries, full latency and training peak memory are unmeasured for these probes.

## 5. Design Rationale
T1 tests whether an overlapping competitor suppresses a missed oracle, with distant-peer removal as control. T2 tests local 3D support against equal-count outside-mask permutation. Failure narrows these exact mechanisms without legitimizing numerical scans.

## 6. Next Batch
No admission. See `pipeline/designs/exp_design_20260926-235700.md`; resume only on an independent inference-available metric-aligned mechanism with all six gates.
