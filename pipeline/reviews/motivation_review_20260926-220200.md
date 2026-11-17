# Motivation Review - AdaRES_project

2026-09-26 22:02 Asia/Shanghai

## 1. Goal
AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515. B1 0.5020; B4-E30 0.5094; gap 0.0056. Goal constraints and bans checked: server-2 GPU0/1 only, one task per GPU, released backbone and project decoder, fixed B2/B4 continuation, full metric/resource reporting, no validation tuning.

## 2. Story Status
| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| Query competition | Partial | B4-E30 0.5094 | Below target |
| Metric-IoU relabeling | Negative | G1 0.4980 | Do not repeat |
| Matched-positive selection | Observational | R1 training positive-restricted oracle 0.89203 | Oracle label unavailable at inference |
| Peer confidence | Negative | R2 frozen product 0.79612 vs B4 0.82254 | Strict repeats unstable |

## 3. Experiment Assessment
B1 0.5020, B4-E30 0.5094, D1 0.5080, F4 0.5060, H5 0.5061, G1 0.4980. R1 and R2 are native read-only probes, not trained experiments.

## 4. Gap Analysis
Training positive labels and mask quality disagree in both directions. A label-free, inference-available discriminator has not been demonstrated. Peer confidence alone has error association but degrades frozen selection, and the previous attention/topology/appearance routes remain rejected. Active queries, full latency and training peak memory are incomplete.

## 5. Design Rationale
R1 audited whether the standard supervision ceiling explains the mask/selection gap; it does not by itself specify a deployable confidence path. R2 tested a structurally distinct confidence communication rule with frozen nulls and was disconfirmed. Any future independent positive-evidence mechanism must be predeclared and pass native controls, backward, resources, independence and constraints.

## 6. Next Batch
No admission. See `pipeline/designs/exp_design_20260926-220200.md`.
