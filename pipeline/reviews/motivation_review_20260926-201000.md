# Motivation Review - AdaRES_project

2026-09-26 20:10 Asia/Shanghai

## 1. Goal
AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515. B1 baseline 0.5020; B4-E30 champion 0.5094, gap 0.0056. Goal constraints: server-2 GPUs 0/1 only, one independent task each, released backbone and project decoder, fixed B2/B4 continuation, report mIoU/Acc@0.25/Acc@0.5/active queries/latency/peak memory, no validation tuning of thresholds, temperatures, weights, features or epochs. Bans: server-6, released checkpoint as B0, ranking-margin or occupancy-weight validation scans.

## 2. Story Status
| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| Competition-aware confidence | Partial | B4-E30 0.5094 | Below target; oracle selection headroom |
| Metric-IoU targets | Negative | G1 0.4980 | No target/weight scan |
| Spatial fragmentation evidence | Negative | P1 oracle more connected 18/107 train errors | No reliable confidence direction |
| Layer-trajectory evidence | Negative | P2 recovers 0/107 train errors | Layer swap is an ablation |

## 3. Experiment Assessment
B1 0.5020, B4-E30 0.5094, D1 0.5080, F4 0.5060, H5 0.5061, G1 0.4980. P1/P2 were native inference diagnoses, not trained experiments; previous O1/O2 were rejected as semantic-changing and global-shift interventions.

## 4. Gap Analysis
Inference-visible evidence that identifies a missed high-IoU query without oracle labels remains unverified. P1 spatial connectedness and P2 confidence depth do not resolve this gap. Active queries, latency and full-run peak memory remain unavailable for historical variants.

## 5. Design Rationale
P1 tested a topology mechanism distinct from compactness/boundary features; P2 tested whether iterative indicator refinement itself discarded a correct selection. Both used frozen training-cohort readouts and repeat controls. Neither supports a new computational confidence pathway. Their negative results narrow the bounded routes but not the overall research space.

## 6. Next Batch
No admission. See `pipeline/designs/exp_design_20260926-201000.md`. Independent future route requires metric-aligned diagnosis, matched native control, backward/resources, independence and goal compliance.
