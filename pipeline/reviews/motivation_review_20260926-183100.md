# Motivation Review - AdaRES_project

2026-09-26 18:31 Asia/Shanghai

## 1. Goal
Goal: "AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515." Baseline B1 0.5020, champion B4-E30 0.5094, target gap 0.0056. Constraints: server-2 GPUs 0/1 only, one independent experiment per GPU, B2/B4 continuation protocol fixed, released SPFormer backbone and project decoder, report mIoU/Acc@0.25/Acc@0.5/active queries/latency/peak memory, no validation tuning of thresholds, temperatures, weights, feature sets or epochs. Bans: no server-6, no released b005/CFNet as B0, no ranking-margin or occupancy-loss-weight scan.

## 2. Story Status
| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| Competition-aware query confidence | Partial | B4-E30 0.5094 | Below target; no fresh matched control |
| Metric-IoU target | Negative | G1 0.4980; paired mask and selection quality decline | No G1 successor without independent evidence |
| Gradient interference | Mixed | N1 cosines -0.868 to +0.194 in trained batches | No repeatable functional benefit from detach-only |
| Cross-expression agreement | Proxy failure | N2 oracle-group inconsistency, language nearest same-object 27.4% | Inference-valid grouping |
| Distinct-target collisions | Sparse | N3 6/365 native pairs | Insufficient occurrence for collision-only loss |

## 3. Experiment Assessment
B1 0.5020, B4-E30 0.5094; B2-E30 0.5058, D1 0.5080, F4 0.5060, H5 0.5061 and G1 0.4980 do not reach the target. G1 fell 0.0114 below B4-E30. Previous E/H/J/K/L/M diagnoses closed their specific routes; the new N1/N2/N3 screens do not reopen them.

## 4. Gap Analysis
The first 16 validation scenes expose substantial oracle-versus-selected headroom, but oracle selection is unavailable at inference. No reproducible, label-free new confidence signal with matched native control and all six hard gates was established this tick. Active queries, full-run latency/peak memory and evaluation noise remain unquantified.

## 5. Design Rationale
N1 tested whether confidence gradients impede mask learning; N2 tested whether a second expression could identify a referent consistently; N3 tested whether distinct-target co-occurrence gave a native exclusivity signal. Each addresses a different causal path. Their bounded tests failed to yield an admissible mechanism; a loss weight, similarity threshold, detach-only ablation or renamed fusion would not repair the missing evidence.

## 6. Next Batch
No training admission. See `pipeline/designs/exp_design_20260926-183100.md`. Resume at the next research window with independent prospective evidence and a concrete native diagnosis/control, while maintaining the six admission gates.
