# Motivation Review - AdaRES_project

2026-09-27 03:14 Asia/Shanghai

## 1. Goal
`goal.json` description: "AdaRES: improve a faithful standard-indicator 3D
referring expression segmentation baseline by learning query confidence from
metric-aligned evidence, targeting ScanRefer mIoU >= 0.515." Metric:
ScanRefer mIoU target 0.515; B1 baseline 0.5020, B4-E30 champion 0.5094,
gap 0.0056. Phase 1 unmet. Constraints: server-2 GPUs [0,1] only, one
independent task per card; no server-6; fixed 30 epochs; released SPFormer
sparse-UNet with project decoder; report mIoU, Acc@0.25/0.5, active queries,
latency and peak memory; no validation scanning of thresholds, temperatures,
weights, features or epoch count. B3 hold was released at its terminal state.
Forbidden: released b005/CFNet as B0 or validation scans of ranking margin /
occupancy weight.

## 2. Story Status
| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| Query competition helps standard indicator | Partial | B4-E30 0.5094 > B1 0.5020 | Still below 0.515 |
| Direct metric targets improve confidence | Negative | G1 0.4980, C1-E30 0.5008 | Do not rename a target loss |
| Point-count distribution distorts superpoint-uniform competition | Testable | W2 real-vs-rotated count contrast on fixed train64; native gradient signal | Full 30-epoch training effect unknown |

## 3. Experiment Assessment
| Experiment | mIoU / native readout | Relation to motivation | Contribution |
|---|---|---|---|
| B1 | 0.5020 | Faithful standard-indicator baseline | Fixed reference |
| B4-E30 | 0.5094 | Overlapping-query context | Current champion |
| D1 / F4 / H5 | 0.5080 / 0.5060 / 0.5061 | Alternative context routes | None surpassed B4 |
| G1 / C1-E30 | 0.4980 / 0.5008 | Direct metric targets / listwise | Negative controls |
| W1 | Prior oracle mean 0.91123 vs final 0.91035 (train64) | Layer mask recoverability | Local degradation, weak useable gap; reject direct swap |
| W2 | 205/388 oracle-margin improvements, shifted 166; fixed mIoU ~0.82256 (train64) | Point-weighted query context | Distinct input and gradient path, not proven efficacy |
| W3 | Product 0.82025 < indicator 0.82256 (train64) | Existing score factorization | Reject frozen fusion |

## 4. Gap Analysis
The B4 indicator uses each superpoint once to calculate query overlap, whereas
point mIoU broadcasts superpoint masks to variable-size point sets. The
superpoint population is available without target labels. W2 directly tests
whether aligning the overlap geometry with point mass changes trained
confidence. W1/W3 did not yield a defensible direct successor; U/V and prior
language/geometry probes remain negative. There is no claim that the W2
diagnostic predicts ScanRefer validation benefit.

## 5. Design Rationale
W2 changes the computation graph's information path by injecting observed
point counts into pairwise query interaction; it leaves B4's supervised
indicator target and model size intact. A true-vs-rotated point-count control
separates correspondence from mere count distribution. The train64 direction
is mixed (208/388 direct wins versus 180 losses), with just 6 changed top-1
queries; this is a risky but falsifiable structural test, not a tuning
opportunity. After training, improvement would motivate a fresh matched
B4 replication; no improvement would reject this precise input path.

## 6. Next Batch
One candidate, AdaRES-W2, contingent on receipt, safe additive commit/sync,
and monitored native launch. GPU1 remains unfilled until a second independent
mechanism passes the same gates. See
`pipeline/designs/exp_design_20260927-031400.md`.
