# Motivation Review: AdaRES-Y42

## 1. Goal
```json
{
  "metric": {
    "name": "mIoU",
    "target": 0.515,
    "dataset": "ScanRefer"
  },
  "baseline_experiment_id": "AdaRES-B1",
  "description": "AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515.",
  "constraints": [
    "Use server-2 GPUs 0 and 1 only; never use server-6.",
    "All experiments launched after AdaRES-B3 must use exactly one GPU; run at most one independent experiment on GPU 0 and one on GPU 1.",
    "Do not design or queue a successor while AdaRES-B3 is running; after B3 reaches a terminal state, admit at most two structurally independent single-GPU candidates.",
    "Use a fixed 30-epoch protocol for the B2 and B4 extension. Resume both from their epoch-20 last checkpoints without resetting the optimizer or LR scheduler; do not tune the epoch count on validation.",
    "Train from the released SPFormer sparse-UNet backbone; the decoder is authored in this project.",
    "Report mIoU, Acc@0.25, Acc@0.5, active queries, latency, and peak memory.",
    "Do not tune thresholds, temperatures, loss weights, feature subsets, or the fixed 30-epoch protocol on ScanRefer validation."
  ],
  "banned": [
    "Do not use server-6.",
    "Do not report the released b005/CFNet checkpoints as AdaRES-B0 results.",
    "Do not scan the ranking margin or the occupancy loss weight on validation."
  ],
  "pipeline_policy": {
    "research_on_capacity_gap": true,
    "steady_ticks_before_design": 1,
    "fill_to_capacity": true,
    "design_buffer": 0,
    "max_candidates_per_design_tick": 2
  }
}
```
Target .515; rounded CSV Y36 .5110/gap .004 is not promotion. W2 retained.

## 2. Story Status
| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| Metric-aligned confidence improves selection | unconfirmed | prior frozen W2 comparisons | trained paired uncertainty |
| Diagonal variance captures joint evidence | incomplete | actual Y42 CPU32 W2/Y39 output delta0, covariance delta2 | native occurrence/gradient/cost |

## 3. Experiment Assessment
| Experiment | mIoU | Relation to motivation | Contribution |
|---|---|---|---|
| Y37 | frozen .5050610 | geometry confidence | inconclusive versus W2; closed |
| Y39 | frozen FIRST .5077059832 | marginal point heterogeneity | complete_inconclusive versus W2; no promotion |
| Y41 | training, raw .4697 at preflight7/30 | probability boundary evidence | preserve original GPU0 run |

## 4. Gap Analysis
All channel marginals can be identical while cross-channel co-occurrence differs. Actual readouts were tested, not conceptually assumed blind. Synthetic controls do not prove sparse-backbone blindness, semantic utility or necessity. Native TRAIN8/init4 responses and finite gradients establish executable realization only. Y39-W2 delta -.0023038668/CI[-.0113603903,.0063383101] remains inconclusive. Keep W2.

## 5. Design Rationale
Y42 adds the complete symmetric covariance statistic, with no channel subset, radius or temperature choice. Native wrong-covariance controls retain diagonal entries, with an additional preregistered PSD reflection. Fixed samples, disabled-path equality, original native two-step entry/resume and paired cost determine feasibility. Released initialization raises mean loss .0183003; two new-only steps learn for both correct and incorrect inputs, not necessity. All adverse/incorrect-control and failed adapter outcomes remain visible; no training-before-gate or diagnostic weights deployed.

## 6. Next Batch
Only Y42, budget1. Seven-section formal design and smell review before any admission. Keep seed1999/30epochs/released backbone/optimizer/loss/native128/one argmax and posttraining FIRST whole-scene protocol. No novelty claim or Phase2.
