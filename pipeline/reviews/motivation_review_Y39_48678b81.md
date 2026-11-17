# Motivation Review - AdaRES_project
This review consolidates already frozen diagnosis/realization and completed
controls. It is not retroactively described as a preregistration. Autonomous
scientific review, not a human approval.

## 1. Goal
Unchanged complete goal.json, SHA53da862754afe00739a75c5ea6b1f184e0358813225b8f8c0c024a355737c966:
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
Target .515. Automatic snapshot ranks rounded Y36 .5110, gap .0040.
Frozen scientific review retains W2: Y36 superiority interval crosses zero.
Phase1 unmet; Phase2 unapproved.

## 2. Story Status
| Hypothesis | State | Evidence | Gap |
|---|---|---|---|
| W2 point-mass competition | retained comparator | full .510009849923 | target unmet, causality not proven |
| Y36 local adjacency | complete_inconclusive | full .510716624292, paired W2 CI crosses0 | no superiority/promotion |
| Y37 global pair distribution | original training preserved | snapshot10/30, rounded .4926 | trained outcome unknown |
| Y38 missing mask history | tested rejection of realization | CPU32 actual Y30 minimum response .005445152521 | proposed missing path already implemented |
| Y39 within-SP feature heterogeneity | native path separated | frozen equal-mean CPU and TRAIN8/init4 | trained utility unknown |

## 3. Experiment Assessment
| Experiment | mIoU | Relation to motivation | Contribution |
|---|---|---|---|
| B1 | .502112417447 | faithful baseline confidence | fixed comparison baseline |
| W2 | .510009849923 | explicit competition evidence | retained trained comparator |
| Y36 | .510716624292 | local adjacency confidence | W2 difference+.000706774369, CI[-.007218154973,.008880424700]; inconclusive |
| Y35 | prior .504801213799 | language evidence confidence | inconclusive prior frozen review, no retest here |
| Y30 | not reloaded | existing explicit mask history/GRU | old medium-harm record preserved; direct new CPU coverage rejects Y38 |

## 4. Gap Analysis
Native W2 averages all32 point features per superpoint. A distribution with
identical means/counts/centroids may have different within-SP variance, while
the final head receives only pooled features and predicted masks/mass.
This is a direct-path information gap, not a claim that upstream learned
means or attention are globally blind to heterogeneity.
TRAIN64/256 exact whole-superpoint mask representability mean .999771170370,
minimum .989761092150, median1; only29 expressions with nonzero gap.
This NEGATIVE diagnosis does not support quantization as the main task
bottleneck. Y39 instead tests learned confidence from point-feature
heterogeneity; GT is never an inference input.

## 5. Design Rationale
All32 centered point-feature variance is computed before native pooling,
then final soft-mask foreground/background point-mass-weighted encoded
distributions inform the existing final indicator. This adds a nonlinear
distribution path without searching thresholds/features/loss settings.
The actual frozen class reacts to wrong SP variance association and wrong
masks, is repeatable and padding-safe, and receives native gradients.
Initial trained-W2 mean loss rises; every adverse/control loss is retained.
Fixed two-step new-path-only AdamW investigates correctability, not necessity
or validation benefit. All wrong-input controls can learn too.
No external literature/novelty search: novelty unknown. No trained efficacy,
no harm, semantic necessity or Pareto claim.

## 6. Next Batch
See pipeline/designs/exp_design_Y39_48678b81.md.
One admission at most, only after actual six-gate review and tested code commit.
