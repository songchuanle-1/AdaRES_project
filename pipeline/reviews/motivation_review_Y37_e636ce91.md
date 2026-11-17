# Motivation Review - AdaRES_project
Tick e636ce91a12e45488093b07fa2fc092b. Autonomous, no human judgment implied.

## 1. Goal
Complete unchanged goal.json, SHA53da862754afe00739a75c5ea6b1f184e0358813225b8f8c0c024a355737c966:
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
Target .515; historical W2 .5100/gap .0050; Phase1 unmet/Phase2 unapproved.

## 2. Story Status
| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| W2 competition confidence | champion, not causal proof | frozen full prior W2 .510007498 | target unmet |
| Word-evidence confidence Y35 | native training done/full primary .504801214 | current two-pass9508/141 | fresh paired harm/cost under evaluation |
| Local-adjacency Y36 | training preserved | current snapshot22/30/rounded .5099 | trained frozen results unknown |
| Global pair-distribution Y37 | CPU information separation | current CPU32/equal moment test | all native hard gates pending |

## 3. Experiment Assessment
| Experiment | mIoU | Motivation relation | Contribution |
|---|---|---|---|
| B1 | prior full .502111562 | faithful standard indicator | baseline, not today's comparator yet |
| W2 | prior full .510007498 | learned competition | current champion, fresh repeated comparison pending |
| Y33 | prior full .505137821 | visual token attention | small/W2 harm already rejected; do not reopen |
| Y35 | today's first .504801214/repeat .504802891 | FG/BG to word attention | target unmet, scientific disposition pending paired intervals/cost |
| Y31 | prior full .505514618 | covariance confidence | inconclusive, no promotion; not a blanket geometry veto |

## 4. Gap Analysis
Current final-head competition/covariance does not explicitly compute weighted
global pair-distance distributions. Upstream spatial attention can encode
related information; no global blindness is asserted. Y37's CPU counterexample
distinguishes its direct readout from actual W2/Y31/Y36 classes on one exact
equal-moment fixture. The uniform masks mean scene shape participates; native
mask-specific incidence and intervention controls are still mandatory.
GT oracle quality is diagnosis only, never an inference feature.

## 5. Design Rationale
Y37 tests whether final-support distribution supplies useful confidence
information beyond first/second moments and nearest-neighbor features.
Fixed Laplace orders1/2/3 are conservative structural choices, not a proven
optimal basis. No validation, numeric, radius or feature selection occurs.
Structural feasibility supports falsifiable training only; trained .515/harm/
paired cost decide benefit. A negative result closes this realization, not all
geometric evidence routes. No external novelty search was performed this tick;
novelty and semantic necessity remain unknown.

## 6. Next Batch
One admission maximum, only after six gates. See
pipeline/designs/exp_design_Y37_e636ce91.md. Preserve Y36GPU1.
