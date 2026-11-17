# Motivation Review - AdaRES_project
Observed locally: 2026-10-14T12:06:44.604524+08:00; owned token 7a40f3879ec449d1837732c6c2cd23b2.

## 1. Goal
Complete unchanged project goal:
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
Target0.515 remains unmet. Rounded ledgerY36 .5110 is not scientific promotion;
W2 remains the scientific comparator.

## 2. Story Status
| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| Y43 all-channel asymmetry | fixed30 training complete | pipeline/analyses/y43_terminal_7a40_7a40f3879ec449d1837732c6c2cd23b2.json | full frozen efficacy/harm/cost pending |
| Y45 metric-confusion decomposition | CPU target mapping only | pipeline/analyses/y45_confusion_cpu_7a40_v2_7a40f3879ec449d1837732c6c2cd23b2.json; pipeline/analyses/y45_point_targets_cpu_7a40_7a40f3879ec449d1837732c6c2cd23b2.json | native TRAIN incidence/controls pending |

## 3. Experiment Assessment
| Experiment | mIoU | Relation to motivation | Contribution |
|---|---|---|---|
| Y43 | .508529961 checkpoint metadata, not frozen evaluation | confidence from point asymmetry | no efficacy conclusion before frozen comparison |
| Y44 | snapshot training30/30, terminal unconfirmed | coordinate-feature association | preserve originalGPU1 run |
| Y45 | unknown; not trained | explicit FP/FN supervision at fixed masks | generated CPU witnesses distinguish error partitions only |

## 4. Gap Analysis
Actual W2 final selection uses matched indicator classification, not scalar
quality regression. The initial scalar-confidence premise is disconfirmed and
retained in its original preregistration. Thirty-two float32 actual-function
cases have identical classification losses/gradients and identical scalar IoU
but different [TP,FP,FN] values [4,2,0] and [6,0,3].
Thirty-two separate point-mapping cases agree with literal point counts;
joint-point/query permutation, padding and empty-mask checks pass; wrong
expression targets differ by at least3 points. These are generated CPU
fixtures with actual parent method bodies, not a captured native TRAIN batch,
not evidence of useful learnability or an admission gate PASS.

## 5. Design Rationale
The corrected stable Y45 asks whether a learned decomposition gives confidence
useful TRAIN supervision beyond matched binary labels. It does not reopen
scalar-IoU/listwise/pairwise ranking implementations, use oracle inference
inputs, tune weights/thresholds/temperatures, or add a higher-moment successor.
Native alignment and empirical task relevance must be checked before model
implementation/admission. Prospective independent-package and frozen
post-training requirements remain mandatory.

## 6. Next Batch
No admission or new training. GPU0 has unregistered PID19723, memory22940MiB;
GPU1 has originalY44 PID15520, memory20772MiB, in the12:02 local observation.
Preserve both processes. The actual GPU0 evaluation attempt was refused by
the outer idleness guard; retained failure log:
pipeline/analyses/y43_frozen_full_7a40_7a40f3879ec449d1837732c6c2cd23b2.log.
The initial float64 CPU fixture failed native float32 label assignment before
results; its original source/log are retained. Corrected v2 did not alter any
model/source/config/checkpoint.
Resume SAME Y43 frozen9508/141 twice EACH/FIRST vs W2/B1, then trained paired
cost, and SAME Y45 native TRAIN correct/wrong target capture when an authorized
GPU is actually idle and its exclusive lease can be acquired. Retry after
2026-10-14T12:36:44.604524+08:00, or sooner on resource/evidence change. No claimed route exhaustion.
