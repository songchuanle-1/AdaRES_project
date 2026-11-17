# Motivation Review - AdaRES_project
Autonomous evidence review, not human approval or backdated preregistration.

## 1. Goal
Complete unchanged goal:
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
Target.515; automatic rounded championY36.5110/gap.0040; scientific W2 retained.

## 2. Story Status
| Hypothesis | State | Evidence | Gap |
|---|---|---|---|
| Y37 global distance confidence | complete_inconclusive | full.505061022921; paired intervals cross0 | target unmet |
| Y40 missing direct word/support path | rejected realization | actualY35 CPU32 coverage | not a ban on language |
| Y41 boundary probability mass | native validation | equal-message CPU and TRAIN8/init4 | trained utility unknown |
| Y39 within-SP feature variance | training preserved | original102267/GPU1 | efficacy unknown |

## 3. Experiment Assessment
| Experiment | mIoU | Relation to motivation | Contribution |
|---|---|---|---|
| W2 | .510107995189 | metric-aligned query competition | current frozen comparator |
| B1 | .502119335285 | faithful standard indicator | full baseline comparator |
| Y37 | .505061022921 | global pair geometry confidence | no confirmed benefit |
| Y36 | prior.510716624292 | normalized local edge features | W2 superiority inconclusive |

## 4. Gap Analysis
Y36's nonempty independently normalized boundary/interior visual pools can
discard probability-weight totals. CPU keeps their visual messages equal
while new boundary descriptors separate; native controls show the new path
is reached. Existing learned features may indirectly encode these quantities.
No population-wide confidence bottleneck, upstream blindness or necessity
has been proven.

## 5. Design Rationale
Test explicit probability flow across nearest spatial edges plus normalized
Dirichlet energy at the final head, not a neighbor-radius/value search.
Native losses initially rise {"trained": {"enabled": 0.0005547478795051575, "wrong_masks": 0.0005493611097335815, "wrong_pairs": 0.0006641075015068054}, "released_init": {"enabled": 0.0022946596145629883, "wrong_masks": 0.0022867918014526367, "wrong_pairs": 0.0031055212020874023}}.
Fixed two new-path-only AdamW steps make mean losses decrease in both correct
and wrong-input conditions; two of eight correct TRAIN scenes rise.
These adverse/negative observations are retained. Learnability is not
necessity or efficacy; diagnostic weights are not deployed. Both failed CPU
controls are retained: finite roundoff and hard alternating empty interior.
The nonempty arithmetic control changes no gate tolerance. No literature
search or novelty claim; peer-review contribution remains unknown.

## 6. Next Batch
See pipeline/designs/exp_design_Y41_5e172bd4.md; only one fixed30 single-GPU candidate after six real gates
and an exact tested seven-file commit. Preserve Y39/GPU1.
