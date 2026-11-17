# Motivation Review: AdaRES-Y44

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

Target.515/Phase1unmet/Phase2unapproved/W2retained. Explicit human GPU0/design
authorization applies only to AdaRES/server2 allowed0,1; preserveY43GPU0.

## 2. Story Status
| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| Explicit feature marginal moments help confidence | unconfirmed | Y39/Y41 frozen reviews; Y42 terminal | trained paired benefit |
| Marginal summaries omit coordinate-feature pairing | controlled witness | 32seed actualW2/Y39/Y42/Y43 invariant0; crossgap>=9 | actual new class/native incidence |
| Joint association improves task selection | untested | stableY44 preregistered before review | native gates and fixed30 terminal test |

## 3. Experiment Assessment
| Experiment | mIoU | Motivation Relation | Contribution |
|---|---|---|---|
| W2 | historical frozen around.510 | point-mass confidence | retained scientific comparator |
| Y39 | frozenFIRST.50770598 | feature heterogeneity | complete_inconclusive |
| Y41 | frozenFIRST.50904712 | boundary confidence | complete_inconclusive |
| Y42 | metadata.50498241; frozen pending | full feature covariance | complete30, not trained superiority |
| Y43 | current structuredunknown; training | feature asymmetry | originalGPU0 retained; not efficacy |

## 4. Gap Analysis
Complete feature multisets and fixed geometry can have different pairings.
Existing controlled extraction/readout inputs discard this explicit joint
association. A geometry-aware backbone may already encode it; upstream
blindness is not claimed. The gap is explicit input, not proven bottleneck.
Native controlled task incidence is still executable and pending.

## 5. Design Rationale
Same stableY44, physically independent W2 copy, raw all3x32 native centered
cross moments, query/point-mass FG/BG final confidence only. No arbitrary
component selection, scalar sweep, new loss or protocol change. Correct versus
withinSP wrong pairing preserves geometry, feature multiset and old summaries.
A falsifiable mechanism separates Y44 from another marginal moment variant.
No literature search this tick; novelty/necessity/human approval unknown.

## 6. Next Batch
ExactlyY44/budget1. See exp_design_Y44_ff347697.md. Native gates remain
pending until execution; no training until six-gate evidence and exact commit.
