# Motivation Review - AdaRES_project

## 1. Goal

Current goal.json, SHA256
53da862754afe00739a75c5ea6b1f184e0358813225b8f8c0c024a355737c966:

```json
{
  "metric": {"name": "mIoU", "target": 0.515, "dataset": "ScanRefer"},
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

Newer user override restricts this project to GPU1; GPU0 stays reserved. Target
0.515, W2 historical champion0.5100, gap0.0050, B1 baseline0.5020.
Phase1 unmet/Phase2 unapproved. B3 terminal, capacity1/active0/budget1 at
2026-10-08 17:49:37.837464+08 preflight; Y31 now canonical done_complete.

## 2. Story Status

| Hypothesis | State | Evidence | Gap |
|---|---|---|---|
| Native confidence sometimes misses higher point-IoU queries | TRAIN diagnosis | Y33 TRAIN64/256:163 oracle gaps | Not proof that the new readout solves them |
| All-query quality supervision establishes a new information pathway | Y32 realization rejected | Actual C1 source and32 executed interventions | Unequal-measure/fallback efficacy unknown; no numerical identity claim |
| Individual token distributions contain evidence beyond FG/BG means | Y33 mechanics and TRAIN incidence | CPU equal-mean delta0.052794486; TRAIN64 wrong-token responses | Native causality, cost and trained benefit pending |

## 3. Experiment Assessment

| Experiment | mIoU | Relation To Motivation | Contribution |
|---|---|---|---|
| W2 | Historical0.5100 | Point-mass overlap competition confidence | Champion below target |
| Y31 | Frozen full0.505514618/0.505507996 | Explicit spatial covariance readout | Protocol complete; paired CIs cross0, not promoted |
| Y30 | Frozen full0.504229394/0.504314270 | Ordered query-mask history | Medium-group harm; same realization rejected |
| Y28 | Frozen full approximately0.505520 | Visual FG/BG mean readback | Small-group harm; same realization rejected, not a universal readback ban |
| Y32 | Not trained | Quality objective replacement | Source/path counterevidence; do not call trained harm |

## 4. Gap Analysis

W2's final indicator accepts query/masks/padding/point mass, not individual final
visual tokens. Its upstream queries already attend to visual content. Y28
first-moment pooling and Y33 distribution-sensitive attention are separable:
equal FG/BG means do not fix a query-conditioned token-attention readout.
Source and CPU discrimination support this distinction but do not prove causal
necessity, generalization, novelty or trained Pareto benefit.

## 5. Design Rationale

Use one stable independent candidate, final readout only, not a pooled-language
cosine ranking rule or a retry of Y28 under a new name. Learned keys interact
with the terminal query over individual tokens; log soft mask support and
point mass constrain the foreground/background aggregation. Width and scaling
follow native256 and dot-product dimension; no numeric search. TRAIN64 is
prediction-blind; no validation used for admission tuning. CPU/source and
TRAIN incidence are real, remaining gates pending. This is an autonomous
scientific review, not a claim of human approval.

## 6. Next Batch

See exp_design_Y33_cde054a6.md. Continue cached TRAIN8/init4 controls, native
two-step backward/resume, paired cost and independence checks. Investigate
unexplained adverse native losses; admit only after six real gates and the
mandatory smell review. GPU0/server6 remain untouched.
