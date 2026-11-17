# Motivation Review: AdaRES-Y28
Owned tick7f5e7b9d138646e1a4458341d88f7124, 2026-10-05.
Diagnostic prereg/stableID preceded this formal review; no backdated ordering.

## 1. Goal
Canonical goal.json is unchanged:
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
  "pipeline_policy": {"research_on_capacity_gap": true, "steady_ticks_before_design": 1, "fill_to_capacity": true, "design_buffer": 0, "max_candidates_per_design_tick": 2}
}
```
Newer user/SERVERS override permits onlyGPU1. B3 alreadyterminal.
Snapshot19:49:05.863150+08: active0/target1/budget1, Y27done_complete3/3.
W2champion.5100/target.5150/gap.0050. No active research hold in recent
question/route_change decisions; Y26 ineligibility is realization-specific.

## 2. Story Status
| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| Overlap-aware confidence | Historical champion | W2.510108 | Target unmet |
| Explicit geometry readback | Complete, inconclusive | Y27full9508/141, allCIs cross0 | No proven benefit |
| Fixed mask-language cosine rank | Rejected | J1only123/240 missed queries favorable | Not retried |
| Learned mask-content readback | Diagnosis, efficacy unknown | Y28TRAIN64/256 | Native causal controls pending |

## 3. Experiment Assessment
| Experiment | mIoU | Motivation Relation | Contribution |
|---|---|---|---|
| W2 | .510107995 | Metric confidence | Champion comparator |
| B1 | .502119335 | Standard indicator | Baseline comparator |
| Y27 | .509987856 | Mask-centroid peer context | Cluster intervals cross0, cost regressions |
| Y25 | .494117094 | Training mask feedback | Realization materially harmful versusW2 |
| Y28 | Unknown | Predicted-mask visual readback | No trained result |

## 4. Gap Analysis
Exact deployed W2 AST has indicator(query,pred_masks,batch_mask,point_mass),
with no direct visual feature argument. This is conditional absence of final
mask-pooled content, NOT semantic blindness: decoder queries already receive
language and scene cross-attention. Prediction-blind native TRAIN64 has163/256
oracle selection gaps, scene-mean.154148389; all163 have distinct chosen/oracle
pooled content. Correct/rolled mean feature distance3.350130, uniform5.352034,
foreground-background5.413076; cached repeat0 and double permutation9.33e-14.
Hash4c61c736 binds raw diagnosis. These facts do not prove missing information,
causal necessity or generalization. J1's fixed cosine result remains negative.

## 5. Design Rationale
Read back final predicted soft-mask weighted, original-point-measure scene
features into each query's confidence. A shared learned foreground/background
interaction supplies a new aggregation dependency; the original overlap path
and selection remain. Dimensions256 derive from parent, no weight/search knob.
Matched rolled/uniform/disabled/detached controls separate this path from
generic additional parameters or numerical normalization. Preserve adverse
initial losses and noise; pretraining controls are not efficacy selection.
Trained success requires unchanged target plus cost; failure rejects this
realization, not all mask pooling. Literature novelty is unverified.

## 6. Next Batch
SAME AdaRES-Y28: implement independent physical W2copy, native TRAIN8 plus
released-init4, cache decoder/criterion causal controls, two optimizer steps,
exact pre-continuation model/AdamW/PolyLR resume, serialstress/latency/memory,
structural smell and all six gates. No training admission until real passes.
Formal design: pipeline/designs/exp_design_Y28_7f5e7b9d.md.
