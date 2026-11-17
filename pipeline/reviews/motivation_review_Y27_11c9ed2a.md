# Motivation Review: AdaRES-Y27
Owned tick11c9ed2af51a4ed9967596b52811d11d, 2026-10-04.

## 1. Goal
Canonical goal.json contents, unchanged:
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
Actual newer user override and SERVERS whitelist permit ONLY GPU1. Historical
B3 terminal; snapshot has active0/target1/budget1. W2.5100, target.5150, gap.0050.
This formal review is persisted after the earlier frozen diagnostic prereg,
before the formal experiment design/admission; do not claim earlier document
ordering. Sampling/intervention were frozen before reading diagnostic outputs.

## 2. Story Status
| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| W2 overlap-aware confidence | Historical champion | .5100, not target | Confidence oracle selection gap |
| Y25 soft mask feedback | Realization rejected | Full9508 scene-CI adverse | No universal feedback ban |
| Y26 qla normalization | Engineering evidence, ineligible | Fresh exactAST norm-only audit | No nontrivial information path |
| Y27 explicit object-relative peer evidence | Diagnosis positive, efficacy unknown | TRAIN64/256,222relation,240other entity | New path might duplicate latent geometry |

## 3. Experiment Assessment
| Experiment | mIoU | Motivation Relation | Contribution |
|---|---|---|---|
| W2 | .510108 current full reference | Confidence from overlap | Comparator |
| B1 | .502122 current full reference | Standard indicator | Comparator |
| Y25 | .494117 | TRAIN feedback | Material harm of realized method |
| Y23 | .507472 | Semantic support | Cluster CI crosses0 |
| Y18 | .507606 | Point-measure supervision | Cluster CI crosses0 |
| Y27 | Unknown, untrained | Post-mask geometry in confidence | Does not inherit a trained gain claim |

## 4. Gap Analysis
Native W2 confidence input contract is geometry-free CONDITIONAL ON fixed
queries/masks/mass; upstream decoder is not geometry-free. TRAIN64 has163
expressions with available oracle improvement, scene-mean.154148389, candidate
centroid spread mean2.378388. The relation incidence is task-relevant but does
not show causality or missing latent information. Need matched actual geometry
controls plus full loss reachability. Initial native TRAIN8 has both beneficial
and adverse loss changes; preserve per-scene data rather than select a subset.
Prior Y21 rotation contrast does not test this post-mask confidence-local path.

## 5. Design Rationale
Mask-derived centroids and ordered-pair learned aggregation create a new
information dependency, not another numerical distribution configuration.
The direct causal prediction is correspondence-sensitive confidence with
common-translation invariance and joint-permutation equivariance, plus actual
centroid-to-mask task-loss gradient. Existing overlap evidence remains.
Dimensions derive from parent256; no hyperparameter search. A trained failure
would reject this realization, not all spatial relations. Trained gain with
Pareto evidence would support the confidence-evidence story. Novelty unverified.

## 6. Next Batch
SAME AdaRES-Y27: physical native controls, cached centroid gradient, native
two-step/resume, serial stress/inference, mandatory smell and six-gate review.
Only after real passes create scoped source commit and receipt; canonical
dispatcher plus increasing native steps. Formal design:
pipeline/designs/exp_design_Y27_11c9ed2a.md. No Y26 dispatch or GPU0 access.
