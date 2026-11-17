# AdaRES-Y7 preregistration: native-positive set separation

Date: 2026-09-29 Asia/Shanghai

## Frozen question

For W2 selection errors whose oracle query belongs to the native
`match_sampled_ins` positive set, does the *correct* positive set separate from
the native negatives more often than an equal-cardinality, position-preserving
cyclic-shift control?

The query-level readout is

`margin(S) = max_{i in S} s_i - max_{j not in S} s_j`,

where `s` is the W2 positive-class softmax score. `separated(S)` is true iff
`margin(S) > 0`. This is a set-level ranking quantity: it asks whether *at least
one* native positive outranks every native negative.

## Fixed cohort and checkpoint

- Model: `adares_w2_pointmass_competition`
- Checkpoint:
  `/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth`
- Primary cohort: first 32 non-augmented ScanRefer validation scenes, seed
  1999, material selection errors (`oracle_iou - selected_iou > 0.05`) for
  which the oracle query is in the true native positive set.
- Secondary cohort: all first-32-scene expressions and all material errors.
- Scene is the sampling and aggregation unit.

## Matched control

For each expression, the true positive set `P` is cyclically shifted through all
`Q - 1` nonzero offsets. This preserves query count and cyclic adjacency but
destroys native positive identity. The control readout is the mean over all
shift offsets, not a selected lucky offset.

## Predeclared decision rule

Pass the diagnosis only if both conditions hold:

1. On the primary cohort, the scene-mean paired difference
   `separated(true) - separated(shifted)` is positive, and a two-sided
   scene-level sign test over nonzero paired differences has `p < 0.05`.
2. On the same cohort, the scene-mean paired difference
   `margin(true) - margin(shifted)` is positive.

If either condition fails, reject the group-separation rationale for a new
loss. A pass is only a diagnosis gate candidate; it does not admit training.

## Independence from C2

C2 optimizes one oracle-best-IoU positive against one hardest confidently wrong
negative. Y7 instead tests whether the native positive *set* contains at least
one query that should outrank every native negative. A future intervention must
use the native positive set and a group max-over-positives objective. It may not
be implemented as another single-positive C2 margin.

## Repeatability

Each scene is forwarded twice without gradients. Any row discrepancy is
reported and prevents fine-grained per-row claims; it does not by itself select
a post-hoc threshold.
