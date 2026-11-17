# AdaRES-Y26 Query-Language Postnorm Preregistration

Created 2026-10-04 Beijing, before diagnostic outcomes. Owned tick
`a1a9809f76d0430b8ad4a7ddaf0ec99e`; snapshot18:29:13, deadline19:29:13.

## Mechanism

W2 `CrossAttentionLayer.forward` calls `self.norm(output)` without assigning
the returned tensor. This observation alone does not establish a performance
bug. Y26 reconnects ONLY `DEC.qla_layers` existing postnorm, in TRAIN and EVAL.
Spatial attention, masks, ranking, language support, matching and losses stay
native. Dropout remains unchanged (configured0); do not bundle other fixes.
Hypothesis: discarded qla postnorm leaves a dead affine calibration path and
passes language-conditioned residual channel mean/scale into query competition.
Reconnection restores that path and removes those nuisance degrees of freedom.
Predicted native readouts: original affine gradients absent; connected finite
nonzero gradients; native residual means/scales differ from connected outputs;
connected outputs differ from repeat/identity control. Task benefit is unknown.

## Frozen Samples And Controls

- Actual native TRAIN first64 distinct scenes in dataset order, max4 expressions,
  seed1999, original augmented batch constructed once per scene.
- Actual trained W2 best checkpoint, hash verified; actual deployed source,
  commit marker94918c1 and raw hash verified. No validation access.
- Cache native decoder inputs once per scene to separate upstream FPS variation.
  Evaluate original, original repeat, identity-return hook, connected existing
  LayerNorm and unit-RMS-only control on the same cache/state.
- Preserve all-scene point-IoU/ranking changes, query statistics, individual
  adverse cases and repeat differences. RMS separates centering/affine from
  generic radial rescaling; it is a diagnostic control, not another candidate.
- Native full backward first8 trained scenes and first4 released-init scenes,
  identical batches/RNG/buffers: native, connected, connected with only affine
  parameters detached, connected repeat. Native loss and all decoder gradients
  checked; connected-v-detached forward equality and norm-affine gradients
  isolate the restored path. No optimization-effect threshold.

## Disposition

Reject exact no-op/reversed claimed path with evidence. Unexplained adverse
native task readouts require causal inspection, not a changed cutoff.
No pretrained cohort gain is a training-efficacy claim. If supported, build a
physical independent W2 copy with only qla postnorm, exact parent config, no
cross-variant import, and test native two steps/resume/resource extremes.
No production training in this snapshot (budget0); leave a concrete next action.

## Frozen Future Evaluation

If later legally admitted: released SPFormer initialization, seed1999,
fixed30epochs, unchanged AdamW/PolyLR/config. Full ScanRefer val9508
expressions/141scenes, two independent passes each candidate/W2/B1; scene
cluster10000bootstrap seed1999. Report mIoU, Acc25/50, spmIoU,128activequeries,
one selected query, serial native latency and peak allocated/reserved memory.
Raw TRAIN target-size cuts1487/3325 stay frozen. A group's upper delta CI below
negative its empirical mean-absolute paired repeat-noise floor constitutes
material harm; crossing boundaries is inconclusive. Preserve largest individual
repeat deviations; no universal noise bound or posthoc subgroup selection.
Parameters must not increase; same singleGPU protocol/finite native stress and
matched measured latency/memory. No reliable p95/Pareto claim from three reads.
Only GPU1 authorized; GPU0 reserved. No novelty claim without literature support.
