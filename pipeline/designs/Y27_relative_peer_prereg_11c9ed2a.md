# Y27 Relative Object Peer Evidence: Preregistration

Created: 2026-10-04, owned tick 11c9ed2af51a4ed9967596b52811d11d.
Candidate ID: AdaRES-Y27. Hypothesis ID: mask-centroid-relative-peer-confidence.

## Hypothesis And Distinction
W2 confidence explicitly reads query features and mask-overlap competitors, but
not mask-derived object-relative spatial arrangements. Earlier decoder query
features already contain spatial processing; this is NOT a geometry-blind model
claim. A new confidence-local information path could distinguish candidates by
their relation to other, including nonoverlapping, scene objects.
This is not Y21 augmentation, X1 context removal/rolling, X2 overlap replacement,
Y25 mask-to-attention feedback, or Y26 normalization. Preserve those outcomes.

## Frozen Diagnosis
Use W2 best checkpoint e3407283384b4899524d055ac146c7bd6cf39cedb10e58a44ad74eb522009f52.
Select FIRST64 DISTINCT training scenes in existing dataset order, seed1999,
native augmented scene, at most4 descriptions per existing chunk. No predicted
quality-based sampling, validation access or relation-subset selection.
Report relation/other-entity map incidence; exact augmented point IoU selected
versus existing128-query oracle; mask-probability/point-mass centroids and their
pairwise spread. Oracle is a diagnosis, not deployable performance.
Cache actual final indicator arguments including batched superpoint positions.
With identical queries/masks/masses, W2 output must be insensitive to centroid
counterfactuals because it has no such input. This identifies an explicit
information gap, not proof that upstream latent features lack it.

## Frozen Intervention And Controls
Physically copy W2 package and config; ONLY confidence-local relative peer path.
c_i=sum_m p_im n_m x_m / sum_m p_im n_m, valid superpoints only.
delta_ij=c_j-c_i. A 3->d->d GELU position MLP, q/k/v and output d->d projections
form learned ordered-pair attention:
a_ij=softmax_j(q_i dot (k_j+pos(delta_ij))/sqrt(d)), diagonal excluded.
r_i=out(sum_j a_ij*(v_j+pos(delta_ij))).
Add r_i to existing overlap context, keep existing confidence-head dimension.
d=existing256; standard dimension-derived scale, no numerical search, no learned
scalar gate, temperature, loss-weight, threshold, extra proposal or schedule.
Single-query case returns zero peer context. Empty area uses existing epsilon.
Invalid padding mass/positions must have no effect; global translation must
leave relative output unchanged within arithmetic tolerance. No rotation
invariance claim: axes in current augmented training coordinate system.
Matched cached controls: native repeat, path disabled, centroid query-index
permutation (fixed one-position cyclic mismatch), zero relative positions,
common translation, and peer-feature detach with forward identity. Report all,
including adverse initial loss and deviations, not only favorable cases.
Joint permutation of query/mask/centroid must preserve equivariance.
No GT or relation maps in inference input; annotation counterfactual must not
affect fixed cached prediction.

## Native And Resource Gates
Native TRAIN8 first distinct scenes and released-initialization TRAIN4:
full native criterion/backward; all12 added trainable tensors finite/nonzero.
Explain adverse native results with cached matched loss/gradient components;
do not pass an unexplained reversal or exact no-op.
Native tools.train two optimizer steps; real model/AdamW/PolyLR checkpoint
save/resume equality BEFORE continuation, report next-step numerical differences.
Cached native prediction on first4 TRAIN scenes, parent identity with path off,
128 queries/one selected. No global FPS determinism claim.
Serial parent/candidate resource test: existing stress indices728/6780
(1055503 augmented points/7618 SP in prior frozen capture), first8 TRAIN scenes;
2 warmups/3 timed readings, isolated allocation/reservation, parameter counts,
native batch1 inference. Freeze implementation dimensions, do not tune on cost.
GPU1 only after live guard, no GPU0 query/use. No training admission until all
six gates AND mandatory structural smell review pass. Novelty unverified.

## Post-Training Comparison Frozen Before Any Admission
Released SPFormer backbone, native seed1999, exactly30 epochs, existing config.
Primary full ScanRefer validation9508 expressions/141 scenes; candidate/W2/B1
each two native passes. Preserve identity(scan_id,object_ids,ann_id).
Whole-scene paired bootstrap10000 draws seed1999; report mIoU, Acc@.25/.5,
spmIoU, active/selected queries, repeated empirical differences and uncertainty.
Existing TRAIN target-size cuts1487/3325 unchanged. Material harm: whole-scene
paired95% CI upper bound below zero for all/small/medium/large against W2,
each subgroup separate; inadequate scenes/CI crossing0 means inconclusive.
Lightweight Pareto requires trained gain without demonstrated latency/peak
memory regression; pretraining resource feasibility alone cannot prove Pareto.
No validation tuning or guaranteed efficacy; only tested realization is judged.
