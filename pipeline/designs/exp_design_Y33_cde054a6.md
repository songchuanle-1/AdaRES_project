# Experiment Design - AdaRES_project

Created: 2026-10-08. Candidate and prospective prereg preceded broad review.

## 1. Motivation

Goal description: "AdaRES: improve a faithful standard-indicator 3D referring
expression segmentation baseline by learning query confidence from
metric-aligned evidence, targeting ScanRefer mIoU >= 0.515."
Capacity gap1/budget1 after canonical Y31 done_complete. Test whether direct
final mask-conditioned token attention supplies useful distribution-sensitive
evidence to native confidence. TRAIN incidence is not benefit or necessity.

## 2. Current Best Results

| ID | Config | State | Best mIoU | Acc@50 | spmIOU | Epochs |
|---|---|---|---|---|---|---|
| W2 | adares_w2_pointmass_competition_res | Historical champion | 0.5100 | Not loaded | Not loaded | 30 |
| Y31 | adares_y31_mask_shape_res | done_complete/protocol complete | Frozen0.505507946 | Not loaded | Not loaded | 30 |
| B1 | Existing frozen comparator | Historical baseline | 0.5020 | Not loaded | Not loaded | Prior protocol |

Y31 full first pass0.505514618 versus W2 repeat0.509909353: delta-0.004394736,
scene-bootstrap CI[-0.012999697,0.004384800]. Not promoted; no frozen subgroup
harm rejection. The monitor's rounded0.5060 is not a substitute for frozen
high-precision checkpoint/full-evaluation metrics.

## 3. Analysis & Direction

W2 remains the comparator. Prior mean-content Y28 and history Y30 realizations
retain their actual subgroup harm rejections. Y32's new all-query pathway
premise is contradicted by actual C1 source/interventions, not by trained
quality. Y33 explicitly differs from first-moment pooling through individual
query/key interactions; CPU equal-mean control and native TRAIN64 have real
responses. Upstream attention and alternative learnable paths mean necessity
is unknown. Phase2 ablations are not authorized; this is a structural Phase1
candidate with one fixed realization.

## 4. Experiment Plan

| # | ID | Config Path | Hypothesis | Expected Result | Priority | Depends On |
|---|---|---|---|---|---|---|
| 1 | AdaRES-Y33 | configs_song/model_paper5/adares_y33_token_attention_res.yaml | Final mask-conditioned token distribution readout | Falsifiable improvement versus W2; no pretraining effect-size gate | P0 | All six gates |

Physical W2 package/config; add MaskTokenAttentionReadout to final indicator
only. q/k/v Linear256-to256 without bias; normalized tokens/query; dot products
divided by sqrt256. Scores plus logsigmoid(mask or -mask) plus log point mass,
padding excluded; attend values and fuse query/FG-minus-BG via512-to256-to2.
Added328450 parameters/seven tensors. Forked initializer after native layers
preserves parent initialization. All masks,128 active queries, single indicator
argmax, auxiliary losses and native loss weight[4] stay unchanged.

Six executable gates: TRAIN64 diagnosis; actual-class32 and cached TRAIN8/init4
controls; native tools.train full two optimizer steps and exact saved
model/AdamW/PolyLR resume; native serial latency/peak memory/parameters including
stress728/6780; physical/config/source independence; frozen goal constraints.
No gate is passed solely from drafting this plan.

Post-training freeze: fixed30/seed1999/released SPFormer backbone, unchanged
ScanRefer full validation9508 expressions/141 scenes. Candidate, historical
W2 and B1 fixed checkpoint/config each twice; first evaluation is primary,
repeats estimate empirical noise without best-repeat selection. Report mIoU,
Acc25/Acc50,128 active queries, measured latency/peak memory. Whole-scene
bootstrap10000 seed1999; size groups use TRAIN raw-target-point cuts1487/3325.
Material subgroup harm requires paired whole-scene interval upper bound below0
against W2 or B1. Intervals crossing0 are inconclusive, not equivalence/no harm.
Promotion requires target and trained lightweight Pareto assessment, not just
positive initial gradients. Noise/seed uncertainty not otherwise quantified.

## 5. GPU Allocation Plan

| Server | GPU | Assignment | Config | Justification |
|---|---|---|---|---|
| server-2 | 1 | Exclusive diagnostics; later gated training if admitted | Y33 | Only current authorized slot |

Live idleness/ownership rechecked by each runner/dispatcher. GPU0 is not an
alternative resource. Preserve foreign jobs; do not broad-sync another project.

## 6. Implementation Checklist

- [x] Prospective stable candidate and first discrimination recorded.
- [x] Independent six-file model/config test bundle created and hash verified.
- [x] CPU actual-class32 and native TRAIN64 completed.
- [ ] Cached native controls, backward/resume and resources reviewed.
- [ ] Commit only candidate/scoped support; verify exact committed deployed bytes.
- [ ] Real admission receipt and canonical gated dispatcher.
- [ ] Unique log shows increasing native training steps.

## 7. Risk Assessment

Memory/time overhead and distribution sensitivity may regress quality, including
small targets. Random initial final confidence can be adverse; investigate with
matched native loss and temporary fixed native optimizer, retaining all results.
Diagnostics never deploy learned temporary weights. Normalized-token projection
does not recover arbitrary raw-feature information; FG-minus-BG cancels on
degenerate identical supports. Exact restore does not imply bitwise continuation
or whole-loader/RNG equivalence. No extra dataset, GT inference, numerical
search, inference oracle or checkpoint rewrite. Failed probes preserve logs;
no blind production restart. Efficacy, necessity, novelty and Pareto unknown.

## 8. Smell Check

| Signal | Assessment |
|---|---|
| 1: <5K without new graph | Not hit:328450 and new token-query graph |
| 2: mere norm/Linear/gate insertion | Not hit: independent key/value interactions and support aggregation, not just a Linear on existing evidence |
| 3: only numeric distribution | Not hit: individual final tokens are read through an additional direct confidence path |
| 4: arbitrary value search | Not hit:256/native dimensions and sqrt dimension; no sweep |
| 5: renamed banned family | Not hit: Y28 remains rejected; equal-mean control separates this realization |
| 6: better numeric configuration | Not hit: causal test concerns readout distribution sensitivity, not scalar tuning |

Three criteria: new nontrivial aggregation graph, dimensions inherited from
native parent, explicit causal hypothesis with wrong-token/equal-mean controls.
This supports structural eligibility, not publication novelty or efficacy.

Status: DRAFT. Admission, production training and human approval not claimed.
