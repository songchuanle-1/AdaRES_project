# Experiment Design - AdaRES_project

2026-10-09; tick8fd5e469dee241da8f21da7c870c0d3f.
Status: PROSPECTIVE, not admission or production launch.

## 1. Motivation

Goal: "AdaRES: improve a faithful standard-indicator 3D referring expression
segmentation baseline by learning query confidence from metric-aligned evidence,
targeting ScanRefer mIoU >= 0.515."
Capacity1/2 after canonical Y33 completion; admission budget1. Test whether
local boundary/interior feature compatibility supplies useful confidence
evidence beyond global competition/covariance. Motivation review above is the
upstream analysis; prereg and actual CPU test preceded broad route review.

## 2. Current Best Results

| ID | Config | State | mIoU | Acc50 | spmIoU | Epochs |
|---|---|---|---|---|---|---|
| W2 | adares_w2_pointmass_competition_res | Historical champion, current primary eval | .510007498 | .562894405 | .473626083 | Fixed30 |
| Y33 | adares_y33_token_attention_res | done_complete; comparison pending | .505137821 firstfull | .556899453 | .470365717 | 30, best20 |
| B1 | adares_b1_standard_res | Historical .5020; fresh eval pending | Unknown here | Unknown | Unknown | Original |

## 3. Analysis & Direction

W2 stays champion; Y33 primary below target. No mean/no-harm/equivalence claim
from unresolved paired CI. Preserve Y28/Y30 subgroup-harm rejections and
Y34 source-overlap rejection. Y36 actual equal-marginal adjacency CPU response
separates this realization from global moments and token attention. Upstream
attention remains; neither necessity nor novelty is established. This is
Phase1 structural exploration, not unapproved Phase2 ablation.

## 4. Experiment Plan

| ID | Config | Hypothesis | Expected Result | Priority | Depends On |
|---|---|---|---|---|---|
| AdaRES-Y36 | configs_song/model_paper5/adares_y36_local_adjacency_res.yaml | Final-mask local edge evidence improves confidence | Falsifiable versus W2/B1; no initialized effect-size gate | P0 | Six real gates |

Independent physical six-file W2 package and exact parent config.
Nearest-distance non-self spatial neighbors, averaged ties, fixed before tests.
Edge MLP consumes feature/neighbor/difference; point-mass soft boundary/interior
summaries plus normalized final query predict a final-only indicator residual.
460034 added parameters/eight tensors/native256 dimensions. New modules are
initialized inside fork_rng after parent modules, preserving all parent bytes
and initialization. Masks/auxiliary losses/native128/oneargmax unchanged.

Freeze: CPU32, native prediction-blind first64 distinct TRAIN scenes/up to4
descriptions; cachedTRAIN8 and releasedinit4 controls; physical native2steps,
exact model/AdamW/PolyLR restoration plus honest numerical continuation;
matched serial resources, first8 scenes and stress dataset indices728/6780.
New-head temporary optimization is investigation only, never deployed.

After production, if admitted: fixed30/seed1999/releasedSPFormer/own decoder;
full ScanRefer9508 expressions/141scenes; Y36/W2/B1 frozen best checkpoints
each twice; first pass primary, never select a better repeat. Scene bootstrap
10000/seed1999, TRAINrawpoint cuts1487/3325. Material subgroup harm if paired
whole-scene interval upper<0 against either comparator. CIcross0 is inconclusive.
Report mIoU/Acc25/Acc50/querycount/latency/peakmem. Quantify empirical repeat
noise and trained paired cost. Promotion requires goal target and no material
harm with measured trained lightweight Pareto; not just added capacity.
Resource increases need trained quality benefit, not an arbitrary tolerance
chosen after observing results. Trained Pareto/seed variance remain unknown.

## 5. GPU Allocation Plan

| Server | GPU | Assignment | Justification |
|---|---|---|---|
| server2 | 0 | Preserve original Y35 | Authorized but occupied; no diagnostic |
| server2 | 1 | Y33 frozen evaluation, then Y36 diagnostic/admission if valid | Exact live idleness and exclusive lock required each use |

## 6. Implementation Checklist

- [x] Stable diagnosis before broad review and exact CPU discrimination.
- [x] Independent local six-file package/config; additive diagnostic upload.
- [ ] Native TRAIN64/controls/backward/resume/resources substantively reviewed.
- [ ] Commit only seven new candidate files; committed/local/deployed byte match.
- [ ] Admission receipt and canonical gated dispatch.
- [ ] Increasing unique native training steps; no blind restart.

## 7. Risk Assessment

Dense distance graph can be expensive on large scenes; resource tests required.
Single nearest-edge realization does not prove optimal neighborhood choice;
no validation search permitted. New readout may worsen confidence/subgroups;
retain individual and aggregate adverse results. Same-position ties average
equivariantly; no coordinate/query correspondence assumption from rejected Y22.
Native diagnosis/backward/resources still unrun at document creation.
No new GT inference input, external data, altered training length or replay.
Exact restore is not whole-loader/RNG bitwise trajectory equivalence.

## 8. Smell Check

| Signal | Assessment |
|---|---|
| 1: small parameter-only change | Not hit:460034 and nontrivial spatial edge graph |
| 2: norm/linear/gate insertion | Not hit: nearest-edge feature/mask relations aggregated into distinct supports |
| 3: numeric distribution only | Not hit: new local adjacency information path |
| 4: arbitrary search | Not hit: nearest-distance definition/native dimensions, fixed/no sweep |
| 5: renamed rejected realization | Not hit: actual equal-global-evidence control distinguishes local edges |
| 6: better numeric configuration | Not hit: causal hypothesis concerns edge compatibility, not scalar settings |

Three criteria: new interaction graph, native inherited dimensions/fixed
neighborhood and explicit matched controls. This establishes structural
eligibility only. No automated document represents human judgment or approval.
