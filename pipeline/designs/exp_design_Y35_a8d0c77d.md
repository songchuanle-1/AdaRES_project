# Experiment Design - AdaRES_project

Status: DRAFT BEFORE ADMISSION. Research Pipeline (autonomous).
Tick a8d0c77d45d142c0a6c46b163b956253; prospective preregistration precedes probes.

## 1. Motivation

"AdaRES: improve a faithful standard-indicator 3D referring expression
segmentation baseline by learning query confidence from metric-aligned evidence,
targeting ScanRefer mIoU >= 0.515."
Upstream review: pipeline/reviews/motivation_review_Y35_a8d0c77d.md.
Capacity2/active1/candidate budget1; explicit GPU0 grant. Investigate whether
foreground-versus-background visual evidence reading individual language tokens
is useful for final confidence. Existing cross-attention is acknowledged.

## 2. Current Best Results

| ID | State | mIoU | Acc50 | spmIoU | Epochs |
|---|---|---|---|---|---|
| W2 | done_complete historical | .5100 | unknown here | unknown here | 30 |
| B1 | done_complete historical | .5020 | unknown here | unknown here | 30 |
| Y31 | frozen complete/inconclusive | .505514618 | unknown here | unknown here | 30 |
| Y33 | running begin snapshot | rounded monitor .5051 | unknown | unknown | monitor22/30 |

Rounded monitor values are not frozen full-cohort efficacy results.
Y31 paired intervals cross zero, not equivalent or proven harmless.

## 3. Analysis And Direction

Y34's CPU32 independent reference has exact output equality to W2's existing
overlap/context/difference path; reject that absent-path realization, not every
nonlinear relational head. Preserve X1/Y8 historical adverse evidence.
Y35 differs from Y33 (query->visual tokens) and rejected Y28 (direct visual means):
FG/BG visual means -> individual word attention -> attended-language contrast
and query -> final confidence residual. CPU equal-language-first-moment control
delta .016250908; native TRAIN64 all64 responsive and repeat max0.
No global language blindness, semantic necessity or trained gain is claimed.

## 4. Experiment Plan

| ID | Config | Purpose | Expected Result | Priority | Dependency |
|---|---|---|---|---|---|
| AdaRES-Y35 | configs_song/model_paper5/adares_y35_language_evidence_res.yaml | mask-evidence-to-word confidence | falsifiable trained comparison; no promised gain | P0 | six gates |

Architecture: point-mass-weighted soft FG/BG means of final visual tokens,
functional LN, shared q-projection, k/v of all unpadded final language tokens,
sqrt(d_model) scaling, FG/BG attended-value difference + normalized final query,
Linear512->256/GELU/Linear256->2 residual. Parent d_model256 fixes width/scale.
New328450 parameters/seven tensors, about .20% of W2 total164629637.
Native parent initialization protected with CPU fork_rng; no diagnostic weights.
Only final indicators change, all masks/auxiliary losses and one argmax preserved.
Frozen30epochs/seed1999/releasedbackbone/native128/AdamW/PolyLR unchanged.

Frozen post-training protocol:
- Full validation9508 expressions/141 scenes, Y35/W2/B1 each two passes.
- First pass primary; mIoU/Acc25/Acc50/active queries/latency/peak memory.
- Whole-scene clustered bootstrap10000 seed1999, paired against both comparators.
- TRAIN target rawpoint cuts1487/3325; no validation subgroup/threshold tuning.
- Material subgroup harm if paired CIupper<0 against W2 or B1.
- CIcrossing0 is inconclusive, not equality/no-harm.
- Measure independent repeat noise and trained paired cost empirically.
- Lightweight trained Pareto remains a measured outcome, not a pre-training claim.

## 5. GPU Allocation

| Server | GPU | Assignment | Conditions |
|---|---|---|---|
| server-2 | 1 | preserve Y33 | no modification/restart/migration |
| server-2 | 0 | Y35 diagnostics; training only if admitted | live idle/exclusive/one task |

No server6, other project or background decision process. Latest SERVERS[0,1].
Each GPU test acquires GPU0 advisory lock, requires <1024MiB/<10% and no compute
process, CUDA_VISIBLE_DEVICES0; actual measurements retained.

## 6. Implementation Checklist

- Independent six-file model package and globally unique native config exist.
- No cross-variant imports; parent classes outside DEC AST-identical.
- Only DEC new member/final readout added; copied other five files unchanged.
- Config parent-equivalent, including epochs/seed/loss/optimizer/query/inference.
- CPU32/nativeTRAIN64/cachedTRAIN8+init4 executed with evidence hashes.
- Native physical train two steps and exact checkpoint model/AdamW/PolyLR restored.
- Short-loss investigation/resource/commit-byte-sync/admission/launch pending here.
- Sync only seven additive Y35 files to server2 under canonical explicit policy.
- Never reset/clean/stash user changes or overwrite running Y33 files.

Pre-measure resource admission budget: new params<1% parent, native stress peak
reserved<24GiB on3090, no OOM/nonfinite gradients. Matched serial timing measured
with2 warmups/3 readings and reported, not assumed speedup or trained Pareto.
No latency winner selected from validation; trained cost must be evaluated later.

## 7. Risk Assessment

Adverse initial W2 loss+.00222339481 and tiny attention gradients require
investigation; released-init mean-.0736477375 does not prove word utility.
Wrong-word and wrong-mask controls can learn; record all scene outcomes.
Checkpoint restore exact before continuation but next-step model max3.05176e-5
and optimizer8.86166e-9 differences remain; no whole-loader bitwise claim.
Native sparse kernels may have inference noise; quantify with frozen repeats.
All-padding words are outside native diagnosis, not a deployment success case.
No GT inference, data collection, feature-subset tuning, or validation changes.
Failure repair bounded at3; failure never authorizes another GPU or gate relaxation.

## 8. Smell Check

| Signal | Judgment |
|---|---|
| <5K params/no new graph | no:328450params, explicit visual-evidence->individual-word path |
| just norm/Linear/gate insertion | surface projections/LN, but two support-conditioned word attentions and contrast define new interactions |
| only numeric distribution | no: changes direct information path, not scale-only |
| arbitrary sweep | no: parent256 and sqrt256; no tunable search |
| renamed banned mechanism | no: no threshold/occupancy/ranking-margin/feature-subset scan |
| better numeric configuration only | no: tests explicit cross-modal evidence readout |

Three standards: nontrivial new interaction, structurally fixed hyperparameters,
explicit causal hypothesis. Surface normalization/projections are defended by
matched equal-language-mean and wrong-value-alignment controls. These justify
testing, not proven novel attention, semantic necessity or guaranteed benefits.
