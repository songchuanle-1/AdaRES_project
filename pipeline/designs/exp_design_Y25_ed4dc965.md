# Experiment Design - AdaRES-Y25
Status: IMPLEMENTED, admission pending receipt validation; 2026-10-03 21:24+08.
motivation_review_ed4dc965.md upstream; scoped candidate source commit
94918c1a008e21c72b9478c28f934e8b329e0c97.

## 1. Motivation
Goal: "AdaRES: improve a faithful standard-indicator 3D referring expression
segmentation baseline by learning query confidence from metric-aligned evidence,
targeting ScanRefer mIoU >= 0.515."
Hypothesis: previous detached Boolean masks prevent geometrically matched
queries from reading excluded annotated target mass during refinement.
Training-only differentiable log mask probability permits that evidence and a
new native loss-to-mask feedback path. No added query/parameter; eval hard gate
and all confidence heads remain the parent W2 implementation.

## 2. Current Best Results
| ID | State | mIoU | Acc50 | spmIoU | Epochs |
|---|---|---|---|---|---|
| W2 | done_complete | Historical .5100 | See immutable full capture | See immutable full capture | 30 |
| B1 | done_complete | Historical .5020 | See immutable full capture | See immutable full capture | 30 |
| Y23 | done_complete | Full .507471535 | .558161548 | .470801008 | 30 |
Y23/Y18 clustered intervals cross0 versus W2; no promotion.

## 3. Analysis And Direction
Native TRAIN64/256expressions: stage mean excluded target mass
.15577655/.12454302/.09708272/.08730793/.08703019/.09141662; 14 fully,
9509 partly locked stage/query occurrences; fallback13; repeat difference0.
Actual custom attention CPU32 controls: hard target attention0, soft minimum
.12741014/target mask gradient minimum .08112176, padding leak0,
detach/full identity0. Mechanism activity only, no trained gain.

## 4. Experiment Plan
| ID | Config | Purpose | Expected Direction | Depends On |
|---|---|---|---|---|
| AdaRES-Y25 | configs_song/model_paper5/adares_y25_mask_feedback_res.yaml | Differentiable TRAIN-only mask-feedback prior | Access/feedback restored; trained efficacy unknown | All six gates |
Freeze before admission: released SPFormer, seed1999, exactly30epochs,
parent optimizer/scheduler/dataloader/losses/thresholds/features unchanged.
TRAIN controls first8 distinct scans at W2 checkpoint and first4 released-init:
hard/soft/soft-detached/full-valid/soft-repeat; report finite downstream
gate gradients, forward/gradient repeat differences, all losses and adverse
readouts. No chosen loss/scene subgroup may be discarded.
Physical copy and actual train two optimizer steps, native save/resume,
TRAIN-only path/eval identity, prediction-blind train8/stress728+6780 resource
measurements follow. No admission from a prototype alone.
Post-training frozen plan: own best after fixed30, full ScanRefer val9508/141
scenes, Y25/W2/B1 each two independent native passes seed1999; raw identity
alignment; mIoU primary, Acc25/Acc50/spmIoU, active queries, latency/p95/peak
memory/parameters. Whole-scene bootstrap10000 seed1999, 95%paired CIs.
Frozen raw TRAIN point-size cuts1487/3325, never val-derived. Material subgroup
harm: negative paired upper95%CI beyond observed paired-repeat noise, with
unquantified noise explicitly unknown; crossing decision boundaries inconclusive.
No single-scene veto. Lightweight Pareto requires no added params/queries and
no reliable inference latency/peakmemory regression; bounded three timing
reads are descriptive, not proof of p95 or Pareto. Thresholds never tuned.

## 5. GPU Allocation Plan
| Server | GPU | Assignment | Justification |
|---|---|---|---|
| server-2 | 1 | Exclusive diagnostics, prospective gated Y25 | Sole current whitelist; verify live occupation each time |
GPU0 not queried/used; no other server/project/scheduler operations.

## 6. Implementation Checklist
- [x] Stable candidate and prediction-blind diagnosis preregistered.
- [x] Actual TRAIN64 and custom attention matched controls executed.
- [x] Native full loss/backward controls.
- [x] Independent physical W2-copy/config and inference identity.
- [x] Native optimizer two-step/save/resume/stress resource tests.
- [ ] Six-gate review, tested scoped commit and receipt.
- [ ] Canonical gated dispatch and native increasing-step verification.

## 7. Risk Assessment
Soft TRAIN and hard eval can create distribution mismatch; target access alone
need not improve IoU or confidence. Gradients may be finite but harmful; preserve
every native reading. Float softmax can underflow for extreme logits; no universal
positive support claim. Fullvalid control may explain an effect instead of
gradient feedback; no fixed task-benefit gate. Large native memory/cost and
resume numerical noise must be measured. No source/weight changes to W2/Y23.

## 8. Smell Check
Looks like an elementwise prior, but the substantive change is a previously
detached Boolean edge becoming a native loss-bearing mask-to-cross-attention
feedback edge and new access to previously excluded target evidence. Actual
matched detach/full tests isolate these mechanisms. Fixed mathematical log
probability, no trainable scale/new tensor/temperature or numerical sweep.
This is a causal structural test, not a confidence-support selector, Q2
relabeling, G1 target calibration or Y20 global detach.

## 9. Executed Native Evidence
Trained8/released-init4 native full controls: all6 downstream bias gradients
finite/nonzero; soft-detach gradient minimum .0238480 versus repeat maximum
.000390291; forward maximum .000190735 equals maximum soft-repeat noise,
not exact full-model identity. Synthetic actual attention detach forward0.
All losses retained: soft-hard mean +.0441206 (trained7positive/1negative)
and +.0316472 (released3positive/1negative). This is an adverse immediate
readout, not silently passed as task benefit. The trained parent was optimized
for hard gates; changing TRAIN support changes its forward distribution.
Released init directions mixed; eval same-source/cached-input identity0 means
the intervention does not change initial hard-eval outputs. These findings
permit a falsifiable training test, not a confident benefit or harmlessness claim.
Native train two actual optimizer steps:400/400 decoder tensors finite/nonzero,
each6 new bias gradients active. Exact model/AdamW/PolyLR restored before
continuation, next loss -4.77e-7/model1.526e-5/optimizer1.456e-7 retained.
Physical six-file package/config independent; only TRAIN get_mask differs,
eval body unchanged; cached4scans eval parent/repeat mask/indicator/sample0.
Stress728/6780 paired released state: parent/candidate median .9043506/.9051482s,
reserved13.402344/13.486328GiB; no OOM/finite gradients. Same W2-checkpoint
serial inference train8+2stressscenes parent/candidate median .0906848/.0909893s,
descriptivep95 .2219310/.2162449, reserved1.7578125GiBboth,164629637params,
128queries/1selected. +0.336%median drift is not reliable regression/Pareto
proof. Formal trained efficacy/Pareto comparisons remain frozen post-training.
