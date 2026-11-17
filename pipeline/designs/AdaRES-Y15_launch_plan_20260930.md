# Y15 Frozen Launch Plan

2026-09-30, lease0c2d6bb3cc524bd59b928b7af3b2f8ec.
This final plan describes the tested implementation, superseding draft
implementation details only, not old freezes/outcomes.

## 1. Motivation

"AdaRES: improve a faithful standard-indicator 3D referring expression
segmentation baseline by learning query confidence from metric-aligned
evidence, targeting ScanRefer mIoU >= 0.515."
Test language-conditioned completion of a missing mask candidate instead
of another confidence coefficient. Training is a falsifiable experiment,
not a declaration of teacher superiority.

## 2. Current Best Results

| ID | Config | State | Best mIoU | Acc50 | spmIoU | Epochs |
|---|---|---|---|---|---|---|
| W2 | adares_w2_pointmass_competition_res | done_complete | .5100 | unknown here | unknown here | 30 |
| B1 | baseline standard indicator | done_complete | .5020 | .553 | .467 | 30 |
| Y15 | adares_y15_query_completion_res | validation | unknown | unknown | unknown | no full run |

## 3. Analysis & Direction

True train64 candidate complementarity exceeds repeated and wrong-language
equal-count pools; teacher/GT extra-query gradients nearly coincide, while
wrong-object supervision changes them. Thus candidate completion is testable,
teacher necessity is unresolved. Source audit verified24 physical Python
files,2 configs and63 evidence hashes. All new9 tensors updated through
native tools.train; largest frozen train pair fits14.71GiB.

Final implementation: one language-conditioned attention query is appended
to the final128 candidates; original six auxiliary layers remain128.
Let t be the frozen B4 oracle binary mask when detached B4 oracle IoU exceeds
the current unchanged-K student oracle by .05 on a training target; otherwise
t is GT. Only the extra-query point-mass-normalized native BCE row uses t.
Native GT dice/score/indicator/semantic/auxiliary objectives remain matched.
For x_s=f_s^T q, L_BCE=sum_s n_s BCE(x_s,t_s)/sum_s n_s,
dL/dq=sum_s n_s f_s(sigmoid(x_s)-t_s)/sum_s n_s.
The native matched-row and instance reductions still apply to this row.
No query-MSE/projection loss or distillation coefficient is implemented.
The early draft's projected q* was a diagnostic, not the final objective.

## 4. Experiment Plan

| ID | Config | Purpose | Expected result | Priority | Dependency |
|---|---|---|---|---|---|
| AdaRES-Y15 | configs_song/model_paper5/adares_y15_query_completion_res.yaml | train completed candidate pool | unknown, goal >=.515 | P0 | six-gate receipt |

Exactly30 epochs, seed1999, native batch2, released SPFormer initialization,
unchanged optimizer/PolyLR/loss weights. Set ONLY this queue entry min_epochs30
so generic early-stop patience cannot truncate the frozen protocol.
Best selection uses existing evaluator rule, no validation epoch tuning.
Evaluate official fullScanRefer mIoU versus W2/B1; Acc25/50, spmIoU,
129 available candidates and one selected RES output, warmed student-only
latency/peak, scene-clustered differences and repeated-evaluation uncertainty.
Use frozen harm/Pareto plan in Y15_stress_resume_prereg_20260930.md.
Train-size tertiles must be frozen before first validation exposure; until
computed do not report size subgroup conclusions. GT package is a causal
control, not a second independent full-training candidate this tick.

## 5. GPU Allocation Plan

| Server | GPU | Assignment | Justification |
|---|---|---|---|
| server-2 | any first free of0/1 | one Y15 task | checked dispatcher capacity/ownership |
| server-2 | remaining card | independent research only | GT is not slot filler |

No server-6, multiple GPUs, new scheduler or manual launch bypass.

## 6. Implementation Checklist

- Source/relative import/remote hashes and frozen-config comparison completed.
- Six-gate review and commit identity required before receipt.
- Gated canonical monitor dispatch only, unique log/PID.
- Launch verified only by increasing native Train[epoch/30] steps.
- Preserve existing dirty work; additive synchronization checks each file.

## 7. Risk Assessment

Teacher is ~43% slower in stress training versus GT and necessity unproved;
no teacher inference dependency. Native continuation not bitwise equal;
small sparse-reduction noise preserved. Full val noise remains unknown.
Historical W2/B1 are not freshly matched30-epoch controls; causal lift requires
subsequent matched studies. Novelty unverified. If native launch faults, retain
exact failure, repair only scoped cause and do not blindly duplicate a process.
Hardware fit establishes executability, not trained Pareto superiority.
