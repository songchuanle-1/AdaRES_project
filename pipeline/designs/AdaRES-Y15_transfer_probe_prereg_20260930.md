# Y15 Native Mask-Transfer Probe

2026-09-30 Asia/Shanghai, before native transfer execution. Same stable Y15 ID,
not an admission or a complete proposed training architecture.

## Question

Does the teacher's missing-mask evidence produce a student mask correction
aligned with ground-truth supervision, beyond wrong-language transfer?
Mask oracle availability alone could be unlearnable or harmful.

## Frozen Units And Test

Use the first item of train distinct scans 0 and 1: dataset indices 0 and 6.
Each scan uses its first four expressions for backward-memory feasibility;
select expressions by existing order, never by observed gain. Both models
retain fixed best weights. Student forward is eval-mode with autograd enabled,
teacher forward is frozen no_grad. No augmentation, exact native feature path.

Hook the final decoder output on the student's actual native model forward.
For each expression, use the B4 training-label oracle mask, find its Hungarian
matched W2 query, and compare a pair of teacher/true binary masks at that query.
Matching and teacher oracle use training data only, never inference labels.
Use point-count-weighted BCE on the student's mask logits:

L_target=sum_s n_s BCEWithLogits(x_s,y_s)/sum_s n_s
L_teacher=sum_s n_s BCEWithLogits(x_s,t_s)/sum_s n_s.

The exact mask-logit gradients are n_s/sum(n)*(sigmoid(x_s)-y_s) and the
analogous teacher gradient. Measure cosine similarity of these gradients,
and independently autograd.grad of both losses into dec.x_mask parameters.
Use the corresponding different-object cyclic B4 expression from Y15 as
wrong-language teacher control, selecting its best mask against that control's
own target, and Hungarian-matching it to the student's current expression.
Compare teacher/GT alignment against wrong-language/GT alignment.
Report an aggregate on all four expressions and the preexisting material-mask
subgroup (teacher oracle minus student oracle >0.05); unavailable groups remain
unknown. Repeat logit-gradient readouts using the teacher's second native
forward. No coefficient tuning, optimizer update or validation evaluation.

## Interpretation

Finite nonzero actual mask-head gradients demonstrate native graph reachability,
not new-path integration in tools/train.py. Higher correct-target gradient
alignment establishes a transferable signal but does not imply trained efficacy
or an independent architecture. Reverse/control-indistinguishable results
reject this realization or require a bounded causal diagnosis, not automatic
PASS. Existing GT supervision can already have stronger gradients: teacher
supervision is not justified merely because it matches GT.

## Next Design Decision

If transfer is feasible, derive an inference-valid student mechanism that
actually fills missing candidates, and compare transfer against equally strong
GT-only supervision. Full training is blocked on that mechanistic control,
native integration, physical independence, resource Pareto and frozen full
ScanRefer evaluation. Do not add an arbitrary extra BCE coefficient and call
ordinary distillation a new contribution.
