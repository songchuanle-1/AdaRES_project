# Y15 Pretraining Admission Review

2026-09-30. Gate calibration is prospective, not a claim of task gain.
No new tiny-cohort significance threshold or validation tuning.

| Gate | Evidence | Decision |
|---|---|---|
| diagnosis | train64 actual scans complementary candidate pools .01168 true,0 repeat,.00141 wrong | PASS: task-relevant controlled information gap |
| controls | same-output loss, language/slot equal-capacity interventions,32 material correct/wrong-object BCE gradients | PASS: intended path active and distinct from matched interventions |
| native_backward | hardened provider two real batches,9 tensors finite nonzero, native tools.train2 updates | PASS: new computation used by training |
| resources | prediction-blind562scan selection,1055503point batch2,14.71GiB teacher vs14.09GT;329728 new params | PASS: frozen structural/single24GiB budget; efficacy frontier unknown |
| independence | AST relative imports,physical B4 teacher,24files/2configs,remote identity | PASS: owned physical candidate and GT-control packages |
| constraints | frozen30epochs/released backbone/seed/loss/native batch, whitelist0/1, prospective full evaluation | PASS conditional on identical committed deployment/receipt |

Source audit performed in this tick against actual artifacts. Controls do not
establish teacher necessity:5/32 teacher targets exactly GT, mean isolated
teacher-GT gradient difference .11536 versus wrong-object2.45926.
Teacher has nonzero conditional mask signal, but a trained benefit is unknown.
An adverse repeat candidate-bit flip is preserved, selected RES output remained
stable in tested inference. No exact determinism or novelty assertion.

The launch plan accurately states final binary-mask BCE and inherited row
normalization; earlier draft projection is diagnostic only.
Post-training harm/Pareto evaluation is frozen in the design, noise still
unknown on the full validation set. Source commit/receipt/dispatcher checks
must be real. This review authorizes one falsifiable Y15 run, not an automatic
second GT run or a method-success statement.
