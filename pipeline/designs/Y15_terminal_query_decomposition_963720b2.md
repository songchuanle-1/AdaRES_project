# Y15 Terminal Query Decomposition

Frozen before decomposition outputs on 2026-10-02 Asia/Shanghai.
Stable ID AdaRES-Y15; post-training diagnosis, not a new candidate/admission.
Official trained best-checkpoint full-validation mIoU0.230319 was already
observed. No selection-decomposition results have been observed.

## Question And Controls

Does the additional language query attract indicator selection despite poorer
target masks, or has the original128-query pool itself degraded?

Keep the native best checkpoint, model, inputs, seed1999, ordering, thresholds,
weights and forward pass unchanged. Hook the final decoder output read-only.
For each original expression record the native selected index/IoU, additional
query IoU, original-pool indicator-argmax IoU, original-pool mask oracle IoU and
full-pool oracle IoU. The original-pool argmax is a diagnostic counterfactual,
not a revised inference method or deployable validation-selected patch.

Record extra-query selection prevalence and grouped readouts by true scan_id.
If original-pool selected performance is also poor, extra-query competition
alone does not explain the deficit. If original-pool oracle remains high but
selection remains low, the result points toward selection quality; it does
not establish teacher causality or authorize changing confidence thresholds.
No fixed effect-size gate, hyperparameter sweep or extra training.

## Execution And Limits

Use all9508 official expressions/141scenes, native batch1, original geometry,
no-grad/eval and one pass. Match expression identities and native selected
IoU to the completed frozen evaluation. Report raw selection and oracle
statistics without treating oracle masks as realizable efficacy.
GPU0 only after fresh idle check and exclusive per-card diagnostic lease;
GPU1 original Y18 remains untouched. Bound timeout well before actual
tick deadline2026-10-02T06:49:29.564457+08:00.

This is an explanatory validation diagnostic. Any future realization needs
independent train-only diagnosis and frozen controls; do not tune on these
validation outputs. Source/outputs/failures are retained as hashed evidence.
