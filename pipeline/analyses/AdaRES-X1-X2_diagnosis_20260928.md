# W2 checkpoint: two fixed-cohort mechanism diagnoses

2026-09-28 11:31 Asia/Shanghai. server-2 single-GPU native inference, W2
30-epoch best checkpoint, first 64 ScanRefer training scenes with augmentation
disabled, 1660 expressions. These results are diagnostic only, not validation
metrics or a matched training effect. Both scripts reconstruct W2's native
indicator and fail if its scores disagree.

| Route | Causal question and paired intervention | Observation | Disposition |
|---|---|---|---|
| X1, competition-context dependence | Replace the W2 competition context by zero, versus rolling that context one query; preserve the original head and all other evidence. If meaningful competitors are needed for selection, deleting context should substantially worsen ranking and a mismatched context should not be systematically superior. | Native 0.76736, zero context 0.76579, rolled context 0.77347 mean point IoU on train64. Zero context improved 60 / worsened 64 expressions by >0.001, rolled improved 91 / worsened 98. 257 and 541 selections flipped respectively. 466 native material selection errors, only 99 had the oracle in the original top 3. One repeated-forward IoU mismatch; GPU0 peak inference reserve 3.477 GiB. | Reject the simple context-removal realization. The rolled control is unexpectedly better on this cohort; it is not an identifiable improvement attributable to correct competition and cannot justify a training admission. Investigate the one repeat mismatch before using finer margins. |
| X2, directed containment | Replace symmetric point-weighted IoU edges with directed intersection / source area, versus a query-rolled containment graph; keep the trained head fixed. If asymmetric inclusion resolves W2's selection deficit, it should separate correct queries and outperform a misaligned graph. | Native 0.76733, containment 0.76608, rolled graph 0.76770. The actual containment intervention improved 4 / worsened 10 expressions by >0.001; 20 selections flipped. The shifted control improved 67 / worsened 69. 466 material errors; repeat mismatch 0; GPU1 peak inference reserve 3.477 GiB. | Reject direct inclusion-edge substitution. Some oracle margins move (283/466 versus shifted 253/466), but task-aligned selections do not improve. A new trainable directed graph would need an independently identified deficit and a native matched control; this frozen substitution is not that evidence. |

Both test definitions were selected before their respective native executions in
this tick, but there is no previously frozen written preregistration; do not
present this as preregistered efficacy. Two independent questions (whether
competition context is necessary, whether overlap direction is missing) yielded
specific negative interventions. The favorable X1 rolled-control cohort mean
could reflect a confound, random query order, or score instability; it is not
evidence for a replacement. Scene-level uncertainty and repeated training
seeds have not been measured. Resume only on a distinct inference-available
mechanism with a prediction-blind deficit and predeclared matched controls.

Raw per-expression evidence:
`AdaRES-X1_context_train64_20260928.json`,
`AdaRES-X2_containment_train64_20260928.json`.
Scripts: `AdaRES/tools/diagnose_adares_x1_competition_context.py`,
`AdaRES/tools/diagnose_adares_x2_containment.py`.
