# AdaRES Y5/Y6: native indicator supervision audit

2026-09-28 14:59 Asia/Shanghai. User-requested foreground design tick,
snapshot 14:46:14, deadline 15:46:14 Asia/Shanghai. W2 30-epoch best
checkpoint, unchanged; first 64 non-augmented training scenes (1660
expressions) and first 32 validation scenes (851 expressions), seed 1999.
Native server-2 GPU0 inference reconstructs the criterion's actual final
positive query set by calling `match_sampled_ins` with expanded superpoint
instance labels and positions. A per-expression cyclic shift of the same
positive set is the negative control. Results are diagnostic subset point
IoUs and label coverage, **not** full ScanRefer validation mIoU, a
trainable improvement, or a population confidence interval.

| ID | Frozen question | Observed result | Disposition |
|---|---|---|---|
| Y5: full-label oracle omission | Does the complete nearest-plus-in-object positive set omit error oracles more often than selected queries, while beating shifted membership? | Train64 material errors 466: oracle positive 239, selected positive 245, shifted oracle 39; repeat mismatch 1. Val32 errors 455: oracle positive 277, selected positive 133, shifted oracle 52; repeat mismatch 0. Val paired oracle-only 220, selected-only 76, both 57, neither 102. | Reject the claimed systematic *omission* by native labels on this fixed cohort. The labels may still be insufficient for ranking; coverage alone is not a remedy. |
| Y6: within-positive quality order | Among expressions with at least two true positives, do positive masks differ in quality while the indicator persistently misorders them relative to a shifted-positive control? | Train64 error subset 307 eligible: true positive mean IoU range 0.1032, score-to-best gap 0.0613, misordered (>0.05) 0.4886; shifted set misordered 0.3192. Val32 errors 379 eligible: true range 0.1498, gap 0.0685, misordered 0.3799; shifted misordered 0.3720. Val scene means misordered 0.3793 vs 0.3679. Train repeat mismatch 1, val 0. | The predicted positive-quality variation exists, but the *predeclared control separation* in misordering does not transfer to val32. Reject this direct rationale for a new loss. |

The shifted Y6 set spans many unrelated masks (val error mean range
0.4903 versus true 0.1498), so its score-to-best gap is not a
like-for-like efficacy estimate. The predeclared misorder comparison
nevertheless lacks specific separation on val32. One train64 repeated
row differs for each route: avoid fine-grained train claims. No independent
training seeds or scene-clustered interval were collected. G1's prior
all-query IoU soft-target training ended at full mIoU 0.4980; C2-E30's
hard-negative margin ended at 0.5014. This audit does not reopen either
variant or tune a threshold/margin on validation.

Neither route supplies a novel implemented inference-time module,
native backward for such a module, a resource profile, independent
package or prospective six-gate admission receipt. The question of a
*distinct* inference-valid ranking mechanism remains open, but these
two specific explanations do not warrant 30-epoch dispatch.

Resume after the short bounded cooldown only with independently justified
prediction-blind evidence for an inference-available signal and a
preregistered, matched, scene-level control. A complete positive-label
audit by itself cannot imply that continuous quality supervision will
help; a future proposal must account for G1's negative training result
and establish a materially different causal mechanism.
