# AdaRES Y1-Y4: paired selection and mechanism audit

2026-09-28 13:07 Asia/Shanghai. All native tests used server-2, the
unchanged W2 30-epoch best checkpoint and fixed non-augmented first 64
training / first 32 validation scenes. B4-E30 was paired with W2 in Y1.
The train and validation cohorts contain 1660 / 851 expressions. Predictions
are diagnostic point IoUs on these subsets, **not** official full ScanRefer
validation mIoU or estimates of a retrained model. Scene is the clustering
unit; no confidence interval or independent training-seed noise floor exists.
No validation hyperparameter or epoch selection was performed.

| Stable ID | Predeclared question and matched control | Native result | Decision |
|---|---|---|---|
| Y1, selection-versus-mask | Match W2 and B4 by scene, expression, scan, annotation; compare selected and oracle point IoU, double forward. | Train selected W2 0.767331 vs B4 0.822562; oracle 0.903576 vs 0.910354. Val selected 0.529628 vs 0.541653; oracle 0.865564 vs 0.864532. All four runs had 0 repeat mismatches. | W2's selection is worse than B4 on these subsets despite similar oracle availability. Y1 is a completed diagnosis, not a candidate trainable mechanism. |
| Y2, target-size moderation | Train Q1=0.005643, Q3=0.023026 target-point fractions frozen before val; compare paired W2-B4 differences by size and one-step within-scene rotated-size control. | Train expression-mean selected delta: small -0.07744, large -0.03372; scene-means -0.04630 / -0.04494. Val expression means -0.01776 / +0.00279, scene-means -0.06578 / +0.01349. Val rotated small / large expression deltas -0.00241 / +0.01994, a similar separation; oracle delta varies by group. | Reject *specific* causal inference from this moderation: train scene means erase the contrast and val rotated control also separates. It motivates no count-transform tuning or training admission. |
| Y3, point-mass identity | Frozen W2 query/mask tensors; compare native mass to uniform or within-expression cyclically rotated valid superpoint masses. | Train native/uniform/rotated 0.767331/0.765718/0.766099; material wins/losses native vs uniform 5/5, vs rotated 7/8. Val 0.529628/0.529628/0.527510; native vs uniform 0/0, vs rotated 4/0. Repeat mismatches 0. | Reject the strong claim that correct point-mass identities drive the trained ranking. Small frozen-weight effects do not prove a retrained variant cannot help. |
| Y4, supervision-support coverage | Compare oracle/selected query sampled-seed in-object support to cyclically shifted support in fixed scenes; instance-ID indexing was corrected after a pre-result failure. | Among train 466 material errors: true in-object oracle 215, selected 217, shifted oracle 38; among val 455 errors: oracle 263, selected 132, shifted oracle 50. Original GT superpoint proxy and corrected instance-ID subset agree in these cohorts. Train corrected pass had 1 repeat-row mismatch; val 0. | Reject the predeclared explanation that error oracles are *less* likely than selected queries to receive in-object support. The complete native loss additionally force-matches a nearest query, which this diagnostic does not reconstruct. |

Y4's first run failed before producing results because `sp_ins_labels`
is collated once per scene, not expression. The second run used
`gt_spmasks` as an explicitly documented proxy. Inspection of native
`scannetv2.py` then allowed a separate, amended instance-ID run;
the original proxy files were preserved. Do not use the proxy as an
exact criterion-label audit. For Y4, a repeat-row mismatch blocks
fine-grained claims on that one training expression.

All four routes are independent questions at the level tested:
source of selection error, target-size moderation, mass-identity
intervention, and in-object seed supervision coverage. Y1 is
observational and Y2 is a stratification analysis; neither supplies
an implemented independent trainable intervention. Y3/Y4 controls
disconfirm the respective immediate mechanisms. None has native
backward, resource, independent-package or goal-constraint admission
evidence for a *new* model. The six gates are not passed and no queue
entry or 30-epoch training job is justified.

Resume research only on a distinct, prediction-blind and inference-valid
mechanism with preregistered matched intervention, repeatability and
native backward; in particular a full seed-label/nearest-match audit
would require exact criterion reconstruction and stable forward results.
Do not rename rejected substitutions, tune validation thresholds or
infer global hypothesis exhaustion from this bounded audit.

Raw files: `AdaRES-y1_{w2,b4}_{train64,val32}_20260928.json`,
`AdaRES-y2_size_{train64,val32}_20260928.json`,
`AdaRES-Y2_size_moderation_20260928.json`,
`AdaRES-y3_mass_{train64,val32}_20260928.json`,
`AdaRES-y4_coverage_{train64,val32}_20260928.json`,
`AdaRES-y4_instance_{train64,val32}_20260928.json`.
