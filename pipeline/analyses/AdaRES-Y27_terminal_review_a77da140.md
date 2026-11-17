# Experiment Result Analysis: AdaRES-Y27
2026-10-05, owned event a77da1408e7d4f289f6dfcb5cc768332.

## 1. Basic Info
- Config: `configs_song/model_paper5/adares_y27_relative_peer_res.yaml`.
- Server: server-2, GPU1 only; original PID9901 absent before/after terminal read.
- Local launch: 2026-10-04 19:30:52.264089+08. Observed completed:
  2026-10-05 17:51:57 to17:52:01+08; exact finish time/duration unknown.
- Native fixed30 completed: finalVal30/30 363/363, last checkpoint metaepoch30,
  best metaepoch23, model/optimizer/scheduler present, stable CPU file hashes.
- Canonical state at17:49:26 preflight remains running/deathwatch1/3.
  This report does not forge queue/experiment lifecycle or rerun preflight.
- Research disposition: complete frozen evaluation protocol; efficacy inconclusive.

## 2. Best Metrics
All models use their trained best checkpoint, full9508expressions/141scenes,
two native passes, seed1999; no thresholds/features/weights/epoch tuning.

| Model | mIoU Pass1 / Pass2 | Acc25 Pass1 | Acc50 Pass1 | spmIoU Pass1 |
|---|---|---|---|---|
| Y27, bestepoch23 | .509987856 / .509987856 | .608960875 | .562999579 | .475034731 |
| W2 | .510107995 / .510107995 | .610748843 | .562999579 | .473719922 |
| B1 | .502119335 / .502221764 | .610328145 | .552797644 | .466681174 |

Y27 checkpoint metadata best_miou=.5099911689758301 is a training-log value,
not substituted for these frozen full-evaluation results. Target.515 unmet.
Available128queries/selected1 for all models.

## 3. Convergence Summary
- Best checkpoint first saved at epoch23; training continued to the fixed30.
- Final bounded log contains Val30/30 completion and last.pth save.
  Last reported mIOU is rounded.505; precise final-epoch auxiliary values unknown.
- No original PID or same-config duplicate process at17:52 and18:10.
  This is physical completion evidence, not trained superiority.
- No Traceback/OOM in the bounded24KB terminal tail; not a whole-log guarantee.

## 4. Comparison
Expression-weighted paired differences, resampling whole scan clusters,
10000draws/seed1999/95%CI. Identity(scan_id,object_ids,ann_id) and target sizes
match across captures. Frozen TRAIN cuts1487/3325, not validation-selected.

| Group | Expressions / Scenes | Delta vsW2 | CI95 vsW2 | Delta vsB1 | CI95 vsB1 |
|---|---|---|---|---|---|
| all | 9508 / 141 | -.000120139 | [-.008415036, .008403480] | +.007868521 | [-.001015767, .017045974] |
| small | 2816 / 124 | -.011868499 | [-.028264348, .004244841] | +.010425228 | [-.007853165, .029319703] |
| medium | 3304 / 136 | -.003622096 | [-.017186898, .010114773] | +.009365900 | [-.005409051, .023415934] |
| large | 3388 / 135 | +.013059862 | [-.000028637, .026172319] | +.004283211 | [-.009576742, .018242669] |

All intervals include zero. The preregistered material-harm rule CIupper<0
is not met in any group; confident benefit is also not demonstrated.
The large-group lower endpoint is negative, not a positive result rounded to0.
No equivalence proof, subgroup selection, posthoc feature filter or repeat search.

Observed maximum expression-repeat differences: Y27=0/W2=0/B1=.973895609;
B1 mean absolute repeat difference all=.000102429/large=.000287454.
These are two-pass observations, not global determinism or universal noise bounds.
One trained seed/best checkpoint per method; bootstrap covers scene sampling,
not training-seed uncertainty. Official unique/multiple metadata unknown.

Frozen trained serial inference cost: TRAIN8 distinct scenes plus728/6780,
batch1/two warmups/three readings, candidate then parent.

| Group | Median Seconds Y27 / W2 | Descriptive p95 Y27 / W2 | Peak Allocated GiB Y27 / W2 | Peak Reserved GiB Y27 / W2 |
|---|---|---|---|---|
| train8 | .078462590 / .075131276 | .123888619 / .119699666 | .961948872 / .960723400 | 1.298828125 / 1.234375000 |
| largest pair | .196855826 / .192487053 | .228708435 / .223807354 | 1.244914532 / 1.243689060 | 1.759765625 / 1.757812500 |

Parameters164958597vs164629637, added328960. Median cost increases about4.4%/
2.3%; allocated/reserved peaks also rise in these frozen measurements.
Three readings/serial order are descriptive, not reliable tail estimates or
hardware-effect isolation. Full-validation allocated/reserved peaks are
4.044630/4.617188GiB vs4.042351/6.486328GiB; do not cherry-pick lower
reserved allocator memory to claim Pareto dominance. Trained lightweight
Pareto benefit is not demonstrated. Historical pretrain cost regressions remain.

## 5. Conclusion
- Inconclusive trained benefit/harm; target unmet; no champion promotion.
- Complete means fixed30 and frozen evaluation protocol completed, not effective.
- No restart, validation tuning, gate weakening, renamed variant or generic
  ban on all relative-geometry mechanisms. Existing adverse controls preserved.
- Canonical lifecycle settles through future ordinary monitor ticks. Current
  preflight budget0 is not raised merely because physical GPU1 is idle.
- After all measurements,18:10:06to18:10:10 GPU1=1MiB/util0/no compute PID;
  original last/best hashes unchanged. GPU0 never queried or used.
- New measurement wrappers/synthetic arithmetic tests affect analysis only.
  Prior lock-check-path and expanded-owner-read limitations are preserved in
  Y27_measurement_checks_a77da140.md; no historical evidence is rewritten.

## 6. Crash Info
No observed native training crash. Bounded terminal evidence only.
Historical TRAIN-only progress-assertion failure8d0dd587/exact phase unknown
and phase-aware repair remain unchanged; they are not reclassified as crashes.
This tick's terminal/full captures/cluster analysis/serial cost/close audit
all returned success. Postflight/receipt/release are still pending at writing.

## Evidence
- y27_terminal_a77da1408e7d4f289f6dfcb5cc768332.json: a45fb8b2.
- y27_y27_full_eval_a77da1408e7d4f289f6dfcb5cc768332.json: a386df85.
- y27_w2_full_eval_a77da1408e7d4f289f6dfcb5cc768332.json: cf67c989.
- y27_b1_full_eval_a77da1408e7d4f289f6dfcb5cc768332.json: 24806191.
- y27_clustered_comparison_a77da1408e7d4f289f6dfcb5cc768332.json: 4384f87f.
- y27_trained_cost_a77da1408e7d4f289f6dfcb5cc768332.json: 234c0c2e.
- y27_posteval_close_a77da1408e7d4f289f6dfcb5cc768332.json: f49c2a59.
- Full SHA256 bindings are recorded on the stable Y27 candidate.
