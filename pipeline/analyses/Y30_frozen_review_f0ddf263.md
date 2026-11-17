# Y30 Frozen Post-Training Review

Owned foreground tick f0ddf263cf184531833a22722eba8c17.
Preflight snapshot 2026-10-07T07:48:40.837768Z; actual measurements below are
later observations, not a rewrite of that snapshot or canonical queue.

## Verified Training Boundary

The original PID39709 and all matching Y30 training processes were absent
before and after the 15:52:51-15:52:54+08 audit. Original last.pth has epoch30,
best.pth epoch20, best_miou 0.5043142437934875. Both retain model, optimizer
and scheduler and were unchanged during CPU reads. Source marker remains
4f5c31f57994b8f9fb640d652c913863dc310a95; all seven repaired candidate files
were byte-verified again. This establishes the fixed30 boundary, not efficacy.

## Frozen Full Evaluation

Each method independently completed two native full passes, each containing
9508 expressions and 141 scenes, native128 queries and one selected query.
Comparators are original trained W2/B1 best checkpoints. No threshold, loss,
feature, seed, epoch, subgroup-cut or selector change was made.

| Method | mIoU Pass1 | mIoU Pass2 | Acc25 Pass1 | Acc50 Pass1 | spmIoU Pass1 |
|---|---|---|---|---|---|
| Y30 | 0.504229394 | 0.504314270 | 0.605279764 | 0.557530501 | 0.470150293 |
| W2 | 0.510107995 | 0.510104850 | 0.610748843 | 0.562999579 | 0.473719922 |
| B1 | 0.502221764 | 0.502111562 | 0.610433319 | 0.552902819 | 0.466782141 |

Frozen whole-scene bootstrap10000/seed1999, expression-weighted paired
deltas, raw TRAIN size cuts1487/3325:

| Population | Expressions / Scenes | Delta vs W2 | Scene-Cluster 95% Interval |
|---|---|---|---|
| All | 9508 / 141 | -0.005878602 | [-0.014236543, 0.002295630] |
| Small | 2816 / 124 | -0.012915696 | [-0.031825345, 0.005038635] |
| Medium | 3304 / 136 | -0.017392276 | [-0.029706029, -0.004614514] |
| Large | 3388 / 135 | +0.011198624 | [-0.003609208, 0.026276323] |

The predeclared meaningful-subgroup harm rule is upper95%CI<0 versus W2.
Medium meets this rule. Overall crosses zero; this is inconclusive, not
equivalence and not population-wide significant harm. Large's positive point
estimate remains retained. Versus B1, overall delta+0.002007629 has interval
[-0.006615791, 0.010683316]; every size-group interval also crosses zero.
Neither pass reaches the unchanged target0.515.

Maximum per-expression repeat differences Y30/W2/B1 are
0.798350155/0.029905617/0.973895609. Mean absolute repeat difference in the
medium comparison is 0.000241631. These empirical observations do not establish
global determinism, a universal noise bound, or training-seed uncertainty.

## Trained Serial Cost

Each method used its own best weights, frozen TRAIN8 plus indices728/6780,
serial native predict batch1, two warmups and three readings per scene.
Separate native batch2 loss/backward/AdamW stress used trained weights and
a fresh native optimizer warmed twice; memory-only updates are not production
resume or checkpoint writes.

| Measurement | Y30 | W2 |
|---|---|---|
| Parameters | 165026951 | 164629637 |
| TRAIN8 median seconds | 0.076077576 | 0.073858140 |
| Largest-pair median seconds | 0.193387847 | 0.192472861 |
| Largest-pair peak reserved GiB | 1.759765625 | 1.757812500 |
| Native train stress median seconds | 0.938473882 | 0.924103791 |
| Native train stress peak reserved GiB | 13.755859375 | 13.400390625 |

All stress losses and reached gradients were finite. This adds397314
parameters. Three readings and fixed candidate-then-parent order do not
isolate transient hardware effects or establish reliable tail significance.
Feasible memory, a small parameter fraction or descriptive timing cannot
establish lightweight Pareto benefit, especially with the frozen medium harm.

## Decision And Scope

REJECT this SAMEY30 trained realization under the unchanged medium-group
material-harm rule. Keep W2 champion and Phase1 unmet; Phase2 remains unapproved.
Do not restart Y30, tune from validation, rebrand it as a new candidate, or
erase the adverse initialization/wrong-history controls and original strict
permutation failures. Repaired graph activity is not temporal necessity.
This is not a universal rejection of every temporal mechanism; causal
necessity, literature novelty and training-seed uncertainty remain unknown.

All original checkpoints were hash-verified unchanged again at
16:08:37-16:08:40+08. GPU1 then had1MiB/util0/no compute process. This is an
observation, not a reservation. GPU0 was never queried or used. Preflight queue
still reported running1; no canonical state/deathwatch counter was forced.
No new admission is authorized by this tick's budget0. Resume design after
canonical capacity opening with fresh whitelist/lease/occupation checks.

## Evidence

- Terminal: pipeline/analyses/y30_terminal_f0ddf263cf184531833a22722eba8c17.json
  SHA256 4c8922f4ffd7626521f98442fdd309a54fb07eb4847cca0797bc906f18f902c6
- Y30 full: pipeline/analyses/y30_y30_full_eval_f0ddf263cf184531833a22722eba8c17.json
  SHA256 1351de5a25bc37b761cb6f81ef1a0a1c63a5d13fb82583ed9e1ce735124b85d5
- W2 full: pipeline/analyses/y30_w2_full_eval_f0ddf263cf184531833a22722eba8c17.json
  SHA256 f4b512d8474628017a12c931817bd900bb61d7b78bce91f77aa4801e2d391445
- B1 full: pipeline/analyses/y30_b1_full_eval_f0ddf263cf184531833a22722eba8c17.json
  SHA256 fbefca006477284334c2e04a3773bc2e4b39429ad4654042ce126f5912483c23
- Clustered: pipeline/analyses/y30_clustered_comparison_f0ddf263cf184531833a22722eba8c17.json
  SHA256 bd055aefe62302caffbb1d04174b5e52aa0d46a6130f6a4d6760201e4d532862
- Trained cost: pipeline/analyses/y30_trained_cost_f0ddf263cf184531833a22722eba8c17.json
  SHA256 6d1f9eae16c9436caeed577455bd91ae121891bb472b3f833e56e8bf1af8778b
- Post-audit: pipeline/analyses/y30_postaudit_f0ddf263cf184531833a22722eba8c17.json
  SHA256 47b7f02585ee53e57de4f75518b44d78eea5eb70092e554582221b5d770a11b4
