# Experiment Result Analysis: AdaRES-Y39

## 1. Basic Info
Canonical done_complete; original PID102267 absent before/after read; GPU1. Config adares_y39_point_distribution_res.yaml. Original start/exit/duration unknown; local frozen evaluation21:59:12.860011->22:10:35.313285+08 on2026-10-11. No original checkpoint overwrite/resume.
Last epoch30, best epoch23, bestSHA4a50225433aa755bf9b0f5643b9bcee8b025c828dcd5aaede2bc092b95fb68e1.

## 2. Best Metrics
| Metric | FIRST frozen full | Source epoch |
|---|---|---|
| mIoU | 0.5077059831652888 | 23 |
| Acc25 | 0.6069625578460244 | 23 |
| Acc50 | 0.5612116112747161 | 23 |
| spmIoU | 0.47172301744895867 | 23 |
9508 expressions/141 scenes/twice EACH Y39/W2/B1/FIRST primary/128 native queries/one argmax; do not substitute original rounded best.

## 3. Convergence Summary
Original best metadata0.5078108310699463 at23/30; original last30 and model/optimizer/scheduler verified. This does not prove optimal epoch or convergence; no validation epoch selection or retune.

## 4. Comparison
FIRST W20.5100098499225277/B10.5022241613397069; Y39-W2-0.0023038667572389342, CI[-0.011360390287967127, 0.006338310071024288]. Y39-B10.005481821825581824, CI[-0.0032655849737820318, 0.014512773945359362].
Whole-scene10000 bootstrap/seed1999/TRAIN cuts1487/3325. Every observed all/size interval crosses0; no predeclared upper<0 harm trigger, not no-harm/equivalence. Small-versus-W2 point estimate is adverse(-.0147890).
Single-object population9508/141; multi-object0, no CI or safety inference. Official unique/multiple metadata unknown, not substituted with object count.
Repeat maximum expression difference Y39/W2/B1=1/.0299056/.9738956; nearly equal aggregate means do not imply reliable expression-level repeatability.
Trained stress median Y39/W2=0.9877400118857622/0.9408668726682663s; reserved=13.513671875/13.400390625GiB; params164901255/164629637. Serial frozen TRAIN8+maxpair728/6780, two warmups/three readings; temporary updates in memory only. No Pareto superiority.

## 5. Conclusion
SAME stableY39 complete_inconclusive. Target.515 unmet; retain W2, no promotion/resume/retune/renamed successor. All initial negative ceiling/adverse losses/incorrect-control learning and repeat limitations remain.

## 6. Crash Info
Not a canonical crash. Bounded terminal tail had no traceback/OOM; not whole-log proof. Y41 PID116906/GPU0 and source retained through probes.
Hashed sources: [{"path": "pipeline/analyses/y39_terminal_current_062ae665ba824deba180a65efb4c44cf.json", "sha256": "e59cfb5d84cb3974e39f34633e6c17d71dcff0f9eac7fa6b9f6ca0187fd83a02"}, {"path": "pipeline/analyses/y39_frozen_full_current_062ae665ba824deba180a65efb4c44cf.json", "sha256": "cdb0188acbae92545e501549ce5102bcd78309f8dde0fc4126e6a1039cad44b7"}, {"path": "pipeline/analyses/y39_clustered_comparison_062ae665ba824deba180a65efb4c44cf.json", "sha256": "3c2d8aa063b4ad3410e3845ba4bf66877deccca635409755953a5ed1aade9b10"}, {"path": "pipeline/analyses/y39_object_groups_062ae665ba824deba180a65efb4c44cf.json", "sha256": "e4eff37b7e5adcc427d68c36bf7b2e2b996e96839dd9bc7a545ba2ea42a710a8"}, {"path": "pipeline/analyses/y39_trained_cost_current_062ae665ba824deba180a65efb4c44cf.json", "sha256": "b4d5145529cdc2354a96c15767b490f5e3fd59aea05dc868177be03487f88ddd"}]
