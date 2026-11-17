# Experiment Result Analysis: AdaRES-Y41
Observed 2026-10-13T02:31:41.353234+08:00.

## 1. Basic Info
Canonical done_complete. Original PID116906 absent; GPU0. Config
adares_y41_boundary_probability_res.yaml. Start/exit/duration unknown.
Original last30/best21; bestSHA490a0a1c50d69293836302f92b5c9b6ebfbbc15655186d2bf9bc8e2195a99c88.
Checkpoints/model/optimizer/scheduler read-only. Evaluation9508/141 twice
EACH Y41/W2/B1; FIRST primary, native128 queries/one argmax.

## 2. Best Metrics
| Metric | FIRST Frozen Full | Checkpoint Epoch |
|---|---|---|
| mIoU | 0.5090471194652967 | 21 |
| Acc25 | 0.6109591922591502 | 21 |
| Acc50 | 0.562263357172907 | 21 |
| spmIoU | 0.47346567242705745 | 21 |
Original best metadata0.5090538263320923 is not a replacement for frozen evaluation.

## 3. Convergence Summary
Fixed30 completed; best first saved at21. Final rounded log mIoU.502.
This is not evidence of an optimal epoch or convergence; no retuning/resume.

## 4. Comparison
Y41-W2-0.0010580409735824786/scene95CI[-0.009716837626550932, 0.007546362684877337].
Y41-B10.006825355109327934/
CI[-0.0030759390807791336, 0.016671534487291105].
Whole-scene10000 bootstrap/seed1999/TRAINcuts1487/3325. No observed all/size
upper<0 harm trigger; crossing0 is inconclusive, not equivalence/no-harm.
Medium versus W2 point estimate-0.010408312330342668
is adverse. Small versus B1 interval is positive, not overall superiority.
Single-object9508/141; multi-object0/0 insufficient. Official unique/multiple
unknown; object count is not a substitute.
Per-model repeat maxima{"y41": 0.0, "w2": 0.9763062000274658, "b1": 0.9738956093788147};
aggregate repeat means{"y41": 0.5090471194652967, "w2": 0.5099093530999647, "b1": 0.5021193352847256}.
Trained paired cost{"candidate": {"train8": {"median_seconds": 0.08503318950533867, "descriptive_p95_seconds": 0.1312100214883685, "peak_allocated_gib": 0.9609737396240234, "peak_reserved_gib": 1.234375}, "largest_pair": {"median_seconds": 0.21997223142534494, "descriptive_p95_seconds": 0.2535836147144437, "peak_allocated_gib": 1.2439393997192383, "peak_reserved_gib": 1.7578125}, "native_train_stress": {"median_seconds": 1.090612981468439, "peak_allocated_gib": 13.28124475479126, "peak_reserved_gib": 13.77734375}}, "parent": {"train8": {"median_seconds": 0.08171113952994347, "descriptive_p95_seconds": 0.12730055321007966, "peak_allocated_gib": 0.9607234001159668, "peak_reserved_gib": 1.234375}, "largest_pair": {"median_seconds": 0.21767617482692003, "descriptive_p95_seconds": 0.2529872730374336, "peak_allocated_gib": 1.2436890602111816, "peak_reserved_gib": 1.7578125}, "native_train_stress": {"median_seconds": 1.0807782839983702, "peak_allocated_gib": 12.850954532623291, "peak_reserved_gib": 13.400390625}}}; parameters{"candidate": 164696711, "parent": 164629637}.
Frozen TRAIN8 plus pressurepair728/6780,2warm/3read; temporary native updates
memory-only. Candidate-then-parent order/descriptive p95/single checkpoint
limit cost interpretation. No trained Pareto superiority.

## 5. Conclusion
SAME Y41 complete_inconclusive. Target.515 unmet; retain W2. No promotion,
retune/resume or renamed equivalent. Native confidence boundary route remains
unproven. All original evidence and failed cost-adapter log preserved.

## 6. Crash Info
No canonical training crash. Cost-adapter first attempt failed before model
execution due to protected-job bytes; v2 corrected Y42 GPU1 binding only.
Bounded tail no traceback/OOM is not whole-log proof. Original Y42 PID91215/
GPU1/source retained. Hashed evidence:
[{"path": "pipeline/analyses/y41_terminal_0422_0422d0d8ad4a4a0288d9e5d7e4e53f73.json", "sha256": "f5a2d3da831c65d64d50d7cddf1a77f6dd30c54c0864d9d3911daeccf7d9b022"}, {"path": "pipeline/analyses/y41_frozen_full_0422_0422d0d8ad4a4a0288d9e5d7e4e53f73.json", "sha256": "f2a0706523b8ea2ec74cdc37af2046d7409cdd7a9a9378d83183117eb137b2bc"}, {"path": "pipeline/analyses/y41_clustered_comparison_0422d0d8ad4a4a0288d9e5d7e4e53f73.json", "sha256": "8527d791d4e25a9d75929f16eb258a3f536aab41c572db0a2231886ecf1b5f00"}, {"path": "pipeline/analyses/y41_object_groups_0422d0d8ad4a4a0288d9e5d7e4e53f73.json", "sha256": "cda9d2f921289f5235bf196a0d47c06694aa73038f6fd4fe598da38bf16d860e"}, {"path": "pipeline/analyses/y41_trained_cost_0422_v2_0422d0d8ad4a4a0288d9e5d7e4e53f73.json", "sha256": "6def3829be9816866b2f0caa30bd8f59d5c374a3f08c8fb6703d35888cb4bd45"}, {"path": "pipeline/analyses/y41_cost_binding_repair_0422d0d8ad4a4a0288d9e5d7e4e53f73.json", "sha256": "8290593cd4e6e4828f4b6226c72528a34b17aa0f36226bb498db0a6dd41d10c7"}]
