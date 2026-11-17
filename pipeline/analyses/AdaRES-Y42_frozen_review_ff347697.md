# Experiment Result Analysis: AdaRES-Y42
Observed 2026-10-13T06:21:02.884974+08:00; same stableID.

## 1. Basic Info
Configconfigs_song/model_paper5/adares_y42_point_covariance_res.yaml.
Server2/GPU1/originalPID91215 absent/no matching original config process.
Canonical done_complete; actual training start/exit/duration unknown.
Last30/best21; bestSHA6d8caaf4bb31e78404b4297e75171258e87505dad938af4f92f8d34cdb8a8172; original last/best
read-only, model/optimizer/scheduler intact. Canonical finished_at is monitor
observation, not verified remote process exit time.
Full9508expressions/141scenes, Y42/W2/B1 twiceEACH/FIRSTprimary/native128/
argmax1. Capturewindow05:59:43.541918->06:11:03.674806+08 on2026-10-13.

## 2. Best Metrics
| Metric | FIRST Frozen Full | Checkpoint Epoch |
|---|---|---|
| mIoU | 0.5049827554145192 | 21 |
| Acc25 | 0.605490113588557 | 21 |
| Acc50 | 0.5567942785023139 | 21 |
| spmIoU | 0.4695762313285676 | 21 |
Original best metadata0.5049824118614197 is not a replacement for frozen full.

## 3. Convergence Summary
Fixed30 completed; best first saved21. Bounded final30log rounded mIoU.500.
No optimal-epoch, convergence or retuning claim. No restart or resumed training.

## 4. Comparison
FIRSTW20.5101048498778671/B10.5022241077375282.
Y42-W2-0.005122094463347903/scene95CI
[-0.01276031528930059, 0.0021337033584317463]; versusB1
0.002758647676990958/CI
[-0.006465807597994274, 0.0121038061802957].
Whole-scene10000bootstrapseed1999/TRAINcuts1487/3325. All/size intervals
cross0/no upper<0 materialharm trigger, NOT equivalence/no-harm. Largeversus
B1 adversepoint-0.011058821050532878/CI
[-0.028201928993217995, 0.005867835941493387] retained.
Objectcountsingle9508/141; multi0/0 insufficient. Officialunique/multiple
unknown; object count is not a substitute.
RepeatsmIoU{"y42": 0.5049846144895332, "w2": 0.5101079951888933, "b1": 0.5021193352847256}; perexprrepeatmax
{"y42": 0.017288565635681152, "w2": 0.029905617237091064, "b1": 0.9738956093788147}.
Trained pairedcost summary{"candidate": {"train8": {"median_seconds": 0.08917046338319778, "descriptive_p95_seconds": 0.13317642053589224, "peak_allocated_gib": 0.9622087478637695, "peak_reserved_gib": 1.30859375}, "largest_pair": {"median_seconds": 0.22249810211360455, "descriptive_p95_seconds": 0.2547558187507093, "peak_allocated_gib": 1.2451744079589844, "peak_reserved_gib": 1.8203125}, "native_train_stress": {"median_seconds": 1.13856553286314, "peak_allocated_gib": 13.413899421691895, "peak_reserved_gib": 14.046875}}, "parent": {"train8": {"median_seconds": 0.08057432528585196, "descriptive_p95_seconds": 0.1197859245352447, "peak_allocated_gib": 0.9607234001159668, "peak_reserved_gib": 1.234375}, "largest_pair": {"median_seconds": 0.19508167263120413, "descriptive_p95_seconds": 0.22497998923063278, "peak_allocated_gib": 1.2436890602111816, "peak_reserved_gib": 1.7578125}, "native_train_stress": {"median_seconds": 0.9366967007517815, "peak_allocated_gib": 12.850881576538086, "peak_reserved_gib": 13.400390625}}}; parameters
{"candidate": 165028231, "parent": 164629637}. TRAIN8+pressurepair728/6780,2warm3read,
serialpredictbatch1/nativebatch2backward/AdamW. Temporary updates memory-only;
immutable original checkp hashes. Candidate-then-parent order,descriptivep95/
singlecheckpoint/seed/split limitinterpretation; no trainedPareto superiority.

## 5. Conclusion
SAMEY42 complete_inconclusive, unfavorable W2 pointestimate; target.515 unmet.
RetainW2. No promotion/retune/restart/renamed equivalent. Full covariance
benefit unproven. New Y44 joint-coordinate-feature pairing is a distinct
preregistered mechanism, not a rescue or tunedY42.

## 6. Crash Info
No canonical training crash. Bounded tail traceback/OOM absence is not whole
log proof. Full/cost probes completed and protected originalY43GPU0/PID13155/
sources unchanged. Y44 localuploadGBK/native-sort test-adapter faults are
separate; neither is a Y42 scientific or training crash.
Hashed evidence:
[{"path": "pipeline/analyses/y42_terminal_ff34_ff3476973900467da752992f8195291b.json", "sha256": "ecb8ce436933e17c0e0f2e2e2a5b0e6a6eccb9ab85ede2458173f7f7a26c58ec"}, {"path": "pipeline/analyses/y42_frozen_full_ff34_ff3476973900467da752992f8195291b.json", "sha256": "490bff7421cac7d15d972ec5d90222d1a59c264c2d62dfcc547922dd584e2fdd"}, {"path": "pipeline/analyses/y42_clustered_comparison_ff3476973900467da752992f8195291b.json", "sha256": "ef7f9d41af2b9967685c89d1c8995d63d7cc204823b4a68569b64b25ccf76c9f"}, {"path": "pipeline/analyses/y42_object_groups_ff3476973900467da752992f8195291b.json", "sha256": "b411cce31884aa7bc9ed7296d54a6619bc392ae8f4a1af760480e803a02ca3ff"}, {"path": "pipeline/analyses/y42_trained_cost_ff34_ff3476973900467da752992f8195291b.json", "sha256": "f85288dd3e1385152a22c06f86637a28522972ecce1998cd262f2365739547c4"}]
