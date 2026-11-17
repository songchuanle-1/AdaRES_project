# Experiment Design: AdaRES-Y44
Created 2026-10-13T06:10:51.996766+08:00; PROSPECTIVE, not admitted.

## 1. Motivation
Project goal description verbatim: AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515.
Snapshotactive1/target2/budget1 afterY42 terminal. The test asks whether an
explicit joint spatial-feature association can improve query confidence.
Stable diagnosis came first; no route rejection based on conceptual similarity.

## 2. Current Best Results
| Experiment ID | Config | State | Best mIoU | Acc50 | spmIoU | Epochs |
|---|---|---|---|---|---|---|
| W2 | adares_w2_pointmass_competition_res | done | historical around.510 | unknowncurrent | unknowncurrent | 30 |
| Y41 | adares_y41_boundary_probability_res | complete_inconclusive | frozenFIRST.50904712 | .56226336 | .47346567 | 30 |
| Y42 | adares_y42_point_covariance_res | done_complete | metadata.50498241/frozenpending | unknownfrozen | unknownfrozen | 30 |
| Y43 | adares_y43_point_asymmetry_res | runningGPU0 | unknownstructured | unknown | unknown | fixed30ongoing |

## 3. Analysis & Direction
No established superiority for feature variance/covariance/third moments.
Y44 adds a distinct joint statistic: E[(coord-mean_coord) outer(feature-mean)].
WithinSP feature pairing preserves every feature marginal moment but changes
this quantity. Actual oldreadout32seed invariance0/crossgap>=9 supports a
narrow explicit information gap. Native task relevance and utility pending.
No banned closed direction reopened or scalar-only configuration variant.

## 4. Experiment Plan
| # | ID | Config Path | Purpose / Hypothesis | Expected Result | Priority | Depends On |
|---|---|---|---|---|---|---|
| 1 | AdaRES-Y44 | configs_song/model_paper5/adares_y44_coordinate_feature_res.yaml | explicit all-coordinate/all-feature association for final confidence | correct/wrong pairing separable; trained benefit unknown | P0 | actual six gates |
Exact raw96-input readout with inherited256 modelwidth, FG/BG point mass and
query; detached masks. Native coords_float gathered point order, SP labels/
expression duplication. FrozenTRAIN8 indices0,44,60,101,115,126,137,171,
released-initfirst4. Native correct/wrongpair/rolled/zero/wrongmasks/disabled/
repeat, directpadding/permutations/detach, all8gradient and optimizerupdate/
checkpointresume/disabledphysicalW2 equality. No diagnostic weight deployment.
ProductionreleasedSPFormer/nativeAdamW,PolyLR/loss/seed1999/fixed30/native128/
singleargmax unchanged; no feature/axis/neighbor/radius/threshold/temperature/
lossweight/hiddenwidth/epoch sweep.
Posttrain Y44/W2/B1 each9508expr/141scenes twice/FIRST primary. Whole-scene
bootstrap10000seed1999; TRAINcuts1487/3325; objectcount groups and official
unique/multiple unknown; empirical repeatfloor nottrainingseeduncertainty.
Frozen materialharm if a meaningful subgroup's scene95CI upper<0 versus
either W2 or B1; crossing0 inconclusive not no-harm/equivalence. Trained
paired TRAIN8/pressurepair728,6780 cost 2warmups/3reads; params<1%added,
trainingpressure reserved<24GiB/noOOM/nonfinite. Report observed time and
memory regressions, no initialized/trained Pareto superiority assumed.
Only goal-check may establish target/Phase2; no rounded-log promotion.

## 5. GPU Allocation Plan
| Server | GPU | Assignment | Config | Justification |
|---|---|---|---|---|
| server-2 | 0 | preserve originalY43/PID13155 | adares_y43_point_asymmetry_res | running authorized original |
| server-2 | 1 | exclusive bounded tests, thenY44 ONLY if admitted | adares_y44_coordinate_feature_res | Y42 terminal/liveidleness still required |
Whitelist0,1/one independent singleGPU jobcard/maxactive2; no server6 orother
project/scheduler/background/newthread. No GPU is assumed free from oldsnapshot.

## 6. Implementation Checklist
- [x] Stable diagnosis and prospective controls frozen before data.
- [x] Seven independent local source files/YAML; raw3x32 cross finalonly.
- [ ] Same-interpreter parentAST/config/source byte identity and actualclassCPU.
- [ ] Native diagnosis/controls/backward/physicalresume/resources verified.
- [ ] Exactly tested7paths immutable commit, rawgitblobs hashes bound.
- [ ] Motivation/review/goalconstraints/sixgate receipt and decision bound.
- [ ] Canonical server2GPU1 scopeddispatch; verify increasing native steps.
All unexecuted checkboxes remain pending. No automatic push/all-server sync.

## 7. Risk Assessment
Raw coordinate units/moment magnitude and geometry already in backbone can
make branch redundant or brittle. Pairing-counterfactual freezes backbone,
not an identical raw-point-cloud intervention. Confidence changes may worsen
selection; initialized adversities require investigation, not threshold edits.
Newparams/memory/time must be measured. Fixed native full/single seedcheckp/
repeat instabilities/unknownofficialuniqueness limit claims. Preserve originals;
if process lacksprogress do not blindlyrestart. No broadened privilege.

## 8. Smell Check
smell2/3: another small confidence attachment/statistic could look incremental.
Defense: all3x32 joint coordinate-feature aggregation exposes associations
unchanged by complete feature marginal summaries; the actual matched pairing
test differs in causal input, not coefficient. Query-conditioned FG/BG cross
readout is a new information interaction, not norm/scaling-only. Dimensions
structurallydetermined/inherited; no numericsearch. The mechanism hypothesis
is explicit and falsifiable, but contribution/novelty/necessity remain unknown.
Status DRAFT; actual tests and review still required for admission.
