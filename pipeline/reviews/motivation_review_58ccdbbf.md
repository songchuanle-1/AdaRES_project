# Motivation Review - AdaRES_project

## 1. Goal
- Current immutable source:pipeline/goal.json, SHA25653da862754afe00739a75c5ea6b1f184e0358813225b8f8c0c024a355737c966.
- Description:"AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515."
- TargetScanRefer.515, W2historical.5100/gap.0050/B1.5020. Phase1unmet.
- All7constraints preserved: server-2GPU0/1 only;singleGPU/onemaxpercard; no successor whileB3running and at most2independentcandidates; fixed30epoch B2/B4continuation without optimizer/schedulerreset; releasedSPFormer; reportmIoU/Acc25/Acc50/queries/latency/memory; no validationthreshold/temperature/weight/feature/epoch tuning.
- All3bans preserved: no server-6, no b005/CFNet-as-B0, no validationrankingmargin/occupancyweightscan. Policyresearch_on_capacity_gap=true/steady_ticks_before_design1/fill_to_capacitytrue/design_buffer0/max_candidates2. No goalchange.

## 2. Story Status
| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| W2 confidence competition | completed .5100 | frozen fullW2 .510108 | .515unmet |
| Y15 extra teacher query | rejected | full .230319/9508extra selections | teacher causal necessity unknown |
| Y18 point-measure mask loss | training | nativeTrain12steps3214->3217 | fixed30epoch efficacy pending |
| Y19 all-query metric labels | rejected | actualG1equivalence | cannot rename calibration |
| Y20 backward-route conflict | rejected | train64cosinepositive | whole-parameter interactions unknown |
| Y21 directional augmentation | rejected | train64directioncontrastnegative | camera-relative semantics unknown |

## 3. Experiment Assessment
| Experiment | mIoU | Relation To Goal | Contribution |
|---|---|---|---|
| G1 | historical .4980 | softquality indicator target | prior trained failure retained |
| Y15 | full .230319 | query completion | strong realized efficacy rejection |
| Y17 | full .504157 | same-target consistency | inconclusive paired intervals |
| Y18 | intermediate .4980 | exactpoint mask supervision | not final efficacy |

## 4. Gap Analysis
- Neither extraqueries nor renamedmetriclabels justify capacityfilling. No new inference-valid mechanism established this tick.
- Y20 conflict globaldirection is falsified despite activegraph; Y21 geometric sensitivity lacks directional-specific degradation. No trained improvement inferred.
- Nativeend-to-endtraining rawlogitnoise8.66407 retained; cachedlocalinput equality does not resolve its upstream cause.

## 5. Design Rationale
- Three stable prospective hypotheses tested independently: objective identity/calibration, backward optimization routing, and language-conditioned geometric augmentation.
- Y19/Y20 overlap with closedG1/N1 was tested, not assumed away. No closed historicalstage/evidence rewritten.
- Further equivalent labels/detach/yaw/wordsubsetprobes lack information gain after these controls; shortcooldown warranted, not universalrouteexhaustion or resourceblock.

## 6. Next Batch
- No new productiontraining/admission. Resume on new native mechanism evidence, Y18terminal/evidence/goal/queuechange, or after2026-10-02T10:30:00+08:00 with a genuinely new preregistered independent hypothesis. Monitor/goalchecks remain active.
- Frozen diagnostics and summaries in pipeline/designs/Y19_Y20_native_diagnosis_58ccdbbf.md, Y20_cached_input_repeat_58ccdbbf.md, Y21_directional_geometry_prereg_58ccdbbf.md; actual outcomes in route/geometryreviews.
