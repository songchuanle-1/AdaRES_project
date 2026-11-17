# AdaRES-Y18 Admission Review

2026-10-01 23:29+08. Scientific status: validated for falsifiable training,
not trained efficacy. Scope only AdaRES_project, server-2 GPU1; preserve Y15.

| Gate | Executed Evidence | Disposition |
|---|---|---|
| diagnosis | 1000value/gradientpoint-expansioncases; train64fixedscans256expressions, two captures | PASS for metric-measure deficit, not efficacy |
| controls | Same logits/matches/GT; original/mass-only/occupancy-only/equal-size measures; exact unit-mass analytic control | PASS for intended information separation |
| native_backward | Released-init8scans4batches finite native totalbackward; native train2steps; resumeddecoder400gradienttensors | PASS for executable graph |
| resources | Isolatedlargest1055503point/7618SP batch2 native steps; candidate/W2reserved14.078125GiB, equal164629637params | PASS for authorized24GiB fit, not Pareto |
| independence | Physical7sourcefiles, no variantimports, remotehashidentity, all modelmethods exceptloss unchanged | PASS |
| constraints | Identicalparentconfig, seed1999/30epochs/releasedSPFormer, nativeinfer, no valused or coefficients searched | PASS |

Train64 final-layer scene-mean gradient alignment with negative softpointIoU:
old .778066, point .914328; repeated .778047/.914329. Difference .136262,
60positive/4negative scenes,12adverseexpressions.256nonzero gradientchanges.
Mass-only .918234, occupancy-only .774834, equal-size .776060; improvement
is not attributable to smoothing alone. Fractionalpointoccupancy is slightly
adverse versusmass-only locally; retainthisrisk, do not posthocselectcontrol.
These are local mask derivatives, not full-parameter descent or learned efficacy.
Released-init augmentedtrain8 meanalignmentold.494875/point.510644; no
unexplainedaggregate reversal, individualadversecases remain.

Exactpoint BCE/Dice replacesnative maskloss only; scoreIoUtargets/matching/
indicator/semantic/sampleobjectives untouched. ClosedG1 scoretarget route
is not reopened. Y16 ceiling rejection is not contradicted; optimization
measure, not representationresolution, is studied here. No claim that
confidencelearning or fractionaloccupancy is necessary, no novelty claim.

Candidate largebatch trainingmedian1.065476s vsW2.975367s (~9.24% higher).
Both peakreserved14.078125GiB; candidatepeakallocated13.549751GiB vs
W2 13.549606GiB. Noextra params or inferdependency. This establishes hardware
executability only; same-sourceinference isnot trainedcostdominance.

Nativegorilla beforecontinuation model/AdamW/PolyLR restoreexactly;
nextlossdifference0, maxmodeldifference3.0517578125e-5, optimizer
7.637936505489051e-9. ExplicitRNGsamebatch diagnostic, not full-loader/
bitwisecontinuation guarantee. Native training entry updates were observed.

Samecheckpointtrain8 inference32expressions: parent/repeat/labelcounterfactual
selectionandbinaryflips0. Firstscene parentprobabilitydelta.001422286 vs
repeat.001422822; counterfactual8.64e-7. Allrawnoise retained; not exactbitwise.
AST of nativepredict methods identical and candidateproductionpointlabels
only consumed by loss. No label inference dependency introduced.

Firstalgebra helperdtypeissue, missingremoteprofilemanifest and native
predictionreaderKeyError recorded in y18_probe_failure_3efff5f3.md;
independentrepairs/rerunsnot overwrites. No altered candidate objective.
Sourceaudit passes; frozenposttrainingplan in exp_design_Y18_20261001.md
specifies groupnoise/materialharm, sceneCI and accuracy/costfrontier.

Decision: admit one Y18 independentmaskobjective variant only after scoped
testedcommit/hashreceipt. Use canonicalgateddispatcher and increasingsteps
verification. No manualtraininglaunch, schedulerchange, extraadmission,
Y15restart, goalrelaxation or validationtuning. Benefits remainunknown.
