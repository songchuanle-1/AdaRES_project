# Motivation Review - AdaRES_project

## 1. Goal
Source: pipeline/goal.json SHA25653da862754afe00739a75c5ea6b1f184e0358813225b8f8c0c024a355737c966.
Description: AdaRES: improve a faithful standard-indicator 3D referring expression segmentation baseline by learning query confidence from metric-aligned evidence, targeting ScanRefer mIoU >= 0.515.
Target ScanRefer mIoU.515; W2historical.5100/gap.0050, B1.5020; Phase1unmet.
All seven constraints: server-2GPU0/1 only; exactlysingleGPU/onemaxpercard; no successor whileB3running and at mosttwoindependentcandidates; B2/B4fixed30extensions resume optimizer/scheduler; releasedSPFormer; reportmIoU/Acc25/Acc50/querycount/latency/peakmemory; no validationthreshold/temperature/weight/feature/epoch tuning.
All three bans: no server-6; no b005/CFNet-as-B0; no validationrankingmargin/occupancyweightscan.
Pipeline policy unchanged: capacitygapresearch, steadythreshold1, fillcapacity, buffer0, maxcandidates2.

## 2. Story Status
| Hypothesis | Status | Evidence | Gap |
|---|---|---|---|
| W2 query confidence | complete | historical.5100 | .515unmet |
| Y18 point-measure supervision | training | preflight16/30best.5070 | fixed30finalefficacy |
| Y22 query correspondence defect | rejected | native cached jointpermutation nearprecision | upstream rawlogitnoise unresolved |
| Y23 valid-token support | validation | train64native padding leak removed | native backward/resources/inference/admission |

## 3. Experiment Assessment
| Experiment | mIoU | Relation To Goal | Contribution |
|---|---|---|---|
| G1/Y19 | historicalG1.4980 | indicatorquality labels | calibrationnotindependent |
| Y15 | full.230319 | extra query completion | realizedharm/selectioncollapse |
| Y17 | full.504157 | same-target consistency | pairedintervalsinconclusive |
| Y20/Y21 | unknown | gradientroute/geometry | frozenpremisesdisconfirmed |
| Y23 | unknown | annotation-valid query/language alignment | removesnuisancegradient, benefitunproved |

## 4. Gap Analysis
Native language masks already define nonexistent padding tokens, but the
semantic alignment denominator includes their projected features.
Train64/256expressions:63paddedscans have nonzero nuisance gradients, which
support masking removes exactly; no valid token is deleted or reweighted.
The average padding-only loss change is small(.00014165), not evidence of
task-level harm or expected gain. Criterion correctness and trained efficacy
are separate questions.

## 5. Design Rationale
Y22 tests slot correspondence rather than reusing S1 query-order training:
native cached criterion equivariance found, reject the repair rationale.
Y23 changes the support of contrastive normalization, not a coefficient,
temperature, valid-language feature subset or inference ranking rule.
It preserves W2 learned indicator and all valid annotations. No claim of
literature novelty; no new search was required to inspect this local defect.

## 6. Next Batch
Physical independent Y23 package/config, exact W2 inference/model modules,
no added parameters. Complete released-init backward, checkpoint-resume,
predict identity and matched large-scene resource tests. Admit only after
all six gates, scoped tested commit, immutable evaluation plan and canonical
dispatcher. Preserve original Y18. See exp_design_Y23_d20c71e4.md.
