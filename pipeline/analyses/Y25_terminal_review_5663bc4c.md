# Experiment Result Analysis - AdaRES_project
Experiment: AdaRES-Y25 | Scientific State: rejected | Canonical State: running/deathwatch1/3 at preflight

## 1. Basic Info
- Config: configs_song/model_paper5/adares_y25_mask_feedback_res.yaml; server-2 GPU1 only.
- Source: 94918c1a008e21c72b9478c28f934e8b329e0c97; original PID98611 absent before/after terminal capture.
- Fixed30 epochs/seed1999 finished: final Val30/30 363/363; CPU last meta30, best meta16, model/optimizer/scheduler present and unchanged during read.
- Actual completion time/duration unknown; locally verified 2026-10-04 17:50:45 to17:50:48+08. Remote checkpoint directory timestamps are not local launch time.
- Last SHA256 6ff87502535e32f4ce4d18223dcffea32ca68dc9ec0a97446ed663cb1704bdab; best b876531f899dd2759ac7e02dcd88e6fa821647efd507799c035854ef2c7298e2.
- No forced queue transition or repeat preflight. Physical training completion and scientific rejection do not impersonate canonical done_complete.

## 2. Best Metrics
Own best selected only after fixed30; full native9508 expressions/141scenes, two passes per model seed1999.

| Metric | Y25 Pass1 | Y25 Pass2 | W2 Pass1 | B1 Pass1 |
|---|---|---|---|---|
| mIoU | .494117094 | .494020751 | .510107995 | .502121793 |
| Acc25 | .613167859 | .613062684 | .610748843 | .610328145 |
| Acc50 | .551114851 | .551009676 | .562999579 | .552902819 |
| spmIoU | .456579818 | .456501793 | .473719922 | .466682976 |

W2 pass2 .510007015; B1 pass2 .502223014. No target/champion promotion.

## 3. Convergence Summary
- Best checkpoint epoch16 meta mIoU.494121432; final epoch30 log rounds to.488.
- Exactly30 epochs completed, no early stop/restart/validation tuning.
- Previous native mechanism activity and finite gradients remain valid but did not establish task benefit. Immediate soft-hard loss increases and TRAINsoft/EVALhard mismatch risk were retained before admission.

## 4. Comparison
Identity-aligned expressions; 10000 whole-scene bootstrap draws seed1999. TRAIN-derived cuts1487/3325 unchanged.

| Versus W2 | Delta mIoU | Scene-Clustered 95%CI | Observed Mean Absolute Repeat Noise | Frozen Harm Rule |
|---|---|---|---|---|
| All9508/141scenes | -.015990902 | [-.023798195,-.008143030] | .000137969 | Met |
| Small2816/124scenes | -.025881101 | [-.042779403,-.008998951] | .000317722 | Met |
| Medium3304/136scenes | -.018097220 | [-.032716532,-.003427960] | .000086846 | Met |
| Large3388/135scenes | -.005716380 | [-.018895457,.007390654] | .000283388 | Inconclusive |

- Versus B1 all delta-.008004700 CI[-.016802654,.000638898]; all frozen size intervals include0. Do not claim reliable harm versus B1.
- Max expression repeat differences Y25/W2/B1=.894704044/.933165193/.973895609; all preserved. Correct mean ABSOLUTE repeat differences, not absolute signed mean; two passes not a universal noise bound.
- Full eval parameters Y25/W2=164629637;128queries/1selected; peakallocated4.042351/reserved6.486328GiB both.
- Frozen train8 plus indices728/6780, serialbatch1/two warmups/three readings each, own trained best weights:
  train8 median Y25/W2=.075167709/.077175695s, descriptivep95=.117561844/.117949866s;
  largestpair median=.191020748/.189681914s, descriptivep95=.225458593/.224446009s;
  largestpair allocated1.243689/reserved1.757813GiB both. Mixed sizes/fixed serial order/three readings do not establish tail precision, equivalence or Pareto superiority.

## 5. Conclusion
- Reject this admitted Y25 realization: target.515 unmet; overall/small/medium regression against W2 satisfies the preregistered material-harm rule.
- Do not tune/restart/promote/relabel this variant or infer that every continuous-feedback mechanism is invalid. The training-soft/eval-hard mismatch is a plausible risk, not an experimentally isolated cause.
- One training seed, best-checkpoint comparison, observed native repeat noise; official unique/multiple unknown. No causal or novelty/Pareto claim.
- Canonical lifecycle awaits normal deathwatch reconciliation. Begin active1/budget0 remains immutable evidence, not authorization for new admission in this tick.

## 6. Execution And Evidence
- Three full capture groups completed17:50:59 to18:01:37+08; cost18:01:48 to18:02:51; four repeat-noise control tests passed; four helpers syntax passed.
- Post-evaluation read-only observation18:03:20+08 verifies hostname dl01-MS-7C60 and GPU1 1MiB/util0/no computePID. GPU0 never queried/used.
- Terminal b0ba0a7b965558661fdb253f7024183a95fe21e97a92148456dbf3d4dada73b7.
- Y25/W2/B1 captures 2f08d7a3ef80e9ca0df05ceda54c4044a9b20b3973469f285c10f9ab49c8973e / e1d3cd32c877c8c697a5d0fdb120453057b226c802cdef30acf1cc6ed26225ee / c3fd1405e09e33181396e619d3f35bec311c6fc3b1c9fc6ff50399da531740a2.
- Cluster analysis c46d3d82951a8e3a342dcf91f1f129b96c92a4fe4eea306c5f84d0a326c3769a; cost d0d0d4d45c11f985974132ae3c06b57b183b846cbcdeff49f1cc4b5df199cf2c.
- Immutable output paths use token5663bc4ccd784a0ab2d0e257af6cdd1c, in pipeline/analyses; raw logs retained. No training crash observed in bounded tail only.
- Tick finish/postflight pending when this review was written; final execution completion must be verified separately.
