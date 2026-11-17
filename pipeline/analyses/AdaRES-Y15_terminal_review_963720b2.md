# Experiment Result Analysis - AdaRES_project
Experiment: AdaRES-Y15 | Scientific state: rejected | Queue: running/deathwatch1/3

## 1. Basic Info
- Config:configs_song/model_paper5/adares_y15_query_completion_res.yaml; server-2 GPU0.
- Observation2026-10-02T06:12:37+08:00: original PID18796 absent, original final Val30 completed, last checkpoint epoch30. Exact wall duration/remote finish time unknown.
- Frozen seed1999,30epochs,released SPFormer. Checkpoints/config/native selection untouched.
- Best checkpoint epoch27 SHA256:6a2e29d7f210bc8cd90511363185867c770f2cd8e151d60ec2911380f1898df5.

## 2. Best Metrics
All9508expressions/141scenes, original full native evaluator, two passes per model:
| Model | mIoU | Acc50 | Acc25 | spmIoU | Checkpoint Epoch |
|---|---|---|---|---|---|
| Y15 | .2303191651 | .1755363904 | .3621161127 | .2105648078 | 27 |
| W2 | .5101079952 | .5629995793 | .6107488431 | .4737199222 | frozen best |
| B1 | .5022242759 | .5530079933 | .6104333193 | .4667840089 | frozen best |

## 3. Convergence Summary
- Original final epoch30 native rounded mIoU.222; best metadata.2303202301 at27. Best-checkpoint full re-evaluation.2303191651; no posthoc epoch selection.
- Training completed its fixed horizon, but the candidate did not approach target.515.

## 4. Comparison
- Whole-scan paired bootstrap10000draws/seed1999: Y15-W2=-.2797888300,95%CI[-.2983370358,-.2609116305]; Y15-B1=-.2719051108,CI[-.2906953096,-.2526473108].
- Frozen train-size cuts1487/3325: small/medium/large all materially adverse versus both comparators under the predeclared group-mean empirical repeat-noise rule.
- Y15 repeat IoU difference0; W2 max expression repeat difference.9331651926/groupmean.0000981453; B1 max.0233703852/groupmean.0000002261. Two passes are not a universal noise bound.
- Params Y15/W2/B1:164959365/164629637/164430473. Firstchunk5warmforward medians.366753/.377105/.313763seconds; allocated4.043580/4.042351/4.040677GiB. Firstchunk timing is not a full latency distribution or proven deployed Pareto frontier.

## 5. Conclusion
- Reject this realized Y15 candidate on concrete disconfirming trained efficacy evidence; no champion promotion, automatic retry or threshold/temperature/weight/architecture tuning on validation.
- Additional preregistered read-only query decomposition:9508/9508 native selections index128 (the added query), native/extraIoU.2303191651; original128 indicator-argmax diagnostic.4950240905; base/full oracle.8459484488/.8470817964.
- Native identities/IoUs matched separately completed full evaluation within1e-6. The query selection collapse is directly observed; training cause/teacher necessity remains unknown without a separately admitted trained causal control.
- Base-only/oracle readouts are diagnostic counterfactuals, not deployed results or validation-selected repairs.
- Official unique/multiple strata unavailable; one training seed. Do not extrapolate universal failure of teacher information or all query-expansion methods.
- Next action: allow normal subsequent monitor observations to reconcile canonical terminal state; preserve full evidence and continue untouched Y18 under its frozen terminal protocol. No extra admission budget in this tick.

## 6. Crash Info
- No observed native training crash. Two diagnostic packaging failures were repaired with retained logs; they are not explanations of the measured efficacy failure.
- Evidence: y15_metadata_963720b2.json; y15_y15_v2_963720b2.json; y15_w2_v2_963720b2.json; y15_b1_v2_963720b2.json; y15_clustered_comparison_963720b2.json; y15_query_decomposition_v2_963720b2.json; Y15_terminal_query_decomposition_963720b2.md; Y15_diagnostic_failures_963720b2.md; native_progress_963720b211ae49b1a59707610cfbda4a.md.
