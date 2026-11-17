# Independent route screens after G1

2026-09-25 17:33 Asia/Shanghai. Read-only fixed first 16 ScanRefer
validation scenes, B4-E30 best checkpoint. These are diagnostic subset
results, not population estimates or validation-selected hyperparameters.

| Route | Fixed question | Observation | Disposition |
|---|---|---|---|
| H1 | Does 128-query sampling omit target-bearing seeds available among 256 FPS seeds? | 0/240 material errors had available-but-unsampled target seeds. 27/441 had no target-bearing FPS seed; their oracle mask IoU averaged 0.803. | Reject sampler-omission mechanism; no query-budget scan. |
| H2 | Does a budget-preserving four-per-query first-layer fine shortlist improve target coverage? | Material-error coverage 0.70936 original versus 0.70842 balanced; 12 improved, 35 worsened. Later original layers already cover >0.89. | Reject this redistribution; no alternative k search. |
| H3 | Does B4's separately trained mask-quality score head provide a useful standalone selection signal? | Score-only mIoU 0.38265 versus indicator 0.49186; among 240 material errors 40 improved and 44 worsened. | Reject standalone score-head reranking; no validation weight scan. |
| H4 | Are relation-bearing expressions enriched among B4 selection errors? | 204/387 relation-marked material errors versus 36/54 unmarked (52.7% versus 66.7%). | Reject simple relation-gated successor; cohort is small and observational. |

H5 is independent of H1-H4: it adds a final all-query message rather than
changing sampler top-k, fine shortlist, score weighting or relation gates.
Its own diagnosis and gates are recorded in the H5 review and receipt.

Raw paired evidence: `AdaRES-H1_sampling_16scenes_20260925.json`,
`AdaRES-H2_fine_16scenes_20260925.json`,
`AdaRES-H3_score_16scenes_20260925.json`,
`AdaRES-H4_relation_16scenes_20260925.json`.
