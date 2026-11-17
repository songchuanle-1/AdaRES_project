# Experiment Result Analysis - AdaRES_project

## 1. Basic Info
- ID: AdaRES-Y18; config: configs_song/model_paper5/adares_y18_point_measure_res.yaml.
- Server: server-2; GPU1 only. Fixed30 epochs/seed1999; no validation tuning.
- Started: original queue launch2026-10-01 23:37:51; exact finish/duration unknown because remote/local clocks differ.
- Observed21:49:55-21:49:58+08: originalPID7449 absent before/after capture, finalVal30/30 363/363, last checkpoint epoch30 and best checkpoint epoch21.
- Training physically complete; canonical lifecycle still running, deathwatch1/3. No forced queue edit/repeated monitor. Y23 remains user-paused until canonical done_complete.

## 2. Best Metrics
Full best-checkpoint native pass1, 9508 expressions/141 scenes; all3 models each have two fresh passes on GPU1.

| Metric | Y18 | W2 | B1 |
|---|---|---|---|
| mIoU | 0.507605986 | 0.510009850 | 0.502222159 |
| Acc25 | 0.608750526 | 0.610643668 | 0.610433319 |
| Acc50 | 0.558266723 | 0.562894405 | 0.552902819 |
| spmIoU | 0.471850907 | 0.473634468 | 0.466782460 |
| Available/selected queries | 128/1 | 128/1 | 128/1 |
| Parameters | 164629637 | 164629637 | 164430473 |
| Peak allocated GiB | 4.042351 | 4.042351 | 4.040677 |
| Peak reserved GiB | 6.486328 | 6.486328 | 6.484375 |
| Warm first-chunk median seconds | 0.335364 | 0.328534 | 0.262520 |

## 3. Convergence Summary
- Best metadata mIoU0.507605910 at epoch21; last.pth saved epoch30 with model/optimizer/scheduler present, unchanged during CPU read.
- Final native log rounds mIoU0.504/Acc50 0.554/Acc25 0.602/spmIoU0.468; not the best-checkpoint result above.
- Y18 two-pass mIoU0.507605986/0.507602997; maximum expression repeat difference0.036709189. No universal deterministic-repeat claim.
- Original log/checkpoints preserved; bounded terminal tail has no Traceback/OOM, not a full-log no-crash proof.

## 4. Comparison
- Target0.515 not reached. Paired expression identity/target geometry checked; train-size cuts1487/3325 remain frozen.
- VsW2 delta-0.002403864, whole-scene bootstrap95%CI[-0.010617414,0.006017898].
- VsB1 delta+0.005383828, CI[-0.003456414,0.014298723]. Draws10000/seed1999, independent units are whole scenes, not expression chunks.
- Allsmall/medium/large CIs crosszero for both comparators. No reliable positive effect or material harm established; crossing-zero does not prove equivalence or safety.
- W2/B1 maximum expression repeat differences0.933165193/0.022280872 retained. Group mean repeat noise is empirical, not a universal bound.

## 5. Conclusion
- Efficacy inconclusive, target unmet; no champion promotion, retraining, favorable subgroup selection or renamed re-admission.
- No parameters added versusW2; inference cost broadly similar in these captures, but first-chunk timing is not a latency distribution, p95 or Pareto proof.
- Officialunique/multiple unavailable; one training seed and best-checkpoint comparisons do not isolate causal mechanism benefit.
- Preserve Y23 user-priority: after canonical Y18 done_complete and freshGPU1 idle/source/checkpoint checks, restore SAMEY23 epoch4 model/AdamW/PolyLR to epoch5/total30. GPU0 remains reserved.

## 6. Evidence And Risks
- Terminal: y18_terminal_45b53a88c2324f3c8a4a1ef63ff59599.json, SHA256 f8daa9a3372e7d06792336b33bb0bbbd2936aafc700e76d3cca4094af31c9254.
- Y18/W2/B1fullcaptures: y18_y18_full_eval_45b53a88c2324f3c8a4a1ef63ff59599.json / y18_w2_full_eval_45b53a88c2324f3c8a4a1ef63ff59599.json / y18_b1_full_eval_45b53a88c2324f3c8a4a1ef63ff59599.json; hashes bound in research_work.json and cluster sources.
- Cluster: y18_clustered_comparison_45b53a88c2324f3c8a4a1ef63ff59599.json, SHA256 fb6dcbb6fc9fcfd6ea3cc4b0be5c47b175c53e9f5b26598b25b71c0d53c8505f.
- Actual evaluation window21:51:23-22:02:24+08. Postevaluation22:03:22GPU1 memory1MiB/util0/no compute process; not a guarantee of future availability.
- No crash repair/new scientific admission gate in this tick. Pipeline tick completion still requires successful finish and independently verified lease release.
