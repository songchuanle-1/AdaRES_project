# Y36 Frozen Post-Training Execution

Tick 1fb792ef616d43b985a0a12024cb9fc8, 2026-10-10.
This binds the earlier Y36 design; it does not change its protocol or claim results.

- Original PID100018/GPU1 is absent; original log and last/best metadata reach
  epoch30. Canonical monitor remains deathwatch1/3, queue running. Do not
  accelerate its counter, rewrite queue/experiments.csv, restart, or admit a successor.
- Frozen best/last are under adares_y36_local_adjacency_res/20261009_060041.
  Best SHA f504f6ed548affa54eb7292cf5a7d930158be5dfaac291417f4852ad0f0698d1.
  Last SHA 9d83e12a3d8a2574c81c5b64f1338fbd1dc4d5db10f0f60b6735921fcd3ba66e.
  Directory timestamps are not reconstructed as start time.
- Y36 source commit 6725f9bd6a8f9b47addae6a4b8336d004f73f1ae remains
  byte-bound to its seven-file LF bundle. Additive marker67f4a1b2 identifies
  the later Y37 deployment, not replacement Y36 source.
- Preserve Y37 PID20348/GPU0, config and source. Recheck GPU1 live occupation
  and acquire the existing exclusive diagnostic lock for every GPU measurement.
- Native full9508 expressions/141scenes, candidate/W2/B1 each twice;
  FIRST pass primary. Native128 queries/one argmax/seed1999/evaluator unchanged.
  Comparators use their original frozen checkpoint hashes, not a favorable repeat.
- Paired expression-weighted delta with whole-scene bootstrap10000, seed1999;
  TRAIN point cuts1487/3325. Material subgroup harm remains interval upper<0
  against either reference. Crossing0 is inconclusive, not equivalence or no harm.
- Trained serial predict cost: unchanged first8 distinct TRAIN scenes plus
  indices728/6780, two warmups and three readings. Separate native batch2
  loss/backward/AdamW stress remains memory-only. No checkpoint write/resume.
- Report empirical repeat differences, raw/subgroup CIs, params, queries,
  latency and peak memory. Tiny serial costs do not prove a population Pareto
  advantage or reliable p95. One training seed cannot prove mechanism necessity.
- Target remains0.515; no validation tuning, human-review impersonation,
  source refactor, new scheduler, or extra training.
