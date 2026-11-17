# Experiment Result Analysis - AdaRES_project
Experiment AdaRES-Y36; scientific state complete_inconclusive.
Observed 2026-10-10T18:11:10.315287+08:00; no postflight claim.
Canonical begin queue state is running/deathwatch1of3; unchanged by this review.

## 1. Basic Info
Original PID100018/server2GPU1/configadares_y36_local_adjacency_res.
Read-only original process/checkpoint capture 2026-10-10T17:50:09.930986+08:00.
Last/best metadata epoch30 and original PID absent; scheduler terminal confirmation
pending. Actual start time, exit time and duration are not inferred from directory
names and remain unknown. Source6725f9bd6a8f9b47addae6a4b8336d004f73f1ae
preserved under later additive marker67f4a1b2.

## 2. Best Metrics
Best checkpoint epoch30; metadata mIoU0.51076740026474.
All9508expressions/141scenes each twice; FIRST pass PRIMARY, no favorable repeat.
| Method | mIoU | Acc25 | Acc50 | spmIoU | Repeated mIoU |
|---|---|---|---|---|---|
| Y36 | 0.510716624292 | 0.609697097181 | 0.561421960454 | 0.474718082593 | 0.510721577420 |
| W2 | 0.510009849923 | 0.610643668490 | 0.562894404712 | 0.473634467799 | 0.510107995189 |
| B1 | 0.502112417447 | 0.610328144720 | 0.552797644089 | 0.466668866304 | 0.502121678666 |
BestSHAf504f6ed548affa54eb7292cf5a7d930158be5dfaac291417f4852ad0f0698d1; lastSHA9d83e12a3d8a2574c81c5b64f1338fbd1dc4d5db10f0f60b6735921fcd3ba66e, post-audited unchanged.

## 3. Convergence Summary
Last30/best30 from metadata. Preflight last visible epoch29/bestrounded.5099 is
an earlier monitoring snapshot, not a contradiction or the frozen primary.
Maximum expression repeat change Y36=0.06025010347366333,
W2=0.9331651926040649,
B1=0.03992509841918945.
Two repetitions quantify empirical noise, not a universal bound or seed variance.

## 4. Comparison
Expression-weighted paired delta, whole-scene bootstrap10000/seed1999;
TRAINrawpointcuts1487/3325. Material subgroup harm if interval upper<0 versus either.
| Reference | Group | Paired delta | Scene CI95 | Expressions/Scenes | Harm CIupper<0 |
|---|---|---|---|---|---|
| versus_w2 | all | 0.000706774369 | [-0.007218154973, 0.008880424700] | 9508/141 | False |
| versus_w2 | small | -0.008495133019 | [-0.024612470092, 0.007851595067] | 2816/124 | False |
| versus_w2 | medium | -0.003578209746 | [-0.017307123148, 0.010347120675] | 3304/136 | False |
| versus_w2 | large | 0.012533857817 | [-0.000994850034, 0.026243154882] | 3388/135 | False |
| versus_b1 | all | 0.008604206844 | [0.000486970108, 0.016646870070] | 9508/141 | False |
| versus_b1 | small | 0.013798593966 | [-0.003038309196, 0.030950116054] | 2816/124 | False |
| versus_b1 | medium | 0.009413328984 | [-0.006934973466, 0.024756762579] | 3304/136 | False |
| versus_b1 | large | 0.003497732912 | [-0.011705865461, 0.018336383582] | 3388/135 | False |
CIcross0 remains inconclusive; official unique/multiple metadata unavailable.
Trained serial cost, original TRAIN8 and largestpair728/6780, two warmups/three
readings; separate native batch2 loss/backward/AdamW stress stays memory-only:
```json
{
  "candidate": {
    "train8": {
      "median_seconds": 0.08170545380562544,
      "descriptive_p95_seconds": 0.12454910231754183,
      "peak_allocated_gib": 0.962437629699707,
      "peak_reserved_gib": 1.234375
    },
    "largest_pair": {
      "median_seconds": 0.20252253580838442,
      "descriptive_p95_seconds": 0.23574117617681623,
      "peak_allocated_gib": 1.2454032897949219,
      "peak_reserved_gib": 1.759765625
    },
    "native_train_stress": {
      "median_seconds": 0.9835909716784954,
      "peak_allocated_gib": 13.4406099319458,
      "peak_reserved_gib": 14.0390625
    }
  },
  "parent": {
    "train8": {
      "median_seconds": 0.08315026480704546,
      "descriptive_p95_seconds": 0.1195365697145462,
      "peak_allocated_gib": 0.9607234001159668,
      "peak_reserved_gib": 1.234375
    },
    "largest_pair": {
      "median_seconds": 0.19535740092396736,
      "descriptive_p95_seconds": 0.22860185941681266,
      "peak_allocated_gib": 1.2436890602111816,
      "peak_reserved_gib": 1.7578125
    },
    "native_train_stress": {
      "median_seconds": 0.9421737398952246,
      "peak_allocated_gib": 12.850881576538086,
      "peak_reserved_gib": 13.400390625
    }
  }
}
```
Params {"candidate": 165089671, "parent": 164629637}; native128/oneargmax.
Tiny serial cost does not establish reliable p95, speedup significance or
population lightweight Pareto. Original checkpoints were not resumed or modified.

## 5. Conclusion
Target unmet; no predeclared subgroup-harm trigger. Intervals crossing zero remain inconclusive, not equivalence or proof of no harm.
No promotion this tick; W2 retained. Same stable Y36 realization, no renamed
rerun, epoch selection, threshold/temperature/loss/feature/neighborhood tuning
or blanket family ban. One trained seed is not causal necessity proof.
Y37 original PID20348/GPU0 remains preserved; its final outcome is unknown.
Do not fill GPU1 until canonical capacity opens and real admission gates pass.

## 6. Crash Info
No crash repair or remote restart. Bounded original log contains final30/30
checkpoint saves, no bounded traceback/OOM; this is not whole-log crash absence.
Canonical deathwatch1of3 is not manually accelerated. Missing local exploratory
path scripts/frozen_terminal.py was not executed; the actual frozen evaluator
and existing hash-bound capture protocol were used.

## Evidence
- pipeline/analyses/y36_terminal_1fb792ef616d43b985a0a12024cb9fc8.json SHA256 e12806ba67973bb920d000729fe7d41cc49235312bbe4d51c5e46d6fd3009779
- pipeline/analyses/y36_y36_full_eval_1fb792ef616d43b985a0a12024cb9fc8.json SHA256 bca35d368b9ca1042904dff44b2d87519b7576d395f7b21b50ab3bf3500c84bd
- pipeline/analyses/y36_w2_full_eval_1fb792ef616d43b985a0a12024cb9fc8.json SHA256 27a0915df5909cb7da9e1b4d2f64753fa4aa398838a2b6c40a4f5028dec231a3
- pipeline/analyses/y36_b1_full_eval_1fb792ef616d43b985a0a12024cb9fc8.json SHA256 87c50aeaebf34a12de3fcb8471034e984cf266ba976dde56acd6dace798a8e3f
- pipeline/analyses/y36_clustered_comparison_1fb792ef616d43b985a0a12024cb9fc8.json SHA256 210387be672e2f547c5db194a0f54e11f184e8aad010dc713a69730fbfaa756e
- pipeline/analyses/y36_trained_cost_1fb792ef616d43b985a0a12024cb9fc8.json SHA256 bfe1a7ff50dd833675738b0ea91fbd1f84586bd7579b30c3ab5e01eb1a7c9a92
- pipeline/analyses/y36_postaudit_1fb792ef616d43b985a0a12024cb9fc8.json SHA256 107458c960b626f7a1c66afa752f10bbe6f206707fda49e67da89727444ba417
