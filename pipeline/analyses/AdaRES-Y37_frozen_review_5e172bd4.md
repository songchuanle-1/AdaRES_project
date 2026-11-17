# Experiment Result Analysis - AdaRES_project
Scientific status complete_inconclusive; canonical lifecycle done_complete confirmed by begin.
Observed 2026-10-11T16:25:57.870974+08:00; no postflight claim.

## 1. Basic Info
Original server2/GPU0/PID20348/configadares_y37_pair_distribution_res.
Original PID absent and same-config train processes absent in direct capture.
Actual start/exit/duration remain unknown; directory names are not timestamps.
Y39 originalPID102267/GPU1/config and sources were preserved.

## 2. Best Metrics
Original last epoch30; best checkpoint epoch19, metadata mIoU0.5051654577255249.
Each candidate/W2/B1 full9508expressions/141scenes twice; FIRST pass primary.
| Method | mIoU | Acc25 | Acc50 | spmIoU | Repeated mIoU |
|---|---|---|---|---|---|
| Y37 | 0.505061022921 | 0.607803954565 | 0.561001262095 | 0.468443808443 | 0.505060097340 |
| W2 | 0.510107995189 | 0.610748843080 | 0.562999579302 | 0.473719922153 | 0.509909353100 |
| B1 | 0.502119335285 | 0.610328144720 | 0.552797644089 | 0.466681173553 | 0.502121678666 |
BestSHA6993fc5dc67f8a7d5c2f5a1d3209a9e59c2b09b5952b64d7b6d8acb1aa1702ca; lastSHAa5b4c4e691536654b101950ddd3b912591876747b31910520383d118154d9fc8.

## 3. Convergence Summary
Last30 and savedbest19 are checkpoint facts, not a full convergence curve.
Rounded automatic CSV .5050 and monitor epochtext30 do not replace metadata.
Maximum expression repeat difference Y37=1.0,
W2=0.9763062000274658,
B1=0.022280871868133545. Near-identical group
means do not prove per-expression determinism. Two repeats are empirical noise,
not universal noise bounds or training-seed uncertainty.

## 4. Comparison
FIRST paired expression-weighted delta, whole-scene bootstrap10000/seed1999.
TRAIN raw-point size cuts1487/3325; subgroup harm if CIupper<0 versus either.
| Reference | Group | Paired delta | Scene CI95 | Expressions/Scenes | Harm |
|---|---|---|---|---|---|
| versus_w2 | all | -0.005046972268 | [-0.012977527251215851, 0.0029095121049355255] | 9508/141 | False |
| versus_w2 | small | -0.006301353262 | [-0.02329066691322021, 0.01040899593155092] | 2816/124 | False |
| versus_w2 | medium | -0.005458316251 | [-0.019172377712305917, 0.007942084273415355] | 3304/136 | False |
| versus_w2 | large | -0.003603224512 | [-0.017206832551819935, 0.009990614024636855] | 3388/135 | False |
| versus_b1 | all | 0.002941687636 | [-0.004362333485613079, 0.010245392957408879] | 9508/141 | False |
| versus_b1 | small | 0.015992373723 | [-0.0009360838920727527, 0.0333564485481082] | 2816/124 | False |
| versus_b1 | medium | 0.007529679863 | [-0.007200363259046116, 0.021654969347848118] | 3304/136 | False |
| versus_b1 | large | -0.012379876218 | [-0.02684134269000966, 0.0019483129986748824] | 3388/135 | False |
Official unique/multiple metadata unknown. Trained serial costs on frozen
TRAIN8 plus maxpair728/6780, two warmups/three readings; independent native
batch2 loss/backward/AdamW stress in memory only, no checkpoint overwritten:
```json
{
  "candidate": {
    "train8": {
      "median_seconds": 0.08428914379328489,
      "descriptive_p95_seconds": 0.1310131786391139,
      "peak_allocated_gib": 0.9617080688476562,
      "peak_reserved_gib": 1.234375
    },
    "largest_pair": {
      "median_seconds": 0.22467300202697515,
      "descriptive_p95_seconds": 0.2678713290952146,
      "peak_allocated_gib": 1.244673728942871,
      "peak_reserved_gib": 1.833984375
    },
    "native_train_stress": {
      "median_seconds": 1.140052093192935,
      "peak_allocated_gib": 14.29847002029419,
      "peak_reserved_gib": 14.767578125
    }
  },
  "parent": {
    "train8": {
      "median_seconds": 0.08759241830557585,
      "descriptive_p95_seconds": 0.12861507264897226,
      "peak_allocated_gib": 0.9607234001159668,
      "peak_reserved_gib": 1.234375
    },
    "largest_pair": {
      "median_seconds": 0.21233489457517862,
      "descriptive_p95_seconds": 0.2461777445860207,
      "peak_allocated_gib": 1.2436890602111816,
      "peak_reserved_gib": 1.7578125
    },
    "native_train_stress": {
      "median_seconds": 1.0815567411482334,
      "peak_allocated_gib": 12.850857734680176,
      "peak_reserved_gib": 13.400390625
    }
  }
}
```
Parameters {"candidate": 164893831, "parent": 164629637}; native128/oneargmax.
Serial descriptive p95 and one trained checkpoint do not establish Pareto.

## 5. Conclusion
Target unmet; 0 predeclared subgroup harm triggers. All relevant
intervals crossing zero remain inconclusive, not equivalence or no harm.
Retain W2; no Y37 promotion, retuning, renamed rerun or whole-family ban.

## 6. Crash Info
No training crash repair or restart. Bounded original final30/30 log saves
lastcheckpoint with no bounded traceback/OOM; not whole-log crash absence.
First evaluator invocation failed child train import before evaluation.
Its raw log is retained; retry inherited paths, same scientific evaluator AST
and checkpoints. Only evaluator's wrong GPU1 lease was delegated to the
authorized outer GPU0 lease; no sampler, metric or population change.

## Evidence
[
  {
    "path": "pipeline/analyses/y37_terminal_current_5e172bd47ee3450dbee928bdecaede7f.json",
    "sha256": "5201c43dffef5554fc24da9db65a12a2e8e697b9715108ac5b4da69dec62bacf"
  },
  {
    "path": "pipeline/analyses/y37_frozen_full_retry_5e172bd47ee3450dbee928bdecaede7f.json",
    "sha256": "2ae9c0cae79a553b6ff9b17d42e77022e3a69bd31b6e75e7eef1b1a337ffefbd"
  },
  {
    "path": "pipeline/analyses/y37_clustered_comparison_5e172bd47ee3450dbee928bdecaede7f.json",
    "sha256": "caa77607949dd021b25c43fea5d9df035c73c0776aa73e5009cee79d57e75b48"
  },
  {
    "path": "pipeline/analyses/y37_trained_cost_5e172bd47ee3450dbee928bdecaede7f.json",
    "sha256": "8186d313d2fb8ef515bcdb68e3e6ef19f907e7137a55660d4ca8e2d522d79241"
  }
]
