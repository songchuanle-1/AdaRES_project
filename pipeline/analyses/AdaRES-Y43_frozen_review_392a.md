# Experiment Result Analysis - AdaRES-Y43
Observed locally 2026-10-14T13:49:41.986768+08:00; original SAME training done_complete30.

## 1. Basic Info
Server2/singleGPU; resumed original run, last/scheduler30, best epoch17.
Original checkpoints immutable. Training start/end/duration clock relation unknown.
Terminal proof: {'path': 'pipeline/analyses/y43_terminal_7a40_392a5a9f5fe242d7a3109a50e6060fda.json', 'sha256': 'd94dda4d14e6ab159ea1ff9c5b56352213bdea89dbb78cf2268850eb2e78e619'}.

## 2. Best Metrics
| Metric | Frozen FIRST | Population |
|---|---|---|
| mIoU | 0.5083255213166099 | 9508 expressions/141 scenes |
| Acc@0.25 | 0.6120109381573412 | same |
| Acc@0.5 | 0.559002944888515 | same |
| spmIoU | 0.4732247954032211 | same |
Checkpoint metadata 0.5085299611091614 is not the frozen efficacy endpoint.

## 3. Convergence Summary
Fixed30 unchanged; best saved epoch17. FIRST is primary, not best repeat.
Second mIoU 0.5084258668707591.
Whole-scene bootstrap10000seed1999; TRAIN cuts1487/3325 unchanged.
Full comparison/noise/subgroup evidence: {'path': 'pipeline/analyses/y43_clustered_392a_392a5a9f5fe242d7a3109a50e6060fda.json', 'sha256': '629ca50995a5e46f0c1fb8690f2975499502e6466cee2da25cbd898bdacb3400'}.

## 4. Comparison
W2: {
  "all": {
    "expressions": 9508,
    "scenes": 141,
    "paired_miou_difference": -0.0017824738722833889,
    "scene_clustered_ci95": [
      -0.010031740678902744,
      0.006381038730220411
    ],
    "max_expression_repeat_iou_difference": 1.0,
    "material_harm_within_observed_noise": false,
    "positive_difference_beyond_observed_noise": false,
    "status": "interval includes zero",
    "historical_abs_mean_repeat_difference": 0.00010034555414919599,
    "observed_group_mean_absolute_repeat_difference": 0.00011414424255709714,
    "frozen_material_harm_ci_upper_below_zero": false
  },
  "small": {
    "expressions": 2816,
    "scenes": 124,
    "paired_miou_difference": -0.01677875926735727,
    "scene_clustered_ci95": [
      -0.031847272004646626,
      -0.0017583243109859244
    ],
    "max_expression_repeat_iou_difference": 1.0,
    "material_harm_within_observed_noise": true,
    "positive_difference_beyond_observed_noise": false,
    "status": "negative interval",
    "historical_abs_mean_repeat_difference": 0.0003551136363636364,
    "observed_group_mean_absolute_repeat_difference": 0.0003551136363636364,
    "frozen_material_harm_ci_upper_below_zero": true
  },
  "medium": {
    "expressions": 3304,
    "scenes": 136,
    "paired_miou_difference": -0.0025002163177813893,
    "scene_clustered_ci95": [
      -0.017778084445265626,
      0.012648083847899677
    ],
    "max_expression_repeat_iou_difference": 0.0,
    "material_harm_within_observed_noise": false,
    "positive_difference_beyond_observed_noise": false,
    "status": "interval includes zero",
    "historical_abs_mean_repeat_difference": 0.0,
    "observed_group_mean_absolute_repeat_difference": 0.0,
    "frozen_material_harm_ci_upper_below_zero": false
  },
  "large": {
    "expressions": 3388,
    "scenes": 135,
    "paired_miou_difference": 0.01138191830966863,
    "scene_clustered_ci95": [
      -0.002861378700659286,
      0.025248719983564663
    ],
    "max_expression_repeat_iou_difference": 0.04617077112197876,
    "material_harm_within_observed_noise": false,
    "positive_difference_beyond_observed_noise": false,
    "status": "interval includes zero",
    "historical_abs_mean_repeat_difference": 1.355208711612886e-05,
    "observed_group_mean_absolute_repeat_difference": 2.517221317381335e-05,
    "frozen_material_harm_ci_upper_below_zero": false
  }
}
B1: {
  "all": {
    "expressions": 9508,
    "scenes": 141,
    "paired_miou_difference": 0.006103703358462506,
    "scene_clustered_ci95": [
      -0.0019631539772504104,
      0.014531589519217573
    ],
    "max_expression_repeat_iou_difference": 1.0,
    "material_harm_within_observed_noise": false,
    "positive_difference_beyond_observed_noise": false,
    "status": "interval includes zero",
    "historical_abs_mean_repeat_difference": 0.00011025620863424613,
    "observed_group_mean_absolute_repeat_difference": 0.00011414424255709714,
    "frozen_material_harm_ci_upper_below_zero": false
  },
  "small": {
    "expressions": 2816,
    "scenes": 124,
    "paired_miou_difference": 0.00551496771721428,
    "scene_clustered_ci95": [
      -0.01052612826954667,
      0.021917576117027594
    ],
    "max_expression_repeat_iou_difference": 1.0,
    "material_harm_within_observed_noise": false,
    "positive_difference_beyond_observed_noise": false,
    "status": "interval includes zero",
    "historical_abs_mean_repeat_difference": 0.0003551136363636364,
    "observed_group_mean_absolute_repeat_difference": 0.0003551136363636364,
    "frozen_material_harm_ci_upper_below_zero": false
  },
  "medium": {
    "expressions": 3304,
    "scenes": 136,
    "paired_miou_difference": 0.01048777979679099,
    "scene_clustered_ci95": [
      -0.004083604278466105,
      0.024730104087563407
    ],
    "max_expression_repeat_iou_difference": 0.011704802513122559,
    "material_harm_within_observed_noise": false,
    "positive_difference_beyond_observed_noise": false,
    "status": "interval includes zero",
    "historical_abs_mean_repeat_difference": 3.542615772736852e-06,
    "observed_group_mean_absolute_repeat_difference": 3.542615772736852e-06,
    "frozen_material_harm_ci_upper_below_zero": false
  },
  "large": {
    "expressions": 3388,
    "scenes": 135,
    "paired_miou_difference": 0.002317661744978943,
    "scene_clustered_ci95": [
      -0.011525876118874158,
      0.016018785646453846
    ],
    "max_expression_repeat_iou_difference": 0.9738956093788147,
    "material_harm_within_observed_noise": false,
    "positive_difference_beyond_observed_noise": false,
    "status": "interval includes zero",
    "historical_abs_mean_repeat_difference": 0.0003059655339968387,
    "observed_group_mean_absolute_repeat_difference": 0.0003059655339968387,
    "frozen_material_harm_ci_upper_below_zero": false
  }
}
Frozen harm triggers versus either: {"w2": ["small"], "b1": []}.
Trained paired costs: {
  "candidate": {
    "train8": {
      "median_seconds": 0.07571047050441848,
      "descriptive_p95_seconds": 0.1213999088096898,
      "peak_allocated_gib": 0.961735725402832,
      "peak_reserved_gib": 1.306640625
    },
    "largest_pair": {
      "median_seconds": 0.19293136800115462,
      "descriptive_p95_seconds": 0.22553741724186693,
      "peak_allocated_gib": 1.2447013854980469,
      "peak_reserved_gib": 1.8203125
    },
    "native_train_stress": {
      "median_seconds": 0.9600591869966593,
      "peak_allocated_gib": 13.027578353881836,
      "peak_reserved_gib": 13.513671875
    }
  },
  "parent": {
    "train8": {
      "median_seconds": 0.07691937000345206,
      "descriptive_p95_seconds": 0.12024531550196116,
      "peak_allocated_gib": 0.9607234001159668,
      "peak_reserved_gib": 1.234375
    },
    "largest_pair": {
      "median_seconds": 0.19163111349917017,
      "descriptive_p95_seconds": 0.22767180124355946,
      "peak_allocated_gib": 1.2436890602111816,
      "peak_reserved_gib": 1.7578125
    },
    "native_train_stress": {
      "median_seconds": 0.9156421160005266,
      "peak_allocated_gib": 12.850881576538086,
      "peak_reserved_gib": 13.400390625
    }
  }
}.
Parameters {"candidate": 164901255, "parent": 164629637};128 queries/singleargmax.
Costs use2warmups/3reads, TRAIN8 and fixed largest pair; stress updates memory-only.
Order/hardware effects and descriptive p95 remain limitations, no Pareto superiority.

## 5. Conclusion
Scientific status rejected_realization_frozen_subgroup_harm; target0.515 not reached. W2 retained, no promotion.
Small-target subgroup harm versus W2 disconfirms this admitted realization under its original freeze; no restart/retune/renamed successor.
Official unique/multiple metadata unknown; single-object9508/141 is not a substitute.
One seed/checkpoint; full-input repeat instability is retained, not explained away.
Y44 reused the current-tick W2/B1 twice-EACH captures; no extra reference passes counted.
