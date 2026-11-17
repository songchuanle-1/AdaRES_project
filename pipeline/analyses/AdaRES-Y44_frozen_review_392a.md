# Experiment Result Analysis - AdaRES-Y44
Observed locally 2026-10-14T13:49:41.986768+08:00; original SAME training done_complete30.

## 1. Basic Info
Server2/singleGPU; resumed original run, last/scheduler30, best epoch18.
Original checkpoints immutable. Training start/end/duration clock relation unknown.
Terminal proof: {'path': 'pipeline/analyses/y44_terminal_392a_392a5a9f5fe242d7a3109a50e6060fda.json', 'sha256': 'da9e8e97dd30a6c580e5a2c876302ad35bee4484b1877af78894d9c87fc7a516'}.

## 2. Best Metrics
| Metric | Frozen FIRST | Population |
|---|---|---|
| mIoU | 0.5102523337714301 | 9508 expressions/141 scenes |
| Acc@0.25 | 0.6116954143878839 | same |
| Acc@0.5 | 0.5629995793016407 | same |
| spmIoU | 0.47409511779856267 | same |
Checkpoint metadata 0.5102537274360657 is not the frozen efficacy endpoint.

## 3. Convergence Summary
Fixed30 unchanged; best saved epoch18. FIRST is primary, not best repeat.
Second mIoU 0.5102544712623306.
Whole-scene bootstrap10000seed1999; TRAIN cuts1487/3325 unchanged.
Full comparison/noise/subgroup evidence: {'path': 'pipeline/analyses/y44_clustered_392a_392a5a9f5fe242d7a3109a50e6060fda.json', 'sha256': '464baefb0d5827732d9035a1dc5fdb822e07843fb35bf344863866dcccf35cbc'}.

## 4. Comparison
W2: {
  "all": {
    "expressions": 9508,
    "scenes": 141,
    "paired_miou_difference": 0.00014433858253680922,
    "scene_clustered_ci95": [
      -0.007431517078138552,
      0.007619130239067876
    ],
    "max_expression_repeat_iou_difference": 0.01891397126019001,
    "material_harm_within_observed_noise": false,
    "positive_difference_beyond_observed_noise": false,
    "status": "interval includes zero",
    "historical_abs_mean_repeat_difference": 2.13749090042694e-06,
    "observed_group_mean_absolute_repeat_difference": 3.4343871669228656e-06,
    "frozen_material_harm_ci_upper_below_zero": false
  },
  "small": {
    "expressions": 2816,
    "scenes": 124,
    "paired_miou_difference": -0.011699453318669839,
    "scene_clustered_ci95": [
      -0.02904200443776662,
      0.0054340726705627024
    ],
    "max_expression_repeat_iou_difference": 0.0,
    "material_harm_within_observed_noise": false,
    "positive_difference_beyond_observed_noise": false,
    "status": "interval includes zero",
    "historical_abs_mean_repeat_difference": 0.0,
    "observed_group_mean_absolute_repeat_difference": 0.0,
    "frozen_material_harm_ci_upper_below_zero": false
  },
  "medium": {
    "expressions": 3304,
    "scenes": 136,
    "paired_miou_difference": 0.00032179156818920997,
    "scene_clustered_ci95": [
      -0.013596155764784231,
      0.014629748039947274
    ],
    "max_expression_repeat_iou_difference": 0.01891397126019001,
    "material_harm_within_observed_noise": false,
    "positive_difference_beyond_observed_noise": false,
    "status": "interval includes zero",
    "historical_abs_mean_repeat_difference": 5.7245675726967345e-06,
    "observed_group_mean_absolute_repeat_difference": 5.7245675726967345e-06,
    "frozen_material_harm_ci_upper_below_zero": false
  },
  "large": {
    "expressions": 3388,
    "scenes": 135,
    "paired_miou_difference": 0.009815475928818507,
    "scene_clustered_ci95": [
      -0.0039901673027129166,
      0.024303218519514825
    ],
    "max_expression_repeat_iou_difference": 0.007574737071990967,
    "material_harm_within_observed_noise": false,
    "positive_difference_beyond_observed_noise": false,
    "status": "interval includes zero",
    "historical_abs_mean_repeat_difference": 4.1596582676190554e-07,
    "observed_group_mean_absolute_repeat_difference": 4.055543660836068e-06,
    "frozen_material_harm_ci_upper_below_zero": false
  }
}
B1: {
  "all": {
    "expressions": 9508,
    "scenes": 141,
    "paired_miou_difference": 0.008030515813282704,
    "scene_clustered_ci95": [
      -0.0009903217903969063,
      0.017054064205161708
    ],
    "max_expression_repeat_iou_difference": 0.9738956093788147,
    "material_harm_within_observed_noise": false,
    "positive_difference_beyond_observed_noise": false,
    "status": "interval includes zero",
    "historical_abs_mean_repeat_difference": 0.00011025620863424613,
    "observed_group_mean_absolute_repeat_difference": 0.00011025620863424613,
    "frozen_material_harm_ci_upper_below_zero": false
  },
  "small": {
    "expressions": 2816,
    "scenes": 124,
    "paired_miou_difference": 0.01059427366590171,
    "scene_clustered_ci95": [
      -0.010492268974689302,
      0.031666337910078314
    ],
    "max_expression_repeat_iou_difference": 0.0,
    "material_harm_within_observed_noise": false,
    "positive_difference_beyond_observed_noise": false,
    "status": "interval includes zero",
    "historical_abs_mean_repeat_difference": 0.0,
    "observed_group_mean_absolute_repeat_difference": 0.0,
    "frozen_material_harm_ci_upper_below_zero": false
  },
  "medium": {
    "expressions": 3304,
    "scenes": 136,
    "paired_miou_difference": 0.013309787682761589,
    "scene_clustered_ci95": [
      -0.001821185130374481,
      0.028636171505727524
    ],
    "max_expression_repeat_iou_difference": 0.01891397126019001,
    "material_harm_within_observed_noise": false,
    "positive_difference_beyond_observed_noise": false,
    "status": "interval includes zero",
    "historical_abs_mean_repeat_difference": 5.7245675726967345e-06,
    "observed_group_mean_absolute_repeat_difference": 5.7245675726967345e-06,
    "frozen_material_harm_ci_upper_below_zero": false
  },
  "large": {
    "expressions": 3388,
    "scenes": 135,
    "paired_miou_difference": 0.0007512193641288193,
    "scene_clustered_ci95": [
      -0.013218858281244453,
      0.015225713201762475
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
Frozen harm triggers versus either: {"w2": [], "b1": []}.
Trained paired costs: {
  "candidate": {
    "train8": {
      "median_seconds": 0.08065403949876782,
      "descriptive_p95_seconds": 0.12344273064663866,
      "peak_allocated_gib": 1.122016429901123,
      "peak_reserved_gib": 1.5546875
    },
    "largest_pair": {
      "median_seconds": 0.19499646249460056,
      "descriptive_p95_seconds": 0.22928575449986965,
      "peak_allocated_gib": 1.5355749130249023,
      "peak_reserved_gib": 2.3359375
    },
    "native_train_stress": {
      "median_seconds": 0.945341398997698,
      "peak_allocated_gib": 12.910467624664307,
      "peak_reserved_gib": 13.392578125
    }
  },
  "parent": {
    "train8": {
      "median_seconds": 0.07461740049620857,
      "descriptive_p95_seconds": 0.11836859750474105,
      "peak_allocated_gib": 0.9607234001159668,
      "peak_reserved_gib": 1.234375
    },
    "largest_pair": {
      "median_seconds": 0.18734846899315016,
      "descriptive_p95_seconds": 0.22085162800431135,
      "peak_allocated_gib": 1.2436890602111816,
      "peak_reserved_gib": 1.7578125
    },
    "native_train_stress": {
      "median_seconds": 0.9184022349945735,
      "peak_allocated_gib": 12.850881576538086,
      "peak_reserved_gib": 13.400390625
    }
  }
}.
Parameters {"candidate": 164917639, "parent": 164629637};128 queries/singleargmax.
Costs use2warmups/3reads, TRAIN8 and fixed largest pair; stress updates memory-only.
Order/hardware effects and descriptive p95 remain limitations, no Pareto superiority.

## 5. Conclusion
Scientific status complete_inconclusive; target0.515 not reached. W2 retained, no promotion.
Intervals cross zero; no benefit, equivalence or no-harm conclusion; do not retune or restart.
Official unique/multiple metadata unknown; single-object9508/141 is not a substitute.
One seed/checkpoint; full-input repeat instability is retained, not explained away.
Y44 reused the current-tick W2/B1 twice-EACH captures; no extra reference passes counted.
