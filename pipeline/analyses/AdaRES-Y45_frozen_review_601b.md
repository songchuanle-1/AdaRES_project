# Y45 Frozen Terminal Review

## 1. Basic Info
- Original fixed30 training is done_complete; last model/optimizer/scheduler
  checkpoint epoch30, best checkpoint epoch16, seed1999/released SPFormer.
- No restart, validation retuning, favorable-repeat choice or checkpoint change.
- Training best0.5076589584; frozen FIRST evaluation is the primary result.

## 2. Best Metrics
| Method | FIRST mIoU | Acc@0.25 | Acc@0.5 | spmIoU |
|---|---|---|---|---|
| Y45 | 0.5076567776 | 0.6159023980 | 0.5574253260 | 0.4726254392 |
| W2 | 0.5101051604 | 0.6107488431 | 0.5629995793 | 0.4737185291 |
| B1 | 0.5022143852 | 0.6104333193 | 0.5529028187 | 0.4667688310 |
- Each method freshly evaluated twice:9508 expressions/141 scenes per pass.
- Second-pass mIoU:Y45 .5076567776/W2 .5101079952/B1 .5022217644.
- Y45 repeat expression differences0; W2 max .0269528031 and B1 .0399250984.
  These repeat observations are not estimates of training-seed uncertainty.

## 3. Convergence Summary
- Complete30/best16; the original final validation ended and saved last.pth.
- Target .515 is unmet. Acc@0.25 improvement is not substituted for this target.
- Training checkpoint meta and frozen evaluation refer to the same best SHA256
  6b37d49d2c5c486c2003058aad67f2917c604fc3300431a2aa7c43065f6a151e.

## 4. Comparison
- Whole-scene bootstrap10000/seed1999, FIRST primary.
- Versus W2:delta -.0024483828, CI95[-.0099743258,.0048384360].
- Versus B1:delta .0054423924, CI95[-.0034831786,.0144328557].
- W2 size strata:small delta-.0080450057/CI[-.0255642836,.0097018352];
  medium-.0069537442/CI[-.0210019507,.0071475734];
  large+.0065970138/CI[-.0058443878,.0186920636].
- All predeclared size-group intervals versus either comparator cross0.
  No frozen CI-upper<0 harm trigger; this is not proof of no harm/equivalence.
- Official unique/multiple metadata remains unknown.

## 5. Conclusion
- Scientific status:complete_inconclusive; failure_scope:inconclusive.
- No promotion, superiority, necessity, no-harm or broad mechanism rejection.
- Keep W2 as the scientific comparator. Do not select a local patch, feature
  subset, coefficient or epoch using this validation result.
- Added197635 parameters over W2's164629637. Trained serial TRAIN8 median
  Y45 .0756801405s/W2 .0745186605s; largest-pair .1926195330/.1929158180s.
- Separate trained native optimizer stress:median .9213630040/.9207984650s,
  peak reserved13.541015625/13.400390625GiB. Updates were memory-only;
  original checkpoint hashes rechecked unchanged. Order/hardware transients
  limit timing comparisons; no causal Pareto-dominance claim.

## 6. Evidence
- Terminal:y45_terminal_601b_601bcda714184133920945f8f067b382.json.
- Six fresh passes:y45_frozen_601b_601bcda714184133920945f8f067b382.json,
  SHA256 ce0c1b844c6b53bb25b717c68849b00f702767dd7d60d93ab5d18acd268871eb.
- Scene/subgroup:y45_clustered_601b_601bcda714184133920945f8f067b382.json.
- Cost:y45_trained_cost_601b_601bcda714184133920945f8f067b382.json,
  SHA256 c459319df3829b2fd6d5991dff9915b086aebc8c7112f509ea1f19e1772f15d3.
- Source, training history and earlier adverse diagnostic evidence retained.
