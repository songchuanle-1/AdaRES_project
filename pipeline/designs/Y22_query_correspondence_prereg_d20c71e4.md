# Y22 Query Correspondence Diagnosis

Owned tick d20c71e418354a6b818d9f37ab72fc7b.
Snapshot 2026-10-02T10:49:03.837056+08:00; deadline 11:49:03.837056+08:00.

## Hypothesis
Joint permutation of native query-indexed predictions must preserve criterion
loss and inverse-permuted gradients when Hungarian matches are unique.
An actual violation would support a correspondence repair, not a new metric
target calibration or an inference threshold change.

## Frozen Controls
- Use the actual W2 criterion/matcher, unchanged loss weights and GT.
- Identity repeat, fixed cyclic permutation, and inverse-permuted gradients.
- Source-derived unique-match and exact-tie cases; ties are explicitly separate.
- If a discrepancy is found, require native training-output confirmation with
  repeat controls before claiming a population defect.
- If exact correspondence holds, reject the alleged implementation defect.

## Source Correction Before Execution
The actual forward uses geometric seed/instance-label correspondence, not
Hungarian assignment. The initial assumption above is retained as an audit
trail. Execute native cached train8 criterion with every layer's query axis
and final sample_inds jointly cyclic-permuted. Compare loss terms and final
mask/indicator/proj-query gradients after inverse permutation; repeat is exact
same cached inputs. This diagnosis concerns criterion correspondence only,
not upstream end-to-end reproducibility.

## Bounds
Train-only; no validation tuning, new production package, admission or dispatch.
Source/synthetic feasibility does not establish trained efficacy. End-to-end
raw-logit repeat noise from Y20 remains unresolved until separately diagnosed.
