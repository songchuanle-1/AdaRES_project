# Y15 Native Integration Summary

2026-09-30 12:06 Asia/Shanghai. Lease a91e889cb5814df086e736bc97f5ad5c.
Scientific stage: validation, NOT ADMITTED. No full-model training launched.

## Implementation

Physical teacher/GT-only packages and configs; no cross-variant production
imports. Final K=128 -> 129 candidates, same competition indicator.
Six auxiliary layers retain 128 candidates and spatial matching.
New cross-attention query has 329728 parameters in 9 tensors.
Final teacher target substitutes extra-query BCE only, no new loss coefficient.
Teacher is a frozen local physical reference, generated on current augmented
training batch, excluded from student state/optimizer, absent in new-instance
predict. Released SPFormer backbone remains full-training initialization.

## Evidence

| Test | Outcome | Limitation |
|---|---|---|
| W2-initialized two real batches / four scans | All 9 new tensors finite nonzero in GT/teacher/language/slot arms | Connectivity, not efficacy |
| Native tools/train.py train loop | Two steps, all 9 tensors updated | No full-model run |
| Released SPFormer, two augmented batches / four scans | Online teacher alignment, detached targets, all 9 tensors nonzero; two optimizer steps | First-four training scenes only |
| Same-output GT/teacher criterion replay | All non-BCE/auxiliary losses exactly equal | Teacher increment is weak |
| Teacher versus GT new-gradient cosine | 0.9999997616 / 0.9999999404; max differences 0.0000571 / 0.0001876 | Necessity not proved; do not inflate weights |
| Fixed dense input language/slot controls | Extra-logit differences 2.417/2.186 and 8.201/9.573; native K masks exactly unchanged | No population improvement claim |
| Fixed decoder inputs repeated | Raw masks/indicators exactly identical | End-to-end sparse reductions still fluctuate |
| Fixed output GT-zeroed prediction | Exactly invariant; fresh predict never constructs teacher | Earlier v1/v2 strict probability equality failures retained |
| Clean-process teacher resources | 5.242/7.611 GiB reserved; .557/.630 s step | No largest-scan guarantee |
| Clean-process GT resources | 4.627/6.992 GiB reserved; .396/.439 s step | Teacher adds ~.62 GiB and 41%-44% training time |
| Warm student inference | enabled .076543/.095135 s; disabled .075497/.093857 s | Two train scans only |

Student total/trainable parameters: 164959365 / 40313733 in both packages.
Control probe retained graph dictionaries: its 8.49/13.08 GiB values are probe
footprints and are preserved, not used as clean resource maxima.
Provider probe first failed because .eval() returned None through an inherited
train() override; fixed probe invocation and reran fully. Failure preserved.

## Gates And Next Test

Diagnosis has real train64 missing-mask evidence. Native backward and physical
independence demonstrated on tested paths. Control signal is reproducible
but teacher-specific gradient nearly redundant; not an efficacy rejection.
Resources measured but lightweight Pareto/harm rule not frozen; admission
receipt absent. No queue mutation, commit or training dispatch.

Next: stable Y15 validation. Freeze evaluation/Pareto and harm policy without
validation tuning; test parent, teacher and independent GT package on
train-selected large scans with optimizer/resume and serialization round trip;
verify correct-versus-shuffled teacher target gradient on material cases and
estimate end-to-end repeat noise. Decide whether teacher cost is justified or
retain GT-only as the same candidate's causal control. Do not multiply BCE,
retune 0.05 or rename a rejected route to fill GPUs.

Live end-of-probe observation 12:06 +08: server-2 GPU0/1 0% /1 MiB;
diagnostic flock free. Windows background remains disabled.
