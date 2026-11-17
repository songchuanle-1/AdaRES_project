# Y15 Provider Probe Failure

Observed 2026-09-30 Asia/Shanghai under lease
a91e889cb5814df086e736bc97f5ad5c.
The process completed two augmented batches and two native train() steps but
did not save its final JSON. These stdout observations are not a finalized
admission artifact.

Failure:
```
File "tools/probe_adares_y15_provider.py", line 147, in main
    fresh(one, mode="predict")
TypeError: 'NoneType' object is not callable
```

The inherited IPDN.train() override has no return. eval() delegates to train(False)
and therefore a chained IPDN(...).cuda().eval() evaluates to None. Fix the probe
to create the model and call eval separately. No production/parent training
behavior is changed and no earlier failure is relabeled PASS.
Rerun the full probe; preserve this failed attempt.
