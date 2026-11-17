# AdaRES-Y6: Quality heterogeneity among native positive queries

Frozen 2026-09-28 14:54 Asia/Shanghai before native Y6 execution. Y5
established that the complete nearest-plus-in-object native positives do
not systematically omit the error oracle. A distinct testable failure is
that the binary indicator treats *all* matched positives alike, even if
their predicted point-mask IoUs and score ranks disagree. This is not a
license to rerun G1's previously failed all-query IoU soft labels (0.4980
after 30 epochs) or C2's margin/loss tuning.

Use unchanged W2 best weights, first 64 unaugmented train scenes and first
32 val scenes, seed 1999. Recover exactly the criterion's final matched
positive set per expression using `match_sampled_ins`. Only expressions
with at least two distinct matched queries are eligible. Compute their
point-IoU range (best minus worst positive), fraction where the native
highest-scored positive is not the best-IoU positive, and the paired
best-versus-highest-scored positive gap. Compare to a cyclically shifted
positive set of the same cardinality as a count-preserving control; also
report scene means and double-forward mismatches. Material errors retain
the previously fixed oracle-selected gap >0.05.

The mechanism is supported only if the true positive set has meaningful
within-set quality variation *and* score misordering is persistent at the
scene level, with a different pattern than shifted membership. A favorable
oracle measurement alone is not an inference-valid selector or a new
training gate. Failure or a near-identical shifted control rejects this
specific rationale. Any future candidate must define a structurally
distinct local, relative, inference-valid interaction and undergo six
native admission gates before a 30-epoch dispatch.
