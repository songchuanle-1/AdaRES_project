# Y15 Augmented Matched Controls

Registered 2026-09-30 11:57 Asia/Shanghai, before this specific test.
Use first four distinct train scans and their exact same sampled augmented
batches. Teacher and GT physical packages load the same released SPFormer
backbone and identical decoder initialization/state. No W2 warm start.

Cache current-batch detached teacher masks once. After capturing the main
criterion inputs, replay the native criterion on the SAME decoder outputs,
so a teacher-target swap cannot be confused with sparse-backbone nondeterminism.
Require GT-only and teacher auxiliary losses equal; final dice/indicator/score
and semantic losses equal; BCE may differ and every new tensor must get finite
nonzero gradients. If every teacher target equals GT, report redundancy rather
than claim teacher-specific information.

For the same augmented inputs, compare full loss/backward with the separate
GT-only package and the online provider. Warm once each, synchronize, clear
allocator between arms, record allocated/reserved memory and wall times.
Do not turn a small-cohort task improvement into an admission threshold.
New query inference remains student-only and labels never enter DEC.
Use correct versus cyclic different-expression language and slot-only queries
on frozen dense inputs. Record raw query/mask changes and teacher/GT gradient
cosine. Feature-attention controls are not population benefit estimates.

Unresolved before launch: a prospective lightweight budget, matched native
large-batch resource checks and post-training material-harm plan. The recorded
two-step warm-start and released-init tests are connectivity, not 30 epochs.
