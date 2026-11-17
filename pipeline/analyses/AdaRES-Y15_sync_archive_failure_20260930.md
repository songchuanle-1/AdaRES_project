# Additive Sync Archive Conversion Failure

2026-09-30, lease0c2d6bb3cc524bd59b928b7af3b2f8ec.
First gated dispatch blocked before launch:
Refusing mismatched existing remote file: Y15 GT config.

Disk, Git blob and remote SHA256 all:
e2c0d7f4f159fcad2a94ec04d7884c2af8e8a22dac8cc64dfc340b1ce61866f9.
Git archive with inherited Windows core.autocrlf=true produced:
c069edca82ba277ca2e5d5693c5790d40b3d4840bd9cec956120e24fd94b5f63.
Git archive with per-command core.autocrlf=false reproduced the actual blob:
e2c0d7f4f159fcad2a94ec04d7884c2af8e8a22dac8cc64dfc340b1ce61866f9.

Scoped fix: disable conversion on all three archive invocations in project
scripts/sync.sh (expected hash, missing-file transfer, post-transfer hash).
No model/config changes, global Git config edits, marker bypass or remote
overwrite. Preserve additive existing-file rejection and every-byte checks.
New source commit includes this deployment-only repair; receipt must bind
that actual HEAD. No training process arose from the failed attempt.
