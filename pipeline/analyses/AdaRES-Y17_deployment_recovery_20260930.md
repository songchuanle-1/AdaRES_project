# Y17 Deployment Recovery

Observed 2026-09-30 22:48-23:07 Asia/Shanghai, event lease91c078a1...
Original deadline23:48:15+08 unchanged. No new candidate admitted this tick.
Only existing admitted Y17 deployment resumed; Y15 running process untouched.

## Fault And Repair

Previous raw diagnostic package CRLF differs from committed LF. Used exact
tested-raw/committed SHA256 restrictions, semantic newline equivalence and
all-file prevalidation followed by atomic replacement under sync flock.
15 candidate files verified, 5 physical copies newline-normalized. The
old source marker was deliberately unchanged during this normalization.
JSON explicitly decoded UTF8 and remote interpreter taken from server-2
SERVERS.yaml project Python3.8.13; bulk payload goes through stdin to avoid
Windows command-line length. Exact isolated remote train-update body passed
foreign-file rejection, atomic update, idempotence, stale-marker rejection
and unchanged marker tests, without changing admission/goal gates.

First repaired dispatch exposed Git Bash MSYS converting remote /bak_home
into a local Git path; hash guard rejected before train replacement.
MSYS_NO_PATHCONV disabled only for this helper, but initially left helper
path in /d/project form; the next dispatch failed to locate local script.
Native cygpath conversion of both local script/root corrected this while
remote POSIX root stays literal. Local argv/path existence and bash syntax
tests passed. Failures were retained, not treated as successful deployment.

Scoped commits262c65f/9da56f0/42ffd58 touch deployment helpers only; candidate
model/config identical to admission90c7519. Current receipt/queue codecommit
42ffd589bfb5c27fa18d735866ddefa3a543e2ee.
Canonical gated monitor successfully verified all16 changed AdaRES files,
performed exact old/new hashed train-entry replacement and advanced marker.
No broad overwrite, reset, scheduler change or other-project action.

## Launch Verification

Y17 PID109632 GPU1 unique log
train_adares_y17_target_equivalent_pairing_res_20260930_230343.log.
Gated queue and experiment launch_verified=true; independent native capture
23:06:54 shows Train[1/30]313->314->315/4686, no traceback in captured tail.
Released backbone/seed1999/config epochs30, min_epochs30/patience30 unchanged.
Y15 PID18796 GPU0 still Train[4/30]780->781->782/4686. Early monitor best.1239
is not completed30epoch result or new goal champion.
GPU0/1 memory15928/17030MiB utilization47/72% at capture time.
All deployed candidate files plus train.py match admitted Git blobs; marker
42ffd58 matches currentHEAD. These are execution facts, NOT efficacy/novelty.

## Next Action

Both authorized cards occupied by independent singleGPU experiments.
Subsequent ticks monitor native progress/checkpoints/crashes and fixed30epoch
completion; no extra capacity filler, unadmitted controls or validation tuning.
Keep frozen scene-clustered/harm/Pareto post-training plan, uncertainty/noise
limitations, generic smoothing and matched regrouped pair-off control need.
