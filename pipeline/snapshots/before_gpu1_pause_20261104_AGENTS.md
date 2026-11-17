# Current User Resource Instructions

## Server-2 GPU1 Reauthorized By User (2026-10-29)

- The owning user said: "server2 GPU1也可以使用".
- AdaRES may use server-2 GPU0 and GPU1 plus server-1 GPU0, one independent experiment per GPU.
- Keep server-2 GPU0 at280W. GPU1 was observed at its existing390W limit; no power/hardware change is authorized or needed.
- This resource grant supersedes earlier GPU1/GPU0-only resource restrictions, including the resource-only sentence in the frozen goal and old heartbeat text.
- Y57's separate experiment pause remains in effect. Preserve its original epoch27 model/optimizer/scheduler checkpoint; resume only on an explicit Y57 continuation request.
- Reuse the existing verified private runtime and exact dataset. Preserve Y65/Y66, foreign jobs, all scientific/data/seed/fixed30 boundaries and server-6 ban.
- The scientific goal file stays byte-identical to preserve signed running receipts; current resource authority is this section, SERVERS.yaml and pipeline/authorizations/server2_gpu1_20261029.json.
- Keep the existing owning thread and schedule frequency; update only the saved resource instructions to match this grant.


## Server-1 Added By User (2026-10-25)

- In the owning chat the user said: "继续推进训练  然后server1 的gpu也可以使用".
- AdaRES may additionally use server-1 (192.168.1.175, verified hostname
  dl02-MS-7B73), GPU0, one independent single-GPU experiment at a time.
- Deploy only this project, reuse the exact existing dataset and released
  backbone, and verify the runtime, source, native checks and live occupation.
- Server-2 remains GPU0-only with its approved 280W limit. Y57/GPU1 remains
  paused; do not resume, migrate, restart it or interfere with foreign jobs.
- This explicit resource grant supersedes earlier server-2-only language for
  server-1 GPU0 only. Server-6 and scheduler changes remain prohibited.
- Authorization record: pipeline/authorizations/server1_gpu0_20261025.json.

## Temporary GPU0-Only Trial (2026-10-21)

- The user explicitly requested pausing Y57/GPU1 and trying GPU0 alone.
- Only server-2 GPU0 is currently authorized for this project's training and
  GPU diagnostics. Keep `allowed_gpus: [0]`; do not refill GPU1 or restore
  its authorization without a new user instruction. Do not change scheduling.
- Preserve the continuing Y56/GPU0 process and its approved 280W runtime
  power limit. Do not migrate Y57 onto it or restart Y56.
- Y57 is user-paused, not rejected, crashed or scientifically complete.
  Preserve its original logs and latest fully saved epoch-27 checkpoint,
  including optimizer/scheduler. Do not automatically resume it or restart
  from pretrain. A future user-approved continuation keeps seed1999/fixed30.
- This temporary restriction supersedes the October 9 two-GPU authorization
  below until the user explicitly changes the trial instruction.

## Latest GPU Authorization (2026-10-09)

- The user explicitly reauthorized server-2 GPU0 for AdaRES_project on
  2026-10-09. This supersedes the GPU0 reservation in the historical
  instructions below. `allowed_gpus: [0, 1]` is now authorized.
- Use exactly one GPU per experiment and at most one independent experiment
  per GPU. GPU1's existing Y33 training must not be moved or restarted.
- Authorization does not imply a free GPU. Verify live occupation before
  diagnostics or dispatch, preserve existing jobs, and retain all admission
  gates. No server-6 or other-project authorization is granted.
- Continue using the existing owning foreground task. Do not change the
  scheduler, resume background execution, or invoke nested `codex exec`.

## Historical Y23 Handoff (2026-10-02)

The instructions below record the original handoff, not a new resume request.
The prior GPU0 reservation was explicitly lifted on 2026-10-09.

- On 2026-10-02 the user requested pausing AdaRES-Y23 and freeing server-2
  GPU0, then resuming the SAME Y23 experiment on GPU1 only AFTER AdaRES-Y18
  completes its fixed 30 epochs.
- GPU0 is reserved outside this project's authorization until the user
  explicitly returns it. Do not refill it, use it for diagnostics, or restore
  `allowed_gpus: [0, 1]` autonomously.
- Preserve Y18 PID7449 and its original training. Do not stop it early to
  satisfy the resume dependency.
- Y23 is paused, not rejected, crashed, or scientifically complete. Preserve
  its original log and last checkpoint. Resume model, optimizer and scheduler,
  total epochs30/seed1999, no restart from pretrain or epoch-count tuning.
- The canonical AdaRES launcher currently lacks a resume argument. Do NOT
  change Y23 to generic waiting: that would start it from scratch.
- Under an owned foreground lease, use the project-local guarded helper
  `scripts/resume_y23_after_y18.py --token <current-token>` once Y18 is
  canonically done_complete. It fails closed on unmet dependencies, occupied
  GPU1, changed checkpoint/source or duplicate process. `--check-only` checks
  the dependency without starting training.
- Continue using the canonical foreground entry and existing heartbeat; no
  new scheduler, background polling process, other-project operation or nested
  codex exec. Record verified native steps before claiming resumed training.
