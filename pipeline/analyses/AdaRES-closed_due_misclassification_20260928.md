# Foreground due-work classification fault

2026-09-28 11:52 Asia/Shanghai. This tick's `begin` produced design mode
and `due_count=12`. All 12 due IDs are historical `closed` entries:
AdaRES-B4, B5, B4-E30, B2-E30, C1, C2, C1-E30, C2-E30, D1, F4, G1, H5.
The project work ledger has 12 closed, 39 rejected and one complete candidate,
and no executable diagnosis/implementation/admission stage.

The preceding tick passed postflight and saved a research cooldown until
2026-09-28 13:32 +08. The cooldown signature still matches the project and
`cooldown_active` returns true. Canonical `closure.pending()` excludes training,
complete and rejected, but omits legacy `closed`, so the context builder sees
those records as due and bypasses the active cooldown. This is not a new
scientific event or reason to reopen closed experiments.

No candidate, goal, gate, queue, scheduler or historical evidence was changed
in this tick. A project-specific stage rewrite would erase historical
provenance; a global skill correction is outside this AdaRES-only heartbeat.
The foreground lease should be aborted as INCOMPLETE with this reason.
