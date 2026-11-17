# Native Resume Probe Save-Path Failure

Lease a1526671b2ae48f7a80c4f8128bf645a, 2026-09-30.
First native update completed, then gorilla.save_checkpoint attempted
`os.makedirs('')` for a basename-only checkpoint and raised FileNotFoundError.
No resume result was produced. Corrected the probe checkpoint to an absolute
path with an existing project parent; no production saver/schema change.
