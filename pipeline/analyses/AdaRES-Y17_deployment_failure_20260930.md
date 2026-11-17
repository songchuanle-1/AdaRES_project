# Y17 Deployment Failure And Resume Plan

Observed 2026-09-30 21:37-21:42 Asia/Shanghai, original lease f922f777...
Deadline was21:41:14+08; no new research tests after this boundary.

Native admission and candidate commit90c75192ce990c66972e82f229b87ddad02421e5
completed. Gated monitor refused deployment on existing remote
ipdn/model_paper5/adares_y17_target_equivalent_pairing/backbone.py
because the tested raw CRLF physical copy differs from committed LF blob.
No train.py replacement or source-marker advance occurred.
Three dispatch attempts gave the same refusal; only the latter attempt
followed an actionable transport repair, which itself failed before writing.
A reconcile-only call preceded dispatch and did NOT launch training.

Normalization helper first exceeded Windows process command-line length;
moving payload into stdin repaired transport. Remote default python3 is3.5:
json.loads(base64.b64decode(...)) rejects bytes. Last diagnostic captured
that exact TypeError; no normalization evidence JSON was emitted and no
candidate bytes changed. A failed helper is not a successful sync.
Guarded train-entry sync helper uses the same default interpreter/bytes
pattern and must ALSO be corrected before dispatch can succeed.

Next owned tick: use /home/songcl/.conda/envs/adares_project/bin/python for
both helper remote bodies, or explicitly decode UTF8 at the JSON boundary;
test isolated guarded-update script using that exact interpreter. Maintain
old/new hashes, stale-marker rejection and path whitelist. Run normalization
only for remote bytes matching previously tested raw or committed hashes and
equal after CRLF->LF normalization. Source marker remains2bb1c107...
After helper fix, candidate-only commit latest tested code, refresh receipt/
queue commit, do not modify unrelated dirty files. Use gated dispatcher,
verify unique log and increasing native steps. Never report PID-only success.

At21:42:38 local observation GPU0 15,788MiB/86%, GPU1 1MiB/0%.
Y15 PID18796 alive, Train[2/30]3920->3921/4686, fixed30epoch protected.
Monitor parsed best mIoU0.0760 at epoch2; this is an early training result,
not the goal champion or a completed30epoch efficacy estimate.
Y17 remains waiting/deployment, no PID/verified training, GPU1 available.
No remote training restart, scheduler change or other-project operation.
