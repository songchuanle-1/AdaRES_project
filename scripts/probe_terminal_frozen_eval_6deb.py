"""Execute one prospectively frozen full two-pass evaluation."""

import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys

assert P["physical_gpu"] == 1
root = Path(P["root"])
terminal_path = Path("/tmp/adares_terminal_bindings_v3_6debf88fed634a13b772f27bdc244bbd.json")
assert hashlib.sha256(terminal_path.read_bytes()).hexdigest() == EVAL["terminal_sha256"]
terminal = json.loads(terminal_path.read_text())
binding = terminal["methods"][EVAL["identity"]]
assert (root / ".codex_source_commit").read_text().strip() == terminal["code_commit"]
for row in terminal["source_bindings"]:
    raw = (root / row["path"]).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw).hexdigest() == row["lf_sha256"], row["path"]
evaluator = root / "tools/evaluate_adares_frozen_full.py"
assert hashlib.sha256(evaluator.read_bytes()).hexdigest() == EVAL["evaluator_sha256"]
assert hashlib.sha256((root / binding["config"]).read_bytes()).hexdigest() == binding["config_sha256"]
for path in Path("/proc").glob("[0-9]*/cmdline"):
    try:
        argv = path.read_bytes().split(b"\0")
    except OSError:
        continue
    assert not (b"tools/train.py" in argv and binding["config"].encode() in argv)
raw_output = Path(P["output"]).with_name(Path(P["output"]).stem + "_capture.json")
assert not raw_output.exists()
environment = os.environ.copy()
environment["PYTHONPATH"] = os.pathsep.join(sys.path)
subprocess.run([
    sys.executable, "-S", "-c", EVAL["adapter"], str(evaluator),
    "--config", binding["config"], "--checkpoint", binding["checkpoint"],
    "--output", str(raw_output), "--gpu-id", "1",
], check=True, env=environment, timeout=780)
capture = json.loads(raw_output.read_text())
assert capture["complete"] and capture["completed_passes"] == 2
assert capture["config_sha256"] == binding["config_sha256"]
assert capture["checkpoint_sha256"] == binding["checkpoint_sha256"]
assert capture["active_queries"] == [128] and capture["selected_queries"] == 1
assert all(p["summary"]["expressions"] == 9508 and p["summary"]["scenes"] == 141
           for p in capture["passes"])
for row in terminal["source_bindings"]:
    assert hashlib.sha256((root / row["path"]).read_bytes().replace(
        b"\r\n", b"\n")).hexdigest() == row["lf_sha256"]
result = dict(
    complete=True, identity=EVAL["identity"], token=P["token"],
    terminal_sha256=EVAL["terminal_sha256"], evaluator_sha256=EVAL["evaluator_sha256"],
    newly_executed_full_passes=2,
    first_primary_summary=capture["passes"][0]["summary"],
    repeat_summary=capture["passes"][1]["summary"],
    repeat_miou_difference=capture["repeat_miou_difference"],
    repeat_max_iou_difference=capture["repeat_max_iou_difference"],
    parameters=capture["parameters"], warm_first_chunk_seconds=capture["warm_first_chunk_seconds"],
    peak_allocated_gib=capture["peak_allocated_gib"],
    peak_reserved_gib=capture["peak_reserved_gib"], rows=capture,
    elapsed_seconds=capture["elapsed_seconds"],
    no_checkpoint_source_training_queue_writes=True,
    limitation="FIRST primary; repeat retained, not best-of-repeats. Comparative conclusion "
               "requires both candidates and all three frozen references. Outer GPU1 lease "
               "replaces redundant inner flock only; scientific evaluator unchanged.")
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
