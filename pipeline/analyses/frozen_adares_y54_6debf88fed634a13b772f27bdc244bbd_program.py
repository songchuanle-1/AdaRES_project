EVAL={'identity': 'AdaRES-Y54', 'terminal_sha256': '87939dea5b7d59955fc6c0f82019a334aa85fd5b2dcea1eb4fa0b5fd54c5f6f6', 'evaluator_sha256': '7824e7807b44c6344b158570626d7ad90fb5fc44eb6852dbd475d34440b1dc13', 'adapter': '\nimport ast,sys\nfrom pathlib import Path\npath=Path(sys.argv[1])\ntree=ast.parse(path.read_text(encoding="utf-8"))\nmain=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="main")\nremoved=[]\nbody=[]\nfor node in main.body:\n    assignment=(isinstance(node,ast.Assign) and len(node.targets)==1\n        and isinstance(node.targets[0],ast.Name) and node.targets[0].id=="lease"\n        and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name)\n        and node.value.func.id=="open" and isinstance(node.value.args[0],ast.Constant)\n        and node.value.args[0].value=="/bak_home/songcl/Project/AdaRES_project/.y15_gpu1_diagnostic.lock")\n    flock=(isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)\n        and isinstance(node.value.func,ast.Attribute)\n        and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=="fcntl"\n        and node.value.func.attr=="flock"\n        and isinstance(node.value.args[0],ast.Name) and node.value.args[0].id=="lease")\n    if assignment or flock:removed.append("assignment" if assignment else "flock")\n    else:body.append(node)\nassert removed==["assignment","flock"]\nmain.body=body\nsys.argv=[str(path)]+sys.argv[2:]\nexec(compile(tree,str(path),"exec"),{"__name__":"__main__","__file__":str(path)})\n'}
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
