"""Execute the unchanged frozen evaluator twice each for Y45, W2 and B1."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(P["root"])
assert P["physical_gpu"] == 1
terminal_path = Path("/tmp/adares_y45_terminal_601b_" + P["token"] + ".json")
terminal = json.loads(terminal_path.read_text())
assert terminal["complete"] and terminal["id"] == "AdaRES-Y45"
best = next(row for row in terminal["checkpoints"] if row["path"].endswith("/best.pth"))
evaluator = root / "tools/evaluate_adares_frozen_full.py"
evaluator_sha = "7824e7807b44c6344b158570626d7ad90fb5fc44eb6852dbd475d34440b1dc13"
assert hashlib.sha256(evaluator.read_bytes()).hexdigest() == evaluator_sha


def preserved():
    for source, sha in terminal["source_hashes"].items():
        assert hashlib.sha256((root / source).read_bytes()).hexdigest() == sha
    assert not Path("/proc/78001").exists()
    for row in terminal["checkpoints"]:
        path = Path(row["path"])
        assert path.stat().st_size == row["bytes"]
    return "original_Y45_finished_no_restart_sources_unchanged"


# The outer runner holds the same exclusive GPU1 lock across all subprocesses.
adapter = r'''
import ast,sys
from pathlib import Path
path=Path(sys.argv[1])
tree=ast.parse(path.read_text(encoding="utf-8"))
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="main")
removed=[]
body=[]
for node in main.body:
    assignment=(isinstance(node,ast.Assign) and len(node.targets)==1
        and isinstance(node.targets[0],ast.Name) and node.targets[0].id=="lease"
        and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name)
        and node.value.func.id=="open" and isinstance(node.value.args[0],ast.Constant)
        and node.value.args[0].value=="/bak_home/songcl/Project/AdaRES_project/.y15_gpu1_diagnostic.lock")
    flock=(isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)
        and isinstance(node.value.func,ast.Attribute)
        and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=="fcntl"
        and node.value.func.attr=="flock"
        and isinstance(node.value.args[0],ast.Name) and node.value.args[0].id=="lease")
    if assignment or flock:removed.append("assignment" if assignment else "flock")
    else:body.append(node)
assert removed==["assignment","flock"]
main.body=body
sys.argv=[str(path)]+sys.argv[2:]
exec(compile(tree,str(path),"exec"),{"__name__":"__main__","__file__":str(path)})
'''
bindings = [
    ("y45", "adares_y45_metric_decomposition_res", best["path"],
     terminal["source_hashes"]["configs_song/model_paper5/adares_y45_metric_decomposition_res.yaml"],
     best["sha256"]),
    ("w2", "adares_w2_pointmass_competition_res",
     "/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth",
     "d4c35ff10712bfb9ece3e25daa9c3b63143ce313f43eaf19735abaa31f6693eb",
     "e3407283384b4899524d055ac146c7bd6cf39cedb10e58a44ad74eb522009f52"),
    ("b1", "adares_b1_standard_res",
     "/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_b1_standard_res/20260918_010854/best.pth",
     "04e3cc580751c72a6fc908b2ff93ad931ab4a3a7d07d1fc9b7ae30a9a993e1dc",
     "cfc89ec0dc64f240913968f307d19c623f5ff0dc51c8e61de64e35abea05dd8b"),
]
result = dict(
    complete=False, id="AdaRES-Y45", rows={}, summaries={},
    evaluator_sha256=evaluator_sha, physical_gpu=1,
    terminal_sha256=hashlib.sha256(terminal_path.read_bytes()).hexdigest(),
    safety_adapter_sha256=hashlib.sha256(adapter.encode()).hexdigest(),
    safety_adapter="Only redundant inner lock removed; outer runner owns GPU1; scientific evaluator unchanged.",
    preserved_before=preserved(), newly_executed_full_passes=0)
for identity, config_name, checkpoint, config_sha, checkpoint_sha in bindings:
    config = "configs_song/model_paper5/" + config_name + ".yaml"
    assert hashlib.sha256((root / config).read_bytes()).hexdigest() == config_sha
    output = Path(P["output"]).with_name(Path(P["output"]).stem + "_" + identity + ".json")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    subprocess.run([
        sys.executable, "-S", "-c", adapter, str(evaluator),
        "--config", config, "--checkpoint", checkpoint,
        "--output", str(output), "--gpu-id", "1"], check=True, env=environment)
    capture = json.loads(output.read_text())
    assert capture["complete"] and capture["completed_passes"] == 2
    assert capture["checkpoint_sha256"] == checkpoint_sha
    assert capture["config_sha256"] == config_sha
    assert all(row["summary"]["expressions"] == 9508 and row["summary"]["scenes"] == 141
               for row in capture["passes"])
    assert capture["active_queries"] == [128] and capture["selected_queries"] == 1
    result["rows"][identity] = capture
    result["summaries"][identity] = capture["passes"][0]["summary"]
    result["newly_executed_full_passes"] += 2
    result["preserved_after_" + identity] = preserved()
    Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
result["complete"] = True
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
