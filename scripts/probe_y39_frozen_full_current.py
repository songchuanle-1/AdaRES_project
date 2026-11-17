"""Frozen full evaluation; outer authorized runner owns the GPU1 lease."""

import hashlib
import sys
from pathlib import Path

root = Path(P["root"])
evaluator = root / "tools/evaluate_adares_frozen_full.py"
expected_evaluator = "7824e7807b44c6344b158570626d7ad90fb5fc44eb6852dbd475d34440b1dc13"
assert hashlib.sha256(evaluator.read_bytes()).hexdigest() == expected_evaluator
assert P["physical_gpu"] == 1


def preserved():
    cmd = Path("/proc/116906/cmdline").read_bytes().split(b"\0")
    assert b"tools/train.py" in cmd
    assert b"configs_song/model_paper5/adares_y41_boundary_probability_res.yaml" in cmd
    assert cmd[cmd.index(b"--gpu_ids") + 1] == b"0"
    return {"pid": 116906, "gpu": 0, "config": "adares_y41_boundary_probability_res"}


adapter = r'''
import ast, sys
from pathlib import Path
path = Path(sys.argv[1])
tree = ast.parse(path.read_text(encoding="utf-8"))
main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
removed = []
body = []
for node in main.body:
    assignment = (
        isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "lease"
        and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "open"
        and isinstance(node.value.args[0], ast.Constant)
        and node.value.args[0].value ==
            "/bak_home/songcl/Project/AdaRES_project/.y15_gpu1_diagnostic.lock"
    )
    flock = (
        isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "fcntl" and node.value.func.attr == "flock"
        and isinstance(node.value.args[0], ast.Name)
        and node.value.args[0].id == "lease"
    )
    if assignment or flock:
        removed.append("assignment" if assignment else "flock")
    else:
        body.append(node)
assert removed == ["assignment", "flock"], removed
main.body = body
sys.argv = [str(path)] + sys.argv[2:]
namespace = {"__name__": "__main__", "__file__": str(path)}
exec(compile(tree, str(path), "exec"), namespace)
'''
bindings = [
    ("y39", "adares_y39_point_distribution_res",
     "adares_y39_point_distribution_res/20261010_074811/best.pth",
     "1a170434f63da98e10f59c5b17473cde5cf3ff6be0e4381e2771f322853ffa03",
     "4a50225433aa755bf9b0f5643b9bcee8b025c828dcd5aaede2bc092b95fb68e1"),
    ("w2", "adares_w2_pointmass_competition_res",
     "adares_w2_pointmass_competition_res/20260926_145041/best.pth",
     "d4c35ff10712bfb9ece3e25daa9c3b63143ce313f43eaf19735abaa31f6693eb",
     "e3407283384b4899524d055ac146c7bd6cf39cedb10e58a44ad74eb522009f52"),
    ("b1", "adares_b1_standard_res",
     "adares_b1_standard_res/20260918_010854/best.pth",
     "04e3cc580751c72a6fc908b2ff93ad931ab4a3a7d07d1fc9b7ae30a9a993e1dc",
     "cfc89ec0dc64f240913968f307d19c623f5ff0dc51c8e61de64e35abea05dd8b"),
]
result = {
    "complete": False, "id": "AdaRES-Y39", "rows": {},
    "evaluator_sha256": expected_evaluator,
    "safety_adapter_sha256": hashlib.sha256(adapter.encode()).hexdigest(),
    "safety_adapter": "Remove only evaluator GPU1 lease assignment/flock; "
                      "outer runner exclusively owns GPU1. Scientific AST unchanged.",
    "preserved_before": preserved(), "physical_gpu": 1,
}
for name, config_name, checkpoint_name, config_hash, checkpoint_hash in bindings:
    config = "configs_song/model_paper5/" + config_name + ".yaml"
    checkpoint = "/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/" + checkpoint_name
    assert hashlib.sha256((root / config).read_bytes()).hexdigest() == config_hash
    output = Path(P["output"]).with_name(Path(P["output"]).stem + "_" + name + ".json")
    child_environment = os.environ.copy()
    child_environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    subprocess.run([
        sys.executable, "-S", "-c", adapter, str(evaluator), "--config", config,
        "--checkpoint", checkpoint, "--output", str(output), "--gpu-id", "1",
    ], check=True, env=child_environment)
    capture = json.loads(output.read_text(encoding="utf-8"))
    assert capture["complete"] and capture["completed_passes"] == 2
    assert capture["checkpoint_sha256"] == checkpoint_hash
    assert all(p["summary"]["expressions"] == 9508 and
               p["summary"]["scenes"] == 141 for p in capture["passes"])
    assert capture["active_queries"] == [128] and capture["selected_queries"] == 1
    result["rows"][name] = capture
    result.setdefault("summaries", {})[name] = capture["passes"][0]["summary"]
    result["preserved_after_" + name] = preserved()
    Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
result["complete"] = True
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
