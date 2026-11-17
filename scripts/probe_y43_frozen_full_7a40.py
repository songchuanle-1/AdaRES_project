"""Frozen Y43/W2/B1 twice each on the outer authorized GPU0 lease."""

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(P["root"])
assert P["physical_gpu"] == 0
evaluator = root / "tools/evaluate_adares_frozen_full.py"
evaluator_sha = "7824e7807b44c6344b158570626d7ad90fb5fc44eb6852dbd475d34440b1dc13"
assert hashlib.sha256(evaluator.read_bytes()).hexdigest() == evaluator_sha
terminal_path = Path("/tmp/adares_y43_terminal_7a40_" + P["token"] + ".json")
terminal = json.loads(terminal_path.read_text())
assert terminal["complete"] and terminal["id"] == "AdaRES-Y43"
best = next(v for v in terminal["checkpoints"] if v["path"].endswith("/best.pth"))
protected = {
    str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (root / "ipdn/model_paper5/adares_y44_coordinate_feature").glob("*.py")
}
protected_config = "configs_song/model_paper5/adares_y44_coordinate_feature_res.yaml"
protected[protected_config] = hashlib.sha256((root / protected_config).read_bytes()).hexdigest()


def preserved():
    for path, sha in {**terminal["source_hashes"], **protected}.items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == sha
    command = Path("/proc/15520/cmdline")
    if command.exists():
        argv = command.read_bytes().split(bytes([0]))
        assert protected_config.encode() in argv and argv[argv.index(b"--gpu_ids") + 1] == b"1"
        return "original_Y44_GPU1_alive"
    import torch
    last = root.parent / "AdaRES_checkpoints/adares_y44_coordinate_feature_res/20261013_072931/last.pth"
    checkpoint = torch.load(str(last), map_location="cpu")
    assert int(checkpoint["meta"]["epoch"]) == 30
    del checkpoint
    return "original_Y44_naturally_finished_saved30"


# Only the redundant hardcoded GPU1 lock is removed; outer runner owns GPU0.
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
    ("y43", "adares_y43_point_asymmetry_res", best["path"],
     terminal["source_hashes"]["configs_song/model_paper5/adares_y43_point_asymmetry_res.yaml"],
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
result = dict(complete=False, id="AdaRES-Y43", rows={}, summaries={},
              evaluator_sha256=evaluator_sha, physical_gpu=0,
              terminal_sha256=hashlib.sha256(terminal_path.read_bytes()).hexdigest(),
              safety_adapter_sha256=hashlib.sha256(adapter.encode()).hexdigest(),
              preserved_before=preserved(), protected_Y44_source_hashes=protected)
for name, config_name, checkpoint, config_sha, checkpoint_sha in bindings:
    config = "configs_song/model_paper5/" + config_name + ".yaml"
    assert hashlib.sha256((root / config).read_bytes()).hexdigest() == config_sha
    output = Path(P["output"]).with_name(Path(P["output"]).stem + "_" + name + ".json")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    subprocess.run([sys.executable, "-S", "-c", adapter, str(evaluator),
                    "--config", config, "--checkpoint", checkpoint,
                    "--output", str(output), "--gpu-id", "0"], check=True, env=environment)
    capture = json.loads(output.read_text())
    assert capture["complete"] and capture["completed_passes"] == 2
    assert capture["checkpoint_sha256"] == checkpoint_sha
    assert all(v["summary"]["expressions"] == 9508 and v["summary"]["scenes"] == 141
               for v in capture["passes"])
    assert capture["active_queries"] == [128] and capture["selected_queries"] == 1
    result["rows"][name] = capture
    result["summaries"][name] = capture["passes"][0]["summary"]
    result["preserved_after_" + name] = preserved()
    Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
result["complete"] = True
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
