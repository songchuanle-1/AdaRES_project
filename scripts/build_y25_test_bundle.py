"""Install an additive, hash-bound Y25 diagnostic bundle only."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor

parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds() > 100
server = monitor.load_servers(str(ROOT))["server-2"]
assert server["allowed_gpus"] == [1] and server["host"] == "192.168.1.163"
parent = ROOT / "AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition"
folder = parent.parent / "adares_y25_mask_feedback"
files = sorted(folder.glob("*.py"))
assert len(files) == 6
for path in files:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    compile(tree, str(path), "exec")
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert "model_paper5" not in (node.module or ""), "Cross-variant import"
        elif isinstance(node, ast.Import):
            assert not any("model_paper5" in v.name for v in node.names)
    if path.name != "dec.py":
        assert path.read_bytes() == (parent / path.name).read_bytes()
old_dec = ast.parse((parent / "dec.py").read_text(encoding="utf-8"))
new_dec = ast.parse((folder / "dec.py").read_text(encoding="utf-8"))
old_method = next(n for n in ast.walk(old_dec) if isinstance(n, ast.FunctionDef) and n.name == "get_mask")
new_method = next(n for n in ast.walk(new_dec) if isinstance(n, ast.FunctionDef) and n.name == "get_mask")
assert len(new_method.body[1].body) == len(old_method.body[1].body) + 1
assert isinstance(new_method.body[1].body[0], ast.If)
new_method.body[1].body.pop(0)
assert ast.dump(old_dec, include_attributes=False) == ast.dump(new_dec, include_attributes=False)
config = ROOT / "AdaRES/configs_song/model_paper5/adares_y25_mask_feedback_res.yaml"
assert config.read_bytes() == (config.parent / "adares_w2_pointmass_competition_res.yaml").read_bytes()
assert len(list((ROOT / "AdaRES").glob("**/" + config.name))) == 1
files.append(config)
records = [dict(path=p.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
           for p in files]
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w") as archive:
    for path in files:
        archive.add(path, arcname=path.relative_to(ROOT).as_posix())
program = """
import ast,hashlib,io,json,pathlib,subprocess,sys,tarfile
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
root=pathlib.Path('/bak_home/songcl/Project/AdaRES_project')
assert (root/'AdaRES/.codex_source_commit').read_text().strip()=='4a24410344b6ab6fe7f56d597013a4c13a9fa317'
archive=tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:')
members=archive.getmembers()
for m in members:
    assert m.isfile()
    assert m.name.startswith('AdaRES/ipdn/model_paper5/adares_y25_mask_feedback/') or m.name=='AdaRES/configs_song/model_paper5/adares_y25_mask_feedback_res.yaml'
    path=(root/m.name).resolve()
    assert root.resolve() in path.parents and not path.exists()
    contents=archive.extractfile(m).read().decode('utf-8')
    if m.name.endswith('.py') and not m.name.endswith('/dec.py'):
        parent=root/'AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition'/path.name
        assert ast.dump(ast.parse(contents))==ast.dump(ast.parse(parent.read_text()))
for m in members:
    path=root/m.name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(archive.extractfile(m).read())
print(json.dumps([dict(path=m.name,sha256=hashlib.sha256((root/m.name).read_bytes()).hexdigest()) for m in members]))
"""
command = server["python"] + " -c '" + program.replace("'", "'\"'\"'") + "'"
output = ROOT / ("pipeline/analyses/y25_source_bundle_" + args.token + ".json")
assert not output.exists()
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None, "Writer busy"
try:
    response = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                               "songcl@192.168.1.163", command],
                              input=buf.getvalue(), capture_output=True, timeout=60)
    if response.returncode:
        raise RuntimeError(response.stderr.decode(errors="replace"))
    assert json.loads(response.stdout) == records
    output.write_text(json.dumps(dict(complete=True, files=records, remote_byte_identity=True,
                                      physical_independence=True, config_exact_parent=True,
                                      only_training_get_mask_change=True,
                                      scope="Additive diagnostic bundle, not admission or production deployment"),
                                 indent=2))
    print(json.dumps(dict(path=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest())))
finally:
    lock.close()
