"""Deploy only new Y31 diagnostic files with a verified foreground lease."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
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
assert (datetime.fromisoformat(owner["deadline_at"])-datetime.now(timezone.utc)).total_seconds() > 100
server = monitor.load_servers(str(ROOT))["server-2"]
assert server["allowed_gpus"] == [1] and server["host"] == "192.168.1.163"
parent = ROOT / "AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition"
folder = parent.parent / "adares_y31_mask_shape"
files = sorted(folder.glob("*.py"))
assert len(files) == 6
for path in files:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    compile(tree, str(path), "exec")
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert "model_paper5" not in (node.module or "")
        elif isinstance(node, ast.Import):
            assert not any("model_paper5" in v.name for v in node.names)
    if path.name != "dec.py":
        assert path.read_bytes() == (parent / path.name).read_bytes()
old = ast.parse((parent / "dec.py").read_text(encoding="utf-8"))
new = ast.parse((folder / "dec.py").read_text(encoding="utf-8"))
old_classes = {n.name: ast.dump(n) for n in old.body if isinstance(n, ast.ClassDef)}
new_classes = {n.name: n for n in new.body if isinstance(n, ast.ClassDef)}
for name in old_classes.keys() - {"DEC"}:
    assert old_classes[name] == ast.dump(new_classes[name])
assert "MaskShapeReadout" in new_classes
config = ROOT / "AdaRES/configs_song/model_paper5/adares_y31_mask_shape_res.yaml"
assert config.read_bytes() == (config.parent / "adares_w2_pointmass_competition_res.yaml").read_bytes()
assert len(list((ROOT / "AdaRES").glob("**/" + config.name))) == 1
files.append(config)
records = [dict(path=p.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
           for p in files]
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w") as archive:
    for path in files:
        archive.add(path, arcname=path.relative_to(ROOT).as_posix())
payload = dict(files=records, marker="4f5c31f57994b8f9fb640d652c913863dc310a95")
program = """
import ast,hashlib,io,json,pathlib,subprocess,sys,tarfile
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
root=pathlib.Path('/bak_home/songcl/Project/AdaRES_project')
assert (root/'AdaRES/.codex_source_commit').read_text().strip()==P['marker']
archive=tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:')
members=archive.getmembers()
assert sorted(m.name for m in members)==sorted(f['path'] for f in P['files'])
for m in members:
    assert m.isfile()
    assert m.name.startswith('AdaRES/ipdn/model_paper5/adares_y31_mask_shape/') or m.name=='AdaRES/configs_song/model_paper5/adares_y31_mask_shape_res.yaml'
    path=(root/m.name).resolve()
    assert root.resolve() in path.parents and not path.exists()
    data=archive.extractfile(m).read()
    assert hashlib.sha256(data).hexdigest()==next(f['sha256'] for f in P['files'] if f['path']==m.name)
    if m.name.endswith('.py') and not m.name.endswith('/dec.py'):
        parent=root/'AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition'/path.name
        assert ast.dump(ast.parse(data.decode('utf-8')))==ast.dump(ast.parse(parent.read_text()))
for m in members:
    path=root/m.name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(archive.extractfile(m).read())
print(json.dumps([dict(path=m.name,sha256=hashlib.sha256((root/m.name).read_bytes()).hexdigest()) for m in members]))
"""
command = server["python"] + " -c " + shlex.quote("P=" + repr(payload) + "\n" + program)
output = ROOT / ("pipeline/analyses/y31_source_bundle_" + args.token + ".json")
assert not output.exists()
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    response = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                               "songcl@192.168.1.163", command],
                              input=buf.getvalue(), capture_output=True, timeout=60)
    if response.returncode:
        raise RuntimeError(response.stderr.decode(errors="replace"))
    assert json.loads(response.stdout) == records
    output.write_text(json.dumps(dict(complete=True, files=records, remote_byte_identity=True,
                                      physical_independence=True, config_exact_parent=True,
                                      scope="Only additive Y31 diagnostics; no production admission"), indent=2),
                      encoding="utf-8")
    print(json.dumps(dict(path=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest())))
finally:
    lock.close()
