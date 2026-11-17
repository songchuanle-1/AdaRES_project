"""Build a scoped additive test bundle; never touch another project or training."""

import ast
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "d20c71e418354a6b818d9f37ab72fc7b"
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == TOKEN
folder = ROOT / "AdaRES/ipdn/model_paper5/adares_y23_valid_token_support"
parent = ROOT / "AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition"
files = sorted(folder.glob("*.py"))
for path in files:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    compile(tree, str(path), "exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            assert not any("model_paper5" in v for v in names), "Cross-variant import"
for name in ("ipdn.py", "dec.py", "backbone.py", "sample_model.py"):
    assert (folder/name).read_bytes() == (parent/name).read_bytes()
config = ROOT / "AdaRES/configs_song/model_paper5/adares_y23_valid_token_support_res.yaml"
assert config.read_bytes() == (ROOT / "AdaRES/configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml").read_bytes()
files.append(config)
buf = io.BytesIO()
records = []
with tarfile.open(fileobj=buf, mode="w") as archive:
    for path in files:
        name = path.relative_to(ROOT).as_posix()
        records.append(dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        archive.add(path, arcname=name)
program = r'''
import hashlib,io,json,pathlib,sys,tarfile
root=pathlib.Path('/bak_home/songcl/Project/AdaRES_project')
archive=tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:')
members=archive.getmembers()
for m in members:
    assert m.isfile()
    assert m.name.startswith('AdaRES/ipdn/model_paper5/adares_y23_valid_token_support/') or m.name=='AdaRES/configs_song/model_paper5/adares_y23_valid_token_support_res.yaml'
    path=(root/m.name).resolve()
    assert root.resolve() in path.parents
    assert not path.exists(), 'Refuse existing remote candidate '+str(path)
for m in members:
    path=root/m.name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(archive.extractfile(m).read())
print(json.dumps([dict(path=m.name,sha256=hashlib.sha256((root/m.name).read_bytes()).hexdigest()) for m in members]))
'''
command = "/home/songcl/.conda/envs/adares_project/bin/python -c '" + program.replace("'", "'\"'\"'") + "'"
r = subprocess.run(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=15","songcl@192.168.1.163",command],
    input=buf.getvalue(),capture_output=True,timeout=60)
if r.returncode:
    raise RuntimeError(r.stderr.decode(errors="replace"))
remote=json.loads(r.stdout)
assert remote == records
out = ROOT / "pipeline/analyses/y23_source_bundle_d20c71e4.json"
assert not out.exists()
out.write_text(json.dumps(dict(files=records, remote_byte_identity=True,
    physical_independence=True, config_exact_parent=True,
    inference_modules_unchanged=True, parameters_added=0,
    scope="Additive test bundle only, not deployment or admission"),indent=2))
print(json.dumps(dict(files=len(records),remote_byte_identity=True)))
