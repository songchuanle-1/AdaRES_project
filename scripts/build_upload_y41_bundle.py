"""Create and upload only a physically independent additive Y41 package."""

from datetime import datetime, timezone
import ast
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
TOKEN = "5e172bd47ee3450dbee928bdecaede7f"
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == TOKEN and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert (datetime.fromisoformat(owner["deadline_at"]) - datetime.now(timezone.utc)).total_seconds() > 240
sys.path.insert(0, "C:/Users/chuanle song/.codex/skills/songcl_research_pipeline")
import pipeline_monitor as monitor

work = json.loads((ROOT / "pipeline/research_work.json").read_text())
assert next(c for c in work["candidates"] if c["id"] == "AdaRES-Y41")["stage"] == "diagnosis"
cpu = json.loads((ROOT / ("pipeline/analyses/y41_boundary_cpu_nonempty_" + TOKEN + ".json")).read_text())
assert cpu["complete"] and cpu["cpu_cases"] == 32 and cpu["descriptor_min"] > 0
parent = ROOT / "AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition"
folder = parent.parent / "adares_y41_boundary_probability"
folder.mkdir(exist_ok=True)
for path in parent.glob("*.py"):
    if path.name == "dec.py":
        continue
    target = folder / path.name
    data = path.read_text(encoding="utf-8").encode("utf-8")
    if target.exists():
        assert target.read_bytes() == data
    else:
        target.write_bytes(data)
source = (parent / "dec.py").read_text(encoding="utf-8")
readout = (ROOT / "scripts/y41_boundary_readout.py").read_text(encoding="utf-8")
node = next(n for n in ast.parse(readout).body if isinstance(n, ast.ClassDef))
assert source.count("class DEC(nn.Module):") == 1
source = source.replace("class DEC(nn.Module):",
                        ast.get_source_segment(readout, node) + "\n\n\nclass DEC(nn.Module):")
anchor = "        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)"
assert source.count(anchor) == 1
source = source.replace(anchor, anchor + "\n        with torch.random.fork_rng(devices=[]):\n"
                        "            self.boundary_readout = MaskBoundaryProbabilityReadout(d_model)")
anchor = "        return {\n            'masks': pred_masks,"
assert source.count(anchor) == 1
source = source.replace(anchor, "        pred_indis = pred_indis + self.boundary_readout(\n"
                        "            query, pred_masks, batch_mask, point_mass, inst_feats, sp_pos)\n\n" + anchor)
decoder = folder / "dec.py"
if decoder.exists():
    assert decoder.read_bytes() == source.encode("utf-8")
else:
    decoder.write_text(source, encoding="utf-8", newline="\n")
config = ROOT / "AdaRES/configs_song/model_paper5/adares_y41_boundary_probability_res.yaml"
config_text = (config.parent / "adares_w2_pointmass_competition_res.yaml").read_text(encoding="utf-8")
if config.exists():
    assert config.read_bytes() == config_text.encode("utf-8")
else:
    config.write_text(config_text, encoding="utf-8", newline="\n")
files = sorted(folder.glob("*.py")) + [config]
assert len(files) == 7
for path in files:
    if path.suffix != ".py":
        continue
    tree = ast.parse(path.read_text(encoding="utf-8"))
    compile(tree, str(path), "exec")
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert "model_paper5" not in (node.module or "")
        if isinstance(node, ast.Import):
            assert not any("model_paper5" in a.name for a in node.names)
records = [{"path": p.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
buffer = io.BytesIO()
with tarfile.open(fileobj=buffer, mode="w") as archive:
    for path in files:
        archive.add(path, arcname=path.relative_to(ROOT).as_posix())
server = monitor.load_servers(str(ROOT))["server-2"]
assert server["host"] == "192.168.1.163" and server["allowed_gpus"] == [0, 1]
program = r'''
import ast,hashlib,io,json,pathlib,subprocess,sys,tarfile
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
root=pathlib.Path('/bak_home/songcl/Project/AdaRES_project')
cmd=pathlib.Path('/proc/102267/cmdline').read_bytes().split(b'\0')
assert b'configs_song/model_paper5/adares_y39_point_distribution_res.yaml' in cmd
assert cmd[cmd.index(b'--gpu_ids')+1]==b'1'
old={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
     for p in (root/'AdaRES/ipdn/model_paper5/adares_y39_point_distribution').glob('*.py')}
archive=tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:')
members=archive.getmembers()
assert sorted(m.name for m in members)==sorted(f['path'] for f in P['files'])
for m in members:
    assert m.isfile()
    assert m.name.startswith('AdaRES/ipdn/model_paper5/adares_y41_boundary_probability/') or m.name=='AdaRES/configs_song/model_paper5/adares_y41_boundary_probability_res.yaml'
    path=(root/m.name).resolve()
    assert root.resolve() in path.parents and not path.exists()
    data=archive.extractfile(m).read()
    assert hashlib.sha256(data).hexdigest()==next(f['sha256'] for f in P['files'] if f['path']==m.name)
    if m.name.endswith('.py'):
        tree=ast.parse(data.decode('utf-8'))
        parent=ast.parse((root/'AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition'/path.name).read_text())
        if m.name.endswith('/dec.py'):
            left={n.name:ast.dump(n) for n in parent.body if isinstance(n,ast.ClassDef)}
            right={n.name:ast.dump(n) for n in tree.body if isinstance(n,ast.ClassDef)}
            assert all(left[k]==right[k] for k in left if k!='DEC')
        else:
            assert ast.dump(tree)==ast.dump(parent)
for m in members:
    path=root/m.name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(archive.extractfile(m).read())
assert pathlib.Path('/proc/102267/cmdline').read_bytes().split(b'\0')==cmd
assert all(hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h for p,h in old.items())
print(json.dumps([dict(path=m.name,sha256=hashlib.sha256((root/m.name).read_bytes()).hexdigest()) for m in members]))
'''
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    command = server["python"] + " -c " + shlex.quote("P=" + repr({"files": records}) + "\n" + program)
    response = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
         "songcl@192.168.1.163", command],
        input=buffer.getvalue(), capture_output=True, timeout=60)
    assert response.returncode == 0, response.stderr.decode(errors="replace")
    assert json.loads(response.stdout) == records
    output = ROOT / ("pipeline/analyses/y41_source_bundle_" + TOKEN + ".json")
    assert not output.exists()
    output.write_text(json.dumps({
        "complete": True, "id": "AdaRES-Y41", "files": records,
        "physical_independence": True, "remote_byte_identity": True,
        "parent_ast_identity_except_final_branch": True, "preserved_y39": True,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"path": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}))
finally:
    lock.close()
