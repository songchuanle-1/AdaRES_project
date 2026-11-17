"""Upload only the seven new Y42 files after structural parent checks."""
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
sys.path.insert(0, 'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import pipeline_monitor as monitor
parser = argparse.ArgumentParser()
parser.add_argument('--token', required=True)
args = parser.parse_args()
owner = json.loads((ROOT / 'pipeline/headless/.foreground.lockdir/owner.json').read_text())
assert owner['token'] == args.token and owner['thread_id'] == os.environ['CODEX_THREAD_ID']
assert (datetime.fromisoformat(owner['deadline_at']) - datetime.now(timezone.utc)).total_seconds() > 100
server = monitor.load_servers(str(ROOT))['server-2']
assert server['allowed_gpus'] == [0, 1] and server['host'] == '192.168.1.163'
parent = ROOT / 'AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition'
folder = parent.parent / 'adares_y43_point_asymmetry'
files = sorted(folder.glob('*.py'))
assert len(files) == 6
for path in files:
    tree = ast.parse(path.read_text(encoding='utf-8'))
    compile(tree, str(path), 'exec')
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert 'model_paper5' not in (node.module or '')
        elif isinstance(node, ast.Import):
            assert not any(('model_paper5' in alias.name for alias in node.names))
    if path.name not in ('dec.py', 'ipdn.py'):
        assert path.read_bytes().replace(b'\r\n', b'\n') == (parent / path.name).read_bytes().replace(b'\r\n', b'\n')
for name, changed in (('dec.py', {'DEC'}), ('ipdn.py', {'IPDN'})):
    old = ast.parse((parent / name).read_text(encoding='utf-8'))
    new = ast.parse((folder / name).read_text(encoding='utf-8'))
    classes = {node.name: node for node in new.body if isinstance(node, ast.ClassDef)}
    for node in old.body:
        if isinstance(node, ast.ClassDef) and node.name not in changed:
            assert ast.dump(node) == ast.dump(classes[node.name])
    if name == 'ipdn.py':
        old_class = next((node for node in old.body if isinstance(node, ast.ClassDef) and node.name == 'IPDN'))
        methods = {node.name: node for node in classes['IPDN'].body if isinstance(node, ast.FunctionDef)}
        for node in old_class.body:
            if isinstance(node, ast.FunctionDef) and node.name not in {'extract_feat', 'loss', 'predict'}:
                assert ast.dump(node) == ast.dump(methods[node.name])
assert 'MaskPointAsymmetryReadout' in {node.name for node in ast.parse((folder / 'dec.py').read_text(encoding='utf-8')).body if isinstance(node, ast.ClassDef)}
config = ROOT / 'AdaRES/configs_song/model_paper5/adares_y43_point_asymmetry_res.yaml'
assert config.read_bytes().replace(b'\r\n', b'\n') == (config.parent / 'adares_w2_pointmass_competition_res.yaml').read_bytes().replace(b'\r\n', b'\n')
files.append(config)
records = [dict(path=path.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in files]
archive_bytes = io.BytesIO()
with tarfile.open(fileobj=archive_bytes, mode='w') as archive:
    for path in files:
        archive.add(path, arcname=path.relative_to(ROOT).as_posix())
payload = dict(files=records, marker='243e0b6827073831e0d68090857755d575efbf16')
program = "\nimport ast,hashlib,io,json,pathlib,subprocess,sys,tarfile\nassert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'\nroot=pathlib.Path('/bak_home/songcl/Project/AdaRES_project')\nassert (root/'AdaRES/.codex_source_commit').read_text().strip()==P['marker']\nrunning=root/'AdaRES/ipdn/model_paper5/adares_y41_boundary_probability'\nbefore={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in running.glob('*.py')}\nassert len(before)==6\narchive=tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:')\nmembers=archive.getmembers()\nassert sorted(m.name for m in members)==sorted(f['path'] for f in P['files'])\nfor m in members:\n    assert m.isfile()\n    assert m.name.startswith('AdaRES/ipdn/model_paper5/adares_y43_point_asymmetry/') or m.name=='AdaRES/configs_song/model_paper5/adares_y43_point_asymmetry_res.yaml'\n    path=(root/m.name).resolve()\n    assert root.resolve() in path.parents and not path.exists()\n    data=archive.extractfile(m).read()\n    assert hashlib.sha256(data).hexdigest()==next(f['sha256'] for f in P['files'] if f['path']==m.name)\n    if m.name.endswith('.py') and not m.name.endswith(('/dec.py','/ipdn.py')):\n        parent=root/'AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition'/path.name\n        assert ast.dump(ast.parse(data.decode('utf-8')))==ast.dump(ast.parse(parent.read_text()))\nfor m in members:\n    path=root/m.name\n    path.parent.mkdir(parents=True,exist_ok=True)\n    path.write_bytes(archive.extractfile(m).read())\nafter={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in running.glob('*.py')}\nassert before==after and (root/'AdaRES/.codex_source_commit').read_text().strip()==P['marker']\nprint(json.dumps(dict(files=[dict(path=m.name,sha256=hashlib.sha256((root/m.name).read_bytes()).hexdigest()) for m in members],y41_before=before,y41_after=after)))\n"
output = ROOT / ('pipeline/analyses/y43_source_bundle_' + args.token + '.json')
assert not output.exists()
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    command = server['python'] + ' -c ' + shlex.quote('P=' + repr(payload) + '\n' + program)
    response = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'songcl@192.168.1.163', command], input=archive_bytes.getvalue(), capture_output=True, timeout=60)
    assert response.returncode == 0, response.stderr.decode(errors='replace')
    result = json.loads(response.stdout)
    assert result['files'] == records
    result.update(complete=True, id='AdaRES-Y43', remote_byte_identity=True, physical_independence=True, config_exact_parent_normalized=True, source_marker=payload['marker'], scope='Additive diagnostic files only; no admission or launch.')
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(path=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest())))
finally:
    lock.close()
