"""Read source fingerprints and the original Y50 final checkpoint metadata."""
import ast
import gc
import hashlib
import json
from pathlib import Path
import re
import subprocess

assert P['physical_gpu'] is None
root = Path(P['root'])
source_rows = []
parent = 'ipdn/model_paper5/adares_w2_pointmass_competition/'
paths = [parent + name for name in ('__init__.py', 'backbone.py', 'dec.py',
                                   'ipdn.py', 'loss.py', 'sample_model.py')]
paths.append('tools/train.py')
for relative in paths:
    data = (root / relative).read_bytes()
    normalized = data.replace(b'\r\n', b'\n')
    source_rows.append(dict(path=relative, raw_sha256=hashlib.sha256(data).hexdigest(),
        normalized_lf_sha256=hashlib.sha256(normalized).hexdigest(),
        ast_sha256=hashlib.sha256(ast.dump(ast.parse(data.decode('utf-8-sig')),
            include_attributes=False).encode()).hexdigest(), crlf_count=data.count(b'\r\n')))

log_path = root / 'logs/train_adares_y50_joint_incidence_res_20261016_145628.log'
with log_path.open('rb') as stream:
    stream.seek(0, 2)
    stream.seek(max(0, stream.tell() - 262144))
    tail = stream.read().decode('utf-8', errors='replace')
progress = re.findall(r'Train\[(\d+)/(\d+)\]:[^\r\n]*?(\d+)/(\d+)\s*\[', tail)
last_progress = list(map(int, progress[-1])) if progress else None
last_paths = re.findall(r'save last pth to ([^\r\n]+)', tail)
assert last_paths, 'No last checkpoint save marker in bounded original log'
last_path = Path(last_paths[-1].strip()).resolve()
checkpoint_root = (root.parent / 'AdaRES_checkpoints').resolve()
assert checkpoint_root in last_path.parents and last_path.name == 'last.pth'
assert last_path.parent.parent.name == 'adares_y50_joint_incidence_res'
process = subprocess.run(['ps', '-ww', '-p', '31539', '-o', 'pid=,args='],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
command = process.stdout.decode().strip()
same_process = process.returncode == 0 and 'tools/train.py configs_song/model_paper5/adares_y50_joint_incidence_res.yaml' in command
assert not same_process, 'Y50 still running; do not read potentially changing terminal checkpoints'

import torch
torch.set_num_threads(2)
checkpoints = []
for path in (last_path, last_path.with_name('best.pth')):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    checkpoint = torch.load(str(path), map_location='cpu')
    meta = checkpoint.get('meta', {})
    values = {}
    for key in ('epoch', 'best_miou'):
        value = meta.get(key, checkpoint.get(key))
        if hasattr(value, 'item'):
            value = value.item()
        assert value is None or isinstance(value, (int, float, str, bool))
        values[key] = value
    after = path.stat()
    assert before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
    checkpoints.append(dict(path=str(path), bytes=after.st_size, sha256=digest.hexdigest(),
                            metadata=values, unchanged_during_read=True))
    del checkpoint
    gc.collect()

crash = any(marker in tail for marker in ('Traceback (most recent call last)', 'CUDA out of memory'))
result = dict(complete=True, project='AdaRES_project', read_only=True, no_GPU=True,
    source_rows=source_rows, original_log=str(log_path), bounded_log_bytes=262144,
    Y50=dict(same_process_present=same_process, process_returncode=process.returncode,
        last_train_progress=last_progress, last_checkpoint_save_marker=str(last_path),
        crash_marker_in_bounded_tail=crash, checkpoints=checkpoints,
        fixed30_normal_exit_evidence=(last_progress == [30, 30, 4686, 4686]
            and checkpoints[0]['metadata']['epoch'] == 30 and not same_process and not crash)),
    no_checkpoint_or_model_writes=True, no_registry_or_queue_change=True,
    limitation='Source identity and terminal metadata only. Not a matched trained comparison, scientific gate PASS, or permission to bypass canonical deathwatch. Use client observation window, not the server clock.')
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
