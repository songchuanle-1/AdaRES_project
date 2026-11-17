"""Run one unchanged two-pass evaluator binding under the outer GPU lease."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

assert P['physical_gpu'] == 1
root = Path(P['root'])
terminal_path = Path('/tmp/adares_y52_source_y50_exit_89ff_' + P['token'] + '.json')
assert hashlib.sha256(terminal_path.read_bytes()).hexdigest() == PAYLOAD['terminal_sha256']
terminal = json.loads(terminal_path.read_text())
assert terminal['complete'] and terminal['Y50']['fixed30_normal_exit_evidence']
assert (root / '.codex_source_commit').read_text().strip() == PAYLOAD['code_commit']
for row in PAYLOAD['source_files']:
    assert hashlib.sha256((root / row['path']).read_bytes()).hexdigest() == row['sha256']
evaluator = root / 'tools/evaluate_adares_frozen_full.py'
assert hashlib.sha256(evaluator.read_bytes()).hexdigest() == PAYLOAD['evaluator_sha256']
assert hashlib.sha256((root / PAYLOAD['config']).read_bytes()).hexdigest() == PAYLOAD['config_sha256']
for row in terminal['Y50']['checkpoints']:
    assert Path(row['path']).stat().st_size == row['bytes']
for command_path in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        argv = command_path.read_bytes().split(b'\x00')
    except OSError:
        continue
    assert not (b'tools/train.py' in argv and b'configs_song/model_paper5/adares_y50_joint_incidence_res.yaml' in argv), 'Y50 train process present'

raw_output = Path(P['output']).with_name(Path(P['output']).stem + '_capture.json')
assert not raw_output.exists()
environment = os.environ.copy()
environment['PYTHONPATH'] = os.pathsep.join(sys.path)
subprocess.run([sys.executable, '-S', '-c', PAYLOAD['adapter'], str(evaluator),
    '--config', PAYLOAD['config'], '--checkpoint', PAYLOAD['checkpoint'],
    '--output', str(raw_output), '--gpu-id', '1'], check=True, env=environment, timeout=780)
capture = json.loads(raw_output.read_text())
assert capture['complete'] and capture['completed_passes'] == 2
assert capture['config_sha256'] == PAYLOAD['config_sha256']
assert capture['checkpoint_sha256'] == PAYLOAD['checkpoint_sha256']
assert capture['active_queries'] == [128] and capture['selected_queries'] == 1
assert all(p['summary']['expressions'] == 9508 and p['summary']['scenes'] == 141 for p in capture['passes'])
for row in PAYLOAD['source_files']:
    assert hashlib.sha256((root / row['path']).read_bytes()).hexdigest() == row['sha256']
result = dict(complete=True, id='AdaRES-Y50', identity=PAYLOAD['identity'],
    evaluator_sha256=PAYLOAD['evaluator_sha256'], terminal_sha256=PAYLOAD['terminal_sha256'],
    newly_executed_full_passes=2, first_primary_summary=capture['passes'][0]['summary'],
    repeat_summary=capture['passes'][1]['summary'], repeat_miou_difference=capture['repeat_miou_difference'],
    repeat_max_iou_difference=capture['repeat_max_iou_difference'],
    parameters=capture['parameters'], warm_first_chunk_seconds=capture['warm_first_chunk_seconds'],
    peak_allocated_gib=capture['peak_allocated_gib'], peak_reserved_gib=capture['peak_reserved_gib'],
    active_queries=capture['active_queries'], selected_queries=1,
    elapsed_seconds=capture['elapsed_seconds'], rows=capture,
    no_model_checkpoint_or_training_writes=True, no_queue_or_registry_changes=True,
    limitation='One of the three frozen comparison bindings, FIRST primary and repeat retained. Not a complete comparative conclusion until Y50, W2 and B1 all finish both full passes. Outer authorized runner owns the exclusive GPU1 lease; only redundant inner flock statements are removed, evaluator science unchanged.')
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
