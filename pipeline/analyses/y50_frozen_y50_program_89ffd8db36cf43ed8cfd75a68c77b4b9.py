PAYLOAD={'identity': 'y50', 'config': 'configs_song/model_paper5/adares_y50_joint_incidence_res.yaml', 'config_sha256': 'd4c35ff10712bfb9ece3e25daa9c3b63143ce313f43eaf19735abaa31f6693eb', 'checkpoint': '/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_y50_joint_incidence_res/20261016_022343/best.pth', 'checkpoint_sha256': 'c6506631625cef154da104b92f056bf31c8a268f7779f849d3fc13898d45452b', 'terminal_sha256': '3017a4a2abcecb29f2459de1f40574abeb8eecf07c52d88d45b420f43411106f', 'code_commit': '26ee1cec57c53ee2f94d6969d0bb637565b6ed59', 'source_files': [{'path': 'ipdn/model_paper5/adares_y50_joint_incidence/__init__.py', 'sha256': 'c540732d6e3334c85008035cc37b8537b0f85383d5f5a6bf663359fbf2b134bb'}, {'path': 'ipdn/model_paper5/adares_y50_joint_incidence/backbone.py', 'sha256': '57db8747d3c38136f568837d70e75ceea23e101900679db4931724ecb57665ca'}, {'path': 'ipdn/model_paper5/adares_y50_joint_incidence/dec.py', 'sha256': '1587458066bd8b33396ad927f8268146ed4d8ac4043f44adf534aeef8fd5ef37'}, {'path': 'ipdn/model_paper5/adares_y50_joint_incidence/ipdn.py', 'sha256': '7cb4b1c5490d0839fcdaf18553983b4c991f69e50474d07e83af9f0f8c937a5a'}, {'path': 'ipdn/model_paper5/adares_y50_joint_incidence/loss.py', 'sha256': '836ff790a0a8b9c55cf465a472c9462699f0753a7e87d0ef74a8946edf31565a'}, {'path': 'ipdn/model_paper5/adares_y50_joint_incidence/sample_model.py', 'sha256': '674e7ef698c6e81ed1c35f14680e1ec748f5c783ed879a6ca72f099b6450bc36'}, {'path': 'configs_song/model_paper5/adares_y50_joint_incidence_res.yaml', 'sha256': 'd4c35ff10712bfb9ece3e25daa9c3b63143ce313f43eaf19735abaa31f6693eb'}], 'evaluator_sha256': '7824e7807b44c6344b158570626d7ad90fb5fc44eb6852dbd475d34440b1dc13', 'adapter': '\nimport ast,sys\nfrom pathlib import Path\npath=Path(sys.argv[1])\ntree=ast.parse(path.read_text(encoding="utf-8"))\nmain=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="main")\nremoved=[]\nbody=[]\nfor node in main.body:\n    assignment=(isinstance(node,ast.Assign) and len(node.targets)==1\n        and isinstance(node.targets[0],ast.Name) and node.targets[0].id=="lease"\n        and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name)\n        and node.value.func.id=="open" and isinstance(node.value.args[0],ast.Constant)\n        and node.value.args[0].value=="/bak_home/songcl/Project/AdaRES_project/.y15_gpu1_diagnostic.lock")\n    flock=(isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)\n        and isinstance(node.value.func,ast.Attribute)\n        and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=="fcntl"\n        and node.value.func.attr=="flock"\n        and isinstance(node.value.args[0],ast.Name) and node.value.args[0].id=="lease")\n    if assignment or flock:removed.append("assignment" if assignment else "flock")\n    else:body.append(node)\nassert removed==["assignment","flock"]\nmain.body=body\nsys.argv=[str(path)]+sys.argv[2:]\nexec(compile(tree,str(path),"exec"),{"__name__":"__main__","__file__":str(path)})\n'}
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
