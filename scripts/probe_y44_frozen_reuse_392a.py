"""Frozen Y43 or Y44/W2/B1 twice EACH; only outer GPU1 lease adaptation."""
assert 'y44_frozen' in P['output']
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
root = Path(P['root'])
assert P['physical_gpu'] == 1
name = 'y43' if 'y43_frozen' in P['output'] else 'y44'
assert name + '_frozen' in P['output']
terminal_name = 'y43_terminal_7a40' if name == 'y43' else 'y44_terminal_392a'
terminal_path = Path('/tmp/adares_' + terminal_name + '_' + P['token'] + '.json')
terminal = json.loads(terminal_path.read_text())
assert terminal['complete'] and terminal['id'] == 'AdaRES-' + name.upper()
best = next((v for v in terminal['checkpoints'] if v['path'].endswith('/best.pth')))
evaluator = root / 'tools/evaluate_adares_frozen_full.py'
expected_evaluator = '7824e7807b44c6344b158570626d7ad90fb5fc44eb6852dbd475d34440b1dc13'
assert hashlib.sha256(evaluator.read_bytes()).hexdigest() == expected_evaluator

def preserved():
    for source, sha in terminal['source_hashes'].items():
        assert hashlib.sha256((root / source).read_bytes()).hexdigest() == sha
    for row in terminal['checkpoints']:
        path = Path(row['path'])
        assert path.stat().st_size == row['bytes']
    for config in ('adares_y43_point_asymmetry_res', 'adares_y44_coordinate_feature_res'):
        encoded = ('configs_song/model_paper5/' + config + '.yaml').encode()
        for item in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                argv = item.read_bytes().split(b'\x00')
            except OSError:
                continue
            assert not (b'tools/train.py' in argv and encoded in argv)
    return 'original_Y43_Y44_training_finished_no_restart'
adapter = '\nimport ast,sys\nfrom pathlib import Path\npath=Path(sys.argv[1])\ntree=ast.parse(path.read_text(encoding="utf-8"))\nmain=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="main")\nremoved=[]\nbody=[]\nfor node in main.body:\n    assignment=(isinstance(node,ast.Assign) and len(node.targets)==1\n        and isinstance(node.targets[0],ast.Name) and node.targets[0].id=="lease"\n        and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name)\n        and node.value.func.id=="open" and isinstance(node.value.args[0],ast.Constant)\n        and node.value.args[0].value=="/bak_home/songcl/Project/AdaRES_project/.y15_gpu1_diagnostic.lock")\n    flock=(isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)\n        and isinstance(node.value.func,ast.Attribute)\n        and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=="fcntl"\n        and node.value.func.attr=="flock"\n        and isinstance(node.value.args[0],ast.Name) and node.value.args[0].id=="lease")\n    if assignment or flock:removed.append("assignment" if assignment else "flock")\n    else:body.append(node)\nassert removed==["assignment","flock"]\nmain.body=body\nsys.argv=[str(path)]+sys.argv[2:]\nexec(compile(tree,str(path),"exec"),{"__name__":"__main__","__file__":str(path)})\n'
candidate_config = 'adares_y43_point_asymmetry_res' if name == 'y43' else 'adares_y44_coordinate_feature_res'
bindings = [(name, candidate_config, best['path'], terminal['source_hashes']['configs_song/model_paper5/' + candidate_config + '.yaml'], best['sha256']), ('w2', 'adares_w2_pointmass_competition_res', '/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth', 'd4c35ff10712bfb9ece3e25daa9c3b63143ce313f43eaf19735abaa31f6693eb', 'e3407283384b4899524d055ac146c7bd6cf39cedb10e58a44ad74eb522009f52'), ('b1', 'adares_b1_standard_res', '/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_b1_standard_res/20260918_010854/best.pth', '04e3cc580751c72a6fc908b2ff93ad931ab4a3a7d07d1fc9b7ae30a9a993e1dc', 'cfc89ec0dc64f240913968f307d19c623f5ff0dc51c8e61de64e35abea05dd8b')]
result = dict(complete=False, id=terminal['id'], rows={}, summaries={}, evaluator_sha256=expected_evaluator, physical_gpu=1, terminal_sha256=hashlib.sha256(terminal_path.read_bytes()).hexdigest(), safety_adapter_sha256=hashlib.sha256(adapter.encode()).hexdigest(), safety_adapter='Remove ONLY redundant hardcoded inner lock; outer runner owns GPU1; scientific AST unchanged.', preserved_before=preserved())
for identity, config_name, checkpoint, config_sha, checkpoint_sha in bindings:
    config = 'configs_song/model_paper5/' + config_name + '.yaml'
    assert hashlib.sha256((root / config).read_bytes()).hexdigest() == config_sha
    if identity in ('w2', 'b1'):
        reuse_path = Path('/tmp/adares_y43_frozen_392a_' + P['token'] + '.json')
        reuse = json.loads(reuse_path.read_text())
        assert reuse['complete'] and reuse['physical_gpu'] == 1
        assert reuse['evaluator_sha256'] == expected_evaluator
        capture = reuse['rows'][identity]
        assert capture['config_sha256'] == config_sha
        assert capture['checkpoint_sha256'] == checkpoint_sha
        assert capture['complete'] and capture['completed_passes'] == 2
        assert all((p['summary']['expressions'] == 9508 and p['summary']['scenes'] == 141 for p in capture['passes']))
        result['rows'][identity] = capture
        result['summaries'][identity] = capture['passes'][0]['summary']
        result.setdefault('reused_comparators', {})[identity] = dict(path=str(reuse_path), sha256=hashlib.sha256(reuse_path.read_bytes()).hexdigest(), newly_executed_passes=0)
        result['preserved_after_' + identity] = preserved()
        Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
        continue
    output = Path(P['output']).with_name(Path(P['output']).stem + '_' + identity + '.json')
    environment = os.environ.copy()
    environment['PYTHONPATH'] = os.pathsep.join(sys.path)
    subprocess.run([sys.executable, '-S', '-c', adapter, str(evaluator), '--config', config, '--checkpoint', checkpoint, '--output', str(output), '--gpu-id', '1'], check=True, env=environment)
    capture = json.loads(output.read_text())
    assert capture['complete'] and capture['completed_passes'] == 2
    assert capture['checkpoint_sha256'] == checkpoint_sha
    assert all((v['summary']['expressions'] == 9508 and v['summary']['scenes'] == 141 for v in capture['passes']))
    assert capture['active_queries'] == [128] and capture['selected_queries'] == 1
    result['rows'][identity] = capture
    result['summaries'][identity] = capture['passes'][0]['summary']
    result['preserved_after_' + identity] = preserved()
    Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
result['complete'] = True
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
