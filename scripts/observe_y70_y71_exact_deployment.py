"""Observe Y70/Y71 across only the exact prepared Y71 additive deployment."""
import argparse
import ast
from datetime import datetime
import hashlib
import inspect
import json
from pathlib import Path
import re
import subprocess
import yaml
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from observe_y70_server2_gpu1_phase import parse_phase, phase_key, progressed, REMOTE as ORIGINAL_REMOTE
from admit_dispatch_y71_gpu0 import validate_additive_delta, validate_keeper
import closure

OLD = '39bdb8d9783512e70cdc2763e9a6097ab0f67a0f'
NEW = '87019e7c2f640f99cb82a3e5b396321fc5b54d25'
NAMES = ('__init__.py', 'backbone.py', 'dec.py', 'ipdn.py', 'loss.py', 'sample_model.py')


def validate_task(task):
    assert task['status'] == 'running' and task['launch_verified'] is True
    if task['id'] == 'AdaRES-Y70':
        validate_keeper(task)
        assert type(task['pid']) is int and task['pid'] == 52561
        return 'adares_y70_query_scale_fusion', 1, 390.0, OLD
    assert task['id'] == 'AdaRES-Y71'
    assert str(task['gpu']) == '0' and type(task['pid']) is int and task['pid'] > 0
    assert task['code_commit'] == NEW
    assert task['config'] == 'configs_song/model_paper5/adares_y71_visual_product_fusion_res.yaml'
    return 'adares_y71_visual_product_fusion', 0, 280.0, NEW


def validate_marker(marker, candidate):
    assert candidate in ('AdaRES-Y70', 'AdaRES-Y71')
    assert marker in ((OLD, NEW) if candidate == 'AdaRES-Y70' else (NEW,))


REMOTE = ORIGINAL_REMOTE
old = "assert (root/'.codex_source_commit').read_text().strip()==task['code_commit']"
new = """marker=(root/'.codex_source_commit').read_text().strip()
validate_marker(marker,task['id'])
deployment=P['deployments'][marker]
for relative,digest in deployment['source_files'].items():
    assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==digest
for relative,expected in deployment['configs'].items():
    assert yaml.safe_load((root/relative).read_text())==expected"""
assert REMOTE.count(old) == 1
REMOTE = REMOTE.replace(old, new).replace("'nvidia-smi','-i','1'", "'nvidia-smi','-i',str(P['gpu'])")
REMOTE = REMOTE.replace('==390.0', "==P['power_limit_watts']")
REMOTE = REMOTE.replace("source_marker=task['code_commit']", 'source_marker=marker')
REMOTE = REMOTE.replace('compute_owners=owners', "additional_source_files_verified=len(deployment['source_files']),additional_YAMLs_verified=len(deployment['configs']),compute_owners=owners")


def observe(token, cid, label):
    server = monitor.load_servers(str(ROOT))['server-2']; closure.validate_adares_server(ROOT, server)
    assert git('rev-parse', 'HEAD').decode().strip() == NEW
    assert git('rev-parse', NEW + '^').decode().strip() == OLD
    rows = [r.split('\t') for r in git('diff', '--no-renames', '--name-status', OLD, NEW).decode().splitlines()]
    validate_additive_delta(rows)
    additions = dict(source_files={}, configs={})
    for status, path in rows:
        blob = git('show', NEW + ':' + path); assert (ROOT / path).read_bytes() == blob
        relative = path[len('AdaRES/'):]
        if path.endswith('.yaml'): additions['configs'][relative] = yaml.safe_load(blob)
        else: additions['source_files'][relative] = hashlib.sha256(blob).hexdigest()
    queue_before, running_before = ref('pipeline/queue.json'), ref('pipeline/running.json')
    task = next(t for t in monitor.load_queue(str(ROOT))['gpu_queues']['server-2'] if t['id'] == cid)
    package, gpu, power, commit = validate_task(task)
    assert monitor.load_running(str(ROOT))['server-2'][str(gpu)] == cid
    closure.admission(ROOT, task)
    source_files = {}
    for name in NAMES:
        relative = 'ipdn/model_paper5/' + package + '/' + name
        raw = (ROOT / 'AdaRES' / relative).read_bytes()
        assert raw == git('show', commit + ':AdaRES/' + relative)
        source_files[relative] = hashlib.sha256(raw).hexdigest()
    config = yaml.safe_load((ROOT / 'AdaRES' / task['config']).read_bytes())
    assert config == yaml.safe_load(git('show', commit + ':AdaRES/' + task['config']))
    deployments = {OLD: dict(source_files={}, configs={}), NEW: additions}
    if cid == 'AdaRES-Y71': deployments.pop(OLD)
    payload = dict(root=server['remote_code_root'], task=task, source_files=source_files,
                   expected_config=config, deployments=deployments, gpu=gpu, power_limit_watts=power)
    functions = '\n\n'.join(inspect.getsource(f) for f in (parse_phase, phase_key, progressed, validate_marker))
    program = 'OLD=' + repr(OLD) + '\nNEW=' + repr(NEW) + '\nP=' + repr(payload) + '\n' + functions + '\n' + REMOTE
    ast.parse(program, feature_version=(3, 8))
    output = ROOT / ('pipeline/checks/y71_deployment_' + cid.lower() + '_' + token + '_' + label + '.json')
    assert not output.exists() and not output.with_suffix('.log').exists()
    started = datetime.now().astimezone().isoformat()
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
        server['user'] + '@' + server['host'], 'PYTHONDONTWRITEBYTECODE=1 ' + server['python'] + ' -B -'],
        input=program.encode(), capture_output=True, timeout=45)
    output.with_suffix('.log').write_bytes(result.stdout + result.stderr)
    assert result.returncode == 0, 'Preserve observer failure; never restart or relax source checks'
    observation = json.loads(result.stdout)
    assert observation['native_progress_confirmed'] == progressed(observation['before'], observation['after'])
    observation.update(id=cid, token=token, observed_start_local=started,
        observed_end_local=datetime.now().astimezone().isoformat(), evidence=[ref(__file__), ref(output.with_suffix('.log'))])
    tick.save(output, observation)
    work = tick.load(ROOT / 'pipeline/research_work.json'); candidate = next(c for c in work['candidates'] if c['id'] == cid)
    assert candidate['stage'] == 'training'
    candidate['evidence'].extend([ref(output), ref(__file__)]); candidate['latest_training_observation'] = ref(output)
    tick.save(ROOT / 'pipeline/research_work.json', work)
    assert ref('pipeline/queue.json') == queue_before and ref('pipeline/running.json') == running_before
    print(json.dumps(dict(evidence=ref(output), before=observation['before'], after=observation['after'],
        native_progress_confirmed=observation['native_progress_confirmed'], source_marker=observation['source_marker'], gpu=observation['gpu'])))


def git(*args): return subprocess.check_output(['git', *args], cwd=ROOT, timeout=30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--token', required=True)
    parser.add_argument('--candidate', required=True, choices=('AdaRES-Y70', 'AdaRES-Y71'))
    parser.add_argument('--label', default='progress'); args = parser.parse_args()
    assert re.fullmatch('[a-z0-9_]{1,40}', args.label)
    lease(args.token, 150)
    lock = monitor.acquire_lock(str(ROOT)); assert lock is not None
    try: observe(args.token, args.candidate, args.label)
    finally: lock.close()
