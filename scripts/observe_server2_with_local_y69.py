"""Observe the unchanged Y67 deployment while exact Y69 additions are local only.

This observer never deploys code or changes a running receipt.  Only the three
listed commits and their verified seven-file additions are accepted.
"""
import argparse
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

import yaml

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
import closure
import observe_y67_server2_gpu1_phase as phases


ORIGINAL = '882d61537fe7ffcb383c8f51f12fea36d1e53a6d'
Y67_COMMIT = 'db35f2870d252e6888e3525b142785938f00ad4a'
Y68_COMMIT = '9af4deca98f121cbbe88e65ac7e233035f76b888'
LOCAL_Y69_COMMIT = '9acd73cdc1b96be003c8450a9e22057248a0f531'
IDENTITIES = {
    'AdaRES-Y65': dict(gpu=0, pid=112356, code_commit=ORIGINAL,
        package='adares_y65_confidence_feedback', watts=280.0,
        receipt_sha256='989dbc53870334632996a142e0f53b4de993c79c03ffbe02d09fe0033f9ab1ce'),
    'AdaRES-Y67': dict(gpu=1, pid=100811, code_commit=Y67_COMMIT,
        package='adares_y67_stage_readout_norm', watts=390.0,
        receipt_sha256='39added73f460c15b16f21c31ea6d6ab8885fd83f8cb2a433f8668c437134d49'),
}
MODULE_FILES = ('__init__.py', 'backbone.py', 'dec.py', 'ipdn.py', 'loss.py', 'sample_model.py')


def expected_additions(package):
    return {'AdaRES/ipdn/model_paper5/' + package + '/' + name for name in MODULE_FILES} | {
        'AdaRES/configs_song/model_paper5/' + package + '_res.yaml'}


def validate_delta(base, target, changed):
    packages = {
        (ORIGINAL, Y67_COMMIT): 'adares_y67_stage_readout_norm',
        (Y67_COMMIT, Y68_COMMIT): 'adares_y68_language_mask_modulation',
    }
    assert (base, target) in packages, 'Unregistered commit edge'
    expected = expected_additions(packages[(base, target)])
    assert len(changed) == 7 and all(len(row) == 2 and row[0] == 'A' for row in changed)
    assert {row[1] for row in changed} == expected, 'Only the exact seven new candidate files are allowed'
    return packages[(base, target)]


def validate_identity(task):
    identity = IDENTITIES[task['id']]
    assert task['status'] == 'running' and task['launch_verified'] is True
    assert str(task['gpu']) == str(identity['gpu']) and task['pid'] == identity['pid']
    assert task['code_commit'] == identity['code_commit']
    assert task['config'] == 'configs_song/model_paper5/' + identity['package'] + '_res.yaml'
    return identity


def git_bytes(commit, path):
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=ROOT)


def adapt(value, server):
    if isinstance(value, dict):
        return {k: adapt(v, server) for k, v in value.items()}
    if isinstance(value, list):
        return [adapt(v, server) for v in value]
    if isinstance(value, str):
        for mapping in server.get('runtime_path_mappings', []):
            old, new = mapping['source'].rstrip('/'), mapping['target'].rstrip('/')
            if value == old or value.startswith(old + '/'):
                value = new + value[len(old):]
    return value


def validate_local_preparation(changed):
    assert len(changed) == 7 and all(len(row) == 2 and row[0] == 'A' for row in changed)
    assert {row[1] for row in changed} == expected_additions('adares_y69_coarse_geometry_bias')


def validate_local_head():
    assert closure.git(ROOT, 'rev-parse', 'HEAD') == LOCAL_Y69_COMMIT
    assert closure.git(ROOT, 'rev-parse', LOCAL_Y69_COMMIT + '^') == Y68_COMMIT
    rows = subprocess.check_output(['git', 'diff', '--no-renames', '--name-status', Y68_COMMIT, LOCAL_Y69_COMMIT], cwd=ROOT).decode().splitlines()
    validate_local_preparation([row.split('\t') for row in rows])
    for path in expected_additions('adares_y69_coarse_geometry_bias'):
        assert (ROOT / path).read_bytes() == git_bytes(LOCAL_Y69_COMMIT, path)


def build_contract(task, server):
    identity = validate_identity(task)
    assert server['host'] == '192.168.1.163' and server['expected_hostname'] == 'dl01-MS-7C60'
    assert server['allowed_gpus'] == [0, 1] and server['max_tasks_per_gpu'] == 1
    validate_local_head()
    cumulative = dict(source_files={}, configs={})
    deployments = {ORIGINAL: dict(source_files={}, configs={})}
    for base, target in ((ORIGINAL, Y67_COMMIT), (Y67_COMMIT, Y68_COMMIT)):
        assert subprocess.run(['git', 'merge-base', '--is-ancestor', base, target], cwd=ROOT).returncode == 0
        changed = subprocess.check_output(
            ['git', 'diff', '--no-renames', '--name-status', base, target], cwd=ROOT).decode().splitlines()
        package = validate_delta(base, target, [row.split('\t') for row in changed])
        for path in expected_additions(package):
            blob = git_bytes(target, path)
            assert (ROOT / path).read_bytes() == blob, 'Local added source changed after commit'
            relative = path[len('AdaRES/'):]
            if path.endswith('.yaml'):
                cumulative['configs'][relative] = adapt(yaml.safe_load(blob), server)
            else:
                cumulative['source_files'][relative] = hashlib.sha256(blob).hexdigest()
        deployments[target] = {k: dict(v) for k, v in cumulative.items()}
    allowed = [ORIGINAL, Y67_COMMIT, Y68_COMMIT] if task['id'] == 'AdaRES-Y65' else [Y67_COMMIT, Y68_COMMIT]
    files = {}
    for name in MODULE_FILES:
        relative = 'ipdn/model_paper5/' + identity['package'] + '/' + name
        digest = ref(ROOT / 'AdaRES' / relative)['sha256']
        for commit in allowed:
            assert hashlib.sha256(git_bytes(commit, 'AdaRES/' + relative)).hexdigest() == digest
        files[relative] = digest
    config = yaml.safe_load((ROOT / 'AdaRES' / task['config']).read_text(encoding='utf-8'))
    for commit in allowed:
        assert yaml.safe_load(git_bytes(commit, 'AdaRES/' + task['config'])) == config
    assert config['train']['seed'] == 1999 and config['train']['epochs'] == 30
    return dict(root=server['remote_code_root'], task=task, identity=identity, source_files=files,
        expected_config=adapt(config, server), allowed_markers=allowed,
        deployments={commit: deployments[commit] for commit in allowed})


REMOTE = r'''
import datetime,hashlib,json,pathlib,subprocess,time,yaml
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
root=pathlib.Path(P['root']);task=P['task'];identity=P['identity']
def verify_runtime():
    marker=(root/'.codex_source_commit').read_text().strip()
    assert marker in P['allowed_markers'], 'Unregistered production marker'
    for relative,digest in P['source_files'].items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==digest
    assert yaml.safe_load((root/task['config']).read_text())==P['expected_config']
    deployment=P['deployments'][marker]
    for relative,digest in deployment['source_files'].items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==digest
    for relative,config in deployment['configs'].items():
        assert yaml.safe_load((root/relative).read_text())==config
    return marker
marker=verify_runtime()
def sample():
    proc=pathlib.Path('/proc')/str(task['pid'])
    assert proc.is_dir() and (proc/'cwd').resolve()==root
    args=(proc/'cmdline').read_bytes().split(b'\0')
    assert task['config'].encode() in args and b'tools/train.py' in args
    log=pathlib.Path(task['log'])
    with log.open('rb') as stream:
        stream.seek(max(0,log.stat().st_size-32000));text=stream.read().decode(errors='replace')
    assert not any(v in text for v in ('Traceback','CUDA out of memory','RuntimeError'))
    return dict(observed_at=datetime.datetime.now().astimezone().isoformat(),
                latest=parse_phase(text),log_bytes=log.stat().st_size)
before=sample();samples=[before];confirmed=False
for attempt in range(3):
    time.sleep(5);after=sample();samples.append(after)
    if progressed(before,after):
        confirmed=True
        break
assert verify_runtime()==marker, 'Deployment changed during observation'
gpu=str(identity['gpu'])
uuid=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=uuid','--format=csv,noheader']).decode().strip()
owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits']).decode()
assert [int(row.split(',')[1]) for row in owners.splitlines() if row.split(',')[0].strip()==uuid]==[int(task['pid'])]
assert float(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=power.limit','--format=csv,noheader,nounits']).decode().strip())==identity['watts']
print(json.dumps(dict(complete=True,before=before,after=after,samples=samples,
    native_progress_confirmed=confirmed,process_identity_verified=True,pid=task['pid'],log=task['log'],
    source_files_verified=len(P['source_files']),native_YAML_semantics_verified=True,
    source_marker=task['code_commit'],production_marker=marker,
    exact_additive_deployment_verified=True,
    additional_source_files_verified=len(P['deployments'][marker]['source_files']),
    additional_configurations_verified=len(P['deployments'][marker]['configs']),
    physical_gpu=identity['gpu'],compute_owners=owners,authorized_gpu_uuid=uuid,
    gpu=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.limit','--format=csv,noheader,nounits']).decode().strip(),
    no_bounded_tail_error=True,training_efficacy='Unknown; full frozen evaluation not executed.',
    limitation='A phase observation never authorizes a restart. A future production marker is not observed until actually deployed.')))
'''


def remote_program(payload):
    source = Path(phases.__file__).read_text(encoding='utf-8')
    functions = '\n\n'.join(ast.get_source_segment(source, node) for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in ('parse_phase', 'phase_key', 'progressed'))
    program = 'P=' + repr(payload) + '\n' + functions + '\n' + REMOTE
    ast.parse(program, feature_version=(3, 8))
    return program


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True, choices=tuple(IDENTITIES))
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 150)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        before = {path: ref(path) for path in ('pipeline/queue.json', 'pipeline/running.json',
            'pipeline/goal.json', 'pipeline/admissions/AdaRES-Y65.json', 'pipeline/admissions/AdaRES-Y67.json')}
        server = monitor.load_servers(str(ROOT))['server-2']
        closure.validate_adares_server(ROOT, server)
        task = next(t for t in monitor.load_queue(str(ROOT))['gpu_queues']['server-2'] if t['id'] == args.candidate)
        identity = validate_identity(task)
        assert ref('pipeline/admissions/' + args.candidate + '.json')['sha256'] == identity['receipt_sha256']
        assert monitor.load_running(str(ROOT))['server-2'][str(identity['gpu'])] == args.candidate
        closure.admission(ROOT, task)
        payload = build_contract(task, server)
        output = ROOT / ('pipeline/checks/server2_y69_local_' + args.candidate.lower() + '_' + args.token + '.json')
        assert not output.exists() and not output.with_suffix('.log').exists()
        started = datetime.now().astimezone().isoformat()
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
            server['user'] + '@' + server['host'], 'PYTHONDONTWRITEBYTECODE=1 ' + server['python'] + ' -B -'],
            input=remote_program(payload).encode(), capture_output=True, timeout=45)
        output.with_suffix('.log').write_bytes(result.stdout + result.stderr)
        assert result.returncode == 0, 'Preserve observation failure; never relaunch'
        observation = json.loads(result.stdout)
        assert observation['native_progress_confirmed'] == phases.progressed(observation['before'], observation['after'])
        observation.update(id=args.candidate, token=args.token, observed_start_local=started,
            observed_end_local=datetime.now().astimezone().isoformat(),
            evidence=[ref(__file__), ref('scripts/observe_server2_y68_additive_phase.py'), ref(phases.__file__), ref(output.with_suffix('.log'))])
        tick.save(output, observation)
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == args.candidate)
        assert candidate['stage'] == 'training'
        candidate['evidence'].extend(e for e in (ref(output), ref(__file__), ref(phases.__file__)) if e not in candidate['evidence'])
        candidate['latest_training_observation'] = ref(output)
        tick.save(ROOT / 'pipeline/research_work.json', work)
        assert all(ref(path) == value for path, value in before.items())
        print(json.dumps(dict(evidence=ref(output), before=observation['before'], after=observation['after'],
            native_progress_confirmed=observation['native_progress_confirmed'],
            production_marker=observation['production_marker'], gpu=observation['gpu'])))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
