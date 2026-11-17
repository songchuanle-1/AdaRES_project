"""Run the retained Y91 native plan only after its original owner is archived."""
import argparse
import ast
from datetime import datetime
import json
import shlex
import subprocess
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from advance_y82 import live, idle, unreserved
from check_ef_execution import validate_frozen
from y91_contract import validate_result
from run_y90_native import OWNERS, validate_terminal
import closure


def runtime(name, candidate, frozen, server):
    assert name in OWNERS and candidate['id'] == frozen['variant']['id'] == 'AdaRES-Y91'
    assert server['allowed_gpus'] == [0] and server['max_tasks_per_gpu'] == 1
    assert frozen['runtime_server'] == name and frozen['runtime_physical_gpu'] == 0
    power = 350. if name == 'server-1' else 280.
    assert frozen['runtime_power_limit_watts'] == power
    if name == 'server-1':
        assert server['expected_hostname'] == 'dl02-MS-7B73'
        assert frozen['runtime_path_mappings'] == server['runtime_path_mappings']
        assert server['python'] == '/data1/songcl/Project/AdaRES_project/runtime/adares_project/bin/python'
    else:
        assert server['expected_hostname'] == 'dl01-MS-7C60'
        assert server['python'] == '/home/songcl/.conda/envs/adares_project/bin/python'
    return power


def run(token, name):
    lease(token, 800)
    server = monitor.load_servers(str(ROOT))[name]
    closure.validate_adares_server(ROOT, server)
    authority = ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')
    assert authority['sha256'] == '3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45'
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(candidate for candidate in work['candidates'] if candidate['id'] == 'AdaRES-Y91')
    assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'exclusive_gpu_slot'
    assert not candidate.get('native_smoke_result') and not candidate.get('admission_receipt')
    key = 'native_smoke_plan_server1' if name == 'server-1' else 'native_smoke_plan_server2_gpu0'
    plan = tick.load(closure.artifact(ROOT, candidate[key]))
    program_path = closure.artifact(ROOT, plan['program'])
    program = program_path.read_text(encoding='utf-8')
    frozen = ast.literal_eval(ast.parse(program).body[0].value)
    assert validate_frozen(candidate, plan, frozen) == 7
    power = runtime(name, candidate, frozen, server)
    closure.artifact(ROOT, frozen['resource_authorization'])
    done = monitor.load_experiment(str(ROOT), OWNERS[name])
    assert done['status'] == 'done_complete' and str(done['gpu']) == '0'
    terminal = tick.load(closure.artifact(ROOT, done['terminal_evidence']))
    validate_terminal(terminal, name)
    unreserved(monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT)), name, 0)
    observation = live(server, 0)
    idle(observation, power)
    basename = 'y91_' + name.replace('-', '') + '_smoke_' + token
    output = ROOT / ('pipeline/analyses/' + basename + '.json')
    log = output.with_suffix('.log')
    activation = ROOT / ('pipeline/checks/' + basename + '_activation.json')
    assert not any(path.exists() for path in (output, log, activation))
    remote_base = server['remote_code_root'].rsplit('/', 1)[0]
    remote_output = remote_base + '/pipeline/probes/' + token + '/' + basename + '.json'
    payload = dict(output=remote_output, root=server['remote_code_root'], physical_gpu=0,
                   token=token, source_sha256=ref(program_path)['sha256'])
    if name == 'server-1':
        payload['path_mappings'] = server['runtime_path_mappings']
    prefix = 'import os,sys,pathlib,json,subprocess,fcntl\nP=' + repr(payload) + '\n'
    prefix += "assert subprocess.check_output(['hostname']).decode().strip()==" + repr(server['expected_hostname']) + '\n'
    prefix += "assert not pathlib.Path(P['output']).exists()\npathlib.Path(P['output']).parent.mkdir(parents=True,exist_ok=True)\n"
    prefix += "os.environ['PYTHONDONTWRITEBYTECODE']='1'\nos.environ['CUDA_VISIBLE_DEVICES']='0'\n"
    prefix += "os.chdir(P['root'])\nsys.path[:0]=" + repr([
        server['remote_code_root'], server['remote_code_root'] + '/tools', server['remote_code_root'] + '/ipdn/lib',
        server['site_packages'], server['pointnet_egg']]) + '\n'
    prefix += 'gpu_lease=open(' + repr(remote_base + '/.foreground_gpu0_diagnostic.lock') + ",'a')\nfcntl.flock(gpu_lease,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
    prefix += "used,util=[int(value.strip()) for value in subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits']).decode().split(',')]\nassert used<1024 and util<10\n"
    prefix += "assert not subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip()\n"
    ast.parse(prefix + program, feature_version=(3, 8))
    tick.save(activation, dict(complete=True, token=token, id=candidate['id'], server=name,
        original_terminal=done['terminal_evidence'], native_plan=candidate[key], observation=observation,
        native_executed=False, GPU1_use_authorized=False, helper=ref(__file__)))
    candidate.update(stage='validation', required_server=name, required_gpu=0, native_smoke_plan=candidate[key])
    for field in ('blocker', 'blocker_kind', 'blocked_from_stage', 'retry_after', 'unblock_condition', 'occupation_evidence'):
        candidate.pop(field, None)
    candidate['evidence'].append(ref(activation))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    command = 'PYTHONDONTWRITEBYTECODE=1 '
    if server.get('runtime_library_path'):
        command += 'LD_LIBRARY_PATH=' + shlex.quote(server['runtime_library_path']) + ' '
    command += 'timeout --signal=TERM --kill-after=10s 600 ' + shlex.quote(server['python']) + ' -S -B -'
    with log.open('xb') as stream:
        response = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
            server['user'] + '@' + server['host'], command], input=(prefix + program).encode(),
            stdout=stream, stderr=subprocess.STDOUT, timeout=625)
    lease(token, 60)
    assert response.returncode == 0, 'Native failed; retain raw log, never replay or relax controls blindly'
    subprocess.run(['scp', '-q', server['user'] + '@' + server['host'] + ':' + remote_output, str(output)],
                   check=True, timeout=30)
    raw = tick.load(output)
    validate_result(raw, candidate, frozen)
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(candidate for candidate in work['candidates'] if candidate['id'] == 'AdaRES-Y91')
    candidate['native_smoke_result'] = ref(output)
    candidate['evidence'].extend([ref(output), ref(log), ref(__file__)])
    work['tick_attempt'] = dict(
        tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id=candidate['id'], kind='native_smoke', result='passed',
        command='python -B scripts/run_y91_native.py --server ' + name + ' --token ' + token,
        observation='Original native TRAIN two finite task losses/backward/updates, fourteen original hidden-gate/final-linear/evidence consumers and70 original MLP-layer calls passed. Efficacy unmeasured.',
        evidence=[ref(output), ref(log)])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(result=ref(output), losses=raw['losses'], peak_reserved_bytes=raw['peak_reserved_bytes'],
                         server=name, observed_at=datetime.now().astimezone().isoformat())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', required=True, choices=('server-1', 'server-2'))
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 800)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        run(args.token, args.server)
    finally:
        writer.close()
