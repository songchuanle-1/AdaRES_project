"""Activate and run exactly the tested Y88 program on truly free server1GPU0."""
import argparse
import ast
from datetime import datetime
import json
import shlex
import subprocess
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from prepare_y69_server1_alternative import observe, idle
from advance_y82 import unreserved
from check_ef_execution import validate_frozen
from y88_contract import validate_result
import closure


def run(token):
    lease(token, 800)
    server = monitor.load_servers(str(ROOT))['server-1']
    closure.validate_adares_server(ROOT, server)
    assert server['allowed_gpus'] == [0]
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(candidate for candidate in work['candidates'] if candidate['id'] == 'AdaRES-Y88')
    assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'exclusive_gpu_slot'
    assert candidate['required_server'] == 'server-1' and candidate['required_gpu'] == 0
    assert not candidate.get('native_smoke_result')
    plan = tick.load(closure.artifact(ROOT, candidate['native_smoke_plan']))
    program_path = closure.artifact(ROOT, plan['program'])
    program = program_path.read_text(encoding='utf-8')
    frozen = ast.literal_eval(ast.parse(program).body[0].value)
    assert validate_frozen(candidate, plan, frozen) == 7
    assert frozen['runtime_server'] == 'server-1' and frozen['runtime_power_limit_watts'] == 350.
    assert frozen['runtime_path_mappings'] == server['runtime_path_mappings']
    closure.artifact(ROOT, frozen['resource_authorization'])
    done = monitor.load_experiment(str(ROOT), 'AdaRES-Y84')
    assert done['status'] == 'done_complete' and str(done['gpu']) == '0'
    terminal = tick.load(closure.artifact(ROOT, done['terminal_evidence']))
    assert terminal['completed_epochs'] == {phase: list(range(1, 31)) for phase in ('Train', 'Val')}
    assert terminal['checkpoints'][0]['epoch'] == 30
    unreserved(monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT)), 'server-1', 0)
    observation = observe(server)
    assert idle(observation), 'Preserve foreign jobs'
    basename = 'y88_server1_smoke_' + token
    output = ROOT / ('pipeline/analyses/' + basename + '.json')
    log = output.with_suffix('.log')
    assert not output.exists() and not log.exists()
    remote_output = '/data1/songcl/Project/AdaRES_project/pipeline/probes/' + token + '/y88_server1_smoke.json'
    payload = dict(output=remote_output, root=server['remote_code_root'], physical_gpu=0,
        path_mappings=server['runtime_path_mappings'], token=token, source_sha256=ref(program_path)['sha256'])
    prefix = "import os,sys,pathlib,json,subprocess,fcntl\nP=" + repr(payload) + "\n"
    prefix += "assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'\n"
    prefix += "assert not pathlib.Path(P['output']).exists()\npathlib.Path(P['output']).parent.mkdir(parents=True,exist_ok=True)\n"
    prefix += "os.environ['PYTHONDONTWRITEBYTECODE']='1'\nos.environ['CUDA_VISIBLE_DEVICES']='0'\n"
    prefix += "os.chdir(P['root'])\nsys.path[:0]=" + repr([server['remote_code_root'], server['remote_code_root'] + '/tools',
        server['remote_code_root'] + '/ipdn/lib', server['site_packages'], server['pointnet_egg']]) + '\n'
    prefix += "gpu_lease=open('/data1/songcl/Project/AdaRES_project/.foreground_gpu0_diagnostic.lock','a')\nfcntl.flock(gpu_lease,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
    prefix += "used,util=[int(x.strip()) for x in subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits']).decode().split(',')]\nassert used<1024 and util<10\n"
    prefix += "assert not subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip()\n"
    ast.parse(prefix + program, feature_version=(3, 8))
    candidate.update(stage='validation')
    for key in ('blocker', 'blocker_kind', 'blocked_from_stage', 'retry_after', 'unblock_condition', 'occupation_evidence'):
        candidate.pop(key, None)
    activation = ROOT / ('pipeline/checks/y88_server1_activation_' + token + '.json')
    assert not activation.exists()
    tick.save(activation, dict(complete=True, token=token, id=candidate['id'], observation=observation,
        original_Y84_terminal=done['terminal_evidence'], native_plan=candidate['native_smoke_plan'],
        native_executed=False, GPU1_use_authorized=False, helper=ref(__file__)))
    candidate['evidence'].append(ref(activation))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    with log.open('xb') as stream:
        response = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
            server['user'] + '@' + server['host'],
            'PYTHONDONTWRITEBYTECODE=1 LD_LIBRARY_PATH=' + shlex.quote(server['runtime_library_path'])
            + ' timeout --signal=TERM --kill-after=10s 600 ' + shlex.quote(server['python']) + ' -S -B -'],
            input=(prefix + program).encode(), stdout=stream, stderr=subprocess.STDOUT, timeout=625)
    lease(token, 60)
    assert response.returncode == 0, 'Native failed; preserve raw log and do not relax guards'
    subprocess.run(['scp', '-q', server['user'] + '@' + server['host'] + ':' + remote_output, str(output)],
        check=True, timeout=30)
    raw = tick.load(output)
    validate_result(raw, candidate, frozen)
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(candidate for candidate in work['candidates'] if candidate['id'] == 'AdaRES-Y88')
    candidate['native_smoke_result'] = ref(output)
    candidate['evidence'].extend([ref(output), ref(log), ref(__file__)])
    work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id=candidate['id'], kind='native_smoke', result='passed',
        command='python -B scripts/run_y88_server1_native.py --token ' + token,
        observation='Original native TRAIN two finite losses/backward/optimizer, fourteen exact quadratic normalized-query projections/mask consumers and unchanged score/head/query_norm/gate algorithms passed.',
        evidence=[ref(output), ref(log)])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(result=ref(output), losses=raw['losses'], peak_reserved_bytes=raw['peak_reserved_bytes'],
        observed_at=datetime.now().astimezone().isoformat())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 800)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        run(args.token)
    finally:
        writer.close()
