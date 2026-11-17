"""Observe continuing training and bind an additional non-GPU Y60 integration test."""

import argparse
import ast
import base64
import csv
from datetime import datetime, timedelta
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess

from prepare_y60_stage_heads import ROOT, FILES, lease, monitor, ref, tick
import closure
from check_ef_execution import idle_gpus, validate_frozen
from dispatch_prepared_candidates import snapshot


def path_for(kind, token, suffix='.json'):
    return ROOT / ('pipeline/checks/y60_' + kind + '_' + token + suffix)


def prepare(token, candidate):
    plan = tick.load(closure.artifact(ROOT, candidate['native_smoke_plan']))
    frozen_program = closure.artifact(ROOT, plan['program'])
    frozen = ast.literal_eval(ast.parse(frozen_program.read_text(encoding='utf-8')).body[0].value)
    assert validate_frozen(candidate, plan, frozen) == 7
    parent_folder = ROOT / 'AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition'
    parent_config = ROOT / 'AdaRES/configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml'
    def encoded(path):
        return dict(base64=base64.b64encode(path.read_bytes()).decode(), sha256=ref(path)['sha256'])
    parent = dict(package=parent_folder.name,
        python_files={name: encoded(parent_folder / name) for name in FILES},
        config=parent_config.relative_to(ROOT / 'AdaRES').as_posix(), config_data=encoded(parent_config))
    for path in [parent_folder / name for name in FILES] + [parent_config]:
        committed = subprocess.check_output(['git', 'show', candidate['code_commit'] + ':' +
            path.relative_to(ROOT).as_posix()], cwd=ROOT)
        assert committed.replace(b'\r\n', b'\n') == path.read_bytes().replace(b'\r\n', b'\n')
    body = ROOT / 'scripts/probe_y60_decoder_cpu.py'
    program_path = path_for('decoder_cpu', token, '_program.py')
    program = 'PACKET=' + repr(dict(frozen=frozen, parent=parent)) + '\n' + body.read_text(encoding='utf-8')
    ast.parse(program, feature_version=(3, 8))
    assert not program_path.exists()
    program_path.write_text(program, encoding='utf-8', newline='\n')
    server = monitor.load_servers(str(ROOT))['server-2']
    assert server['host'] == '192.168.1.163' and server['allowed_gpus'] == [0, 1]
    assert server['max_tasks_per_gpu'] == 1
    queue = monitor.load_queue(str(ROOT))
    tasks = [t for t in queue['gpu_queues']['server-2'] if t['status'] in ('running', 'claiming')]
    live = snapshot(server, tasks, [])
    assert live['gpu_query_rc'] == live['compute_query_rc'] == 0
    inventory = {int(r[0]): r for r in csv.reader(io.StringIO(live['authorized_device_inventory']))}
    assert set(inventory) == {0, 1}
    compute = list(csv.reader(io.StringIO(live['compute'])))
    reserved = {int(t['gpu']) for t in tasks if t.get('gpu') is not None}
    available = idle_gpus(inventory, compute, reserved)
    progress = []
    for task in tasks:
        row = live['tasks'][task['id']]
        steps = closure.steps(row['tail'])
        real = (row['process_live'] and row['cwd'] == server['remote_code_root']
                and task['config'] in row['cmdline'] and 'tools/train.py' in row['cmdline'])
        epoch = re.findall(r'(Train|Val)\[(\d+)/(\d+)\]', row['tail'])
        metric = monitor.parse_running_miou([task['log']], task['pid'], Path(task['config']).stem,
            server['work_dir'], server['host'], server['user'], registered_native=True) if real else None
        progress.append(dict(id=task['id'], gpu=task.get('gpu'), pid=task.get('pid'),
            process_verified=bool(real), native_progress_verified=bool(real and len(set(steps)) >= 2),
            phase=epoch[-1][0] if epoch else None, epoch=int(epoch[-1][1]) if epoch else None,
            total_epochs=int(epoch[-1][2]) if epoch else None,
            first_step=steps[0] if steps else None, last_step=steps[-1] if steps else None,
            parsed_current_segment_metric=metric,
            observed_error=any(s in row['tail'] for s in ('Traceback', 'CUDA out of memory', 'RuntimeError'))))
    health_program = """
import json,pathlib,subprocess
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
out={'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
for name,cmd in {
 'thermal':['nvidia-smi','--query-gpu=index,temperature.gpu,fan.speed,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],
 'performance':['nvidia-smi','-q','-d','PERFORMANCE'],
 'kernel':['dmesg','-T'],'disk':['df','-B1','/bak_home']}.items():
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=10)
 text=r.stdout
 if name=='kernel':text='\\n'.join(v for v in text.splitlines() if 'NVRM: Xid' in v or 'fallen off' in v)
 out[name]={'rc':r.returncode,'stdout':text,'stderr':r.stderr}
print(json.dumps(out))
"""
    health = json.loads(monitor.ssh(server['host'], server['user'],
        shlex.quote(server['python']) + ' -B -c ' + shlex.quote(health_program), timeout=45))
    observation = path_for('integration_occupation', token)
    assert not observation.exists()
    tick.save(observation, dict(complete=True, observed_at=datetime.now().astimezone().isoformat(),
        live=live, training=progress, health=health, idle_authorized_gpus=available,
        queue_changed=False, GPU_test_executed=False,
        note='No running job, power/fan setting, production source or queue mutated.'))
    print(json.dumps(dict(program=ref(program_path), resource=ref(observation), training=progress,
        idle_authorized_gpus=available, thermal=health['thermal'], kernel=health['kernel']), indent=2))


def record(token, work, candidate):
    test = ROOT / ('pipeline/analyses/y60_decoder_cpu_' + token + '.json')
    result = tick.load(test)
    assert result['complete'] and result['id'] == candidate['id']
    assert result['code_commit'] == candidate['code_commit']
    assert result['goal_sha256'] == ref('pipeline/goal.json')['sha256']
    assert result['actual_decoder_class_executed'] and result['actual_seven_stage_head_calls'] == [1] * 7
    assert result['all_stage_component_backward_finite'] and result['final_only_loss_no_sibling_head_gradient']
    assert result['native_smoke_passed'] is False and result['CUDA_initialized'] is False
    observation = path_for('integration_occupation', token)
    observed = tick.load(observation)
    new_evidence = [ref(test), ref(test.with_suffix('.log')), ref(observation),
        ref(path_for('decoder_cpu', token, '_program.py')), ref('scripts/probe_y60_decoder_cpu.py'), ref(__file__)]
    for entry in new_evidence:
        if entry not in candidate['evidence']:
            candidate['evidence'].append(entry)
    candidate['decoder_integration_check'] = ref(test)
    candidate['next_action'] = ('Keep unchanged tested Y60 source/YAML/commit and native_smoke_plan. '
        'Full native sparse TRAIN two-step loss/backward and GPU peak remain required; execute with the '
        'current foreground lease only when canonical reservations and actual compute show an exclusive '
        'authorized whole GPU free, then admit and try canonical dispatch the same tick. '
        'CPU fixture integration is not native task PASS or evidence of useful efficacy.')
    if not observed['idle_authorized_gpus']:
        now = datetime.now().astimezone()
        candidate.update(stage='blocked', blocked_from_stage='validation', required_server='server-2',
            blocker_kind='exclusive_gpu_slot', occupation_evidence=ref(observation),
            retry_after=(now + timedelta(hours=1)).isoformat(),
            blocker='Both authorized whole-GPU slots are still reserved and occupied by original training. '
                    'Y60 full decoder CPU integration passed; native sparse task/GPU check remains unrun.',
            unblock_condition='Canonical queue/running change opens a slot plus live idle GPU confirmation, '
                              'or the bounded resource recheck is due.')
    work['tick_attempt'] = dict(
        tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id=candidate['id'], kind='implementation_test', result='passed',
        command='python -B scripts/run_experiment_first_probe.py y60_decoder_cpu ' +
            path_for('decoder_cpu', token, '_program.py').relative_to(ROOT).as_posix() +
            ' --token ' + token + ' --seconds 300',
        observation='Actual full DEC graph on synthetic CPU fixtures passed common-initial-state/all-stage '
                    'output parity and seven private-head execution/gradient isolation. '
                    'Not a native sparse task backward or trained gain; actual authorized GPUs occupied.',
        evidence=[ref(test), ref(test.with_suffix('.log')), ref(observation)])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(candidate=candidate['id'], evidence=ref(test), new_evidence=len(new_evidence),
        admitted=False, native_smoke_passed=False)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('prepare', 'record'))
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 120)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y60')
        assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'exclusive_gpu_slot'
        if args.operation == 'prepare':
            prepare(args.token, candidate)
        else:
            record(args.token, work, candidate)
    finally:
        lock.close()


if __name__ == '__main__':
    main()
