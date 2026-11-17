"""Freeze the tested Y60 native smoke and observe its actual resource gate."""

import argparse
import ast
import csv
from datetime import datetime, timedelta
import hashlib
import io
import json
import re
import shlex
import subprocess

from prepare_y60_stage_heads import ROOT, files, lease, monitor, ref, tick
import closure
from check_ef_execution import idle_gpus
from dispatch_prepared_candidates import snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 180)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        prefix = 'pipeline/checks/y60_'
        version_path = ROOT / (prefix + 'version_' + args.token + '.json')
        version = tick.load(version_path)
        cpu_path = ROOT / ('pipeline/analyses/y60_cpu_' + args.token + '.json')
        cpu = tick.load(cpu_path)
        assert cpu['complete'] and cpu['implementation_test_passed']
        prepared = ROOT / (prefix + 'cpu_' + args.token + '_program.py')
        payload = ast.literal_eval(ast.parse(prepared.read_text(encoding='utf-8')).body[0].value)
        row = payload['variant']
        for path in files():
            expected = row['config_data'] if path.suffix == '.yaml' else row['python_files'][path.name]
            assert ref(path)['sha256'] == expected['sha256']
            committed = subprocess.check_output(['git', 'show', version['code_commit'] + ':' +
                path.relative_to(ROOT).as_posix()], cwd=ROOT)
            assert hashlib.sha256(committed).hexdigest() == expected['sha256']
        frozen = dict(variant=row, code_commit=version['code_commit'],
                      goal_sha256=ref('pipeline/goal.json')['sha256'],
                      train_source_lf_sha256=payload['train_source_lf_sha256'])
        body = ROOT / 'scripts/probe_y60_native_smoke.py'
        program = 'FROZEN=' + repr(frozen) + '\n' + body.read_text(encoding='utf-8')
        ast.parse(program, feature_version=(3, 8))
        refused = False
        try:
            exec(compile(program, '<y60-native>', 'exec'), dict(P=dict(physical_gpu=None)))
        except AssertionError:
            refused = True
        assert refused
        program_path = ROOT / (prefix + 'smoke_' + args.token + '_program.py')
        assert not program_path.exists()
        program_path.write_text(program, encoding='utf-8', newline='\n')
        plan_path = ROOT / (prefix + 'smoke_' + args.token + '_plan.json')
        assert not plan_path.exists()
        command = ('python -B scripts/run_experiment_first_probe.py y60_smoke ' +
                   program_path.relative_to(ROOT).as_posix() +
                   ' --token <CURRENT_TOKEN> --seconds 600 --gpu --gpu-id <IDLE_AUTHORIZED_GPU>')
        tick.save(plan_path, dict(complete=True, id=row['id'], profile='experiment_first',
            program=ref(program_path), config=row['config'], code_commit=version['code_commit'],
            goal_sha256=frozen['goal_sha256'], planned_command=command,
            required_resource='Exclusive authorized server-2 GPU0/1, idle compute and reconciled queue reservations.',
            checks='Native YAML and TRAIN loader, native tools/train.py::train for two finite optimizer steps; '
                   'each of seven heads executes twice and receives finite nonzero final-weight gradients; '
                   'same released backbone/seed1999/fixed30; measure GPU peak below card capacity.',
            Python38_compile=True, missing_GPU_entry_refused=True, native_executed=False))
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
            observed = live['tasks'][task['id']]
            steps = closure.steps(observed['tail'])
            real = (observed['process_live'] and observed['cwd'] == server['remote_code_root']
                    and task['config'] in observed['cmdline'] and 'tools/train.py' in observed['cmdline'])
            epoch = re.findall(r'Train\[(\d+)/(\d+)\]', observed['tail'])
            progress.append(dict(id=task['id'], gpu=task.get('gpu'), pid=task.get('pid'),
                process_verified=bool(real), native_progress_verified=bool(real and len(set(steps)) >= 2),
                epoch=int(epoch[-1][0]) if epoch else None, total_epochs=int(epoch[-1][1]) if epoch else None,
                first_step=steps[0] if steps else None, last_step=steps[-1] if steps else None,
                observed_error=any(s in observed['tail'] for s in ('Traceback', 'CUDA out of memory', 'RuntimeError'))))
        health_program = """
import json,pathlib,subprocess
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
commands={
 'thermal':['nvidia-smi','--query-gpu=index,temperature.gpu,fan.speed,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],
 'performance':['nvidia-smi','-q','-d','PERFORMANCE'],
 'kernel':['dmesg','-T'],
 'disk':['df','-B1','/bak_home']}
out={'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
for name,cmd in commands.items():
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=10)
 text=r.stdout
 if name=='kernel':text='\\n'.join(line for line in text.splitlines() if 'NVRM: Xid' in line or 'fallen off' in line)
 out[name]={'rc':r.returncode,'stdout':text,'stderr':r.stderr}
print(json.dumps(out))
"""
        health = json.loads(monitor.ssh(server['host'], server['user'],
            shlex.quote(server['python']) + ' -B -c ' + shlex.quote(health_program), timeout=45))
        now = datetime.now().astimezone()
        evidence_path = ROOT / (prefix + 'resource_' + args.token + '.json')
        assert not evidence_path.exists()
        tick.save(evidence_path, dict(complete=True, observed_at=now.isoformat(),
            command='python -B scripts/freeze_and_check_y60.py --token ' + args.token,
            live=live, health=health, training=progress, idle_authorized_gpus=available,
            result='ready_for_native' if available else 'blocked_no_exclusive_GPU',
            native_smoke_executed=False, queue_changed=False,
            limitations='CPU/component PASS is not sparse task PASS or efficacy. No GPU workload, '
                        'production synchronization, fan/power changes or training termination performed.'))
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == row['id'])
        candidate['native_smoke_plan'] = ref(plan_path)
        candidate['evidence'].extend([ref(body), ref(program_path), ref(plan_path), ref(evidence_path), ref(__file__)])
        if available:
            candidate['next_action'] = ('Recheck current exact source/commit/goal and queue reservations; execute '
                'the frozen native_smoke_plan on a verified exclusive free authorized GPU; admit and try '
                'canonical dispatch in this tick only after actual native two-step smoke passes.')
        else:
            candidate.update(stage='blocked', blocked_from_stage='validation',
                blocker_kind='exclusive_gpu_slot', occupation_evidence=ref(evidence_path),
                blocker='Both authorized whole-GPU slots are reserved by the continuing original experiments; '
                        'live compute is present. Full Y60 sparse task smoke cannot safely share a running GPU.',
                retry_after=(now + timedelta(hours=1)).isoformat(),
                unblock_condition='Canonical queue/running change and live whole-card compute/memory checks '
                                  'confirm a genuinely idle authorized GPU, or the bounded resource retry is due.',
                wake_paths=['pipeline/queue.json', 'pipeline/running.json', 'SERVERS.yaml'],
                next_action='Use the current foreground lease to recheck frozen source/commit/goal and current '
                            'queue reservations. Run native_smoke_plan only on a genuinely idle exclusive '
                            'server-2 GPU0/1, then admit and try gated dispatch after actual two-step PASS. '
                            'Do not interrupt Y56/Y57, reopen old routes or ask for already-covered operations.')
        work['tick_attempt'] = dict(
            tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
            candidate_id=row['id'], kind='implementation_test', result='passed',
            command='python -B scripts/run_experiment_first_probe.py y60_cpu ' +
                prepared.relative_to(ROOT).as_posix() + ' --token ' + args.token + ' --seconds 300',
            observation='Native YAML full CPU construction, released backbone and real TRAIN collate passed; '
                        'two-dtype independent-head forward/gradient controls passed. No native sparse task '
                        'backward or admission. GPU resource observation: ' +
                        ('free authorized slot found.' if available else 'both whole-GPU slots occupied.'),
            evidence=[ref(cpu_path), ref(cpu_path.with_suffix('.log')), ref(evidence_path)])
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(plan=ref(plan_path), evidence=ref(evidence_path), training=progress,
            idle_authorized_gpus=available, thermal=health['thermal'], kernel=health['kernel']), indent=2))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
