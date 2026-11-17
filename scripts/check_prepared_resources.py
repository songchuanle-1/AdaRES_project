"""Bounded live recheck for already implemented native-smoke candidates."""

import argparse
import ast
import csv
from datetime import datetime, timedelta
import io
import json
from pathlib import Path
import re
import shlex

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
import closure
from check_ef_execution import idle_gpus, validate_frozen
from dispatch_prepared_candidates import snapshot

IDS = ('AdaRES-Y58', 'AdaRES-Y59', 'AdaRES-Y60')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 180)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        work = tick.load(ROOT / 'pipeline/research_work.json')
        now = datetime.now().astimezone()
        candidates = [c for c in work['candidates'] if c['id'] in IDS and
            (c['stage'] == 'validation' or (c['stage'] == 'blocked' and
             c.get('blocker_kind') == 'exclusive_gpu_slot' and
             datetime.fromisoformat(c['retry_after']) <= now))]
        assert candidates, 'No due eligible resource check; preserve unchanged holds'
        checks = []
        for candidate in candidates:
            plan = tick.load(closure.artifact(ROOT, candidate['native_smoke_plan']))
            program = closure.artifact(ROOT, plan['program'])
            frozen = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
            checks.append(dict(id=candidate['id'], exact_files=validate_frozen(candidate, plan, frozen),
                code_commit=candidate['code_commit'], config=candidate['config'],
                frozen_program=ref(program), current_version_and_goal_match=True))
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
            real = (observed['process_live'] and observed['cwd'] == server['remote_code_root'] and
                    task['config'] in observed['cmdline'] and 'tools/train.py' in observed['cmdline'])
            epochs = re.findall(r'(Train|Val)\[(\d+)/(\d+)\]', observed['tail'])
            metric = monitor.parse_running_miou([task['log']], task['pid'], Path(task['config']).stem,
                server['work_dir'], server['host'], server['user'], registered_native=True) if real else None
            progress.append(dict(id=task['id'], gpu=task.get('gpu'), pid=task.get('pid'),
                process_verified=bool(real), native_progress_verified=bool(real and len(set(steps)) >= 2),
                phase=epochs[-1][0] if epochs else None, epoch=int(epochs[-1][1]) if epochs else None,
                total_epochs=int(epochs[-1][2]) if epochs else None,
                first_step=steps[0] if steps else None, last_step=steps[-1] if steps else None,
                parsed_current_segment_metric=metric,
                observed_error=any(s in observed['tail'] for s in ('Traceback', 'CUDA out of memory', 'RuntimeError'))))
        health_program = """
import json,pathlib,subprocess
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
out={'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
for name,cmd in {
 'devices':['nvidia-smi','--query-gpu=index,uuid,power.limit,power.draw,temperature.gpu,fan.speed,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],
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
        assert health['devices']['rc'] == 0
        devices = {int(r[0]): dict(uuid=r[1].strip(), power_limit_watts=float(r[2]),
            draw_watts=float(r[3]), temperature_c=float(r[4]), fan_percent=float(r[5]),
            utilization_percent=float(r[6]), memory_used_mib=int(r[7]), memory_total_mib=int(r[8]))
            for r in csv.reader(io.StringIO(health['devices']['stdout']))}
        assert set(devices) == {0, 1}
        output = ROOT / ('pipeline/checks/prepared_resources_' + args.token + '.json')
        assert not output.exists()
        observed_at = datetime.now().astimezone()
        result = dict(complete=True, observed_at=observed_at.isoformat(),
            command='python -B scripts/check_prepared_resources.py --token ' + args.token,
            candidate_identity_checks=checks, live=live, training=progress, health=health,
            devices=devices, idle_authorized_gpus=available,
            result='ready_for_native_smoke' if available else 'blocked_no_exclusive_GPU',
            native_smoke_executed=False, queue_changed=False, hardware_settings_changed=False,
            limitation='Source/goal/commit identity and live resources only. No new CPU diagnostic, '
                       'native sparse backward, admission or efficacy result. Slot opening requires '
                       'a fresh native GPU check and verified launch, not just this observation.')
        tick.save(output, result)
        evidence = ref(output)
        for candidate in candidates:
            for entry in (evidence, ref(__file__)):
                if entry not in candidate['evidence']:
                    candidate['evidence'].append(entry)
            if available:
                candidate['stage'] = 'validation'
                for key in ('blocker', 'blocker_kind', 'retry_after', 'unblock_condition'):
                    candidate.pop(key, None)
            else:
                candidate.update(stage='blocked', blocked_from_stage='validation', required_server='server-2',
                    blocker_kind='exclusive_gpu_slot', occupation_evidence=evidence,
                    blocker='Exact prepared source/YAML/commit/goal still match. Both authorized whole '
                            'GPUs remain occupied or reserved by continuing original jobs; native sparse '
                            'two-step smoke cannot share these GPUs.',
                    retry_after=(observed_at + timedelta(hours=1)).isoformat(),
                    unblock_condition='Canonical queue/running opening plus confirmed live idle GPU, '
                                      'or a bounded resource retry is due.')
            candidate['next_action'] = ('Keep this tested implementation and native_smoke_plan. With '
                'the current foreground lease, check actual exclusive free GPU0/1 and queue reservations, '
                'then execute native YAML/TRAIN two-step loss/backward, intended-path and peak checks. '
                'After actual PASS, admit and try existing gated dispatch in the same tick. '
                'No repeated routine approval or prior scientific-gain proof; preserve all running jobs.')
        work['tick_attempt'] = dict(
            tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
            candidate_id=candidates[0]['id'], kind='resource_check', result='passed' if available else 'blocked',
            command=result['command'],
            observation='Actual current committed/prepared YAML/source/goal identities checked for ' +
                ', '.join(c['id'] for c in candidates) + '; ' +
                ('idle authorized card detected; native smoke not yet run.' if available else
                 'both authorized whole GPUs still occupied/reserved; no native GPU smoke executed.'),
            evidence=[evidence])
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(evidence=evidence, candidates=[c['id'] for c in candidates],
            training=progress, devices=devices, idle_authorized_gpus=available,
            kernel=health['kernel'], native_smoke_executed=False), indent=2))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
