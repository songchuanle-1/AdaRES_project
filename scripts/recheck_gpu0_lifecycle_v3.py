"""Observe current GPU0 resources with lifecycle-aware, stage-based due work."""

import argparse
import ast
import csv
from datetime import datetime, timedelta
import io
import json
import shlex

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from check_ef_execution import idle_gpus, validate_frozen
from continue_y61_decoder_cpu import source_payload
from dispatch_prepared_candidates import snapshot
import closure


def work_kind(candidate, experiment):
    if candidate.get('blocker_kind') not in (None, 'exclusive_gpu_slot'):
        return None
    if candidate['stage'] not in ('validation', 'implementation', 'blocked'):
        return None
    if candidate.get('scientific_status') == 'fixed30_complete_frozen_review_pending':
        assert experiment.get('status') == 'done_complete' and candidate.get('post_training_freeze')
        return 'frozen_evaluation'
    if candidate.get('admission_receipt'):
        return None
    if candidate.get('native_smoke_plan'):
        return 'native_smoke'
    if candidate['id'] == 'AdaRES-Y61':
        assert not candidate.get('code_commit') and candidate.get('decoder_integration_check')
        return 'version_then_native_smoke'
    raise AssertionError('Unknown pending lifecycle; preserve it for review')


def is_due(candidate, now):
    if candidate['stage'] in ('validation', 'implementation'):
        return True
    retry = candidate.get('retry_after')
    return retry is None or datetime.fromisoformat(retry) <= now


def update_resource(candidate, kind, available, observed_at, occupants):
    result = dict(candidate)
    result['resource_work_kind'] = kind
    result['required_server'] = 'server-2'
    stage = 'implementation' if kind == 'version_then_native_smoke' else 'validation'
    if available:
        result['stage'] = stage
        for key in ('blocker', 'blocker_kind', 'blocked_from_stage', 'retry_after', 'unblock_condition'):
            result.pop(key, None)
    else:
        result.update(stage='blocked', blocked_from_stage=stage, blocker_kind='exclusive_gpu_slot',
            blocker='Only authorized GPU0 is occupied/reserved; current verified or reserved tasks: ' +
                    ', '.join(occupants) + '. Preserve foreign jobs and exclude GPU1.',
            retry_after=(observed_at + timedelta(hours=1)).isoformat(),
            unblock_condition='Canonical queue/running opens GPU0 plus fresh exclusive-idle verification.')
    if kind == 'frozen_evaluation':
        result['next_action'] = ('Preserve completed fixed30 best/last/model/optimizer/scheduler and '
            'the immutable post_training_freeze. Only run its registered full frozen evaluation '
            'and trained-cost comparison on exclusive authorized GPU0. No training restart, '
            'repeat admission, validation tuning or promotion from the training log.')
    elif kind == 'native_smoke':
        result['next_action'] = ('Preserve the exact committed source/YAML and frozen native_smoke_plan. '
            'On exclusive idle authorized GPU0 with the current lease, run actual native TRAIN '
            'two-step loss/backward/path/resource checks; only a real PASS permits admission '
            'and existing gated dispatch. Reuse scoped authorization.')
    else:
        result['next_action'] = ('Preserve the exact seven tested Y61 files and CPU evidence. Save '
            'only this scoped bundle as a version without invalidating admitted waiting work, '
            'bind its actual native YAML/TRAIN two-step plan, and execute on exclusive idle '
            'authorized GPU0 before admission. CPU checks do not replace native backward; '
            'do not disturb current training, use GPU1 or ask for repeated routine approval.')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 180)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        server = monitor.load_servers(str(ROOT))['server-2']
        assert server['host'] == '192.168.1.163' and server['allowed_gpus'] == [0]
        assert server['max_tasks_per_gpu'] == 1
        work = tick.load(ROOT / 'pipeline/research_work.json')
        now = datetime.now().astimezone()
        selected = []
        for candidate in work['candidates']:
            if candidate['id'] not in ('AdaRES-Y56', 'AdaRES-Y58', 'AdaRES-Y59', 'AdaRES-Y60', 'AdaRES-Y61'):
                continue
            experiment = monitor.load_experiment(str(ROOT), candidate['id']) or {}
            kind = work_kind(candidate, experiment)
            if kind is None or not is_due(candidate, now):
                continue
            if kind == 'frozen_evaluation':
                freeze = closure.artifact(ROOT, candidate['post_training_freeze'])
                terminal = closure.artifact(ROOT, experiment['terminal_evidence'])
                metadata = tick.load(terminal)['checkpoints']
                assert any(c['epoch'] == c['scheduler_last_epoch'] == 30 for c in metadata)
                checks = dict(freeze=ref(freeze), terminal=ref(terminal), status=experiment['status'])
            elif kind == 'native_smoke':
                plan = tick.load(closure.artifact(ROOT, candidate['native_smoke_plan']))
                program = closure.artifact(ROOT, plan['program'])
                frozen = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
                assert validate_frozen(candidate, plan, frozen) == 7
                checks = dict(exact_files=7, source_goal_commit_match=True)
            else:
                _, original = source_payload(candidate)
                decoder = tick.load(closure.artifact(ROOT, candidate['decoder_integration_check']))
                assert decoder['complete'] and not decoder['native_smoke_passed']
                assert decoder['goal_sha256'] == ref('pipeline/goal.json')['sha256']
                checks = dict(exact_files=7, original_cpu_payload=original, native_smoke_executed=False)
            selected.append((candidate, kind, checks))
        assert selected, 'No due eligible resource work; preserve unchanged holds'
        queue = monitor.load_queue(str(ROOT))
        tasks = [t for t in queue['gpu_queues']['server-2'] if t['status'] in ('running', 'claiming')]
        live = snapshot(server, tasks, [])
        assert live['gpu_query_rc'] == live['compute_query_rc'] == 0
        inventory = {int(row[0]): row for row in csv.reader(io.StringIO(live['authorized_device_inventory']))
                     if int(row[0]) in server['allowed_gpus']}
        assert set(inventory) == {0}
        reserved = {int(t['gpu']) for t in tasks if t.get('gpu') is not None}
        available = idle_gpus(inventory, list(csv.reader(io.StringIO(live['compute']))), reserved)
        power_program = ("import json,subprocess;assert subprocess.check_output(['hostname']).decode().strip()"
            "=='dl01-MS-7C60';print(json.dumps({'gpu0':subprocess.check_output(['nvidia-smi','-i','0',"
            "'--query-gpu=power.limit,power.draw,temperature.gpu,memory.used','--format=csv,noheader,nounits']).decode().strip()}))")
        hardware = json.loads(monitor.ssh(server['host'], server['user'],
            shlex.quote(server['python']) + ' -B -c ' + shlex.quote(power_program), timeout=30))
        assert float(hardware['gpu0'].split(',')[0]) == 280
        observed_at = datetime.now().astimezone()
        records = [dict(id=c['id'], kind=kind, checks=checks) for c, kind, checks in selected]
        output = ROOT / ('pipeline/checks/gpu0_lifecycle_v3_' + args.token + '.json')
        assert not output.exists()
        tick.save(output, dict(complete=True, observed_at=observed_at.isoformat(), records=records,
            live=live, hardware=hardware, idle_authorized_gpus=available,
            native_smoke_executed=False, queue_changed=False, hardware_changed=False,
            command='python -B scripts/recheck_gpu0_lifecycle_v3.py --token ' + args.token))
        additions = [ref(output), ref(__file__), ref('scripts/test_gpu0_lifecycle_v3.py')]
        for candidate, kind, _ in selected:
            candidate.update(update_resource(candidate, kind, available, observed_at, [t['id'] for t in tasks]))
            candidate['occupation_evidence'] = ref(output)
            candidate['evidence'].extend(e for e in additions if e not in candidate['evidence'])
        generated_at = tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at']
        previous = work.get('tick_attempt', {})
        if not (previous.get('tick_started_at') == generated_at
                and previous.get('kind') == 'native_smoke' and previous.get('result') == 'passed'):
            work['tick_attempt'] = dict(tick_started_at=generated_at,
                candidate_id=selected[0][0]['id'], kind='resource_check',
                result='passed' if available else 'blocked',
                command='python -B scripts/recheck_gpu0_lifecycle_v3.py --token ' + args.token,
                observation='Current lifecycle/source/freeze checks and real authorized GPU0 '
                    'occupation observed. No native smoke or efficacy claim; preserve live jobs.',
                evidence=[ref(output)])
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(evidence=ref(output), records=records, idle_authorized_gpus=available,
                             hardware=hardware, running_ids=[t['id'] for t in tasks]), indent=2))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
