"""Recheck admitted FIFO work and the immutable CPU-prepared Y61 while GPU0 is busy."""

import argparse
import ast
from datetime import datetime, timedelta
import json
import subprocess

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
from continue_y61_decoder_cpu import source_payload
import closure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 90)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        server = monitor.load_servers(str(ROOT))['server-2']
        assert server['host'] == '192.168.1.163' and server['allowed_gpus'] == [0]
        assert server['max_tasks_per_gpu'] == 1
        resource_path = ROOT / ('pipeline/checks/prepared_resources_' + args.token + '.json')
        resource = tick.load(resource_path)
        observed_at = datetime.fromisoformat(resource['observed_at'])
        age = (datetime.now().astimezone() - observed_at).total_seconds()
        assert 0 <= age < 900, 'Use a current measured resource check, not an old busy snapshot'
        assert resource['complete'] and not resource['idle_authorized_gpus']
        assert not resource['native_smoke_executed'] and resource['devices']['0']['power_limit_watts'] == 280
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidates = {c['id']: c for c in work['candidates'] if c['id'] in
                      ('AdaRES-Y58', 'AdaRES-Y59', 'AdaRES-Y61')}
        queue = monitor.load_queue(str(ROOT))
        tasks = {t['id']: t for t in queue['gpu_queues']['server-2'] if t['id'] in
                 ('AdaRES-Y58', 'AdaRES-Y59')}
        assert tasks['AdaRES-Y59']['status'] == 'waiting' and int(tasks['AdaRES-Y59']['gpu']) == 0
        assert not tasks['AdaRES-Y59'].get('dispatch_hold') and not tasks['AdaRES-Y59'].get('admission_error')
        assert any(t['id'] == 'AdaRES-Y58' and t['native_progress_verified'] for t in resource['training'])
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
        checks = []
        for name in ('AdaRES-Y58', 'AdaRES-Y59'):
            candidate, task = candidates[name], tasks[name]
            closure.admission(ROOT, task)
            plan = tick.load(closure.artifact(ROOT, candidate['native_smoke_plan']))
            program = closure.artifact(ROOT, plan['program'])
            frozen = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
            assert validate_frozen(candidate, plan, frozen) == 7
            assert head == task['code_commit']
            checks.append(dict(id=name, status=task['status'], receipt_verified=True,
                               exact_files=7, HEAD_matches_receipt=True))
        candidate = candidates['AdaRES-Y61']
        assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'exclusive_gpu_slot'
        payload, original = source_payload(candidate)
        checks.append(dict(id=candidate['id'], exact_files=len(payload['variant']['python_files']) + 1,
                           immutable_CPU_payload=original, source_identity_verified=True,
                           native_smoke_passed=False, source_committed=False))
        output = ROOT / ('pipeline/checks/ef_waiting_recheck_' + args.token + '.json')
        assert not output.exists()
        now = datetime.now().astimezone()
        tick.save(output, dict(complete=True, observed_at=now.isoformat(),
            resource_observed_at=resource['observed_at'], resource_evidence=ref(resource_path),
            command='python -B scripts/recheck_ef_waiting_gpu0.py --token ' + args.token,
            checks=checks, current_HEAD=head, result='blocked_GPU0_occupied',
            waiting_id='AdaRES-Y59', waiting_order_preserved=True, native_smoke_executed=False,
            queue_changed=False, remote_mutations=False, new_candidates=0,
            limitation='Current receipts/source/payload/resource checks only; no new CPU/GPU model '
                       'test, admission, launch or trained-efficacy result.'))
        evidence = ref(output)
        for name in ('AdaRES-Y59', 'AdaRES-Y61'):
            for entry in (evidence, ref(__file__)):
                if entry not in candidates[name]['evidence']:
                    candidates[name]['evidence'].append(entry)
        candidate.update(occupation_evidence=ref(resource_path),
            blocker='Y58 still occupies the sole authorized GPU0; admitted Y59 remains FIFO waiting. '
                    'Y61 tested source/payload is unchanged. Actual native GPU smoke is still unrun.',
            last_resource_check_at=resource['observed_at'],
            retry_after=(observed_at + timedelta(hours=1)).isoformat())
        work['tick_attempt'] = dict(
            tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
            candidate_id='AdaRES-Y59', kind='resource_check', result='blocked',
            command='python -B scripts/recheck_ef_waiting_gpu0.py --token ' + args.token,
            observation='Current Y58/Y59 receipts, seven files, YAML and HEAD match; immutable Y61 '
                        'CPU-tested source matches. Measured authorized GPU0 is occupied by growing '
                        'Y58; preserve admitted Y59 FIFO waiting without a stage or queue change.',
            evidence=[evidence, ref(resource_path)])
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(evidence=evidence, checks=checks, result='blocked_GPU0_occupied',
                              queue_changed=False), indent=2))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
