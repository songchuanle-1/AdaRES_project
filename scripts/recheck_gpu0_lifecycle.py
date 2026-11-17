"""Recheck prepared work without sending completed training back to admission."""

import argparse
import ast
from copy import deepcopy
from datetime import datetime, timedelta
import json
import sys

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
from continue_y61_decoder_cpu import source_payload
import check_prepared_resources_gpu0_trial as legacy
import closure


IDS = ('AdaRES-Y56', 'AdaRES-Y58', 'AdaRES-Y60', 'AdaRES-Y61')
FIELDS = ('stage', 'blocked_from_stage', 'scientific_status', 'blocker_kind',
          'blocker', 'retry_after', 'next_action', 'occupation_evidence')


def role(candidate, experiment):
    if candidate.get('blocker_kind') != 'exclusive_gpu_slot':
        return None
    if candidate.get('scientific_status') == 'fixed30_complete_frozen_review_pending':
        assert experiment.get('status') == 'done_complete'
        assert candidate.get('post_training_freeze')
        return 'frozen_evaluation'
    if candidate['id'] == 'AdaRES-Y60':
        assert not candidate.get('admission_receipt') and candidate.get('native_smoke_plan')
        return 'native_smoke'
    if candidate['id'] == 'AdaRES-Y61':
        assert not candidate.get('code_commit') and candidate.get('decoder_integration_check')
        return 'version_then_native_smoke'
    raise AssertionError('Unknown resource-held lifecycle; do not rewrite it')


def disposition(candidate, kind, resource, resource_ref):
    out = deepcopy(candidate)
    busy = not resource['idle_authorized_gpus']
    live_ids = [t['id'] for t in resource['training'] if t['process_verified']]
    out['occupation_evidence'] = resource_ref
    out['required_server'] = 'server-2'
    out['resource_work_kind'] = kind
    if busy:
        out.update(stage='blocked', blocker_kind='exclusive_gpu_slot',
            blocker='Only authorized GPU0 is occupied or reserved; verified live task(s): ' +
                    ', '.join(live_ids) + '. GPU1 is excluded by the user single-card trial.',
            retry_after=(datetime.fromisoformat(resource['observed_at']) + timedelta(hours=1)).isoformat(),
            unblock_condition='Canonical queue/running opens GPU0 and fresh live checks confirm exclusive idle use.')
        out['blocked_from_stage'] = 'implementation' if kind == 'version_then_native_smoke' else 'validation'
    else:
        out['stage'] = 'implementation' if kind == 'version_then_native_smoke' else 'validation'
        for key in ('blocker', 'blocker_kind', 'blocked_from_stage', 'retry_after', 'unblock_condition'):
            out.pop(key, None)
    if kind == 'frozen_evaluation':
        out['next_action'] = ('Preserve completed fixed30 best/last checkpoints, optimizer/scheduler and '
            'the immutable post_training_freeze. Execute only the registered full frozen evaluation '
            'and trained-cost comparison on exclusive authorized GPU0. Do not restart training, '
            'repeat training admission, tune validation or promote from the log best.')
    elif kind == 'native_smoke':
        out['next_action'] = ('Prioritize the already committed Y60 and its unchanged native_smoke_plan. '
            'With a current lease and exclusive idle GPU0, execute native YAML/TRAIN two-step '
            'loss/backward, intended-path and peak checks. Only a real PASS permits admission '
            'and existing gated dispatch in that tick. No repeated routine authorization.')
    else:
        out['next_action'] = ('Preserve the seven tested Y61 files and CPU evidence. Y59 is already '
            'running, not waiting. Prioritize committed Y60 and preserve its HEAD-bound plan; '
            'then save only the exact tested Y61 bundle, bind and execute native YAML/TRAIN '
            'two-step smoke with explicit --gpu-id 0 on an exclusive idle authorized slot. '
            'CPU evidence does not substitute for native backward or admission. Do not use GPU1.')
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    parser.add_argument('--reuse-current-observation', action='store_true')
    args = parser.parse_args()
    lease(args.token, 180)
    if not args.reuse_current_observation:
        # The immutable collector is safe only for unadmitted Y60, never completed Y58.
        old_ids, old_argv = legacy.IDS, sys.argv
        try:
            legacy.IDS = ('AdaRES-Y60',)
            sys.argv = [legacy.__file__, '--token', args.token]
            legacy.main()
        finally:
            legacy.IDS, sys.argv = old_ids, old_argv
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        server = monitor.load_servers(str(ROOT))['server-2']
        assert server['host'] == '192.168.1.163' and server['allowed_gpus'] == [0]
        resource_path = ROOT / ('pipeline/checks/prepared_resources_' + args.token + '.json')
        resource = tick.load(resource_path)
        resource_ref = ref(resource_path)
        age = (datetime.now().astimezone() - datetime.fromisoformat(resource['observed_at'])).total_seconds()
        assert resource['complete'] and 0 <= age < 600
        assert not resource['native_smoke_executed'] and not resource['hardware_settings_changed']
        assert set(resource['idle_authorized_gpus']).issubset({0})
        assert resource['devices']['0']['power_limit_watts'] == 280
        work = tick.load(ROOT / 'pipeline/research_work.json')
        now = datetime.now().astimezone()
        records, updates = [], []
        for candidate in work['candidates']:
            if candidate['id'] not in IDS:
                continue
            experiment = monitor.load_experiment(str(ROOT), candidate['id']) or {}
            kind = role(candidate, experiment)
            if not kind:
                continue
            # The legacy collector may already have advanced this tick's retry.
            checked_now = candidate.get('occupation_evidence') == resource_ref
            if not checked_now and datetime.fromisoformat(candidate['retry_after']) > now:
                continue
            checks = {}
            if kind == 'frozen_evaluation':
                freeze = closure.artifact(ROOT, candidate['post_training_freeze'])
                terminal = closure.artifact(ROOT, experiment['terminal_evidence'])
                metadata = tick.load(terminal)['checkpoints']
                assert any(c['epoch'] == 30 and c['scheduler_last_epoch'] == 30 for c in metadata)
                checks.update(freeze=ref(freeze), terminal=ref(terminal),
                              canonical_status=experiment['status'], fixed30_checkpoint_verified=True)
            elif kind == 'native_smoke':
                plan = tick.load(closure.artifact(ROOT, candidate['native_smoke_plan']))
                program = closure.artifact(ROOT, plan['program'])
                frozen = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
                assert validate_frozen(candidate, plan, frozen) == 7
                checks.update(exact_files=7, current_source_goal_commit_match=True)
            else:
                payload, original = source_payload(candidate)
                decoder = tick.load(closure.artifact(ROOT, candidate['decoder_integration_check']))
                assert decoder['goal_sha256'] == ref('pipeline/goal.json')['sha256']
                assert decoder['complete'] and not decoder['native_smoke_passed']
                checks.update(exact_files=7, original_cpu_payload=original,
                              decoder_evidence=candidate['decoder_integration_check'], native_smoke_passed=False)
            updated = disposition(candidate, kind, resource, resource_ref)
            records.append(dict(id=candidate['id'], kind=kind, checks=checks,
                before={key: candidate.get(key) for key in FIELDS},
                after={key: updated.get(key) for key in FIELDS}))
            updates.append((candidate, updated))
        assert any(r['id'] == 'AdaRES-Y60' for r in records)
        output = ROOT / ('pipeline/checks/gpu0_lifecycle_' + args.token + '.json')
        assert not output.exists()
        tick.save(output, dict(complete=True, observed_at=datetime.now().astimezone().isoformat(),
            resource_evidence=resource_ref, records=records, legacy_collector=ref(legacy.__file__),
            preserved_legacy_observation=True, source_changed=False, queue_changed=False,
            hardware_changed=False, native_smoke_executed=False,
            command='python -B scripts/recheck_gpu0_lifecycle.py --token ' + args.token +
                    (' --reuse-current-observation' if args.reuse_current_observation else '')))
        additions = [resource_ref, ref(output), ref(__file__), ref('scripts/test_gpu0_lifecycle.py')]
        for candidate, updated in updates:
            candidate.update(updated)
            candidate['evidence'].extend(e for e in additions if e not in candidate['evidence'])
        work['tick_attempt'] = dict(
            tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
            candidate_id='AdaRES-Y60', kind='resource_check',
            result='blocked' if not resource['idle_authorized_gpus'] else 'passed',
            command=tick.load(output)['command'],
            observation='Current exact Y60 source/YAML/goal/commit verified; completed Y56/Y58 '
                'remain frozen-evaluation work, not new training admissions. Y61 tested source '
                'preserved. Actual authorized GPU0 is occupied/reserved; no GPU smoke run.'
                if not resource['idle_authorized_gpus'] else
                'Fresh exclusive-idle resource observation recorded; native smoke remains unrun.',
            evidence=[resource_ref, ref(output)])
        attempt = work['tick_attempt']
        y60 = next(c for c in work['candidates'] if c['id'] == attempt['candidate_id'])
        assert all(e in y60['evidence'] for e in attempt['evidence'])
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(evidence=ref(output), roles=[dict(id=r['id'], kind=r['kind']) for r in records],
                             native_smoke_executed=False, queue_changed=False), indent=2))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
