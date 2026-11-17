"""Preserve Y80's original evaluation while verified Y83 occupies its GPU0."""
import argparse
from datetime import datetime, timedelta
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
import closure


def park(token):
    lease(token, 180)
    writer = monitor.acquire_lock(str(ROOT)); assert writer is not None
    try:
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y80')
        owner = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y83')
        assert candidate['stage'] == 'validation' and candidate['resource_work_kind'] == 'frozen_evaluation'
        assert candidate['required_server'] == 'server-2' and candidate['required_gpu'] == 0
        assert monitor.load_experiment(str(ROOT), 'AdaRES-Y80')['status'] == 'done_complete'
        assert monitor.load_running(str(ROOT))['server-2']['0'] == 'AdaRES-Y83'
        task = next(t for t in monitor.load_queue(str(ROOT))['gpu_queues']['server-2'] if t['id'] == 'AdaRES-Y83')
        assert task['status'] == 'running' and task['launch_verified'] and task['pid'] == 20849 and str(task['gpu']) == '0'
        closure.admission(ROOT, task)
        observation = tick.load(closure.artifact(ROOT, owner['latest_training_observation']))
        assert observation['token'] == token and observation['native_progress_confirmed']
        for field in ('terminal_evidence', 'terminal_archive_binding', 'post_training_freeze'):
            closure.artifact(ROOT, candidate[field])
        freeze = tick.load(closure.artifact(ROOT, candidate['post_training_freeze']))
        resource = freeze['resource_condition']
        assert resource['server'] == 'server-2' and resource['gpu'] == 0 and resource['power_limit_watts'] == 280
        out = ROOT / ('pipeline/checks/y80_frozen_evaluation_hold_' + token + '.json'); assert not out.exists()
        tick.save(out, dict(complete=True, token=token, id='AdaRES-Y80', original_server='server-2',
            original_gpu=0, power_limit_watts=280, current_reserved_owner='AdaRES-Y83', current_task=task,
            native_owner_observation=owner['latest_training_observation'], terminal=candidate['terminal_evidence'],
            full_frozen_evaluation_executed=False, fresh_canonical_physical_occupation_required=True, helper=ref(__file__)))
        candidate.update(stage='blocked', blocked_from_stage='validation', blocker_kind='exclusive_gpu_slot',
            blocker='Original Y80 fixed30 and stable finite checkpoints verified and canonically archived. Original full frozen FIRST/cost requires server-2 GPU0/280W, now occupied by verified independent Y83. Preserve artifacts and all live training.',
            retry_after=(datetime.now().astimezone() + timedelta(hours=1)).isoformat(),
            unblock_condition='Fresh exclusive original server-2 GPU0/280W with no compute or canonical reservation; execute the original full frozen protocol.')
        candidate['evidence'].append(ref(out))
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(__import__('json').dumps(dict(hold=ref(out), full_frozen_evaluation_pending=True)))
    finally:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--token', required=True)
    park(parser.parse_args().token)
