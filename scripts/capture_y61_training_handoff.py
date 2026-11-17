"""Observe Y61 and preserve completed evaluations behind its occupied GPU0."""

import argparse
from datetime import datetime
import json
import shlex
import unittest

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from check_gpu0_single_card_trial import REMOTE, evaluate
from recheck_gpu0_lifecycle_v3 import update_resource
import closure


CID = 'AdaRES-Y61'
EVALUATIONS = ('AdaRES-Y56', 'AdaRES-Y58', 'AdaRES-Y59', 'AdaRES-Y60')


def validate_evaluation_state(candidate):
    assert candidate['id'] in EVALUATIONS
    assert candidate['scientific_status'] == 'fixed30_complete_frozen_review_pending'
    assert candidate['stage'] in ('validation', 'blocked')
    assert candidate.get('blocker_kind') in (None, 'exclusive_gpu_slot')


def validate_progress(task, observation, root):
    assert task['id'] == CID and task['status'] == 'running'
    assert task['launch_verified'] and int(task['gpu']) == 0
    assert task['fixed_epochs'] == task['min_epochs'] == task['patience'] == 30
    assessment = evaluate(observation, task, root)
    for key in ('process_identity_verified', 'native_progress_verified',
                'power_cap_preserved', 'gpu0_compute_owner_verified'):
        assert assessment[key], key
    assert not assessment['tail_error_observed']
    before, after = [sample['progress'] for sample in observation['samples']]
    assert before[0] == after[0] == 'Train' and before[2] == after[2] == '30'
    assert int(before[4]) == int(after[4]) > 2
    assert (int(after[1]), int(after[3])) > (int(before[1]), int(before[3]))
    return assessment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 180)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        tests = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(
            str(ROOT / 'scripts'), pattern='test_y61_training_handoff.py'))
        assert tests.wasSuccessful() and tests.testsRun > 0
        server = monitor.load_servers(str(ROOT))['server-2']
        assert server['host'] == '192.168.1.163' and server['allowed_gpus'] == [0]
        tasks = monitor.load_queue(str(ROOT))['gpu_queues']['server-2']
        paused = next(t for t in tasks if t['id'] == 'AdaRES-Y57')
        assert paused['status'] == 'paused' and paused['dispatch_hold']
        assert paused['resume_requires_user_authorization']
        active = [t for t in tasks if t['status'] in ('running', 'claiming', 'waiting')]
        assert len(active) == 1 and active[0]['id'] == CID
        task = active[0]
        closure.admission(ROOT, task)
        assert monitor.load_running(str(ROOT))['server-2']['0'] == CID
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == CID)
        assert candidate['stage'] == 'training'
        dispatch = ref('pipeline/checks/y61_gpu0_dispatch_' + args.token + '.json')
        assert tick.load(closure.artifact(ROOT, dispatch))['task']['launch_verified']
        completed = []
        for item in work['candidates']:
            if item['id'] not in EVALUATIONS:
                continue
            experiment = monitor.load_experiment(str(ROOT), item['id'])
            assert experiment['status'] == 'done_complete'
            validate_evaluation_state(item)
            terminal = closure.artifact(ROOT, experiment['terminal_evidence'])
            proof = tick.load(terminal)
            assert any(c['epoch'] == c['scheduler_last_epoch'] == 30
                       for c in proof['checkpoints'])
            freeze = closure.artifact(ROOT, item['post_training_freeze'])
            completed.append(dict(id=item['id'], terminal=ref(terminal), freeze=ref(freeze)))
        assert {c['id'] for c in completed} == set(EVALUATIONS)
        program = 'TASK=' + repr(dict(pid=task['pid'], config=task['config'], log=task['log'])) + '\n' + REMOTE
        observation = json.loads(monitor.ssh(server['host'], server['user'],
            shlex.quote(server['python']) + ' -B -c ' + shlex.quote(program), timeout=75))
        assessment = validate_progress(task, observation, server['remote_code_root'])
        now = datetime.now().astimezone()
        output = ROOT / ('pipeline/checks/y61_training_handoff_' + args.token + '.json')
        assert not output.exists()
        record = dict(complete=True, observed_at=now.isoformat(), id=CID, pid=task['pid'],
            observation=observation, assessment=assessment, completed_evaluations=completed,
            dispatch=dispatch, CPU_guard_tests=tests.testsRun,
            Y57_still_paused=True, remote_mutations=False, hardware_changed=False,
            queue_changed=False, scientific_promotion=False,
            previous_local_failure=dict(exit_code=1,
                command='python -B scripts/capture_y61_training_handoff.py --token ' + args.token,
                observation='The initial guard rejected validation-stage evaluation records '
                    'with retained exclusive_gpu_slot fields before any SSH observation or writes.',
                failure_scope='local handoff schema; not native smoke or production training',
                repair='Accept only validation/blocked with absent or exclusive_gpu_slot blocker. '
                    'Refresh completed evaluation holds from actual occupied GPU0; clear only '
                    'Y61 stale resource fields after verified live training progress.'),
            evidence=[ref(__file__), ref('scripts/test_y61_training_handoff.py'),
                      ref('scripts/check_gpu0_single_card_trial.py')],
            limitation='Short samples only, not hardware repair, full-run stability or trained efficacy.')
        lease(args.token, 60)
        tick.save(output, record)
        additions = [ref(output), ref(__file__), ref('scripts/test_y61_training_handoff.py')]
        for item in work['candidates']:
            if item['id'] in EVALUATIONS:
                item.update(update_resource(item, 'frozen_evaluation', [], now, [CID]))
                item['occupation_evidence'] = ref(output)
            elif item['id'] == CID:
                item['latest_runtime_observation'] = ref(output)
                for key in ('blocker', 'blocker_kind', 'blocked_from_stage', 'retry_after', 'unblock_condition'):
                    item.pop(key, None)
            else:
                continue
            item['evidence'].extend(e for e in additions if e not in item['evidence'])
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(evidence=ref(output), assessment=assessment,
            samples=[dict(progress=s['progress'], device=s['device']) for s in observation['samples']],
            parked_evaluation_ids=list(EVALUATIONS)), indent=2))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
