"""Reconcile only the stale slot after Y60's already committed canonical archive."""

import argparse
import copy
from datetime import datetime
import json
import unittest

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from dispatch_prepared_candidates import snapshot
from finalize_y60_fixed30 import validate_proof
import closure


def reconciled_mapping(running, tasks, experiment):
    assert experiment['id'] == 'AdaRES-Y60' and experiment['status'] == 'done_complete'
    assert not any(t['id'] == 'AdaRES-Y60' for t in tasks)
    assert not any(t['status'] in ('running', 'claiming') and int(t.get('gpu', -1)) == 0 for t in tasks)
    paused = next(t for t in tasks if t['id'] == 'AdaRES-Y57')
    assert paused['status'] == 'paused' and paused['dispatch_hold']
    result = copy.deepcopy(running)
    assert result.get('server-2', {}).get('0') == 'AdaRES-Y60'
    del result['server-2']['0']
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 180)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        tests = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(
            str(ROOT / 'scripts'), pattern='test_y60_terminal_mapping.py'))
        assert tests.wasSuccessful() and tests.testsRun > 0
        server = monitor.load_servers(str(ROOT))['server-2']
        assert server['host'] == '192.168.1.163' and server['allowed_gpus'] == [0]
        experiment = monitor.load_experiment(str(ROOT), 'AdaRES-Y60')
        proof_path = closure.artifact(ROOT, experiment['terminal_evidence'])
        proof = tick.load(proof_path)
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y60')
        assert candidate['scientific_status'] == 'fixed30_complete_frozen_review_pending'
        handoff = tick.load(closure.artifact(ROOT, candidate['terminal_handoff_plan']))
        validate_proof(proof, handoff['payload'])
        assert experiment['terminal_checkpoint_metadata'] == proof['checkpoints']
        live = snapshot(server, [experiment], [])
        assert live['gpu_query_rc'] == live['compute_query_rc'] == 0
        assert not live['tasks']['AdaRES-Y60']['process_live']
        tasks = monitor.load_queue(str(ROOT))['gpu_queues']['server-2']
        before = monitor.load_running(str(ROOT))
        after = reconciled_mapping(before, tasks, experiment)
        output = ROOT / ('pipeline/checks/y60_terminal_mapping_' + args.token + '.json')
        assert not output.exists()
        record = dict(complete=True, observed_at=datetime.now().astimezone().isoformat(),
            original_command='python -B scripts/finalize_y60_fixed30.py finalize --token ' + args.token,
            original_exit_code=1,
            observed_failure='Final assertion found running.json server-2/0 still set to AdaRES-Y60.',
            failure_scope='local bookkeeping after canonical archive; not model, checkpoint or training failure',
            canonical_archive_committed=True, terminal_evidence=ref(proof_path),
            before=before, after=after, live=live, CPU_guard_tests=tests.testsRun,
            queue_changed=False, archive_replayed=False, remote_mutations=False,
            evidence=[ref(__file__), ref('scripts/test_y60_terminal_mapping.py')])
        tick.save(output, record)
        monitor.save_running(str(ROOT), after)
        assert monitor.load_running(str(ROOT)) == after
        candidate['evidence'].extend([ref(output), ref(__file__), ref('scripts/test_y60_terminal_mapping.py')])
        candidate['terminal_mapping_reconciliation'] = ref(output)
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(evidence=ref(output), canonical_status=experiment['status'],
            stale_slot_removed=True, archive_replayed=False, CPU_guard_tests=tests.testsRun), indent=2))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
