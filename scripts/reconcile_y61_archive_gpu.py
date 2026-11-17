"""Correct only the verified Y61 archival GPU field without replaying archival."""

import argparse
import copy
import csv
from datetime import datetime
import io
import json
from pathlib import Path
import unittest

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
import closure

CID = 'AdaRES-Y61'


def corrected_experiment(before, expected_wrong_gpu):
    assert before['id'] == CID and before['status'] == 'done_complete'
    assert before['server'] == 'server-2' and str(before['gpu']) == str(expected_wrong_gpu)
    assert any(h.get('action') == 'launched' and str(h.get('gpu')) == '0'
               for h in before['history'])
    after = copy.deepcopy(before)
    after['gpu'] = '0'
    assert {k for k in before if before[k] != after[k]} == {'gpu'}
    return after


def corrected_ledger(raw, expected_wrong_gpu):
    lines = raw.splitlines(keepends=True)
    header = next(csv.reader([lines[0]]))
    id_index, gpu_index = header.index('id'), header.index('gpus')
    matches = []
    for index, line in enumerate(lines[1:], 1):
        row = next(csv.reader([line]))
        if row and row[id_index] == CID:
            matches.append((index, row))
    assert len(matches) == 1
    index, row = matches[0]
    assert row[gpu_index] == str(expected_wrong_gpu)
    row[gpu_index] = '0'
    ending = lines[index][len(lines[index].rstrip('\r\n')):]
    out = io.StringIO(newline='')
    csv.writer(out, lineterminator=ending).writerow(row)
    lines[index] = out.getvalue()
    return ''.join(lines)


def corrected_analysis(raw, expected_wrong_gpu):
    old = '  - Server: server-2 | GPU: ' + str(expected_wrong_gpu)
    assert raw.count(old) == 1 and '  - ID: ' + CID in raw
    return raw.replace(old, '  - Server: server-2 | GPU: 0', 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 120)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        suite = unittest.defaultTestLoader.discover(str(ROOT / 'scripts'), pattern='test_y61_archive_gpu.py')
        tests = unittest.TextTestRunner(verbosity=2).run(suite)
        assert tests.wasSuccessful() and tests.testsRun > 0
        protected = [ref('pipeline/queue.json'), ref('pipeline/running.json')]
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == CID)
        before = monitor.load_experiment(str(ROOT), CID)
        proof_path = closure.artifact(ROOT, before['terminal_evidence'])
        proof = tick.load(proof_path)
        assert proof['complete'] and proof['process_absent']
        assert proof['checkpoints'][0]['epoch'] == proof['checkpoints'][0]['scheduler_last_epoch'] == 30
        expected_wrong = str(30 - proof['checkpoints'][1]['epoch'])
        runtime_ref = next(e for e in candidate['evidence']
            if e['path'] == 'pipeline/checks/ef_gpu0_runtime_be3a6c9c4ae04d489329781059e74e60.json')
        runtime = tick.load(closure.artifact(ROOT, runtime_ref))
        assert runtime['id'] == CID and runtime['pid'] == before['pid'] == 78290
        assert runtime['assessment']['gpu0_compute_owner_verified']
        assert all(s['device']['index'] == 0 for s in runtime['observation']['samples'])
        after = corrected_experiment(before, expected_wrong)
        ledger_path = ROOT / 'pipeline/experiments.csv'
        ledger_before = ledger_path.read_bytes()
        ledger_after = corrected_ledger(ledger_before.decode('utf-8'), expected_wrong).encode('utf-8')
        analysis_path = ROOT / ('pipeline/analyses/adares_y61_peer_value_gradient_res_done_complete_' + before['archive_id'] + '.md')
        analysis_before = analysis_path.read_bytes()
        analysis_after = corrected_analysis(analysis_before.decode('utf-8'), expected_wrong).encode('utf-8')
        prefix = ROOT / ('pipeline/checks/y61_archive_gpu_repair_' + args.token)
        output = prefix.with_suffix('.json')
        assert not output.exists()
        snapshots = {}
        for name, value in (('before_experiment', before), ('after_experiment', after)):
            path = Path(str(prefix) + '_' + name + '.json')
            assert not path.exists()
            tick.save(path, value)
            snapshots[name] = ref(path)
        for name, value, suffix in (
                ('before_ledger', ledger_before, '.csv'), ('after_ledger', ledger_after, '.csv'),
                ('before_analysis', analysis_before, '.md'), ('after_analysis', analysis_after, '.md')):
            path = Path(str(prefix) + '_' + name + suffix)
            assert not path.exists()
            path.write_bytes(value)
            snapshots[name] = ref(path)
        monitor.save_experiment(str(ROOT), after)
        ledger_path.write_bytes(ledger_after)
        analysis_path.write_bytes(analysis_after)
        assert monitor.load_experiment(str(ROOT), CID) == after
        assert ledger_path.read_bytes() == ledger_after and analysis_path.read_bytes() == analysis_after
        assert protected == [ref('pipeline/queue.json'), ref('pipeline/running.json')]
        tick.save(output, dict(complete=True, id=CID, observed_at=datetime.now().astimezone().isoformat(),
            observed_wrong_gpu=expected_wrong, verified_gpu='0', terminal_proof=ref(proof_path),
            actual_runtime=runtime_ref, snapshots=snapshots, CPU_guard_tests=tests.testsRun,
            explanation='Canonical finalize_task accepts GPU after ep; the Y61 helper passed '
                '30-best_epoch there. Correct only this metadata using verified GPU0 run identity.',
            fields_changed=['experiment.gpu', 'Y61 CSV row.gpus', 'Y61 generated analysis GPU line'],
            archive_replayed=False, model_or_checkpoint_changed=False, queue_or_running_changed=False,
            remote_mutations=False, source=ref(__file__), tests=ref('scripts/test_y61_archive_gpu.py')))
        additions = [ref(output), ref(__file__), ref('scripts/test_y61_archive_gpu.py'),
            ref('pipeline/checks/y61_terminal_import_failure_' + args.token + '.json'),
            ref('pipeline/checks/y61_terminal_import_failure_' + args.token + '_program.py')]
        candidate['evidence'].extend(e for e in additions if e not in candidate['evidence'])
        candidate['archival_gpu_metadata_repair'] = ref(output)
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(evidence=ref(output), gpu='0', archive_replayed=False, CPU_guard_tests=tests.testsRun)))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
