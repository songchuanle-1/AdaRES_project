"""Rebind unchanged native capture to observed newline-equivalent parent bytes."""
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SKILL = Path('C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
sys.path.insert(0, str(SKILL))
import foreground_tick as tick
import pipeline_monitor as monitor

TOKEN = '89ffd8db36cf43ed8cfd75a68c77b4b9'


def ref(relative):
    path = (ROOT / relative).resolve()
    assert ROOT in path.parents and path.is_file()
    return dict(path=relative, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None, 'Pipeline writer busy'
    try:
        owner = tick.load(ROOT / 'pipeline/headless/.foreground.lockdir/owner.json')
        assert owner['token'] == TOKEN and owner['thread_id'] == os.environ['CODEX_THREAD_ID']
        assert (datetime.fromisoformat(owner['deadline_at']) - datetime.now(timezone.utc)).total_seconds() > 900
        for name, key in (('.tick_context.json', 'context_sha256'), ('.tick_context_state.next.json', 'pending_sha256')):
            assert ref('pipeline/headless/' + name)['sha256'] == owner[key]
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y52')
        assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'exclusive_gpu_slot'
        for row in candidate['evidence']:
            assert ref(row['path']) == row, row['path']
        old_program = 'pipeline/analyses/y52_native_matched_capture_program_5b882c0928b14a3bab6799e9d15a5bf4.py'
        old_text = (ROOT / old_program).read_text(encoding='utf-8')
        tree = ast.parse(old_text)
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'PAYLOAD' for t in n.targets))
        payload = ast.literal_eval(assignment.value)
        body_path = 'scripts/probe_y52_native_matched_capture.py'
        body = (ROOT / body_path).read_text(encoding='utf-8')
        assert old_text.endswith(body)
        inventory_path = 'pipeline/analyses/y52_source_y50_exit_89ff_' + TOKEN + '.json'
        inventory = tick.load(ROOT / inventory_path)
        assert inventory['complete'] and inventory['no_GPU'] and inventory['read_only']
        observed = {r['path']: r for r in inventory['source_rows']}
        comparisons = []
        for row in payload['parent_files']:
            local = (ROOT / 'AdaRES' / row['path']).read_bytes()
            assert hashlib.sha256(local).hexdigest() == row['sha256']
            remote = observed[row['path']]
            canonical = hashlib.sha256(local.replace(b'\r\n', b'\n')).hexdigest()
            assert canonical == remote['normalized_lf_sha256'], 'Not newline-equivalent'
            comparisons.append(dict(path=row['path'], frozen_local_raw=row['sha256'],
                observed_remote_raw=remote['raw_sha256'], canonical_lf_sha256=canonical))
            row['sha256'] = remote['raw_sha256']
        assert observed['tools/train.py']['raw_sha256'] == payload['train_source_sha256']
        resource_path = 'pipeline/analyses/y52_resources_exit_89ff_' + TOKEN + '.json'
        resource = tick.load(ROOT / resource_path)
        assert resource['complete'] and 1 in resource['idle_eligible_gpu_ids']
        failure_path = 'pipeline/analyses/y52_native_capture_89ff_' + TOKEN + '.log'
        assert 'AssertionError' in (ROOT / failure_path).read_text(encoding='utf-8')
        program_path = 'pipeline/analyses/y52_native_matched_capture_binding_v2_' + TOKEN + '.py'
        plan_path = 'pipeline/analyses/y52_native_binding_v2_prereg_' + TOKEN + '.json'
        assert not (ROOT / program_path).exists() and not (ROOT / plan_path).exists()
        source = 'PAYLOAD=' + repr(payload) + '\n' + body
        ast.parse(source, feature_version=(3, 8))
        compile(source, program_path, 'exec')
        plan = dict(complete=True, id='AdaRES-Y52', token=TOKEN,
            preregistered_at=datetime.now().astimezone().isoformat(),
            failure_scope='implementation: diagnostic source binding only',
            reason='Original strict raw-parent guard stopped before model construction because deployment uses different CRLF/LF encodings. Every parent canonical-LF SHA256 agrees exactly; no model source semantics changed.',
            change='Only payload parent_files raw SHA256 values are rebound to measured remote bytes. The runtime raw-byte assertions and capture body are unchanged. No relaxed source guard.',
            prediction='The same strict raw-byte checks will accept the measured newline-equivalent deployment, then run the SAME eight-scene/32-expression native TRAIN capture; every remaining assertion stays active.',
            comparisons=comparisons, preserved_failure=ref(failure_path),
            inventory=ref(inventory_path), original_program=ref(old_program), body=ref(body_path),
            resource=ref(resource_path), prototype=ref('pipeline/analyses/y52_prototype_bundle_4567_4567f5a09ecb4aa89df8f0c2932652df.json'),
            previous_design_cycle=copy.deepcopy(work.get('design_cycle')),
            frozen_test='Unchanged seed1999/released backbone/eight TRAIN scenes/32 expressions/batch_size2/actual sparse forward, FPS and final+auxiliary matches. No backward, optimizer, saved weights, efficacy or admission claim.',
            result='pending', candidate_model_revisions=0,
            limitation='AST digests from different Python versions are not compared. Canonical source-byte equality, exact observed raw binding and unchanged runtime assertions are required; actual GPU idleness and exclusive lease must still be checked again.')
        # All inputs and original hashes are verified before persisting the repair plan.
        (ROOT / program_path).write_text(source, encoding='utf-8', newline='\n')
        plan['program'] = ref(program_path)
        tick.save(ROOT / plan_path, plan)
        for path in (resource_path, failure_path, inventory_path,
                     'scripts/probe_y52_source_y50_exit_89ff.py',
                     'scripts/prepare_y52_native_binding_v2_89ff.py', program_path, plan_path):
            if path not in {e['path'] for e in candidate['evidence']}:
                candidate['evidence'].append(ref(path))
        candidate['previous_resource_blocker'] = {k: candidate[k] for k in
            ('blocker', 'blocker_kind', 'required_server', 'retry_after', 'unblock_condition', 'last_resource_check_at') if k in candidate}
        candidate['stage'] = 'validation'
        candidate['failure_scope'] = 'implementation'
        candidate['next_action'] = 'Run SAME frozen native matched capture with preregistered strict source-binding repair on freshly idle authorized GPU1 under exclusive lease. Then continue paired backward/shared-query, optimizer, full restore and resource controls; none are PASS or admission yet.'
        for key in ('blocker', 'blocker_kind', 'retry_after', 'unblock_condition'):
            candidate.pop(key, None)
        candidate.setdefault('diagnostic_revision_history', []).append(dict(
            preregistered_at=plan['preregistered_at'], scope='source_binding_not_model',
            change=plan['change'], prediction=plan['prediction'], evidence=[ref(plan_path)], result='pending'))
        work.pop('design_cycle', None)
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(program=ref(program_path), plan=ref(plan_path),
            preserved_failure=ref(failure_path), stage=candidate['stage'],
            model_revisions=0, runtime_assertions_unchanged=True), indent=2))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
