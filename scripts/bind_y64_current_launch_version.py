"""Bind identical tested Y64 bytes to current HEAD, retaining the earlier smoke."""
import argparse
import ast
import copy
from datetime import datetime
import hashlib
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
import closure


def bind(token):
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y64')
    assert candidate['stage'] == 'validation' and candidate['required_server'] == 'server-1'
    assert not candidate.get('admission_receipt')
    old_plan_ref = candidate['native_smoke_plan']
    assert old_plan_ref == candidate['native_smoke_plan_server1']
    old_plan = tick.load(closure.artifact(ROOT, old_plan_ref))
    old_program = closure.artifact(ROOT, old_plan['program'])
    source = old_program.read_text(encoding='utf-8')
    frozen = ast.literal_eval(ast.parse(source).body[0].value)
    assert validate_frozen(candidate, old_plan, frozen) == 7
    previous_result = candidate['native_smoke_result']
    raw = tick.load(closure.artifact(ROOT, previous_result))
    assert raw['complete'] and raw['steps'] == raw['optimizer_steps'] == 2
    assert raw['code_commit'] == candidate['code_commit'] == frozen['code_commit']
    head = closure.git(ROOT, 'rev-parse', 'HEAD')
    assert head == '9706a8c36bead4459668ab66a3d1abc39344ef8b' and head != candidate['code_commit']
    closure.git(ROOT, 'merge-base', '--is-ancestor', candidate['code_commit'], head)
    new_frozen = copy.deepcopy(frozen)
    new_frozen['code_commit'] = head
    new_plan = copy.deepcopy(old_plan)
    new_plan.update(code_commit=head, previous_plan=old_plan_ref,
        previous_native_result=previous_result, native_executed=False,
        planned_command='python -B scripts/run_y64_server1_native_probe.py AdaRES-Y64 --token <CURRENT_TOKEN> --attempt 2')
    checked = dict(candidate, code_commit=head)
    assert validate_frozen(checked, new_plan, new_frozen) == 7
    assert new_frozen['variant'] == frozen['variant']
    assert new_frozen['goal_sha256'] == frozen['goal_sha256']
    base = ROOT / ('pipeline/checks/y64_current_launch_version_' + token)
    program_path = base.with_name(base.name + '_program.py')
    plan_path = base.with_name(base.name + '_plan.json')
    binding_path = base.with_name(base.name + '_binding.json')
    assert not any(p.exists() for p in (program_path, plan_path, binding_path))
    body = source.split('\n', 1)[1]
    updated = 'FROZEN=' + repr(new_frozen) + '\n' + body
    ast.parse(updated, feature_version=(3, 8))
    assert updated.split('\n', 1)[1] == body
    program_path.write_text(updated, encoding='utf-8', newline='\n')
    new_plan['program'] = ref(program_path)
    tick.save(plan_path, new_plan)
    tick.save(binding_path, dict(complete=True, id=candidate['id'], token=token,
        observed_at=datetime.now().astimezone().isoformat(), previous_plan=old_plan_ref,
        previous_native_result=previous_result, previous_commit=frozen['code_commit'],
        code_commit=head, exact_files_unchanged=7, variant_and_native_YAML_identical=True,
        current_HEAD_files_match_original_tested_payload=True, previous_smoke_unchanged=True,
        execution_body_sha256=hashlib.sha256(body.encode()).hexdigest(),
        active_plan=ref(plan_path), native_result_reused_for_admission=False,
        fresh_native_steps_required=True, hardware_changed=False, queue_changed=False,
        production_sync=False, admitted=False))
    candidate.update(code_commit=head, native_smoke_plan=ref(plan_path),
        native_smoke_plan_server1=ref(plan_path), previous_native_smoke_result=previous_result,
        next_action=new_plan['planned_command'] + '; require actual current-version native PASS before admission/dispatch.')
    candidate.pop('native_smoke_result')
    candidate['evidence'].extend([ref(program_path), ref(plan_path), ref(binding_path), ref(__file__)])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(binding=ref(binding_path), active_plan=ref(plan_path), code_commit=head,
        exact_files_unchanged=7, actual_current_version_native_check_pending=True)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--token', required=True)
    a = p.parse_args()
    lease(a.token, 180)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        bind(a.token)
    finally:
        lock.close()
