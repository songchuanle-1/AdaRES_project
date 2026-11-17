"""Rebind identical prepared source to the deployment HEAD before native tests."""

import argparse
import ast
import base64
import copy
from datetime import datetime
import hashlib
import subprocess

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
import closure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 180)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        work = tick.load(ROOT / 'pipeline/research_work.json')
        head = closure.git(ROOT, 'rev-parse', 'HEAD')
        rows = []
        for candidate in work['candidates']:
            if candidate['id'] not in ('AdaRES-Y58', 'AdaRES-Y59'):
                continue
            assert candidate['stage'] == 'validation'
            old_plan_ref = candidate['native_smoke_plan']
            old_plan = tick.load(closure.artifact(ROOT, old_plan_ref))
            old_program = closure.artifact(ROOT, old_plan['program'])
            source = old_program.read_text(encoding='utf-8')
            assignment = ast.parse(source).body[0]
            assert assignment.targets[0].id == 'FROZEN'
            frozen = ast.literal_eval(assignment.value)
            validate_frozen(candidate, old_plan, frozen)
            old_commit = candidate['code_commit']
            subprocess.run(['git', 'merge-base', '--is-ancestor', old_commit, head], cwd=ROOT, check=True)
            variant = frozen['variant']
            files = {'AdaRES/ipdn/model_paper5/' + variant['package'] + '/' + name: value
                     for name, value in variant['python_files'].items()}
            files['AdaRES/' + variant['config']] = variant['config_data']
            for path, value in files.items():
                blob = subprocess.check_output(['git', 'show', head + ':' + path], cwd=ROOT)
                assert hashlib.sha256(blob).hexdigest() == value['sha256']
                assert blob == base64.b64decode(value['base64'])
            stem = candidate['id'].lower().replace('-', '_')
            program = ROOT / ('pipeline/checks/' + stem + '_head_smoke_' + args.token + '_program.py')
            plan_path = program.with_name(program.name.replace('_program.py', '_plan.json'))
            binding = program.with_name(program.name.replace('_program.py', '_binding.json'))
            assert not any(p.exists() for p in (program, plan_path, binding))
            frozen['code_commit'] = head
            body = ''.join(source.splitlines(keepends=True)[assignment.end_lineno:])
            program.write_text('FROZEN=' + repr(frozen) + '\n' + body, encoding='utf-8', newline='\n')
            ast.parse(program.read_text(encoding='utf-8'), feature_version=(3, 8))
            tick.save(binding, dict(complete=True, id=candidate['id'],
                observed_at=datetime.now().astimezone().isoformat(), old_commit=old_commit, code_commit=head,
                original_plan=old_plan_ref, original_program=old_plan['program'],
                candidate_files=[ref(path) for path in files], exact_seven_files_unchanged=True,
                native_program_body_unchanged=True, only_frozen_commit_identity_changed=True,
                implementation_changed=False, tests_not_yet_executed=True))
            plan = copy.deepcopy(old_plan)
            plan.update(code_commit=head, program=ref(program), original_plan=old_plan_ref,
                        deployment_binding=ref(binding), native_executed=False)
            tick.save(plan_path, plan)
            candidate.setdefault('version_binding_history', []).append(dict(
                previous_code_commit=old_commit, previous_smoke_plan=old_plan_ref, binding=ref(binding)))
            candidate.update(code_commit=head, native_smoke_plan=ref(plan_path))
            for entry in (ref(binding), ref(program), ref(plan_path), ref(__file__)):
                candidate['evidence'].append(entry)
            validate_frozen(candidate, plan, frozen)
            rows.append(dict(id=candidate['id'], program=ref(program), binding=ref(binding), code_commit=head))
        assert len(rows) == 2
        tick.save(ROOT / 'pipeline/research_work.json', work)
        import json
        print(json.dumps(rows, indent=2))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
