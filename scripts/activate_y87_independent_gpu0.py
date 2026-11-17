"""Retain identical Y87 bytes directly on Y85 and preserve Y86's failed guard."""
import argparse
import ast
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from advance_y82 import live, idle, unreserved
from check_ef_execution import validate_frozen
from y85_deployment import Y85
import closure

OLD = '053aea85a03be4244879c90362455c45327a7d64'
BRANCH = 'refs/heads/adares-y87-independent-gpu0'


def git(*args, env=None, data=None):
    return subprocess.check_output(['git', *args], cwd=ROOT, env=env, input=data, timeout=30)


def activate(token):
    lease(token, 600)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        work = tick.load(ROOT / 'pipeline/research_work.json')
        get = lambda cid: next(candidate for candidate in work['candidates'] if candidate['id'] == cid)
        candidate, failed = get('AdaRES-Y87'), get('AdaRES-Y86')
        assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'exclusive_gpu_slot'
        assert candidate['code_commit'] == OLD and not candidate.get('admission_receipt')
        assert not candidate.get('native_smoke_result') and candidate['implementation_test_passed']
        assert failed['stage'] == 'blocked' and failed['blocker_kind'] == 'numerical_control_failure'
        failure = closure.artifact(ROOT, failed['native_smoke_failure'])
        diagnostic_log = ROOT / ('pipeline/analyses/y86_gpu0_diagnosis_' + token + '.log')
        text = diagnostic_log.read_text(encoding='utf-8')
        line = next(line for line in text.splitlines() if line.startswith('AssertionError: {'))
        observed = ast.literal_eval(line.split(': ', 1)[1])
        assert observed['step'] == 1 and observed['stage'] == 4
        assert observed['weight_nonzero'] and observed['original_assertion_unchanged']
        assert observed['confidence_delta_max'] == 0 and observed['overlap_max'] > 0
        assert all(abs(value) > 0 for value in observed['weight'])
        server = monitor.load_servers(str(ROOT))['server-2']
        closure.validate_adares_server(ROOT, server)
        assert server['allowed_gpus'] == [0]
        unreserved(monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT)), 'server-2', 0)
        done = monitor.load_experiment(str(ROOT), 'AdaRES-Y85')
        assert done['status'] == 'done_complete' and str(done['gpu']) == '0'
        proof = tick.load(closure.artifact(ROOT, done['terminal_evidence']))
        assert proof['completed_epochs'] == {phase: list(range(1, 31)) for phase in ('Train', 'Val')}
        resource = live(server, 0)
        idle(resource, 280.)
        assert git('rev-parse', 'HEAD').decode().strip() == Y85
        original_manifest = candidate['source_manifest']
        version = tick.load(closure.artifact(ROOT, original_manifest))
        original_plan = candidate['native_smoke_plan']
        plan = tick.load(closure.artifact(ROOT, original_plan))
        original_program = closure.artifact(ROOT, plan['program'])
        source = original_program.read_text(encoding='utf-8')
        frozen = ast.literal_eval(ast.parse(source).body[0].value)
        assert validate_frozen(candidate, plan, frozen) == 7
        paths = [row[1] for row in version['exact_delta']]
        assert len(paths) == 7 and all('adares_y87_self_language_product' in path for path in paths)
        assert not git('ls-files', '--', *paths).strip()
        assert subprocess.run(['git', 'show-ref', '--verify', '--quiet', BRANCH], cwd=ROOT).returncode == 1
        base = ROOT / ('pipeline/checks/y87_independent_gpu0_' + token)
        manifest, index = base.with_suffix('.json'), base.with_suffix('.index')
        program = base.with_name(base.name + '_program.py')
        plan_path = base.with_name(base.name + '_plan.json')
        diagnosis = base.with_name(base.name + '_y86_diagnosis.json')
        assert not any(path.exists() for path in (manifest, index, program, plan_path, diagnosis))
        index_path = Path(git('rev-parse', '--git-path', 'index').decode().strip())
        if not index_path.is_absolute():
            index_path = ROOT / index_path
        before = hashlib.sha256(index_path.read_bytes()).hexdigest()
        env = dict(os.environ, GIT_INDEX_FILE=str(index))
        git('read-tree', Y85, env=env)
        for path in paths:
            raw = (ROOT / path).read_bytes()
            assert raw == git('show', OLD + ':' + path)
            blob = git('hash-object', '-w', '--stdin', data=raw).decode().strip()
            git('update-index', '--add', '--cacheinfo', '100644,' + blob + ',' + path, env=env)
        tree = git('write-tree', env=env).decode().strip()
        delta = [row.split('\t') for row in git(
            'diff-tree', '-r', '--no-commit-id', '--name-status', Y85, tree).decode().splitlines()]
        assert sorted(delta) == sorted([['A', path] for path in paths])
        commit = git('commit-tree', tree, '-p', Y85, '-m',
            'Retain tested Y87 independently without unadmitted Y86 source').decode().strip()
        git('update-ref', BRANCH, commit, '0' * 40)
        assert git('rev-parse', 'HEAD').decode().strip() == Y85
        assert hashlib.sha256(index_path.read_bytes()).hexdigest() == before
        new_version = copy.deepcopy(version)
        new_version.update(code_commit=commit, parent_commit=Y85, retained_branch=BRANCH,
            supersedes=original_manifest, exact_delta=delta, isolated_index=ref(index),
            model_and_YAML_bytes_unchanged=True, metadata_only_independent_parent=True,
            scientific_revision=False, helper=ref(__file__))
        tick.save(manifest, new_version)
        updated_frozen = copy.deepcopy(frozen)
        updated_frozen['code_commit'] = commit
        body = source.partition('\n')[2]
        updated_source = 'FROZEN=' + repr(updated_frozen) + '\n' + body
        assert ast.dump(ast.parse(source).body[1:][0]) == ast.dump(ast.parse(updated_source).body[1:][0])
        assert updated_source.partition('\n')[2] == body
        ast.parse(updated_source, feature_version=(3, 8))
        program.write_text(updated_source, encoding='utf-8', newline='\n')
        new_plan = copy.deepcopy(plan)
        new_plan.update(code_commit=commit, retained_branch=BRANCH, program=ref(program),
            supersedes=original_plan, source_manifest=ref(manifest),
            metadata_only_independent_parent=True, controls_unchanged=True)
        candidate.update(code_commit=commit, retained_branch=BRANCH, source_manifest=ref(manifest))
        assert validate_frozen(candidate, new_plan, updated_frozen) == 7
        tick.save(plan_path, new_plan)
        tick.save(diagnosis, dict(complete=True, token=token, id='AdaRES-Y86', failed_native=failure.relative_to(ROOT).as_posix(),
            raw_log=ref(diagnostic_log), observation=observed, original_assertion_preserved=True,
            result='failed', failure_scope='implementation', native_PASS=False, admitted=False,
            interpretation='The nonzero learned coefficient times tiny overlap is below float32 readout resolution at the observed stage. Frozen positive-response guard remains failed; no hypothesis or trained-efficacy rejection.',
            helper=ref(__file__)))
        failed['evidence'].extend([ref(diagnosis), ref(diagnostic_log)])
        failed.update(native_failure_diagnosis=ref(diagnosis),
            next_action='Preserve original failed native log and unchanged positive-response guard. Tiny overlap contribution rounded to zero at observed second-batch stage4; no admission or trained-efficacy claim. Only an evidence-backed bounded implementation revision may proceed; independent work continues.')
        candidate.update(stage='validation', required_server='server-2', required_gpu=0,
            native_smoke_plan=ref(plan_path), native_smoke_plan_server2_gpu0=ref(plan_path),
            independent_parent_binding=ref(manifest),
            next_action='Execute identical original native YAML/TRAIN controls on fresh server-2 GPU0/280W, then actual PASS only may admit/dispatch.')
        for key in ('blocker', 'blocker_kind', 'blocked_from_stage', 'retry_after', 'unblock_condition', 'occupation_evidence'):
            candidate.pop(key, None)
        candidate['evidence'].extend([original_manifest, original_plan, ref(manifest), ref(plan_path),
                                     ref(program), ref(index), ref(__file__)])
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(activation=ref(manifest), code_commit=commit, program=ref(program),
            plan=ref(plan_path), unchanged_controls=True, Y86_diagnosis=ref(diagnosis), native_executed=False)))
    finally:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    activate(parser.parse_args().token)
