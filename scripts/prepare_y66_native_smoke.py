"""Save CPU-tested Y66 and bind unexecuted Y65/Y66 native plans to that HEAD."""
import argparse
import ast
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from prepare_y66_peer_state_feedback import files
from check_ef_execution import validate_frozen
import closure
import experiment_first


def git(*args):
    return subprocess.check_output(['git', '-c', 'core.autocrlf=false', *args], cwd=ROOT, timeout=120)


def frozen_from(program):
    text = program.read_text(encoding='utf-8')
    frozen = ast.literal_eval(ast.parse(text).body[0].value)
    return frozen, text.partition('\n')[2]


def refuse_tampering(candidate, plan, frozen):
    for field in ('goal', 'source'):
        bad = copy.deepcopy(frozen)
        if field == 'goal': bad['goal_sha256'] = '0' * 64
        else: bad['variant']['python_files']['dec.py']['base64'] = 'Y2hhbmdlZA=='
        refused = False
        try: validate_frozen(candidate, plan, bad)
        except AssertionError: refused = True
        assert refused


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 240)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        server = monitor.load_servers(str(ROOT))['server-2']
        assert server['host'] == '192.168.1.163' and server['allowed_gpus'] == [0]
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y66')
        y65 = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y65')
        for item in (candidate, y65):
            assert item['stage'] == 'blocked' and item['blocker_kind'] == 'exclusive_gpu_slot'
            assert not item.get('admission_receipt')
        assert not candidate.get('code_commit') and not candidate.get('native_smoke_plan')
        manifest = tick.load(closure.artifact(ROOT, candidate['source_manifest']))
        assert manifest['complete'] and manifest['id'] == candidate['id'] and manifest['uncommitted']
        assert manifest['exact_files'] == [ref(p) for p in files()]
        cpu = tick.load(closure.artifact(ROOT, manifest['cpu_test']))
        for key in ('complete', 'implementation_test_passed', 'full_model_CPU_constructed',
                    'configured_TRAIN_data_collated', 'released_backbone_loaded',
                    'decoder_zero_anchor_all_stages_exact', 'decoder_feedback_gradient_nonzero',
                    'nonzero_feedback_changes_refinement'):
            assert cpu[key], key
        assert cpu['id'] == candidate['id'] and not cpu['CUDA_initialized'] and not cpu['native_smoke_passed']
        original = next(e for e in candidate['evidence'] if e['path'].startswith('pipeline/checks/y66_cpu_')
                        and e['path'].endswith('_program.py'))
        payload = ast.literal_eval(ast.parse(closure.artifact(ROOT, original).read_text(encoding='utf-8')).body[0].value)
        assert payload['goal_sha256'] == cpu['goal_sha256'] == ref('pipeline/goal.json')['sha256']
        assert hashlib.sha256((ROOT / 'AdaRES/tools/train.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest() == payload['train_source_lf_sha256']
        reference_path = ROOT / ('pipeline/analyses/y66_native_reference_cpu_' + args.token + '.json')
        reference = tick.load(reference_path)
        assert reference['complete'] and reference['id'] == candidate['id'] and reference['native_reference_CPU_checked']
        assert len(reference['cases']) == 8 and not reference['CUDA_initialized'] and not reference['native_smoke_passed']
        assert all(v['scalar_reference_passed'] and v['actual_component_matches_native_reference'] for v in reference['cases'])
        test_log = ROOT / ('pipeline/checks/y66_native_plan_guards_' + args.token + '.log')
        assert not test_log.exists()
        tested = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'scripts',
            '-p', 'test_y66_native_smoke_plan.py', '-v'], cwd=ROOT, capture_output=True, timeout=60)
        test_log.write_bytes(tested.stdout + tested.stderr)
        assert tested.returncode == 0 and b'Ran 10 tests' in test_log.read_bytes()
        queue_before, running_before = ref('pipeline/queue.json'), ref('pipeline/running.json')
        queue = monitor.load_queue(str(ROOT))
        assert not any(t['status'] in ('waiting', 'claiming') for ts in queue['gpu_queues'].values() for t in ts)
        live = next(t for t in queue['gpu_queues']['server-1'] if t['id'] == 'AdaRES-Y64')
        assert live['status'] == 'running' and live['launch_verified']
        assert monitor.load_running(str(ROOT))['server-1']['0'] == live['id']
        closure.admission(ROOT, live)
        y64 = next(c for c in work['candidates'] if c['id'] == live['id'])
        y64_plan = tick.load(closure.artifact(ROOT, y64['native_smoke_plan']))
        y64_frozen, _ = frozen_from(closure.artifact(ROOT, y64_plan['program']))
        assert validate_frozen(y64, y64_plan, y64_frozen) == 7
        old65_ref = copy.deepcopy(y65['native_smoke_plan'])
        old65_plan = tick.load(closure.artifact(ROOT, old65_ref))
        old65_frozen, old65_body = frozen_from(closure.artifact(ROOT, old65_plan['program']))
        assert validate_frozen(y65, old65_plan, old65_frozen) == 7
        assert not old65_plan['native_executed']
        assert old65_body == (ROOT / 'scripts/probe_y65_native_smoke.py').read_text(encoding='utf-8')
        assert not git('diff', '--cached', '--name-only').strip(), 'Preserve staged user work'
        parent = git('rev-parse', 'HEAD').decode().strip()
        assert parent == manifest['parent_head'] == y65['code_commit'] == live['code_commit']
        assert parent == '9706a8c36bead4459668ab66a3d1abc39344ef8b'
        paths = [p.relative_to(ROOT).as_posix() for p in files()]
        for path in paths:
            value = payload['variant']['config_data'] if path.endswith('.yaml') else payload['variant']['python_files'][Path(path).name]
            assert ref(path)['sha256'] == value['sha256']
        authorization = experiment_first.effective(ROOT)['authorization']
        closure.artifact(ROOT, authorization)
        base = ROOT / ('pipeline/checks/y66_native_' + args.token)
        version_path = base.with_name(base.name + '_version.json')
        program_path = base.with_name(base.name + '_program.py')
        plan_path = base.with_name(base.name + '_plan.json')
        y65_base = ROOT / ('pipeline/checks/y65_native_current_' + args.token)
        y65_program_path = y65_base.with_name(y65_base.name + '_program.py')
        y65_plan_path = y65_base.with_name(y65_base.name + '_plan.json')
        inputs_path = base.with_name(base.name + '_inputs.json')
        assert not any(p.exists() for p in (version_path, program_path, plan_path,
            y65_program_path, y65_plan_path, inputs_path))
        tick.save(inputs_path, dict(id=candidate['id'], parent_commit=parent,
            exact_files=[ref(p) for p in files()], original_cpu_manifest=candidate['source_manifest'],
            original_y65_plan=old65_ref, original_y65_program=old65_plan['program'],
            Y64_live_task_sha256=queue_before, running_registry=running_before,
            authorization=authorization, native_executed=False, admitted=False))
        git('add', '--', *paths)
        assert set(git('diff', '--cached', '--name-only').decode().splitlines()) == set(paths)
        for path in paths: assert hashlib.sha256(git('show', ':' + path)).hexdigest() == ref(path)['sha256']
        git('commit', '-m', 'Add Y66 peer-state refinement feedback experiment')
        commit = git('rev-parse', 'HEAD').decode().strip()
        assert git('rev-parse', commit + '^').decode().strip() == parent
        assert set(git('diff-tree', '--no-commit-id', '--name-only', '-r', commit).decode().splitlines()) == set(paths)
        assert not git('diff', '--cached', '--name-only').strip()
        frozen = dict(variant=payload['variant'], code_commit=commit,
            goal_sha256=payload['goal_sha256'], train_source_lf_sha256=payload['train_source_lf_sha256'])
        checked = dict(candidate, code_commit=commit)
        plan = dict(complete=True, id=candidate['id'], config=candidate['config'], profile='experiment_first',
            code_commit=commit, goal_sha256=frozen['goal_sha256'],
            required_resource='Fresh exclusive authorized idle server-2 GPU0, current lease and exact '
                'hostname, no compute PID, memory<1024MiB/util<10, approved 280W unchanged. Preserve Y64/Y57.',
            planned_command='python -B scripts/run_experiment_first_probe.py y66_smoke ' +
                program_path.relative_to(ROOT).as_posix() + ' --token <CURRENT_TOKEN> --seconds 600 --gpu --gpu-id 0',
            checks='Native YAML/TRAIN/tools.train two finite loss/backward/optimizer steps; six peer '
                'feedback/seven shared indicators per step; prior native tensor identity; independent '
                'per-point union/IoU/raw-peer context for rows0/64/127; actual refinement consumption; '
                'exact zero first-step anchor, nonzero second-step projection/response and finite '
                'nonzero projection gradients, finite query/mask/input gradients, native modes and peak memory.',
            Python38_compile=True, CPU_guard_tests_passed=10, CPU_reference_cases_passed=8,
            native_executed=False, CPU_checks_do_not_replace_native_backward=True)
        assert validate_frozen(checked, plan, frozen) == 7
        refuse_tampering(checked, plan, frozen)
        body = ROOT / 'scripts/probe_y66_native_smoke.py'
        program = 'FROZEN=' + repr(frozen) + '\n' + body.read_text(encoding='utf-8')
        ast.parse(program, feature_version=(3, 8))
        program_path.write_text(program, encoding='utf-8', newline='\n')
        plan['program'] = ref(program_path)
        tick.save(plan_path, plan)
        # Only the unexecuted Y65 launch identity advances; preserve the entire old plan/program.
        frozen65 = copy.deepcopy(old65_frozen)
        frozen65['code_commit'] = commit
        plan65 = copy.deepcopy(old65_plan)
        plan65.update(code_commit=commit, previous_plan=old65_ref, native_executed=False,
            required_resource=plan['required_resource'],
            planned_command='python -B scripts/run_experiment_first_probe.py y65_smoke ' +
                y65_program_path.relative_to(ROOT).as_posix() + ' --token <CURRENT_TOKEN> --seconds 600 --gpu --gpu-id 0')
        assert {k: v for k, v in frozen65.items() if k != 'code_commit'} == {
            k: v for k, v in old65_frozen.items() if k != 'code_commit'}
        assert validate_frozen(dict(y65, code_commit=commit), plan65, frozen65) == 7
        refuse_tampering(dict(y65, code_commit=commit), plan65, frozen65)
        y65_program_path.write_text('FROZEN=' + repr(frozen65) + '\n' + old65_body, encoding='utf-8', newline='\n')
        assert frozen_from(y65_program_path)[1] == old65_body
        ast.parse(y65_program_path.read_text(encoding='utf-8'), feature_version=(3, 8))
        plan65['program'] = ref(y65_program_path)
        tick.save(y65_plan_path, plan65)
        closure.admission(ROOT, live)
        assert validate_frozen(y64, y64_plan, y64_frozen) == 7
        assert ref('pipeline/queue.json') == queue_before and ref('pipeline/running.json') == running_before
        assert git('rev-parse', 'HEAD').decode().strip() == commit
        version = dict(complete=True, observed_at=datetime.now().astimezone().isoformat(),
            id=candidate['id'], parent_commit=parent, code_commit=commit,
            exact_files=[ref(p) for p in files()], original_cpu_manifest=candidate['source_manifest'],
            original_cpu_program=original, cpu_test=manifest['cpu_test'], authorization=authorization,
            source_unchanged_since_CPU_tests=True, CPU_guard_tests_passed=10,
            CPU_reference_cases_passed=8, guard_log=ref(test_log), reference_result=ref(reference_path),
            stale_goal_control_refused=True, tampered_source_control_refused=True,
            native_smoke_plan=ref(plan_path), Y65_previous_plan=old65_ref,
            Y65_current_plan=ref(y65_plan_path), Y65_body_and_payload_unchanged=True,
            native_smoke_passed=False, admitted=False, production_sync=False, pushed=False,
            queue_changed=False, running_changed=False, Y64_admission_and_version_preserved=True,
            hardware_changed=False)
        tick.save(version_path, version)
        additions = [ref(version_path), ref(plan_path), ref(program_path), ref(body), ref(__file__),
            ref(test_log), ref('scripts/test_y66_native_smoke_plan.py'), ref(inputs_path),
            ref(reference_path), ref(reference_path.with_suffix('.log')),
            ref('scripts/check_y66_native_reference_cpu.py'),
            ref('pipeline/checks/y66_native_reference_cpu_' + args.token + '_program.py')]
        candidate['evidence'].extend(additions)
        candidate.update(code_commit=commit, native_smoke_plan=ref(plan_path), resource_work_kind='native_smoke',
            next_action='Validate current goal/frozen seven-file plan, then on fresh exclusive authorized '
                'idle server-2 GPU0 execute planned native YAML/TRAIN two-step smoke with current token. '
                'Verify six feedback/seven indicators, independent point-mass peer context, exact native '
                'tensor correspondence/actual refinement consumption, finite nonzero projection gradient '
                'and second-step response. Only actual native PASS plus original frozen full post-training '
                'protocol permit admission and same-tick dispatch. Preserve Y64 marker9706, Y57 and 280W.')
        y65['evidence'].extend([ref(version_path), ref(y65_plan_path), ref(y65_program_path), ref(__file__)])
        y65.update(code_commit=commit, native_smoke_plan=ref(y65_plan_path),
            next_action='Validate the current HEAD-bound plan and run its exact native YAML/TRAIN '
                'two-step smoke on fresh exclusive idle authorized server-2 GPU0 with explicit --gpu '
                '--gpu-id0. Original payload and diagnostic body unchanged; old plan remains immutable. '
                'Only actual native PASS plus frozen full post-training protocol permit admission/dispatch. '
                'Preserve Y64 running marker9706, Y57 pause, 280W and foreign jobs.')
        work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
            candidate_id=candidate['id'], kind='implementation_test', result='passed',
            command='python -B scripts/prepare_y66_native_smoke.py --token ' + args.token,
            observation='Y66 exact CPU-tested seven-file version saved;10 native guards and8 independent '
                'CPU formula cases passed, stale-goal/source controls refused. Y65 unexecuted identical '
                'payload/body bound to new HEAD; Y64 preserved. No native GPU execution or efficacy claim.',
            evidence=additions)
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(version=ref(version_path), Y66_plan=ref(plan_path),
            Y65_current_plan=ref(y65_plan_path), code_commit=commit, scoped_files=7,
            native_executed=False, admitted=False, production_sync=False,
            Y64_admission_and_version_preserved=True), indent=2))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
