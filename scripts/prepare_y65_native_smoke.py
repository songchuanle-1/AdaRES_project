"""Version the CPU-tested Y65 bundle and freeze an unexecuted native GPU0 plan."""
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
from prepare_y65_confidence_feedback import files
from check_ef_execution import validate_frozen
import closure
import experiment_first


def git(*args):
    return subprocess.check_output(['git', '-c', 'core.autocrlf=false', *args], cwd=ROOT, timeout=120)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--token', required=True)
    args = p.parse_args()
    lease(args.token, 240)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        server = monitor.load_servers(str(ROOT))['server-2']
        assert server['host'] == '192.168.1.163' and server['allowed_gpus'] == [0]
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y65')
        assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'exclusive_gpu_slot'
        assert not candidate.get('code_commit') and not candidate.get('admission_receipt')
        manifest = tick.load(closure.artifact(ROOT, candidate['source_manifest']))
        assert manifest['complete'] and manifest['id'] == candidate['id'] and manifest['uncommitted']
        assert manifest['exact_files'] == [ref(path) for path in files()]
        cpu_path = closure.artifact(ROOT, manifest['cpu_test'])
        cpu = tick.load(cpu_path)
        assert cpu['complete'] and cpu['implementation_test_passed'] and cpu['id'] == candidate['id']
        assert cpu['full_model_CPU_constructed'] and cpu['configured_TRAIN_data_collated']
        assert cpu['decoder_zero_anchor_all_stages_exact'] and cpu['decoder_feedback_gradient_nonzero']
        assert not cpu['native_smoke_passed'] and not cpu['CUDA_initialized']
        originals = [e for e in candidate['evidence'] if e['path'].startswith('pipeline/checks/y65_cpu_')
                     and e['path'].endswith('_program.py')]
        assert len(originals) == 1
        original = closure.artifact(ROOT, originals[0])
        payload = ast.literal_eval(ast.parse(original.read_text(encoding='utf-8')).body[0].value)
        assert payload['goal_sha256'] == ref('pipeline/goal.json')['sha256']
        assert cpu['goal_sha256'] == payload['goal_sha256']
        assert payload['variant']['id'] == candidate['id']
        body = ROOT / 'scripts/probe_y65_native_smoke.py'
        ast.parse(body.read_text(encoding='utf-8'), feature_version=(3, 8))
        test_log = ROOT / ('pipeline/checks/y65_native_plan_guards_' + args.token + '.log')
        assert not test_log.exists()
        tested = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'scripts',
            '-p', 'test_y65_native_smoke_plan.py', '-v'], cwd=ROOT, capture_output=True, timeout=60)
        test_log.write_bytes(tested.stdout + tested.stderr)
        assert tested.returncode == 0 and b'Ran 8 tests' in test_log.read_bytes()
        assert hashlib.sha256((ROOT / 'AdaRES/tools/train.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest() == payload['train_source_lf_sha256']
        queue_before, running_before = ref('pipeline/queue.json'), ref('pipeline/running.json')
        queue = monitor.load_queue(str(ROOT))
        assert not any(t['status'] in ('waiting', 'claiming') for ts in queue['gpu_queues'].values() for t in ts)
        live = next(t for t in queue['gpu_queues']['server-1'] if t['id'] == 'AdaRES-Y63')
        assert live['status'] == 'running' and live['launch_verified']
        assert monitor.load_running(str(ROOT))['server-1']['0'] == live['id']
        closure.admission(ROOT, live)
        assert not git('diff', '--cached', '--name-only').strip(), 'Preserve staged user work'
        paths = [p.relative_to(ROOT).as_posix() for p in files()]
        assert len(paths) == 7
        parent = git('rev-parse', 'HEAD').decode().strip()
        assert parent == 'e27895f45b6f172454c50d40d1c0184c6aa8dfcb'
        assert manifest['parent_head'] == live['code_commit'] == '82fbd9fb1fc56da6afee5b145f59cfa533464707'
        git('merge-base', '--is-ancestor', live['code_commit'], parent)
        y64 = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y64')
        y64_plan = tick.load(closure.artifact(ROOT, y64['native_smoke_plan']))
        y64_program = closure.artifact(ROOT, y64_plan['program'])
        y64_frozen = ast.literal_eval(ast.parse(y64_program.read_text(encoding='utf-8')).body[0].value)
        assert validate_frozen(y64, y64_plan, y64_frozen) == 7
        for path in paths:
            value = payload['variant']['config_data'] if path.endswith('.yaml') else payload['variant']['python_files'][Path(path).name]
            assert ref(path)['sha256'] == value['sha256']
        base = ROOT / ('pipeline/checks/y65_native_' + args.token)
        version_path = base.with_name(base.name + '_version.json')
        program_path = base.with_name(base.name + '_program.py')
        plan_path = base.with_name(base.name + '_plan.json')
        assert not any(p.exists() for p in (version_path, program_path, plan_path))
        authorization = experiment_first.effective(ROOT)['authorization']
        closure.artifact(ROOT, authorization)
        git('add', '--', *paths)
        assert set(git('diff', '--cached', '--name-only').decode().splitlines()) == set(paths)
        for path in paths: assert hashlib.sha256(git('show', ':' + path)).hexdigest() == ref(path)['sha256']
        git('commit', '-m', 'Add Y65 confidence feedback refinement experiment')
        commit = git('rev-parse', 'HEAD').decode().strip()
        assert git('rev-parse', commit + '^').decode().strip() == parent
        assert set(git('diff-tree', '--no-commit-id', '--name-only', '-r', commit).decode().splitlines()) == set(paths)
        assert not git('diff', '--cached', '--name-only').strip()
        frozen = dict(variant=payload['variant'], code_commit=commit,
                      goal_sha256=payload['goal_sha256'], train_source_lf_sha256=payload['train_source_lf_sha256'])
        checked = dict(candidate, code_commit=commit)
        plan = dict(complete=True, id=candidate['id'], config=candidate['config'], profile='experiment_first',
            code_commit=commit, goal_sha256=frozen['goal_sha256'],
            required_resource='Fresh exclusive authorized idle server-2 GPU0, exact hostname/current lease, '
                'no compute PID, memory<1024MiB/util<10, keep approved 280W. Preserve running Y63 and paused Y57.',
            planned_command='python -B scripts/run_experiment_first_probe.py y65_smoke ' +
                program_path.relative_to(ROOT).as_posix() + ' --token <CURRENT_TOKEN> --seconds 600 --gpu --gpu-id 0',
            checks='Native YAML/TRAIN/tools.train two finite loss/backward/optimizer steps; '
                'seven shared indicators and six feedback calls each step; independent binary-logistic '
                'feedback formula; actual feedback tensor consumed by subsequent coarse attention; '
                'zero first-step anchor and nonzero second-step response; finite nonzero feedback '
                'direction gradients, input gradients, native layer modes and peak memory.',
            Python38_compile=True, CPU_guard_tests_passed=8, native_executed=False,
            CPU_checks_do_not_replace_native_backward=True)
        assert validate_frozen(checked, plan, frozen) == 7
        for field in ('goal', 'source'):
            tampered = copy.deepcopy(frozen)
            if field == 'goal': tampered['goal_sha256'] = '0' * 64
            else: tampered['variant']['python_files']['dec.py']['base64'] = 'Y2hhbmdlZA=='
            refused = False
            try: validate_frozen(checked, plan, tampered)
            except AssertionError: refused = True
            assert refused
        program = 'FROZEN=' + repr(frozen) + '\n' + body.read_text(encoding='utf-8')
        ast.parse(program, feature_version=(3, 8))
        program_path.write_text(program, encoding='utf-8', newline='\n')
        plan['program'] = ref(program_path)
        tick.save(plan_path, plan)
        assert ref('pipeline/queue.json') == queue_before and ref('pipeline/running.json') == running_before
        closure.admission(ROOT, live)
        assert validate_frozen(y64, y64_plan, y64_frozen) == 7
        version = dict(complete=True, observed_at=datetime.now().astimezone().isoformat(),
            id=candidate['id'], parent_commit=parent, code_commit=commit,
            exact_files=[ref(p) for p in files()], original_cpu_manifest=candidate['source_manifest'],
            original_cpu_program=originals[0], cpu_test=manifest['cpu_test'], authorization=authorization,
            source_unchanged_since_CPU_tests=True, CPU_guard_tests_passed=8,
            guard_log=ref(test_log), stale_goal_control_refused=True, tampered_source_control_refused=True,
            native_smoke_plan=ref(plan_path), native_smoke_passed=False, admitted=False,
            production_sync=False, pushed=False, queue_changed=False, running_changed=False,
            running_Y63_admission_preserved=True, current_Y64_plan_identity_preserved=True, hardware_changed=False)
        tick.save(version_path, version)
        additions = [ref(version_path), ref(plan_path), ref(program_path), ref(body), ref(__file__),
                     ref(test_log), ref('scripts/test_y65_native_smoke_plan.py')]
        candidate['evidence'].extend(additions)
        candidate.update(code_commit=commit, native_smoke_plan=ref(plan_path), resource_work_kind='native_smoke',
            next_action='On fresh exclusive idle authorized server-2 GPU0 run the frozen native YAML/TRAIN '
                'two-step plan with current token and explicit --gpu --gpu-id 0. Verify binary-logistic '
                'formula, six feedback/seven indicators per step, actual refinement consumption, '
                'finite nonzero feedback gradients and second-step response. '
                'Only actual native PASS and frozen full post-training protocol permit admission/dispatch. '
                'Preserve live Y63/source marker82f, Y57 pause, 280W and all foreign jobs.')
        work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
            candidate_id=candidate['id'], kind='implementation_test', result='passed',
            command='python -B scripts/prepare_y65_native_smoke.py --token ' + args.token,
            observation='CPU-tested exact seven-file version saved; 8 native execution guards and '
                'stale-goal/tampered-source negative controls passed. Native GPU steps unexecuted; '
                'no production sync, queue change or efficacy claim.', evidence=additions)
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(version=ref(version_path), plan=ref(plan_path), code_commit=commit,
            scoped_files=7, native_smoke_executed=False, admitted=False, production_sync=False,
            Y63_admission_preserved=True), indent=2))
    finally:
        lock.close()


if __name__ == '__main__': main()
