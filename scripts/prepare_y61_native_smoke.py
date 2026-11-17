"""Version only the unchanged CPU-tested Y61 bundle and freeze its GPU0 plan."""

import argparse
import ast
import copy
from datetime import datetime
import hashlib
import json
import subprocess
import sys

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from prepare_y61_peer_values import files
from continue_y61_decoder_cpu import source_payload
from check_ef_execution import validate_frozen
import closure
import experiment_first


def git(*args):
    return subprocess.check_output(['git', '-c', 'core.autocrlf=false', *args], cwd=ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 180)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        server = monitor.load_servers(str(ROOT))['server-2']
        assert server['host'] == '192.168.1.163' and server['allowed_gpus'] == [0]
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y61')
        assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'exclusive_gpu_slot'
        assert not candidate.get('code_commit') and not candidate.get('admission_receipt')
        payload, original = source_payload(candidate)
        decoder_path = closure.artifact(ROOT, candidate['decoder_integration_check'])
        decoder = tick.load(decoder_path)
        assert decoder['complete'] and decoder['all_stage_backward_finite']
        assert decoder['all_stage_forward_bitwise_equal'] and not decoder['native_smoke_passed']
        assert decoder['goal_sha256'] == ref('pipeline/goal.json')['sha256']
        cpu_refs = [e for e in candidate['evidence'] if
                    e['path'].startswith('pipeline/analyses/y61_cpu_') and e['path'].endswith('.json')]
        assert len(cpu_refs) == 1
        cpu = tick.load(closure.artifact(ROOT, cpu_refs[0]))
        assert cpu['complete'] and cpu['implementation_test_passed']
        assert cpu['full_model_CPU_constructed'] and cpu['configured_TRAIN_data_collated']
        assert not cpu['native_smoke_passed']
        body = ROOT / 'scripts/probe_y61_native_smoke.py'
        ast.parse(body.read_text(encoding='utf-8'), feature_version=(3, 8))
        subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'scripts',
                        '-p', 'test_y61_native_smoke_plan.py', '-v'], cwd=ROOT, check=True)
        assert hashlib.sha256((ROOT / 'AdaRES/tools/train.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest() == payload['train_source_lf_sha256']
        assert len(list((ROOT / 'AdaRES/configs_song').rglob(files()[-1].name))) == 1
        queue_before = ref('pipeline/queue.json')
        running_before = ref('pipeline/running.json')
        queue = monitor.load_queue(str(ROOT))
        assert not any(t['status'] in ('waiting', 'claiming') for tasks in queue['gpu_queues'].values() for t in tasks), 'Preserve admitted waiting/claiming versions'
        assert not git('diff', '--cached', '--name-only').strip(), 'Do not mix staged user work'
        paths = [p.relative_to(ROOT).as_posix() for p in files()]
        assert len(paths) == 7
        parent = git('rev-parse', 'HEAD').decode().strip()
        version_path = ROOT / ('pipeline/checks/y61_version_' + args.token + '.json')
        program_path = ROOT / ('pipeline/checks/y61_smoke_' + args.token + '_program.py')
        plan_path = ROOT / ('pipeline/checks/y61_smoke_' + args.token + '_plan.json')
        assert not any(p.exists() for p in (version_path, program_path, plan_path))
        git('add', '--', *paths)
        assert set(git('diff', '--cached', '--name-only').decode().splitlines()) == set(paths)
        for path in paths:
            assert hashlib.sha256(git('show', ':' + path)).hexdigest() == ref(path)['sha256']
        git('commit', '-m', 'Add Y61 peer-value gradient isolation experiment')
        commit = git('rev-parse', 'HEAD').decode().strip()
        assert git('rev-parse', commit + '^').decode().strip() == parent
        assert set(git('diff-tree', '--no-commit-id', '--name-only', '-r', commit).decode().splitlines()) == set(paths)
        assert not git('diff', '--cached', '--name-only').strip()
        frozen = dict(variant=payload['variant'], code_commit=commit,
            goal_sha256=ref('pipeline/goal.json')['sha256'],
            train_source_lf_sha256=payload['train_source_lf_sha256'])
        checked = dict(candidate, code_commit=commit)
        plan = dict(complete=True, id=candidate['id'], profile='experiment_first',
            config=candidate['config'], code_commit=commit, goal_sha256=frozen['goal_sha256'],
            required_resource='Exclusive idle authorized server-2 GPU0, current foreground lease, reconciled queue and live compute checks; keep 280W.',
            planned_command='python -B scripts/run_experiment_first_probe.py y61_smoke ' +
                program_path.relative_to(ROOT).as_posix() + ' --token <CURRENT_TOKEN> --seconds 600 --gpu --gpu-id 0',
            checks='Native YAML/TRAIN/tools.train two finite loss/backward/optimizer steps; shared head seven calls per step; live focal/detached peer query_norm inputs, trainable affine and mask input gradients, GPU memory.',
            Python38_compile=True, CPU_guard_tests_passed=8, native_executed=False,
            CPU_checks_do_not_replace_native_backward=True)
        assert validate_frozen(checked, plan, frozen) == 7
        stale = copy.deepcopy(frozen)
        stale['goal_sha256'] = '0' * 64
        rejected_goal = False
        try:
            validate_frozen(checked, plan, stale)
        except AssertionError:
            rejected_goal = True
        assert rejected_goal
        tampered = copy.deepcopy(frozen)
        tampered['variant']['python_files']['dec.py']['base64'] = 'Y2hhbmdlZA=='
        rejected_source = False
        try:
            validate_frozen(checked, plan, tampered)
        except AssertionError:
            rejected_source = True
        assert rejected_source
        program = 'FROZEN=' + repr(frozen) + '\n' + body.read_text(encoding='utf-8')
        ast.parse(program, feature_version=(3, 8))
        program_path.write_text(program, encoding='utf-8', newline='\n')
        plan['program'] = ref(program_path)
        tick.save(plan_path, plan)
        assert ref('pipeline/queue.json') == queue_before and ref('pipeline/running.json') == running_before
        authorization = experiment_first.effective(ROOT)['authorization']
        closure.artifact(ROOT, authorization)
        version = dict(complete=True, observed_at=datetime.now().astimezone().isoformat(),
            id=candidate['id'], parent_commit=parent, code_commit=commit,
            exact_files=[ref(p) for p in files()], original_cpu_payload=original,
            cpu_test=cpu_refs[0], decoder_test=ref(decoder_path), authorization=authorization,
            source_unchanged_since_CPU_tests=True, CPU_guard_tests_passed=8,
            stale_goal_control_refused=True, tampered_source_control_refused=True,
            native_smoke_plan=ref(plan_path), native_smoke_passed=False, admitted=False,
            production_sync=False, pushed=False, queue_changed=False, running_changed=False,
            hardware_changed=False)
        tick.save(version_path, version)
        additions = [ref(version_path), ref(plan_path), ref(program_path), ref(body),
                     ref(__file__), ref('scripts/test_y61_native_smoke_plan.py')]
        candidate['evidence'].extend(e for e in additions if e not in candidate['evidence'])
        candidate.update(code_commit=commit, native_smoke_plan=ref(plan_path),
            blocked_from_stage='validation', resource_work_kind='native_smoke',
            next_action='On fresh verified exclusive idle authorized GPU0, run the frozen native YAML/TRAIN two-step plan with current token and explicit --gpu-id 0. Only real finite losses/backward/path/resource PASS permits receipt, admission and gated dispatch. Do not change or restart Y60, use GPU1, or confuse CPU guard tests with native smoke.')
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(json.dumps(dict(version=ref(version_path), plan=ref(plan_path), code_commit=commit,
            scoped_files=7, native_smoke_executed=False, admitted=False, queue_changed=False), indent=2))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
