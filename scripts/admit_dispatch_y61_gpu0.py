"""Admit Y61 only from its real native smoke; use the existing gated launcher."""

import argparse
import ast
import copy
import csv
from datetime import datetime
import io
import json
import math
import shlex
import unittest

import yaml

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from check_ef_execution import idle_gpus, validate_frozen
from dispatch_prepared_candidates import snapshot
import closure
import experiment_first


CID = 'AdaRES-Y61'


def validate_result(raw, candidate, frozen):
    assert raw['complete'] and raw['id'] == CID == candidate['id']
    assert raw['config'] == candidate['config'] == frozen['variant']['config']
    assert raw['code_commit'] == candidate['code_commit'] == frozen['code_commit']
    assert raw['goal_sha256'] == frozen['goal_sha256']
    assert raw['native_YAML_sha256'] == frozen['variant']['config_data']['sha256']
    assert raw['steps'] == raw['optimizer_steps'] == 2
    assert len(raw['losses']) == 2 and all(math.isfinite(v) for v in raw['losses'])
    assert raw['native_entry'] == 'tools/train.py::train' and raw['split'] == 'train'
    assert raw['native_data_loaded'] and raw['backward_finite'] and raw['intended_path_observed']
    assert raw['validation_used'] is False and raw['model_weights_saved'] is False
    assert raw['shared_head_calls'] == 14
    assert raw['query_norm_input_requires_grad'] == [True, False] * 14
    assert len(raw['gradient_checks']) == 2
    for check in raw['gradient_checks']:
        assert check['all_finite'] and check['indicator_tensors'] == 10
        assert check['indicator_calls'] == 7
        assert check['query_norm_input_requires_grad'] == [True, False] * 7
        assert check['query_and_mask_input_gradients_finite']
        assert check['trainable_query_norm_gradient_nonzero']
    assert 0 < raw['peak_reserved_bytes'] < 24 * 2**30
    assert len(raw['batches']) == 2
    assert all(len(b['scan_ids']) == 2 and b['expressions'] > 0 and b['points'] > 0
               for b in raw['batches'])
    assert raw['data_source']['dataset_class'] and raw['data_source']['config_data_root']


def exact_candidate():
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == CID)
    assert candidate['stage'] in ('blocked', 'validation')
    assert candidate.get('blocker_kind') in (None, 'exclusive_gpu_slot')
    assert not candidate.get('admission_receipt') and not candidate.get('revision_history')
    assert candidate['code_commit'] == closure.git(ROOT, 'rev-parse', 'HEAD')
    plan = tick.load(closure.artifact(ROOT, candidate['native_smoke_plan']))
    program = closure.artifact(ROOT, plan['program'])
    frozen = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
    assert validate_frozen(candidate, plan, frozen) == 7
    return work, candidate, plan, program, frozen


def protected_queue():
    server = monitor.load_servers(str(ROOT))['server-2']
    assert server['host'] == '192.168.1.163' and server['allowed_gpus'] == [0]
    assert server['max_tasks_per_gpu'] == 1
    queue = monitor.load_queue(str(ROOT))
    tasks = queue['gpu_queues']['server-2']
    paused = next(t for t in tasks if t['id'] == 'AdaRES-Y57')
    assert paused['status'] == 'paused' and paused['dispatch_hold']
    return server, queue, tasks


def require_idle(server):
    live = snapshot(server, [], [])
    assert live['gpu_query_rc'] == live['compute_query_rc'] == 0
    inventory = {int(row[0]): row for row in csv.reader(io.StringIO(live['authorized_device_inventory']))
                 if int(row[0]) == 0}
    assert idle_gpus(inventory, list(csv.reader(io.StringIO(live['compute']))), set()) == [0]
    command = "import subprocess;assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60';print(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=power.limit','--format=csv,noheader,nounits']).decode().strip())"
    power = monitor.ssh(server['host'], server['user'],
        shlex.quote(server['python']) + ' -B -c ' + shlex.quote(command), timeout=30)
    assert float(power.strip()) == 280
    return live


def check(token):
    work, candidate, plan, program, frozen = exact_candidate()
    _, _, tasks = protected_queue()
    assert monitor.load_experiment(str(ROOT), CID) is None
    assert not (ROOT / 'pipeline/admissions/AdaRES-Y61.json').exists()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'scripts'), pattern='test_y61_admission_handoff.py')
    tests = unittest.TextTestRunner(verbosity=2).run(suite)
    assert tests.wasSuccessful() and tests.testsRun > 0
    # A precheck can pass while admission remains blocked; never synthesize raw results.
    raw = ROOT / ('pipeline/analyses/y61_smoke_' + token + '.json')
    output = ROOT / ('pipeline/checks/y61_admission_handoff_' + token + '.json')
    assert not output.exists()
    tick.save(output, dict(complete=True, observed_at=datetime.now().astimezone().isoformat(),
        id=CID, exact_files=7, source_goal_commit_match=True, CPU_guard_tests=tests.testsRun,
        CPU_guard_tests_passed=True, real_current_tick_smoke_exists=raw.exists(),
        original_smoke_plan=candidate['native_smoke_plan'],
        running_ids=[t['id'] for t in tasks if t['status'] in ('running', 'claiming')],
        native_smoke_executed=False, admitted=False, queue_changed=False, production_sync=False,
        next_command='python -B scripts/admit_dispatch_y61_gpu0.py admit --token <CURRENT_TOKEN>',
        command='python -B scripts/admit_dispatch_y61_gpu0.py check --token ' + token,
        evidence=[ref(__file__), ref('scripts/test_y61_admission_handoff.py'), ref(program)],
        limitation='Precheck and negative controls only. Real GPU0 idle checks/native two-step '
                   'results/admission and increasing production steps remain required.'))
    candidate['admission_handoff_precheck'] = ref(output)
    candidate['evidence'].extend(e for e in [ref(output), ref(__file__),
        ref('scripts/test_y61_admission_handoff.py')] if e not in candidate['evidence'])
    candidate['next_action'] = ('On exclusive authorized idle GPU0, use the unchanged native_smoke_plan '
        'with current token and --gpu-id 0. Only after actual native two-step PASS, run '
        'scripts/admit_dispatch_y61_gpu0.py admit then dispatch under that lease. '
        'The precheck is not native backward or admission. Preserve Y60/Y57 and 280W.')
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(evidence=ref(output), tests=tests.testsRun, admitted=False,
                         native_smoke_executed=False, queue_changed=False), indent=2))


def admit(token):
    work, candidate, plan, program, frozen = exact_candidate()
    server, queue, tasks = protected_queue()
    assert not any(t['status'] in ('running', 'claiming', 'waiting') for t in tasks)
    assert monitor.load_experiment(str(ROOT), 'AdaRES-Y60')['status'] == 'done_complete'
    assert monitor.load_experiment(str(ROOT), CID) is None
    assert candidate['stage'] == 'validation'
    view = tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')
    assert view['queue']['candidate_budget'] >= 1
    raw_path = ROOT / ('pipeline/analyses/y61_smoke_' + token + '.json')
    raw = tick.load(raw_path)
    validate_result(raw, candidate, frozen)
    assert raw_path.with_suffix('.log').is_file()
    idle = require_idle(server)
    cfg = yaml.safe_load((ROOT / 'AdaRES' / candidate['config']).read_text(encoding='utf-8'))
    assert cfg['train']['seed'] == 1999 and cfg['train']['epochs'] == 30
    assert cfg['dataloader']['train']['batch_size'] == 2
    assert cfg['model']['sampling_module']['num_proposal'] == 128
    prefix = 'pipeline/checks/' + CID
    smoke_path = ROOT / (prefix + '_native_smoke_' + token + '.json')
    freeze_path = ROOT / (prefix + '_post_training_freeze_' + token + '.json')
    receipt_path = ROOT / ('pipeline/admissions/' + CID + '.json')
    assert not any(p.exists() for p in (smoke_path, freeze_path, receipt_path))
    command = ('python -B scripts/run_experiment_first_probe.py y61_smoke ' + ref(program)['path'] +
               ' --token ' + token + ' --seconds 600 --gpu --gpu-id 0')
    smoke = copy.deepcopy(raw)
    smoke.update(command=command, physical_gpu=0,
        admission_resource_check=dict(observed_at=datetime.now().astimezone().isoformat(),
            live=idle, power_limit_watts=280, required_exclusive_gpu=0),
        evidence=[ref(raw_path), ref(raw_path.with_suffix('.log')), ref(program), candidate['native_smoke_plan']])
    tick.save(smoke_path, smoke)
    tick.save(freeze_path, dict(complete=True, id=CID, frozen_at=datetime.now().astimezone().isoformat(),
        goal=ref('pipeline/goal.json'), code_commit=candidate['code_commit'],
        training='Released SPFormer; seed1999; fixed30; native YAML losses/optimizer/scheduler; '
                 'one GPU; Q128; single argmax; no validation tuning.',
        evaluation=dict(split='Full frozen ScanRefer val9508 expressions/141 scenes',
            methods=[CID, 'AdaRES-Y36', 'AdaRES-W2', 'AdaRES-B1'], repeats=2, primary='FIRST',
            metrics=['mIoU', 'Acc@0.25', 'Acc@0.5', 'active_queries', 'latency', 'peak_memory'],
            scene_cluster_bootstrap=10000, bootstrap_seed=1999, TRAIN_size_cuts=[1487, 3325],
            material_harm='Prespecified target-size subgroup difference CI upper<0 versus Y36/W2; retain all negatives.',
            uncertainty='Crossing intervals or too few independent scenes are inconclusive; repeats do not measure training-seed uncertainty.'),
        resource_condition='GPU0 runtime cap280W; disclose for trained-cost comparison. Two batches '
                           'do not bound full-dataset memory; external GPU1 load is not controlled.',
        scientific_efficacy='Unknown until trained and full frozen evaluation is complete.'))
    receipt = dict(profile='experiment_first', id=CID, candidate_id=CID, config=candidate['config'],
        code_commit=candidate['code_commit'], goal_sha256=raw['goal_sha256'],
        authorization=experiment_first.effective(ROOT)['authorization'], native_smoke=ref(smoke_path),
        post_training_plan=ref(freeze_path), admitted_at=datetime.now().astimezone().isoformat(),
        gates={gate: dict(passed=True, evidence=[ref(smoke_path), ref(freeze_path), ref(program)])
               for gate in experiment_first.GATES})
    experiment_first.validate_smoke(ROOT, receipt)
    decision = monitor.log_decision(str(ROOT), 'next_batch', 'Y61是否通过真实原生两步检查并可接续GPU0？',
        '原生YAML/TRAIN两步及共享头/peer断开输入/有限梯度/资源检查通过，按预算准入Y61并尝试既有门控派发；Y57保持暂停。',
        '真实原始结果及日志/current goal/精确版本一致；Y60已完整终结且GPU0当前独占空闲，保持280W和固定30轮/seed1999，不预证效益。',
        ref(smoke_path)['path'])
    receipt['decision'] = ref(decision)
    tick.save(receipt_path, receipt)
    task = dict(id=CID, label=CID + ' fixed30', config=candidate['config'], status='waiting', gpu=0,
        depends_on=[], code_commit=candidate['code_commit'], admission_receipt=ref(receipt_path)['path'],
        scientific_candidate_id=CID, patience=30, min_epochs=30, fixed_epochs=30)
    closure.admission(ROOT, task)
    candidate.update(stage='admission', admission_receipt=task['admission_receipt'],
        native_smoke=ref(smoke_path), post_training_freeze=ref(freeze_path),
        next_action='Run this helper dispatch through the canonical gated monitor; require actual '
                    'runtime source and increasing native steps. Never relaunch an uncertain task.')
    additions = [ref(raw_path), ref(raw_path.with_suffix('.log')), ref(smoke_path), ref(freeze_path),
                 ref(receipt_path), ref(decision), ref(__file__)]
    candidate['evidence'].extend(e for e in additions if e not in candidate['evidence'])
    work['tick_attempt'] = dict(tick_started_at=view['generated_at'], candidate_id=CID,
        kind='native_smoke', result='passed', command=command,
        observation='Real native two-step loss/backward/shared-head path passed. Admitted; '
                    'production training still requires gated launch and verified increasing steps.',
        evidence=[ref(raw_path), ref(smoke_path)])
    tasks.append(task)
    tick.save(ROOT / 'pipeline/research_work.json', work)
    monitor.save_queue(str(ROOT), queue)
    print(json.dumps(dict(admitted=CID, production_training_verified=False)))


def dispatch(token):
    server, queue, tasks = protected_queue()
    assert not any(t['status'] in ('running', 'claiming') for t in tasks), 'Reconcile uncertain launches'
    assert [t['id'] for t in tasks if t['status'] == 'waiting'] == [CID]
    assert monitor.load_experiment(str(ROOT), 'AdaRES-Y60')['status'] == 'done_complete'
    task = next(t for t in tasks if t['id'] == CID)
    closure.admission(ROOT, task)
    before = require_idle(server)
    events = []
    monitor._process_queue(queue, 'server-2', server['host'], server['user'], str(ROOT), set(), {'0'}, events,
        max_tasks_per_gpu=1, allow_external_overlay=False, remote_code_root=server['remote_code_root'],
        current_tasks=0, max_tasks_total=1)
    queue = monitor.load_queue(str(ROOT))
    task = next(t for t in queue['gpu_queues']['server-2'] if t['id'] == CID)
    after = snapshot(server, [task] if task.get('pid') else [], [])
    output = ROOT / ('pipeline/checks/y61_gpu0_dispatch_' + token + '.json')
    assert not output.exists()
    tick.save(output, dict(complete=True, observed_at=datetime.now().astimezone().isoformat(),
        before=before, after=after, task=task, events=events, Y57_still_paused=True,
        GPU1_not_used=True, hardware_settings_changed=False))
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == CID)
    candidate.update(stage='training' if task['status'] == 'running' and task.get('launch_verified') else 'deployment',
        next_action='Observe exact Y61 fixed30 PID/log and native step growth. Reconcile any '
                    'uncertain launch; never retry blindly, migrate or reset optimizer/scheduler.')
    candidate['evidence'].append(ref(output))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(evidence=ref(output), status=task['status'], pid=task.get('pid'),
        launch_verified=task.get('launch_verified'), events=events), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('check', 'admit', 'dispatch'))
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 300)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        dict(check=check, admit=admit, dispatch=dispatch)[args.operation](args.token)
    finally:
        lock.close()


if __name__ == '__main__':
    main()
