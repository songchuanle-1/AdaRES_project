"""Admit Y68 only from its real native smoke; use the existing gated launcher."""

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
import subprocess
import sys

import yaml

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from check_ef_execution import idle_gpus, validate_frozen
from dispatch_prepared_candidates import snapshot
import closure
import experiment_first


CID = 'AdaRES-Y68'


def validate_result(raw, candidate, frozen):
    assert raw['complete'] is True and raw['id'] == CID == candidate['id']
    assert raw['config'] == candidate['config'] == frozen['variant']['config']
    assert raw['code_commit'] == candidate['code_commit'] == frozen['code_commit']
    assert raw['goal_sha256'] == frozen['goal_sha256']
    assert raw['native_YAML_sha256'] == frozen['variant']['config_data']['sha256']
    assert raw['steps'] == raw['optimizer_steps'] == 2
    assert len(raw['losses']) == 2 and all(math.isfinite(v) for v in raw['losses'])
    assert raw['native_entry'] == 'tools/train.py::train' and raw['split'] == 'train'
    assert all(raw[k] is True for k in ('native_data_loaded','backward_finite','intended_path_observed'))
    assert raw['validation_used'] is False and raw['model_weights_saved'] is False
    assert raw['shared_head_calls'] == raw['prediction_consumption_calls'] == 14
    assert raw['modulation_calls'] == 2
    assert raw['query_norm_input_requires_grad'] == [True] * 14
    assert all(type(v) is bool for v in raw['query_norm_input_requires_grad'])
    assert len(raw['gradient_checks']) == len(raw['batches']) == 2
    from test_y68_native_smoke_plan import N
    for check in raw['gradient_checks']:
        assert check['all_finite'] is True and check['tensors'] > 0 and check['indicator_tensors'] == 10
        N['check_path_calls'](check['query_norm_input_requires_grad'], check['indicator_calls'],
            check['modulation_calls'], check['prediction_consumption_calls'])
        assert check['feature_and_language_gradients_finite'] is True
        for key in ('modulation_weight_gradient_max', 'parameter_update_max'):
            assert math.isfinite(check[key]) and check[key] > 0
    N['check_response_rows'](raw['modulation_paths'])
    assert 0 < raw['peak_reserved_bytes'] < 24 * 2**30
    assert all(len(b['scan_ids']) == 2 and b['expressions'] > 0 and b['points'] > 0 for b in raw['batches'])
    assert len({scene for b in raw['batches'] for scene in b['scan_ids']}) == 4
    assert raw['data_source']['dataset_class'] == 'ipdn.dataset.scannetv2_test.ScanNetDataset_sample_graph_edge'
    assert raw['data_source']['config_data_root'].rstrip('/') == '/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data'



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
    assert candidate.get('required_gpu') == plan.get('gpu') == frozen['runtime_physical_gpu'] == 0
    assert frozen['runtime_power_limit_watts'] == 280.0
    closure.artifact(ROOT, frozen['resource_authorization'])
    return work, candidate, plan, program, frozen


def protected_queue():
    server = monitor.load_servers(str(ROOT))['server-2']
    assert server['host'] == '192.168.1.163' and server['allowed_gpus'] == [0, 1]
    assert server['max_tasks_per_gpu'] == 1
    grant=tick.load(closure.artifact(ROOT,ref('pipeline/authorizations/server2_gpu1_20261029.json')))
    assert grant['authorization_granted'] and grant['newly_authorized_gpu'] == 1 and grant['Y57_resume_authorized'] is False
    queue = monitor.load_queue(str(ROOT))
    tasks = queue['gpu_queues']['server-2']
    paused = next(t for t in tasks if t['id'] == 'AdaRES-Y57')
    assert paused['status'] == 'paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']
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



def admit(token):
    work, candidate, plan, program, frozen = exact_candidate()
    server, queue, tasks = protected_queue()
    assert not any(t['status'] in ('running', 'claiming', 'waiting') and str(t.get('gpu')) == '0' for t in tasks)
    assert monitor.load_experiment(str(ROOT), CID) is None
    assert candidate['stage'] == 'validation'
    view = tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')
    assert view['queue']['candidate_budget'] >= 1
    raw_path = ROOT / ('pipeline/analyses/y68_gpu0_smoke_' + token + '.json')
    raw = tick.load(raw_path)
    validate_result(raw, candidate, frozen)
    candidate['native_smoke_result'] = ref(raw_path)
    assert candidate.get('required_gpu') == 0
    assert raw_path.with_suffix('.log').is_file()
    idle = require_idle(server)
    guard_log = ROOT / ('pipeline/checks/y68_admission_guards_' + token + '.log')
    assert not guard_log.exists()
    tested = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'scripts', '-p', 'test_y68_gpu0_admission.py', '-v'], cwd=ROOT, capture_output=True, timeout=60)
    guard_log.write_bytes(tested.stdout + tested.stderr)
    assert tested.returncode == 0 and b'Ran 14 tests' in guard_log.read_bytes()
    cfg = yaml.safe_load((ROOT / 'AdaRES' / candidate['config']).read_text(encoding='utf-8'))
    assert cfg['train']['seed'] == 1999 and cfg['train']['epochs'] == 30
    assert cfg['dataloader']['train']['batch_size'] == 2
    assert cfg['model']['sampling_module']['num_proposal'] == 128
    prefix = 'pipeline/checks/' + CID
    smoke_path = ROOT / (prefix + '_native_smoke_' + token + '.json')
    freeze_path = ROOT / (prefix + '_post_training_freeze_' + token + '.json')
    receipt_path = ROOT / ('pipeline/admissions/' + CID + '.json')
    assert not any(p.exists() for p in (smoke_path, freeze_path, receipt_path))
    command = ('python -B scripts/run_experiment_first_probe.py y68_gpu0_smoke ' + ref(program)['path'] +
               ' --token ' + token + ' --seconds 600 --gpu --gpu-id 0')
    smoke = copy.deepcopy(raw)
    smoke.update(command=command, physical_gpu=0,
        admission_resource_check=dict(observed_at=datetime.now().astimezone().isoformat(),
            live=idle, power_limit_watts=280, required_exclusive_gpu=0),
        evidence=[ref(raw_path), ref(raw_path.with_suffix('.log')), ref(program), candidate['native_smoke_plan']])
    tick.save(smoke_path, smoke)
    parent = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y64')['post_training_freeze']
    frozen_parent = tick.load(closure.artifact(ROOT, parent))
    freeze = copy.deepcopy(frozen_parent)
    freeze.update(id=CID, frozen_at=datetime.now().astimezone().isoformat(),
        goal=ref('pipeline/goal.json'), code_commit=candidate['code_commit'], parent_protocol=parent,
        resource_condition=dict(server='server-2', gpu=0, power_limit_watts=280,
            note='Keep approved280W; paired trained-cost comparisons on same device. Two batches are not a full-dataset memory guarantee.'),
        scientific_efficacy='Unknown until training and the frozen full evaluation are complete.')
    freeze['resource_authorization'] = ref('pipeline/authorizations/server2_gpu1_20261029.json')
    freeze['evaluation']['methods'][0] = CID
    restored = copy.deepcopy(freeze['evaluation']);restored['methods'][0] = 'AdaRES-Y64'
    assert restored == frozen_parent['evaluation']
    assert freeze['training'] == frozen_parent['training']
    tick.save(freeze_path, freeze)

    receipt = dict(resource_authorization=ref('pipeline/authorizations/server2_gpu1_20261029.json'), profile='experiment_first', id=CID, candidate_id=CID, config=candidate['config'],
        code_commit=candidate['code_commit'], goal_sha256=raw['goal_sha256'],
        authorization=experiment_first.effective(ROOT)['authorization'], native_smoke=ref(smoke_path),
        post_training_plan=ref(freeze_path), admitted_at=datetime.now().astimezone().isoformat(),
        gates={gate: dict(passed=True, evidence=[ref(smoke_path), ref(freeze_path), ref(program)])
               for gate in experiment_first.GATES})
    experiment_first.validate_smoke(ROOT, receipt)
    decision = monitor.log_decision(str(ROOT), 'next_batch', 'Y68是否通过真实原生两步检查并可接续GPU0？',
        '原生YAML/TRAIN两步及有效语言词均值的mask特征调制/每步七次实际mask预测消费/有限非零梯度及真实参数更新/资源检查通过，按预算准入Y68并尝试既有门控派发；Y57保持暂停。',
        '真实原始结果及日志/current goal/精确版本一致；其他服务器原运行保持且server-2 GPU0当前独占空闲且Y65已真实归档，GPU0保持280W/GPU1既有390W不变和固定30轮/seed1999，不预证效益。',
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
                 ref(receipt_path), ref(decision), ref(__file__), ref(guard_log), ref('scripts/test_y68_gpu0_admission.py')]
    candidate['evidence'].extend(e for e in additions if e not in candidate['evidence'])
    work['tick_attempt'] = dict(tick_started_at=view['generated_at'], candidate_id=CID,
        kind='native_smoke', result='passed', command=command,
        observation='Real native two-step loss/backward/valid-language mask-feature modulation consumed by all seven predictions passed. Admitted; '
                    'production training still requires gated launch and verified increasing steps.',
        evidence=[ref(raw_path), ref(smoke_path)])
    tasks.append(task)
    tick.save(ROOT / 'pipeline/research_work.json', work)
    monitor.save_queue(str(ROOT), queue)
    print(json.dumps(dict(admitted=CID, production_training_verified=False)))


def dispatch(token):
    server, queue, tasks = protected_queue()
    server1_before = copy.deepcopy(queue['gpu_queues']['server-1'])
    others_before = copy.deepcopy([t for t in tasks if t['id'] != CID])
    assert not any(t['status'] in ('running', 'claiming') and str(t.get('gpu')) == '0' for t in tasks), 'Reconcile uncertain launches'
    assert [t['id'] for t in tasks if t['status'] == 'waiting'] == [CID]
    task = next(t for t in tasks if t['id'] == CID)
    closure.admission(ROOT, task)
    before = require_idle(server)
    events = []
    monitor._process_queue(queue, 'server-2', server['host'], server['user'], str(ROOT), {'1'}, {'0'}, events,
        max_tasks_per_gpu=1, allow_external_overlay=False, remote_code_root=server['remote_code_root'],
        current_tasks=1, max_tasks_total=2)
    queue = monitor.load_queue(str(ROOT))
    task = next(t for t in queue['gpu_queues']['server-2'] if t['id'] == CID)
    assert queue['gpu_queues']['server-1'] == server1_before
    assert [t for t in queue['gpu_queues']['server-2'] if t['id'] != CID] == others_before
    after = snapshot(server, [task] if task.get('pid') else [], [])
    output = ROOT / ('pipeline/checks/y68_gpu0_dispatch_' + token + '.json')
    assert not output.exists()
    tick.save(output, dict(complete=True, observed_at=datetime.now().astimezone().isoformat(),
        before=before, after=after, task=task, events=events, Y57_still_paused=True,
        GPU1_Y67_preserved=True, hardware_settings_changed=False))
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == CID)
    candidate.update(stage='training' if task['status'] == 'running' and task.get('launch_verified') else 'deployment',
        next_action='Observe exact Y68 fixed30 PID/log and native step growth. Reconcile any '
                    'uncertain launch; never retry blindly, migrate or reset optimizer/scheduler.')
    candidate['evidence'].append(ref(output))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(evidence=ref(output), status=task['status'], pid=task.get('pid'),
        launch_verified=task.get('launch_verified'), events=events), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('admit', 'dispatch'))
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 300)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        dict(admit=admit, dispatch=dispatch)[args.operation](args.token)
    finally:
        lock.close()


if __name__ == '__main__':
    main()
