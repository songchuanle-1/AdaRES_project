"""Verify real Y84 native evidence, preserve the full protocol and dispatch GPU0."""
import argparse
import ast
import copy
from datetime import datetime
import json
import math
import subprocess
import sys
import yaml
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
from advance_y82 import live, idle, unreserved
from y84_deployment import Y84
import closure
import experiment_first

CID = 'AdaRES-Y84'
NAME, GPU, POWER = 'server-1', 0, 350.
GRANT = 'pipeline/authorizations/server1_gpu0_20261025.json'


def prepared(work):
    candidate = next(candidate for candidate in work['candidates'] if candidate['id'] == CID)
    assert candidate['code_commit'] == Y84
    assert (candidate['required_server'], candidate['required_gpu']) == (NAME, GPU)
    assert candidate['native_smoke_plan'] == candidate['native_smoke_plan_server1']
    plan = tick.load(closure.artifact(ROOT, candidate['native_smoke_plan']))
    program = closure.artifact(ROOT, plan['program'])
    frozen = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
    assert validate_frozen(candidate, plan, frozen) == 7
    assert (frozen['runtime_server'], frozen['runtime_physical_gpu'],
            frozen['runtime_power_limit_watts']) == (NAME, GPU, POWER)
    assert frozen['resource_authorization'] == ref(GRANT)
    closure.artifact(ROOT, frozen['resource_authorization'])
    return candidate, plan, program, frozen


def validate_result(raw, candidate, frozen):
    assert raw['complete'] is True and raw['id'] == candidate['id'] == CID == frozen['variant']['id']
    assert raw['config'] == candidate['config'] == frozen['variant']['config']
    assert raw['code_commit'] == frozen['code_commit'] == Y84
    assert raw['goal_sha256'] == frozen['goal_sha256']
    assert raw['native_YAML_sha256'] == frozen['variant']['config_data']['sha256']
    assert raw['steps'] == raw['optimizer_steps'] == 2
    assert len(raw['losses']) == 2 and all(math.isfinite(value) for value in raw['losses'])
    assert raw['native_entry'] == 'tools/train.py::train' and raw['split'] == 'train'
    for key in ('native_data_loaded', 'backward_finite', 'intended_path_observed',
                'native_modes_verified', 'static_mask_features_preserved',
                'original_hard_gate_algorithm_preserved'):
        assert raw[key] is True
    assert raw['validation_used'] is False and raw['model_weights_saved'] is False
    assert raw['shared_head_calls'] == raw['query_norm_consumption_calls'] == 14
    assert raw['initialization_calls'] == 2
    assert len(raw['gradient_checks']) == len(raw['batches']) == 2
    from test_y84_native_smoke_plan import N
    N['check_initialization_rows'](raw['initialization_paths'])
    for index, row in enumerate(raw['gradient_checks']):
        assert type(row['step']) is int and row['step'] == index
        assert row['all_finite'] is True and row['tensors'] > 0 and row['indicator_tensors'] == 10
        assert row['prediction_calls'] == row['indicator_calls'] == row['query_norm_calls'] == 7
        assert row['both_query_and_scene_gradients_finite'] is True
        assert all(math.isfinite(row[key]) and row[key] > 0
                   for key in ('new_gradient_max', 'parameter_update_max'))
    assert 0 < raw['peak_reserved_bytes'] < 24 * 2**30
    assert all(len(batch['scan_ids']) == 2 and batch['expressions'] > 0 and batch['points'] > 0
               for batch in raw['batches'])
    assert len({scene for batch in raw['batches'] for scene in batch['scan_ids']}) == 4
    assert raw['data_source']['dataset_class'] == 'ipdn.dataset.scannetv2_test.ScanNetDataset_sample_graph_edge'
    assert raw['data_source']['config_data_root'].rstrip('/') == '/data1/songcl/Project/AdaRES_project/data'


def verify(token):
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate, _, _, frozen = prepared(work)
    assert candidate['stage'] == 'validation' and candidate.get('native_smoke_result')
    path = closure.artifact(ROOT, candidate['native_smoke_result'])
    raw = tick.load(path)
    validate_result(raw, candidate, frozen)
    print(json.dumps(dict(result=ref(path), losses=raw['losses'],
                          peak_reserved_bytes=raw['peak_reserved_bytes'], actual_native_verified=True)))


def admit(token):
    server = monitor.load_servers(str(ROOT))[NAME]
    closure.validate_adares_server(ROOT, server)
    assert server['allowed_gpus'] == [0]
    servers = monitor.load_servers(str(ROOT))
    assert servers['server-2']['allowed_gpus'] == [0] and servers['server-6']['allowed_gpus'] == []
    authority = ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')
    assert authority['sha256'] == '3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45'
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate, _, program, frozen = prepared(work)
    assert candidate['stage'] == 'validation' and not candidate.get('admission_receipt')
    assert closure.git(ROOT, 'rev-parse', 'HEAD') == Y84
    rawpath = closure.artifact(ROOT, candidate['native_smoke_result'])
    raw = tick.load(rawpath)
    validate_result(raw, candidate, frozen)
    queue = monitor.load_queue(str(ROOT))
    unreserved(queue, monitor.load_running(str(ROOT)), NAME, GPU)
    assert monitor.load_experiment(str(ROOT), CID) is None
    current = live(server, GPU)
    idle(current, POWER)
    cfg = yaml.safe_load((ROOT / 'AdaRES' / candidate['config']).read_bytes())
    assert cfg['train']['seed'] == 1999 and cfg['train']['epochs'] == 30
    assert cfg['dataloader']['train']['batch_size'] == 2 and cfg['model']['sampling_module']['num_proposal'] == 128
    tests = ROOT / ('pipeline/checks/adares-y84_admission_guards_' + token + '.log')
    assert not tests.exists()
    response = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'scripts',
        '-p', 'test_y84_admission.py', '-v'], cwd=ROOT, capture_output=True, timeout=60)
    tests.write_bytes(response.stdout + response.stderr)
    assert response.returncode == 0
    parentref = next(candidate for candidate in work['candidates']
                     if candidate['id'] == 'AdaRES-Y64')['post_training_freeze']
    parent = tick.load(closure.artifact(ROOT, parentref))
    prefix = 'pipeline/checks/' + CID + '_' + token
    smoke_path = ROOT / (prefix + '_native_smoke.json')
    freeze_path = ROOT / (prefix + '_post_training_freeze.json')
    receipt_path = ROOT / ('pipeline/admissions/' + CID + '.json')
    assert not any(path.exists() for path in (smoke_path, freeze_path, receipt_path))
    command = 'python -B scripts/y84_server1_alternative.py run --token ' + token
    smoke = copy.deepcopy(raw)
    smoke.update(command=command, server=NAME, physical_gpu=GPU, admission_resource_check=current,
        evidence=[ref(rawpath), ref(rawpath.with_suffix('.log')), ref(program), candidate['native_smoke_plan']])
    tick.save(smoke_path, smoke)
    freeze = copy.deepcopy(parent)
    freeze.update(id=CID, frozen_at=datetime.now().astimezone().isoformat(),
        goal=ref('pipeline/goal.json'), code_commit=Y84, parent_protocol=parentref,
        resource_condition=dict(server=NAME, gpu=GPU, power_limit_watts=POWER,
            note='Existing power preserved; full paired evaluation/cost on this device; two batches are not a full-dataset memory bound.'),
        resource_authorization=ref(GRANT), scientific_efficacy='Unknown until original full frozen evaluation.')
    freeze['evaluation']['methods'][0] = CID
    restored = copy.deepcopy(freeze['evaluation'])
    restored['methods'][0] = 'AdaRES-Y64'
    assert restored == parent['evaluation'] and freeze['training'] == parent['training']
    tick.save(freeze_path, freeze)
    receipt = dict(profile='experiment_first', id=CID, candidate_id=CID, config=candidate['config'],
        code_commit=Y84, goal_sha256=raw['goal_sha256'],
        authorization=experiment_first.effective(ROOT)['authorization'], native_smoke=ref(smoke_path),
        post_training_plan=ref(freeze_path), resource_authorization=ref(GRANT),
        admitted_at=datetime.now().astimezone().isoformat(),
        gates={gate: dict(passed=True, evidence=[ref(smoke_path), ref(freeze_path), ref(program)])
               for gate in experiment_first.GATES})
    experiment_first.validate_smoke(ROOT, receipt)
    decision = monitor.log_decision(str(ROOT), 'next_batch', CID + ' native TRAIN admission',
        'Original native YAML/TRAIN two finite backward and optimizer steps passed; admit independent visual query initialization.',
        'Original full frozen evaluation, seed1999/fixed30, exact data and350W preserved. Efficacy unknown; GPU1/Y57 stay paused.',
        ref(smoke_path)['path'])
    receipt['decision'] = ref(decision)
    tick.save(receipt_path, receipt)
    task = dict(id=CID, label=CID + ' fixed30', config=candidate['config'], status='waiting',
        gpu=GPU, server=NAME, depends_on=[], code_commit=Y84, admission_receipt=ref(receipt_path)['path'],
        scientific_candidate_id=CID, patience=30, min_epochs=30, fixed_epochs=30)
    closure.admission(ROOT, task)
    candidate.update(stage='admission', admission_receipt=task['admission_receipt'],
        native_smoke=ref(smoke_path), post_training_freeze=ref(freeze_path),
        next_action='Canonical gated dispatch; verify increasing native steps. Preserve Y83 and both user pauses.')
    candidate['evidence'].extend([ref(smoke_path), ref(freeze_path), ref(receipt_path), ref(decision),
                                 ref(tests), ref(__file__), authority])
    work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id=CID, kind='native_smoke', result='passed', command=command,
        observation='Real original native two-step paths/backward/optimizer passed; queued with unchanged full evaluation.',
        evidence=[ref(smoke_path), ref(freeze_path)])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    queue['gpu_queues'][NAME].append(task)
    monitor.save_queue(str(ROOT), queue)
    print(json.dumps(dict(admitted=CID, receipt=ref(receipt_path), losses=raw['losses'])))


def dispatch(token):
    server = monitor.load_servers(str(ROOT))[NAME]
    closure.validate_adares_server(ROOT, server)
    assert server['allowed_gpus'] == [0]
    queue = monitor.load_queue(str(ROOT))
    tasks = queue['gpu_queues'][NAME]
    task = next(task for task in tasks if task['id'] == CID)
    assert task['status'] == 'waiting'
    closure.admission(ROOT, task)
    assert not monitor.load_running(str(ROOT)).get(NAME, {}).get('0')
    assert not any(task['id'] != CID and task['status'] in ('waiting', 'claiming', 'running')
                   and str(task.get('gpu')) == '0' for task in tasks)
    others = {name: copy.deepcopy([task for task in tasks if task['id'] != CID])
              for name, tasks in queue['gpu_queues'].items()}
    current = live(server, GPU)
    idle(current, POWER)
    active = {str(task['gpu']) for task in tasks if task['status'] in ('running', 'claiming')}
    events = []
    monitor._process_queue(queue, NAME, server['host'], server['user'], str(ROOT), active, {'0'}, events,
        max_tasks_per_gpu=1, allow_external_overlay=False, remote_code_root=server['remote_code_root'],
        current_tasks=len(active), max_tasks_total=1)
    queue = monitor.load_queue(str(ROOT))
    task = next(task for task in queue['gpu_queues'][NAME] if task['id'] == CID)
    assert {name: [task for task in tasks if task['id'] != CID]
            for name, tasks in queue['gpu_queues'].items()} == others
    out = ROOT / ('pipeline/checks/adares-y84_dispatch_' + token + '.json')
    assert not out.exists()
    tick.save(out, dict(complete=True, token=token, id=CID, before=current, task=task,
                       events=events, other_queue_entries_preserved=True, helper=ref(__file__)))
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(candidate for candidate in work['candidates'] if candidate['id'] == CID)
    candidate.update(stage='training' if task['status'] == 'running' and task.get('launch_verified')
        else 'deployment', next_action='Observe exact native counter increase; reconcile uncertainty without restart.')
    candidate['evidence'].append(ref(out))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(evidence=ref(out), status=task['status'], pid=task.get('pid'),
                          launch_verified=task.get('launch_verified'), events=events)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('verify', 'admit', 'dispatch'))
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 300)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        dict(verify=verify, admit=admit, dispatch=dispatch)[args.operation](args.token)
    finally:
        writer.close()
