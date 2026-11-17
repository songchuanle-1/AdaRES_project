"""Bind the real Y60 smoke, admit once and dispatch through the existing monitor."""

import argparse
import ast
import copy
from datetime import datetime
import json

import yaml

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
from dispatch_prepared_candidates import snapshot
import closure
import experiment_first


CID = 'AdaRES-Y60'


def admit(token):
    lease(token, 300)
    server = monitor.load_servers(str(ROOT))['server-2']
    assert server['host'] == '192.168.1.163' and server['allowed_gpus'] == [0]
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == CID)
    queue = monitor.load_queue(str(ROOT))
    tasks = queue['gpu_queues']['server-2']
    assert not any(t['status'] in ('running', 'claiming', 'waiting') for t in tasks)
    assert next(t for t in tasks if t['id'] == 'AdaRES-Y57')['status'] == 'paused'
    assert monitor.load_experiment(str(ROOT), 'AdaRES-Y59')['status'] == 'done_complete'
    assert monitor.load_experiment(str(ROOT), CID) is None
    head = closure.git(ROOT, 'rev-parse', 'HEAD')
    assert candidate['stage'] == 'validation' and candidate['code_commit'] == head
    assert not candidate.get('revision_history')
    view = tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')
    assert view['queue']['candidate_budget'] >= 1
    plan = tick.load(closure.artifact(ROOT, candidate['native_smoke_plan']))
    program = closure.artifact(ROOT, plan['program'])
    frozen = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
    assert validate_frozen(candidate, plan, frozen) == 7
    raw_path = ROOT / ('pipeline/analyses/y60_smoke_' + token + '.json')
    raw = tick.load(raw_path)
    assert raw['complete'] and raw['id'] == CID and raw['config'] == candidate['config']
    assert raw['code_commit'] == head and raw['goal_sha256'] == ref('pipeline/goal.json')['sha256']
    assert raw['steps'] == raw['optimizer_steps'] == 2
    assert len(raw['losses']) == 2 and raw['native_entry'] == 'tools/train.py::train'
    assert raw['native_data_loaded'] and raw['backward_finite'] and raw['intended_path_observed']
    assert raw['stage_head_forward_calls'] == [2] * 7
    assert len(raw['gradient_checks']) == 2
    assert all(row['all_finite'] and row['indicator_tensors_by_stage'] == [10] * 7
               for row in raw['gradient_checks'])
    assert 0 < raw['peak_reserved_bytes'] < 24 * 2**30 and raw['validation_used'] is False
    assert raw['native_YAML_sha256'] == frozen['variant']['config_data']['sha256']
    cfg = yaml.safe_load((ROOT / 'AdaRES' / candidate['config']).read_text(encoding='utf-8'))
    assert cfg['train']['seed'] == 1999 and cfg['train']['epochs'] == 30
    assert cfg['dataloader']['train']['batch_size'] == 2
    assert cfg['model']['sampling_module']['num_proposal'] == 128
    smoke_path = ROOT / ('pipeline/checks/' + CID + '_native_smoke_' + token + '.json')
    receipt_path = ROOT / ('pipeline/admissions/' + CID + '.json')
    freeze_path = ROOT / ('pipeline/checks/' + CID + '_post_training_freeze_' + token + '.json')
    assert not any(p.exists() for p in (smoke_path, receipt_path, freeze_path))
    smoke = copy.deepcopy(raw)
    command = ('python -B scripts/run_experiment_first_probe.py y60_smoke ' + ref(program)['path'] +
               ' --token ' + token + ' --seconds 600 --gpu --gpu-id 0')
    smoke.update(command=command, physical_gpu=0,
        evidence=[ref(raw_path), ref(raw_path.with_suffix('.log')), ref(program), candidate['native_smoke_plan']],
        note='Real native observations plus invocation binding; immutable raw result retained.')
    tick.save(smoke_path, smoke)
    tick.save(freeze_path, dict(complete=True, id=CID, frozen_at=datetime.now().astimezone().isoformat(),
        goal=ref('pipeline/goal.json'), code_commit=head,
        training='Released SPFormer; seed1999; fixed30; native YAML losses/optimizer/scheduler; '
                 'one GPU; Q128; single argmax; no validation tuning.',
        evaluation=dict(split='Full frozen ScanRefer val9508 expressions/141 scenes',
            methods=[CID, 'AdaRES-Y36', 'AdaRES-W2', 'AdaRES-B1'], repeats=2, primary='FIRST',
            metrics=['mIoU', 'Acc@0.25', 'Acc@0.5', 'active_queries', 'latency', 'peak_memory'],
            scene_cluster_bootstrap=10000, bootstrap_seed=1999, TRAIN_size_cuts=[1487, 3325],
            material_harm='Prespecified target-size subgroup difference CI upper<0 versus Y36/W2; retain all negatives.',
            uncertainty='Crossing intervals or too few independent scenes are inconclusive; repeats do not measure training-seed uncertainty.'),
        resource_condition='GPU0 runtime power cap280W; disclose for latency comparison. '
                           'Two initialization batches do not bound full-dataset memory.',
        scientific_efficacy='Unknown until trained and full frozen evaluation is complete.'))
    receipt = dict(profile='experiment_first', id=CID, candidate_id=CID, config=candidate['config'],
        code_commit=head, goal_sha256=raw['goal_sha256'],
        authorization=experiment_first.effective(ROOT)['authorization'],
        native_smoke=ref(smoke_path), post_training_plan=ref(freeze_path),
        admitted_at=datetime.now().astimezone().isoformat(), gates={
            gate: dict(passed=True, evidence=[ref(smoke_path), ref(freeze_path), ref(program)])
            for gate in experiment_first.GATES})
    experiment_first.validate_smoke(ROOT, receipt)
    decision = monitor.log_decision(str(ROOT), 'next_batch',
        'Y59完整固定30轮已归档，Y60是否满足真实检查并接续唯一授权GPU0？',
        'Y60原生YAML/真实TRAIN两步optimizer及有限loss/backward、七头执行与资源检查通过，按本轮预算准入一项并尝试既有GPU0门控派发；Y57继续暂停。',
        '精确七文件/current goal/commit一致，CPU和原始两步证据保留；不要求预证指标改善，不改seed1999/fixed30/数据边界/280W或调度。',
        ref(smoke_path)['path'])
    receipt['decision'] = ref(decision)
    tick.save(receipt_path, receipt)
    task = dict(id=CID, label=CID + ' fixed30', config=candidate['config'], status='waiting', gpu=0,
        depends_on=[], code_commit=head, admission_receipt=receipt_path.relative_to(ROOT).as_posix(),
        scientific_candidate_id=CID, patience=30, min_epochs=30, fixed_epochs=30)
    closure.admission(ROOT, task)
    tasks.append(task)
    candidate.update(stage='admission', admission_receipt=task['admission_receipt'],
        post_training_freeze=ref(freeze_path), native_smoke=ref(smoke_path),
        next_action='Use the existing gated GPU0 dispatcher; require exact runtime source and '
                    'increasing native steps. Preserve fixed30/seed1999, Y57 pause and 280W.')
    additions = [ref(raw_path), ref(raw_path.with_suffix('.log')), ref(smoke_path), ref(freeze_path),
                 ref(receipt_path), ref(decision), ref(__file__)]
    candidate['evidence'].extend(e for e in additions if e not in candidate['evidence'])
    work['tick_attempt'] = dict(tick_started_at=view['generated_at'], candidate_id=CID,
        kind='native_smoke', result='passed', command=command,
        observation='Two actual native TRAIN optimizer steps with finite loss/backward and '
                    'all seven independent confidence heads exercised. Admitted, not yet verified training.',
        evidence=[ref(raw_path), ref(smoke_path)])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    monitor.save_queue(str(ROOT), queue)
    print(json.dumps(dict(admitted=CID, code_commit=head, training_started=False)))


def dispatch(token):
    lease(token, 300)
    server = monitor.load_servers(str(ROOT))['server-2']
    assert server['host'] == '192.168.1.163' and server['allowed_gpus'] == [0]
    queue = monitor.load_queue(str(ROOT))
    tasks = queue['gpu_queues']['server-2']
    assert not any(t['status'] in ('running', 'claiming') for t in tasks), 'Reconcile, never retry blindly'
    assert [t['id'] for t in tasks if t['status'] == 'waiting'] == [CID]
    paused = next(t for t in tasks if t['id'] == 'AdaRES-Y57')
    assert paused['status'] == 'paused' and paused['dispatch_hold']
    before = snapshot(server, [], [])
    assert before['gpu_query_rc'] == before['compute_query_rc'] == 0
    assert 'GPU-e0b11432-b7aa-2006-1903-dcc739f26146' not in before['compute']
    events = []
    monitor._process_queue(queue, 'server-2', server['host'], server['user'], str(ROOT), set(), {'0'}, events,
        max_tasks_per_gpu=1, allow_external_overlay=False, remote_code_root=server['remote_code_root'],
        current_tasks=0, max_tasks_total=1)
    queue = monitor.load_queue(str(ROOT))
    task = next(t for t in queue['gpu_queues']['server-2'] if t['id'] == CID)
    live = snapshot(server, [task] if task.get('pid') else [], [])
    output = ROOT / ('pipeline/checks/y60_gpu0_dispatch_' + token + '.json')
    assert not output.exists()
    tick.save(output, dict(complete=True, observed_at=datetime.now().astimezone().isoformat(),
        before=before, after=live, task=task, events=events, Y57_still_paused=True,
        GPU1_not_used=True, hardware_settings_changed=False))
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == CID)
    candidate.update(stage='training' if task['status'] == 'running' and task.get('launch_verified') else 'deployment',
        next_action='Monitor the exact fixed30 Y60 native run with actual step-growth evidence; '
                    'if launch is uncertain, reconcile original PID/log instead of relaunching.')
    candidate['evidence'].append(ref(output))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(evidence=ref(output), id=CID, status=task['status'], pid=task.get('pid'),
        launch_verified=task.get('launch_verified'), admission_error=task.get('admission_error'),
        dispatch_hold=task.get('dispatch_hold'), events=events), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('admit', 'dispatch'))
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 300)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        (admit if args.operation == 'admit' else dispatch)(args.token)
    finally:
        lock.close()


if __name__ == '__main__':
    main()
