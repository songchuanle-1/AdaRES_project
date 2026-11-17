"""Admit actually tested Y58/Y59 and use the existing GPU0 gated dispatcher."""

import argparse
import ast
import copy
from datetime import datetime, timedelta
import json
from pathlib import Path

import yaml

from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
from dispatch_prepared_candidates import snapshot
import closure
import experiment_first


IDS = ('AdaRES-Y58', 'AdaRES-Y59')


def admit(token):
    lease(token, 300)
    server = monitor.load_servers(str(ROOT))['server-2']
    assert server['host'] == '192.168.1.163' and server['allowed_gpus'] == [0]
    assert server['max_tasks_per_gpu'] == 1
    work = tick.load(ROOT / 'pipeline/research_work.json')
    queue = monitor.load_queue(str(ROOT))
    assert not any(t['status'] in ('running', 'claiming', 'waiting')
                   for t in queue['gpu_queues']['server-2'])
    assert next(t for t in queue['gpu_queues']['server-2'] if t['id'] == 'AdaRES-Y57')['status'] == 'paused'
    head = closure.git(ROOT, 'rev-parse', 'HEAD')
    policy = experiment_first.effective(ROOT)
    view = tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')
    assert view['queue']['candidate_budget'] >= 2
    records = []
    for cid in IDS:
        candidate = next(c for c in work['candidates'] if c['id'] == cid)
        assert candidate['stage'] == 'validation' and candidate['code_commit'] == head
        assert not candidate.get('revision_history')
        assert all(t['id'] != cid for t in queue['gpu_queues']['server-2'])
        assert monitor.load_experiment(str(ROOT), cid) is None
        plan = tick.load(closure.artifact(ROOT, candidate['native_smoke_plan']))
        program = closure.artifact(ROOT, plan['program'])
        frozen = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
        assert validate_frozen(candidate, plan, frozen) == 7
        stem = cid.lower().replace('-', '_') + '_head_smoke'
        raw_path = ROOT / ('pipeline/analyses/' + stem + '_' + token + '.json')
        raw = tick.load(raw_path)
        assert raw['complete'] and raw['id'] == cid and raw['code_commit'] == head
        assert raw['goal_sha256'] == ref('pipeline/goal.json')['sha256']
        assert raw['steps'] == raw['optimizer_steps'] == 2
        assert raw['native_YAML_sha256'] == frozen['variant']['config_data']['sha256']
        assert raw['native_data_loaded'] and raw['backward_finite'] and raw['intended_path_observed']
        assert all(row['all_finite'] and row['indicator_tensors'] > 0 for row in raw['gradient_checks'])
        assert 0 < raw['peak_reserved_bytes'] < 24 * 2**30
        config = ROOT / 'AdaRES' / candidate['config']
        cfg = yaml.safe_load(config.read_text(encoding='utf-8'))
        assert cfg['train']['seed'] == 1999 and cfg['train']['epochs'] == 30
        assert cfg['dataloader']['train']['batch_size'] == 2
        assert cfg['model']['sampling_module']['num_proposal'] == 128
        smoke_path = ROOT / ('pipeline/checks/' + cid + '_native_smoke_' + token + '.json')
        receipt_path = ROOT / ('pipeline/admissions/' + cid + '.json')
        freeze_path = ROOT / ('pipeline/checks/' + cid + '_post_training_freeze_' + token + '.json')
        assert not any(p.exists() for p in (smoke_path, receipt_path, freeze_path))
        smoke = copy.deepcopy(raw)
        smoke.update(command='python -B scripts/run_experiment_first_probe.py ' + stem + ' ' +
            ref(program)['path'] + ' --token ' + token + ' --seconds 600 --gpu --gpu-id 0',
            physical_gpu=0, evidence=[ref(raw_path), ref(raw_path.with_suffix('.log')),
                                      ref(program), plan['deployment_binding']],
            note='Verbatim native observations plus invocation/evidence binding; raw output retained unchanged.')
        tick.save(smoke_path, smoke)
        tick.save(freeze_path, dict(complete=True, id=cid, frozen_at=datetime.now().astimezone().isoformat(),
            goal=ref('pipeline/goal.json'), code_commit=head,
            training='Released SPFormer; seed1999; fixed30; native YAML losses/optimizer/scheduler; one GPU; Q128; single argmax; no validation tuning.',
            evaluation=dict(split='Full frozen ScanRefer val9508 expressions/141 scenes',
                methods=[cid, 'AdaRES-Y36', 'AdaRES-W2', 'AdaRES-B1'], repeats=2, primary='FIRST',
                metrics=['mIoU','Acc@0.25','Acc@0.5','active_queries','latency','peak_memory'],
                scene_cluster_bootstrap=10000, bootstrap_seed=1999, TRAIN_size_cuts=[1487,3325],
                material_harm='Prespecified target-size subgroup difference CI upper<0 versus Y36/W2; retain all negatives.',
                uncertainty='Crossing intervals or too few independent scenes are inconclusive; repeats do not measure training-seed uncertainty.'),
            resource_condition='GPU0 runtime power cap280W; report this when comparing latency; bounded two-batch smoke is not a full-dataset resource guarantee.',
            scientific_efficacy='Unknown until trained and frozen evaluation is complete.'))
        receipt = dict(profile='experiment_first', id=cid, candidate_id=cid, config=candidate['config'],
            code_commit=head, goal_sha256=raw['goal_sha256'], authorization=policy['authorization'],
            native_smoke=ref(smoke_path), post_training_plan=ref(freeze_path),
            admitted_at=datetime.now().astimezone().isoformat(), gates={
                name: dict(passed=True, evidence=[ref(smoke_path), ref(freeze_path), plan['deployment_binding']])
                for name in experiment_first.GATES})
        experiment_first.validate_smoke(ROOT, receipt)
        records.append((candidate, receipt_path, receipt, smoke_path, freeze_path))
    decision = monitor.log_decision(str(ROOT), 'next_batch',
        'Y56固定30轮已完整归档，唯一授权GPU0已释放，如何续做已实现候选？',
        'Y58/Y59分别通过当前部署HEAD下原生YAML/真实TRAIN两步有限后向、意图路径及资源检查，按本轮预算2准入；只尝试server-2 GPU0既有门控派发，GPU1与Y57暂停不变。',
        '每项精确7文件与先前已测试提交及当前HEAD逐字节一致，旧计划/原始结果保留；最小可运行性不代表训练上涨。保持fixed30/seed1999、数据边界和280W，不重复索权。',
        '; '.join(ref(row[3])['path'] for row in records))
    for candidate, receipt_path, receipt, smoke_path, freeze_path in records:
        receipt['decision'] = ref(decision)
        tick.save(receipt_path, receipt)
        task = dict(id=candidate['id'], label=candidate['id'] + ' fixed30', config=candidate['config'],
            status='waiting', gpu=0, depends_on=[], code_commit=head,
            admission_receipt=receipt_path.relative_to(ROOT).as_posix(), scientific_candidate_id=candidate['id'],
            patience=30, min_epochs=30, fixed_epochs=30)
        closure.admission(ROOT, task)
        queue['gpu_queues']['server-2'].append(task)
        candidate.update(stage='admission', admission_receipt=task['admission_receipt'],
            post_training_freeze=ref(freeze_path), native_smoke=ref(smoke_path),
            next_action='Use existing gated dispatch on authorized GPU0 only; verify source and native '
                        'step growth. Preserve Y57 pause. If GPU0 busy, retain FIFO waiting without oversubscription.')
        candidate['evidence'].extend([ref(receipt_path), ref(smoke_path), ref(freeze_path), ref(decision), ref(__file__)])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    monitor.save_queue(str(ROOT), queue)
    print(json.dumps(dict(admitted=list(IDS), code_commit=head, training_started=False), indent=2))


def dispatch(token):
    lease(token, 300)
    server = monitor.load_servers(str(ROOT))['server-2']
    assert server['host'] == '192.168.1.163' and server['allowed_gpus'] == [0]
    queue = monitor.load_queue(str(ROOT))
    tasks = queue['gpu_queues']['server-2']
    paused = next(t for t in tasks if t['id'] == 'AdaRES-Y57')
    assert paused['status'] == 'paused' and paused['dispatch_hold']
    assert not any(t['status'] in ('claiming','running') for t in tasks), 'Reconcile an existing launch, never retry blindly'
    assert {t['id'] for t in tasks if t['status'] == 'waiting'} == set(IDS)
    before = snapshot(server, [], [])
    assert before['gpu_query_rc'] == before['compute_query_rc'] == 0
    assert 'GPU-e0b11432-b7aa-2006-1903-dcc739f26146' not in before['compute']
    events = []
    monitor._process_queue(queue, 'server-2', server['host'], server['user'], str(ROOT), set(), {'0'}, events,
        max_tasks_per_gpu=1, allow_external_overlay=False, remote_code_root=server['remote_code_root'],
        current_tasks=0, max_tasks_total=1)
    queue = monitor.load_queue(str(ROOT))
    tasks = [t for t in queue['gpu_queues']['server-2'] if t['id'] in IDS]
    live = snapshot(server, [t for t in tasks if t.get('pid')], [])
    output = ROOT / ('pipeline/checks/ef_gpu0_dispatch_' + token + '.json')
    assert not output.exists()
    tick.save(output, dict(complete=True, observed_at=datetime.now().astimezone().isoformat(),
        before=before, after=live, tasks=tasks, events=events,
        Y57_still_paused=True, GPU1_not_used=True, hardware_settings_changed=False))
    work = tick.load(ROOT / 'pipeline/research_work.json')
    for candidate in work['candidates']:
        if candidate['id'] not in IDS:
            continue
        task = next(t for t in tasks if t['id'] == candidate['id'])
        if task['status'] == 'running' and task.get('launch_verified'):
            candidate.update(stage='training', next_action='Monitor this exact fixed30 native run and preserve '
                'checkpoint/log lineage. Report trained efficacy only after the frozen evaluation; do not tune validation.')
        else:
            candidate.update(stage='deployment', next_action='Retain existing FIFO waiting admission. '
                'Use only genuinely idle GPU0 with the canonical dispatcher; uncertain launches require reconciliation.')
        candidate['evidence'].append(ref(output))
    # Preserve untested Y60 as a real GPU wait once the admitted work fills the sole slot.
    if any(t['status'] in ('running','claiming','waiting') for t in tasks):
        candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y60')
        if candidate['stage'] == 'validation':
            candidate.update(stage='blocked', blocked_from_stage='validation', blocker_kind='exclusive_gpu_slot',
                required_server='server-2', retry_after=(datetime.now().astimezone()+timedelta(hours=1)).isoformat(),
                blocker='GPU0 is occupied or reserved by the newly admitted FIFO Y58/Y59 work. GPU1 remains unauthorized.',
                unblock_condition='An actual canonical queue opening on GPU0; recheck all live compute PIDs and exclusive lease.',
                next_action='Preserve the exact prepared Y60 source/YAML and native_smoke_plan. After Y58/Y59 release '
                    'the sole authorized GPU0 slot, run actual native two-step smoke under the current lease before admission.')
            candidate['evidence'].append(ref(output))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(evidence=ref(output), tasks=[dict(id=t['id'], status=t['status'],
        gpu=t.get('gpu'), pid=t.get('pid'), launch_verified=t.get('launch_verified'),
        admission_error=t.get('admission_error'), dispatch_hold=t.get('dispatch_hold')) for t in tasks], events=events), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('admit','dispatch'))
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
