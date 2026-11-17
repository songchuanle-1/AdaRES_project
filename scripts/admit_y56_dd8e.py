"""Admit the explicitly committed, tested Y56 through canonical receipts."""

from datetime import datetime
import hashlib
import json
import subprocess
from research_y56_dd8e import ROOT, TOKEN, lease, ref, save, bind, tick, monitor
import closure

lease(300)
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work = tick.load(ROOT/'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y56')
    queue = monitor.load_queue(str(ROOT))
    assert not queue['gpu_queues']['server-2']
    assert candidate['stage'] == 'validation' and candidate['revision_history'][-1]['result'] == 'passed'
    review_ref = ref('pipeline/analyses/y56_gate_review_'+TOKEN+'.json')
    review = tick.load(ROOT/review_ref['path'])
    assert ref('pipeline/goal.json')['sha256'] == review['goal_sha256']
    head = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    assert head == candidate['code_commit'] == '200486ee12bcf3d7e82393d009d06214dfe83e38'
    binding = ref('pipeline/analyses/AdaRES-Y56_commit_binding_'+TOKEN+'.json')
    authorization = ref('pipeline/authorizations/AdaRES-Y56_commit.json')
    for row in review['files']:
        assert ref(row['path']) == row
        raw = subprocess.check_output(['git','-C',str(ROOT),'show',head+':'+row['path']])
        assert hashlib.sha256(raw).hexdigest() == row['sha256']
    for row in candidate['evidence']:
        closure.artifact(ROOT,row)
    plan = save('pipeline/analyses/AdaRES-Y56_post_training_freeze_'+TOKEN+'.json',dict(
        complete=True,id='AdaRES-Y56',frozen_at=datetime.now().astimezone().isoformat(),code_commit=head,
        preregistered_design=ref('pipeline/designs/exp_design_Y56_dd8e.md'),
        train=dict(initialization='Released SPFormer sparse-UNet only; not trained-W2 diagnostic weights',
                   seed=1999,epochs=30,batch_size=2,queries=128,selected_queries=1,
                   optimizer_loss_scheduler='Unchanged W2 YAML',single_gpu=True,no_validation_tuning=True),
        evaluation=dict(split='Complete frozen ScanRefer val9508expressions/141scenes',
            methods=['AdaRES-Y56','AdaRES-Y36','AdaRES-W2','AdaRES-B1'],repeats=2,primary='FIRST',
            metrics=['mIoU','Acc@0.25','Acc@0.5','active_queries','latency','peak_memory'],
            scene_cluster_bootstrap=10000,bootstrap_seed=1999,TRAIN_size_cuts=[1487,3325],
            material_harm='Prespecified target-size subgroup difference CI upper<0 vs Y36/W2; retain all negatives.',
            noise='Independent inference repeat; not training-seed uncertainty.',
            lightweight_Pareto='Compare matched full-evaluation latency/peak memory/parameters, no metric-free promotion.',
            uncertainty='Crossing intervals or inadequate independent scenes are inconclusive; official unique/multiple unknown.'),
        limitations=review['limitations']))
    decision = monitor.log_decision(str(ROOT),'manual_adares-y56',
        'Y56七文件已获本会话明确本地提交授权，是否按既有科学和资源门禁准入？',
        '精确测试字节已提交；六项实测门槛复核后只准入Y56，交规范调度器实时验证server-2单卡及原生进度。',
        '保留两次有界性能修订、原始高开销、失败驱动日志、复制病例时延风险和训练效益未知；不改目标、协议或调度。',
        review_ref['path']+'; '+binding['path']+'; '+plan['path'])
    decision_ref=ref(decision)
    gates=json.loads(json.dumps(review['gates']))
    gates['independence']['evidence'].append(binding)
    gates['constraints']['evidence'].extend([plan,ref('pipeline/goal.json')])
    receipt_path='pipeline/admissions/AdaRES-Y56.json'
    receipt=dict(id='AdaRES-Y56',candidate_id='AdaRES-Y56',
        config='configs_song/model_paper5/adares_y56_unique_competitors_res.yaml',
        code_commit=head,goal_sha256=review['goal_sha256'],decision=decision_ref,gates=gates,
        implementation_revision_sha256=closure.revision_digest(candidate['revision_history'][-1]),
        admitted_at=datetime.now().astimezone().isoformat(),post_training_plan=plan,
        local_commit_authorization=authorization,limitations=review['limitations'])
    save(receipt_path,receipt)
    task=dict(id='AdaRES-Y56',label='AdaRES-Y56 unique competitors fixed30',config=receipt['config'],
        status='waiting',gpu=None,depends_on=[],code_commit=head,admission_receipt=receipt_path,
        patience=30,min_epochs=30,fixed_epochs=30,requires_exclusive_ownership=True,
        scientific_candidate_id='AdaRES-Y56')
    closure.admission(ROOT,task)
    queue['gpu_queues']['server-2'].append(task)
    candidate.update(stage='admission',admission_receipt=receipt_path,post_training_freeze=plan,
        next_action='Use canonical gated server-2 dispatch for SAME admitted fixed30 Y56; verify real native steps. No blind retry.')
    bind(candidate,ref(receipt_path),plan,decision_ref,binding,authorization,ref(__file__))
    work.pop('design_cycle',None)
    tick.save(ROOT/'pipeline/research_work.json',work)
    monitor.save_queue(str(ROOT),queue)
    closure.validate_work(ROOT,tick.load(ROOT/'pipeline/headless/.tick_context.json'),work['candidates'],queue)
    print(json.dumps(dict(admitted='AdaRES-Y56',code_commit=head,receipt=ref(receipt_path),status='waiting',launched=False),indent=2))
finally:
    lock.close()
