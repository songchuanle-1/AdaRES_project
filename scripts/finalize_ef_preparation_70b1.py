"""Record tested preparation and a genuine whole-GPU wait, not a smoke PASS."""

import argparse
import csv
from datetime import datetime,timedelta
import io
import json
from experiment_first_70b1 import ROOT,TOKEN,lease,ref,bind,tick,monitor

parser=argparse.ArgumentParser()
parser.add_argument('--occupation',required=True)
args=parser.parse_args()
lease()
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    occupation=tick.load(ROOT/args.occupation)
    now=datetime.now().astimezone()
    assert occupation['complete'] and 0<=(now-datetime.fromisoformat(occupation['observed_at'])).total_seconds()<180
    live=occupation['live']
    inventory={int(row[0]):row for row in csv.reader(io.StringIO(live['authorized_device_inventory']))}
    processes={int(row[1]):row for row in csv.reader(io.StringIO(live['compute']))}
    assert set(inventory)=={0,1} and set(processes)=={83638,95947}
    queue=monitor.load_queue(str(ROOT))
    tasks={t['id']:t for t in queue['gpu_queues']['server-2']}
    assert set(tasks)=={'AdaRES-Y56','AdaRES-Y57'}
    for row in occupation['summary']:
        task=tasks[row['id']]
        assert task['status']=='running' and task['pid']==row['pid']
        gpu=int(task['gpu'])
        assert processes[row['pid']][0].strip()==inventory[gpu][1].strip()
        assert float(inventory[gpu][4])>=1024
    work=tick.load(ROOT/'pipeline/research_work.json')
    actual='pipeline/analyses/ef_cpu_'+TOKEN+'.json'
    cpu=tick.load(ROOT/actual)
    assert cpu['complete'] and cpu['implementation_test_passed'] and not cpu['native_smoke_passed']
    current_resource=ref(args.occupation)
    decision=monitor.log_decision(str(ROOT),'next_batch',
        '实验优先新授权已绑定；已有Y56/Y57真实训练，两项就绪waiting为0。',
        '在原Y58/Y59下完成独立包、原生YAML、完整CPU构造/预训练/真实TRAIN拼批/模式实测，精确本地保存14文件；保留两卡训练，原生两步检查等待整卡空闲。',
        '新授权复用常规实验操作许可，不重复审批；CPU及静态入口检查不代替真实GPU后向，不发准入收据或伪造waiting。',
        actual+'; '+args.occupation+'; pipeline/analyses/ef_version_'+TOKEN+'.json')
    for c in work['candidates']:
        if c['id'] in ('AdaRES-Y58','AdaRES-Y59'):
            c.update(stage='blocked',blocked_from_stage='validation',blocker_kind='exclusive_gpu_slot',
                required_server='server-2',retry_after=(now+timedelta(hours=1)).isoformat(),
                blocker='The exact Y56/Y57 live training processes occupy both authorized GPUs. '
                        'Independent implementation, native YAML, released-backbone CPU construction, '
                        'actual TRAIN collate and mode tests passed. Two-step sparse native GPU smoke '
                        'cannot run without oversubscribing protected training; no authorization or mount fault.',
                unblock_condition='A queue opening wakes this same ID; immediately recheck live authorized '
                    'GPU memory/utilization/all compute PIDs and exclusive lease, exact committed bytes and '
                    'current goal, then run the saved two-step native smoke and admit/dispatch if it passes.',
                occupation_evidence=current_resource,
                next_action='Run native_smoke_plan on the first genuinely idle server-2 GPU0/1 with the '
                    'CURRENT owning token and exact tested commit/YAML/goal. Require two actual finite '
                    'TRAIN loss/backward optimizer steps and observed mode path; then save smoke receipt, '
                    'admit to the existing queue and try gated dispatch in that same tick. Do not request '
                    'another routine commit/train approval or rerun the old trained-W2 efficacy diagnosis first.',
                preparation_scope='Independent implementation and native YAML CPU checks complete; '
                    'GPU two-step smoke, peak-resource check, admission and training pending.',
                implementation_test=ref(actual))
            bind(c,current_resource,ref(decision),ref(__file__))
        elif c['id'] in tasks:
            bind(c,current_resource)
    old=work.pop('design_cycle',None)
    if old:
        work.setdefault('legacy_design_cycle_history',[]).append(old)
    work['tick_attempt']=dict(tick_started_at=tick.load(ROOT/'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id='AdaRES-Y59',kind='implementation_test',result='passed',
        command='python -B scripts/run_ef_cpu_70b1.py',
        observation='Both exact independent packages resolved via native YAML, fully constructed on CPU, '
                    'loaded released backbone and collated real TRAIN data. Y58 kept 45 backbone BN fixed '
                    'and 14 decoder BN training; Y59 restored frozen encoder eval with repeat error0 and '
                    'preserved a trainable encoder. No sparse GPU task backward executed; both slots busy.',
        evidence=[ref(actual),ref(actual.replace('.json','.log'))])
    lease()
    tick.save(ROOT/'pipeline/research_work.json',work)
    print(json.dumps(dict(implementation_tests_passed=2,admissions=0,new_training=0,
        waiting=0,resource=current_resource,decision=ref(decision),
        pending_native_smoke=['AdaRES-Y58','AdaRES-Y59']),ensure_ascii=False,indent=2))
finally:
    lock.close()
