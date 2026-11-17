"""Park only the prepared native diagnostics against verified full GPU occupancy."""

import argparse
import csv
from datetime import datetime,timedelta,timezone
import io
import json
from research_b780 import ROOT,TOKEN,lease,ref,bind,tick,monitor
import closure

parser=argparse.ArgumentParser()
parser.add_argument('--occupation',required=True)
args=parser.parse_args()
lease()
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work=tick.load(ROOT/'pipeline/research_work.json')
    queue=monitor.load_queue(str(ROOT))
    evidence=ref(args.occupation)
    observed=tick.load(ROOT/args.occupation)
    now=datetime.now(timezone.utc).astimezone()
    age=(now-datetime.fromisoformat(observed['observed_until'])).total_seconds()
    assert observed['complete'] and 0<=age<180
    live=observed['live']
    assert live['hostname']=='dl01-MS-7C60'
    assert live['source_marker']=='05dc2c0424e33f460f96a4ab4938b0d05e7dc6ea'
    devices={int(row[0]):row for row in csv.reader(io.StringIO(live['authorized_device_inventory']))}
    processes={int(row[1]):row for row in csv.reader(io.StringIO(live['compute']))}
    tasks={t['id']:t for t in queue['gpu_queues']['server-2'] if t['id'] in ('AdaRES-Y56','AdaRES-Y57')}
    assert len(devices)==2 and set(devices)=={0,1} and len(tasks)==2
    for row in observed['summary']:
        task=tasks[row['id']]
        gpu=int(task['gpu'])
        assert task['status']=='running' and task['pid']==row['pid'] and task['pid'] in processes
        assert processes[task['pid']][0].strip()==devices[gpu][1].strip()
        assert float(devices[gpu][4])>=1024 and row['distinct_native_steps']>=2
    review=ref('pipeline/reviews/motivation_review_Y58_Y59_b780.md')
    design=ref('pipeline/designs/exp_design_Y58_Y59_b780.md')
    for number in (58,59):
        c=next(c for c in work['candidates'] if c['id']=='AdaRES-Y'+str(number))
        assert c['stage']=='diagnosis'
        plan=ref('pipeline/analyses/y'+str(number)+'_native_'+TOKEN+'_prereg.json')
        static=ref('pipeline/analyses/y'+str(number)+'_native_static_'+TOKEN+'.json')
        assert tick.load(ROOT/static['path'])['native_executed'] is False
        c.update(stage='blocked',blocked_from_stage='diagnosis',blocker_kind='exclusive_gpu_slot',
            required_server='server-2',retry_after=(now+timedelta(hours=1)).isoformat(),
            blocker='All authorized GPUs0/1 are occupied by the exact live Y56/Y57 training processes; '
                    'CPU/data/language controls and source-bound native preparation are complete. '
                    'No permission, mount or connection blocker is being claimed.',
            unblock_condition='A queue opening wakes this SAME ID early; then freshly verify an authorized '
                              'GPU is idle, no compute process is present, the current foreground lease and '
                              'frozen source/checkpoint/dataset match, and acquire an exclusive diagnostic lease.',
            native_manifest=plan,occupation_evidence=evidence,
            next_action='Resume the SAME frozen native diagnostic from diagnosis on a genuinely free '
                        'authorized GPU under the new current token. Preserve all repeat/negative outcomes; '
                        'evaluate task relevance before candidate implementation. No admission or training yet.',
            attempted_alternatives=[
                dict(test='Non-GPU matched component/language/data controls',
                     result='Completed; CPU scope only, native task effect remains pending.'),
                dict(test='Source-bound native test preparation and static fail-closed entry guard',
                     result='Completed without CUDA; actual execution is not claimed.'),
                dict(test='Fresh exact PID/cwd/config/native-step and all authorized GPU compute queries',
                     result='Both authorized slots occupied; preserve ongoing training, no spare allowed GPU.')],
            preparation_scope='Diagnostic reserve only. All six full candidate admission gates remain pending.')
        bind(c,evidence,review,design,plan,static,ref(__file__),ref('scripts/snapshot_reserve_b780.py'))
    for c in work['candidates']:
        if c['id'] in tasks:
            bind(c,evidence)
    work['design_cycle']=dict(stop_reason='external_resource_blocked',
        attempted_ids=['AdaRES-Y58','AdaRES-Y59'],
        evidence=[evidence,review,design],retry_after=(now+timedelta(hours=1)).isoformat(),
        resume_condition='Bounded two-ID reserve preparation is complete. Continue the same native '
                         'diagnostics on a verified authorized queue/GPU opening; no new train admission '
                         'or scheduler change in this reserve-only tick.')
    closure.validate_work(ROOT,tick.load(ROOT/'pipeline/headless/.tick_context.json'),work['candidates'],queue)
    decision=monitor.log_decision(str(ROOT),'next_batch',
        'Y56/Y57 两卡真实训练满载，本轮准入预算0、储备准备预算2；如何推进研究？',
        '完成Y58归一化同伴依赖、Y59冻结语言模式两条独立CPU诊断及原生预案；按实占用等待独占空卡，不增加队列或训练。',
        'CPU见证不是任务收益；两项原生诊断未运行、六项准入门槛仍待完成。原训练进程及固定协议不改，无新增人工授权要求。',
        evidence['path']+'; '+review['path']+'; '+design['path'])
    for c in work['candidates']:
        if c['id'] in ('AdaRES-Y58','AdaRES-Y59'):
            bind(c,ref(decision))
    lease()
    tick.save(ROOT/'pipeline/research_work.json',work)
    print(json.dumps(dict(parked=['AdaRES-Y58','AdaRES-Y59'],occupation=evidence,
                          decision=ref(decision),queue_changed=False,training_launched=False),ensure_ascii=False,indent=2))
finally:
    lock.close()
