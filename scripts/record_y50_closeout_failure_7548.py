"""Preserve exact failed closeout; leave the verified training untouched."""
from datetime import datetime,timezone
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import foreground_tick as tick
import pipeline_monitor as monitor
TOKEN='75486f9263d34214b3bd2e1412c69639'
owner=tick.load(ROOT/'pipeline/headless/.foreground.lockdir/owner.json')
assert owner['token']==TOKEN and owner['thread_id']==os.environ['CODEX_THREAD_ID']
assert datetime.now(timezone.utc)<datetime.fromisoformat(owner['deadline_at'])
def ref(p):return dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest())
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    queue=monitor.load_queue(str(ROOT))
    task=next(t for t in queue['gpu_queues']['server-2'] if t['id']=='AdaRES-Y50')
    assert task['status']=='running' and task['launch_verified'] and task['pid']==31539
    path='pipeline/analyses/y50_closeout_failure_7548_'+TOKEN+'.json'
    assert not (ROOT/path).exists()
    tick.save(ROOT/path,dict(complete=True,recorded_at_local=datetime.now().astimezone().isoformat(),
        tick_execution_status='INCOMPLETE',scientific_status='training',
        actual_finish_exit=1,actual_finish_output='CLOSURE_INCOMPLETE: External resource stop lacks blocked candidate\nPOSTFLIGHT FAILED: lease retained; continue this tick or explicitly abort',
        unchanged_verified_task=task,
        interpretation='User-scoped tested commit, admission and actual training launch succeeded. Generic capacity=2 closeout cannot accept observed foreign-occupied second card without a separate durable blocked scientific candidate; no such candidate was invented.',
        response='Preserve all evidence; canonical abort releases the control lease only, never stops the already authorized PID31539 training. Do not repeat finish with unchanged evidence or weaken the checker.',
        prior_authorization=ref('pipeline/authorizations/AdaRES-Y50_commit.json'),
        actual_training=ref('pipeline/analyses/y50_started_7548_'+TOKEN+'.json')))
    work=tick.load(ROOT/'pipeline/research_work.json')
    candidate=next(c for c in work['candidates'] if c['id']=='AdaRES-Y50')
    assert candidate['stage']=='training'
    candidate['evidence'].append(ref(path))
    tick.save(ROOT/'pipeline/research_work.json',work)
    memory=ROOT/'pipeline/codex_memory.md'
    text=memory.read_text(encoding='utf-8')
    text=text.replace('TickfinishPENDING, user-scopedcommit/launchalreadyexecuted.','TickINCOMPLETE: postflight rejected external-capacity disposition; scopedcommit/launchalreadyexecuted. Abort/releasepending, trainingnotstopped.')
    text=text.replace('- TickfinishstillPENDING; ifpostflightcannotacceptfixedcapacity/scopedaction, recordexactfailureandabort/releasewithoutstoppingPID31539. No postreleaseprojectwrites.',
        '- Exactfinishfailure: CLOSURE_INCOMPLETE: External resource stop lacks blocked candidate. No fabricatedsecondcandidate orrepeatedfinish; canonicalabort/releasePENDING, neverstopPID31539. Evidence'+path+'. No postreleaseprojectwrites.')
    assert len(text.splitlines())<120
    memory.write_text(text,encoding='utf-8')
    notes=ROOT/('pipeline/analyses/tick_notes_7548_'+TOKEN+'.json')
    record=tick.load(notes)
    record['actions'].append('实际finish返回1：CLOSURE_INCOMPLETE: External resource stop lacks blocked candidate。用户限定提交、准入和递增训练启动均已实施；tick执行状态INCOMPLETE，科学状态training。保留证据并按契约abort释放租约，不停止或重启PID31539。')
    record['decisions'].append('不为通用容量校验补造阻塞候选或更改目标/门槛，不重复相同finish；解除控制租约不撤销已授权Y50训练，也不需要用户再次授权。')
    tick.save(notes,record)
    print(path)
finally:lock.close()
