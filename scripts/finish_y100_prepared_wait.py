"""Close an evidenced GPU0 wait for the two already CPU-tested candidates."""
import argparse,json
from datetime import datetime
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from finish_y92_binding_tick import command,SKILL
from finish_y95_preparation_tick import fresh,GOAL,PAUSE
from y100_deployment import Y99,Y100,git
import closure,gpu_occupation
RUNS=(('server-1','AdaRES-Y99',2980836,Y99,'AdaRES-Y101'),('server-2','AdaRES-Y100',19305,Y100,'AdaRES-Y102'))
PREPARATIONS={
    'AdaRES-Y101':'88d07bb84969209cc7604d419898c0965537c63b0674c656e855dccb1b4a549d',
    'AdaRES-Y102':'120467eec12f97daef45155adcf151a682eec49407ba40b09704e11d0f71b63b'}

def evidence(r):return r['path']+' SHA'+r['sha256']

def record(token):
    lease(token,180);lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:
        assert git('rev-parse','HEAD').decode().strip()==Y100
        before={p:ref(p) for p in ('pipeline/queue.json','pipeline/running.json','pipeline/goal.json','SERVERS.yaml')};index=git('ls-files','--stage','-z')
        assert before['pipeline/goal.json']['sha256']==GOAL and ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')['sha256']==PAUSE
        queue=monitor.load_queue(str(ROOT));assert monitor.load_running(str(ROOT))=={'server-1':{'0':'AdaRES-Y99'},'server-2':{'0':'AdaRES-Y100'}}
        work=tick.load(ROOT/'pipeline/research_work.json');c={r['id']:r for r in work['candidates']}
        view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json');context=tick.load(ROOT/'pipeline/headless/.tick_context.json')
        assert not view['preflight']['operational_issues'] and not view['changes']['new_terminal_ids'] and view['queue']['ready_waiting_count']==0
        paused=next(r for r in queue['gpu_queues']['server-2'] if r['id']=='AdaRES-Y57')
        assert paused['status']=='paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization'] is True
        observations={};occupations={};prepared={}
        for name,cid,pid,commit,heldid in RUNS:
            task=next(r for r in queue['gpu_queues'][name] if r['id']==cid)
            assert (task['status'],task['launch_verified'],task['pid'],task['code_commit'],str(task['gpu']))==('running',True,pid,commit,'0');closure.admission(ROOT,task)
            entry=c[cid]['latest_training_observation'];v=tick.load(closure.artifact(ROOT,entry))
            assert v['token']==token and v['native_progress_confirmed'] is True and v['pid']==pid and v['source_marker'] in (commit,Y100)
            fresh(v,'observed_local');observations[cid]=dict(evidence=entry,phase=v['after']['latest'],pid=pid)
            held=c[heldid];assert held['stage']=='blocked' and held['blocker_kind']=='exclusive_gpu_slot' and held['resource_work_kind']=='native_smoke'
            assert held['required_server']==name and held['required_gpu']==0 and not held.get('native_smoke_result') and not held.get('admission_receipt')
            assert held['implementation_test_passed'] is True and held['native_preparation']['sha256']==PREPARATIONS[heldid]
            prep=tick.load(closure.artifact(ROOT,held['native_preparation']))
            assert prep['guard_tests_passed']==10 and prep['runtime_guard_tests_passed']==8 and not prep['native_executed'] and not prep['admitted']
            for r in (held['CPU_result'],held['source_manifest'],held['additive_version_binding'],held['native_smoke_plan_server1'],held['native_smoke_plan_server2_gpu0']):closure.artifact(ROOT,r)
            binding=tick.load(closure.artifact(ROOT,held['additive_version_binding']))
            assert binding['parent_commit']==binding['working_HEAD']==Y100 and binding['matching_current_owners']=={'server-1':'AdaRES-Y99','server-2':'AdaRES-Y100'}
            prepared[heldid]=dict(CPU_result=held['CPU_result'],source=held['source_manifest'],native_preparation=held['native_preparation'])
            proof=ref('pipeline/checks/gpu_occupation_'+name+'_release_'+token+'.json');fresh(tick.load(closure.artifact(ROOT,proof)),'observed_at')
            sv=next(r for r in context['queue']['servers'] if r['name']==name);state=gpu_occupation.validated_state(ROOT,context,held,sv,queue)
            assert state['busy_gpu_ids']==state['reserved_gpu_ids']==[0] and not state['idle_gpu_ids'];occupations[name]=proof
        attempt=work['tick_attempt'];assert attempt['tick_started_at']==view['generated_at'] and attempt['kind']=='resource_check' and attempt['result']=='blocked'
        out=ROOT/('pipeline/checks/y100_prepared_wait_'+token+'.json');assert not out.exists()
        tick.save(out,dict(complete=True,token=token,observed_at=datetime.now().astimezone().isoformat(),observations=observations,occupations=occupations,prepared_candidates=prepared,
            original_runs_preserved=True,waiting_native_order=['AdaRES-Y101','AdaRES-Y102'],new_scientific_candidates=0,scientific_CPU_replays=0,native_GPU_tasks=0,admissions=0,launches=0,archives=0,
            checkpoints_loaded=False,repeated_scientific_or_binding_tests=False,formal_FIRST_or_cost_executed=False,
            observation='Two independent CPU-tested versions and native plans already preserved; both original GPU0 owners still run. No legally available native slot. Preserve their bounded preparation and scientific evidence without equivalent tests or a quota-forced successor.',
            evidence=[ref(__file__)]+[r['native_preparation'] for r in prepared.values()]+[r['evidence'] for r in observations.values()]+list(occupations.values())))
        memory=ROOT/'pipeline/codex_memory.md';backup=ROOT/('pipeline/snapshots/codex_memory_before_'+token+'.md');assert not backup.exists()
        old_bytes=memory.read_bytes();backup.write_bytes(old_bytes);assert backup.read_bytes()==old_bytes
        old=old_bytes.decode('utf-8').splitlines();lines=[];updated=set()
        for line in old[:old.index('## Last Tick')]:
            if line.startswith('- Current token '):line='- Current token '+token+'; snapshot '+view['generated_at']+'; deadline '+view['deadline_at']+'; never extend.';updated.add('token')
            elif line.startswith('- Exact previous seven-section memory:'):line='- Exact previous seven-section memory: '+evidence(ref(backup))+'; complete historical protocols, source, negative evidence, bans and dedicated recovery byte-preserved; never reopening authority.';updated.add('backup')
            elif line.startswith('- AdaRES-Y99 original PID') or line.startswith('- AdaRES-Y100 original PID'):
                cid='AdaRES-Y99' if line.startswith('- AdaRES-Y99 ') else 'AdaRES-Y100';row=observations[cid];p=row['phase'];server='server1GPU0/350W' if cid=='AdaRES-Y99' else 'server2GPU0/280W'
                line='- '+cid+' original PID'+str(row['pid'])+'/'+server+', '+p['phase']+str(p['epoch'])+'/30 step'+str(p['step'])+'/'+str(p['count'])+'; exact original source/YAML/private runtime/power/increasing native counters '+evidence(row['evidence'])+'; original receipt/freeze unchanged.';updated.add(cid)
            elif line.startswith('- Fresh canonical occupation '):line='- Fresh canonical occupation '+'; '.join(evidence(r) for r in occupations.values())+'; both authorizedGPU0 busy/reserved by originalY99/Y100. Y101/Y102 FIRST native tasks remain waiting, existing CPU/version/plans not replayed; original GPU1 frozen FIRST/cost remain user-held and never moved/substituted. Collector actual tick_attempt retained.';updated.add('occupation')
            lines.append(line)
        assert updated=={'token','backup','AdaRES-Y99','AdaRES-Y100','occupation'}
        lines+=['## Last Tick','- '+token+': canonical observation/dispatch of existing qualified work first; no new physical terminal/operational issue. Original Y99/Y100 increasing counters; exact PIDs/source/YAML/private runtimes/350W280W confirmed, fresh physical busy/reserved bothGPU0. Two existing independent Y101/Y102 CPU versions and native plans preserved, first native checks await their exact original-owner completion/archive/exclusive idle resources. Y101 first; no preparation/binding/scientific replay.',
            '- New scientificIDs0/CPUscientificreplays0/nativeGPUtasks0/admissions0/launches0/archives0/FIRST0/cost0. No checkpoints loaded, restart/migration, tuning, resource/pause/power/data/scheduler change or new user action. Closeout '+evidence(ref(out))+'; official finish/receipt/release authoritative.']
        assert len(lines)<120 and sum(r.startswith('## ') for r in lines)==7;memory.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
        notes=ROOT/('pipeline/reports/notes_'+token+'.json');assert not notes.exists()
        phases=[cid+' '+r['phase']['phase']+str(r['phase']['epoch'])+'/30 step'+str(r['phase']['step'])+'/'+str(r['phase']['count']) for cid,r in observations.items()]
        tick.save(notes,dict(actions=[
            '运行：'+'；'.join(phases)+'。两条原训练按原PID/源码/原生YAML/私有环境/功耗继续，原生步数递增。先检查已有合格队列，待跑0，无新终态或运行故障。',
            '资源核验：两张授权GPU0均实测busy/reserved，恰由原Y99/Y100占用，无可执行的原生GPU空槽。已有两个独立Y101/Y102完成CPU实现、隔离版本和原生计划，原证据保留，Y101优先；本轮不重跑已通过的检查，也不为凑额度生成等价方案。',
            '本轮新科学ID0、科学CPU重跑0、原生GPU检查0、准入0、新训练0、归档0、FIRST0、成本0。未加载活跃权重，训练效果仍待完整原协议后评。'
        ],decisions=[
            '保留原Y99/Y100固定30轮训练。Y101/Y102首次原生检查继续要求匹配原owner自然30轮、完整精确有限原检查点核验和规范归档，再验授权独占空闲GPU0；改变HEAD/owner须新测试绑定，不能复用旧parent。原生通过后同轮完整原设备协议/原运行兼容保护/准入/安全派发与原生步数核验。',
            '仅AdaRES；server2GPU1全部使用与Y57单独暂停保持，不迁移冻结后评/成本、不重启受保护实验、不用server6、不改数据/功耗/调度、不干预外部任务。无新增用户处理事项。'
        ],evidence=[ref(out),ref(backup)]))
        assert all(ref(p)==r for p,r in before.items()) and git('ls-files','--stage','-z')==index
        return out,notes,len(lines)
    finally:lock.close()

def finish(token):
    lease(token,240)
    work=tick.load(ROOT/'pipeline/research_work.json');c={r['id']:r for r in work['candidates']}
    for _,cid,_,_,_ in RUNS:
        value=tick.load(closure.artifact(ROOT,c[cid]['latest_training_observation']))
        age=(datetime.now().astimezone()-datetime.fromisoformat(value['observed_local'])).total_seconds()
        if value['token']!=token or age>=180:
            command(['scripts/y100_deployment.py','observe','--id',cid,'--token',token,'--label','release_refresh'])
    for name,_,_,_,_ in RUNS:command([SKILL+'/gpu_occupation.py','AdaRES_project','--token',token,'--server',name,'--label','release'])
    out,notes,lines=record(token);command([SKILL+'/foreground_tick.py','AdaRES_project','report','--token',token,'--notes',str(notes)]);command([SKILL+'/foreground_tick.py','AdaRES_project','finish','--token',token])
    receipt=tick.load(ROOT/('pipeline/tick_receipts/'+token+'.json'))
    assert receipt['status']=='FINALIZED' and receipt['postflight']=='PASS' and receipt['context_committed'] is True and receipt['lease_release']=='released'
    assert not (ROOT/'pipeline/headless/.foreground.lockdir').exists() and not (ROOT/'pipeline/headless/.tick_context_state.next.json').exists()
    pending=ROOT/'pipeline/archive_pending';assert not pending.exists() or not any(p.is_file() for p in pending.rglob('*'))
    assert ref('pipeline/goal.json')['sha256']==GOAL
    print(json.dumps(dict(closeout=ref(out),status=receipt['status'],postflight=receipt['postflight'],context_committed=True,lease_release=receipt['lease_release'],memory_lines=lines,observations={cid:r['phase'] for cid,r in tick.load(out)['observations'].items()},new_scientific_candidates=0,native_GPU_tasks=0,admissions=0,launches=0,archives=0)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);finish(p.parse_args().token)
