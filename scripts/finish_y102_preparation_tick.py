"""Close first Y102 CPU preparation without replaying Y101 or original terminals."""
import argparse,ast,base64,json
from datetime import datetime
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from finish_y92_binding_tick import command,SKILL
from finish_y95_preparation_tick import fresh,GOAL,PAUSE
from y100_deployment import Y99,Y100,git
from prepare_y102_geometry_mask_skip import files
from y102_contract import validate_cpu
from run_y102_native import runtime,validate_binding
from check_ef_execution import validate_frozen
import closure,gpu_occupation
RUNS=(('server-1','AdaRES-Y99',2980836,Y99),('server-2','AdaRES-Y100',19305,Y100))

def evidence(r):return r['path']+' SHA'+r['sha256']

def record(token):
    lease(token,180);lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:
        assert git('rev-parse','HEAD').decode().strip()==Y100
        before={p:ref(p) for p in ('pipeline/queue.json','pipeline/running.json','pipeline/goal.json','SERVERS.yaml')};index=git('ls-files','--stage','-z')
        assert before['pipeline/goal.json']['sha256']==GOAL and ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')['sha256']==PAUSE
        work=tick.load(ROOT/'pipeline/research_work.json');c={r['id']:r for r in work['candidates']};new=c['AdaRES-Y102'];earlier=c['AdaRES-Y101']
        for item in (new,earlier):
            assert item['stage']=='blocked' and item['blocker_kind']=='exclusive_gpu_slot' and item['resource_work_kind']=='native_smoke'
            assert not item.get('native_smoke_result') and not item.get('admission_receipt')
        assert new['required_server']=='server-2' and earlier['required_server']=='server-1'
        assert earlier['native_preparation']['sha256']=='88d07bb84969209cc7604d419898c0965537c63b0674c656e855dccb1b4a549d'
        closure.artifact(ROOT,earlier['native_preparation']);closure.artifact(ROOT,earlier['CPU_result'])
        validate_cpu(tick.load(closure.artifact(ROOT,new['CPU_result'])))
        version=tick.load(closure.artifact(ROOT,new['source_manifest']));validate_binding(version,new,Y100)
        assert version['CPU_result']==new['CPU_result'] and version['code_commit']==new['code_commit']
        assert version['main_index_unchanged'] and version['original_Y99_Y100_preserved'] and len(version['exact_delta'])==7
        assert git('rev-parse',new['code_commit']+'^').decode().strip()==Y100
        source=tick.load(closure.artifact(ROOT,version['source_snapshot']));payload=source['CPU_tested_payload']
        assert payload==ast.literal_eval(ast.parse(closure.artifact(ROOT,new['CPU_program']).read_text(encoding='utf-8')).body[0].value)
        for p in files():
            v=payload['variant']['config_data'] if p.suffix=='.yaml' else payload['variant']['python_files'][p.name]
            assert p.read_bytes()==base64.b64decode(v['base64'])==git('show',new['code_commit']+':'+p.relative_to(ROOT).as_posix())
            assert ref(p)['sha256']==v['sha256']
        prep=tick.load(closure.artifact(ROOT,new['native_preparation']))
        assert prep['token']==token and prep['CPU_result']==new['CPU_result'] and prep['guard_tests_passed']==10 and prep['runtime_guard_tests_passed']==8
        assert prep['same_callbacks_actually_executed_on_CPU'] and prep['original_Y99_Y100_preserved']
        assert not prep['native_executed'] and not prep['admitted'] and not prep['production_sync']
        assert 'Ran 10 tests' in closure.artifact(ROOT,prep['guard_tests']).read_text(encoding='utf-8')
        assert 'Ran 8 tests' in closure.artifact(ROOT,prep['runtime_guard_tests']).read_text(encoding='utf-8')
        servers=monitor.load_servers(str(ROOT))
        for name,key in (('server-1','native_smoke_plan_server1'),('server-2','native_smoke_plan_server2_gpu0')):
            plan=tick.load(closure.artifact(ROOT,new[key]));f=ast.literal_eval(ast.parse(closure.artifact(ROOT,plan['program']).read_text(encoding='utf-8')).body[0].value)
            assert validate_frozen(new,plan,f)==7 and runtime(name,new,f,servers[name])==(350. if name=='server-1' else 280.)
            assert plan['execution_helper']==ref('scripts/run_y102_native.py')
        queue=monitor.load_queue(str(ROOT));assert monitor.load_running(str(ROOT))=={'server-1':{'0':'AdaRES-Y99'},'server-2':{'0':'AdaRES-Y100'}}
        context=tick.load(ROOT/'pipeline/headless/.tick_context.json');view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json');assert not view['preflight']['operational_issues']
        observations={};occupations={}
        for name,cid,pid,commit in RUNS:
            task=next(r for r in queue['gpu_queues'][name] if r['id']==cid)
            assert (task['status'],task['launch_verified'],task['pid'],task['code_commit'],str(task['gpu']))==('running',True,pid,commit,'0');closure.admission(ROOT,task)
            entry=c[cid]['latest_training_observation'];v=tick.load(closure.artifact(ROOT,entry))
            assert v['token']==token and v['pid']==pid and v['native_progress_confirmed'] is True
            fresh(v,'observed_local');observations[cid]=dict(evidence=entry,phase=v['after']['latest'],pid=pid)
            proof=ref('pipeline/checks/gpu_occupation_'+name+'_release_'+token+'.json');fresh(tick.load(closure.artifact(ROOT,proof)),'observed_at')
            held=earlier if name=='server-1' else new;sv=next(r for r in context['queue']['servers'] if r['name']==name)
            state=gpu_occupation.validated_state(ROOT,context,held,sv,queue)
            assert state['busy_gpu_ids']==state['reserved_gpu_ids']==[0] and not state['idle_gpu_ids'];occupations[name]=proof
        paused=next(r for r in queue['gpu_queues']['server-2'] if r['id']=='AdaRES-Y57')
        assert paused['status']=='paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']
        out=ROOT/('pipeline/checks/y102_preparation_closeout_'+token+'.json');assert not out.exists()
        tick.save(out,dict(complete=True,token=token,observed_at=datetime.now().astimezone().isoformat(),Y102_CPU=new['CPU_result'],Y102_source=new['source_manifest'],Y102_native_preparation=new['native_preparation'],preserved_Y101_preparation=earlier['native_preparation'],observations=observations,occupations=occupations,
            new_scientific_candidates=1,CPU_scientific_attempts=1,scientific_CPU_replays=0,native_GPU_tasks=0,admissions=0,launches=0,archives=0,checkpoint_loaded=False,formal_FIRST_or_cost_executed=False,
            evidence=[ref(__file__),new['CPU_result'],new['CPU_program'],new['source_manifest'],new['native_preparation'],earlier['native_preparation']]+[r['evidence'] for r in observations.values()]+list(occupations.values())))
        memory=ROOT/'pipeline/codex_memory.md';backup=ROOT/('pipeline/snapshots/codex_memory_before_'+token+'.md');assert not backup.exists()
        old_bytes=memory.read_bytes();backup.write_bytes(old_bytes);assert backup.read_bytes()==old_bytes
        old=old_bytes.decode('utf-8').splitlines();lines=[];updated=set()
        for line in old[:old.index('## Last Tick')]:
            if line.startswith('- Current token '):line='- Current token '+token+'; snapshot '+view['generated_at']+'; deadline '+view['deadline_at']+'; never extend.';updated.add('token')
            elif line.startswith('- Exact previous seven-section memory:'):line='- Exact previous seven-section memory: '+evidence(ref(backup))+'; complete historical routes/protocols/negative evidence/bans/dedicated recoveries byte-preserved; never reopening authority.';updated.add('backup')
            elif line.startswith('- AdaRES-Y99 original PID') or line.startswith('- AdaRES-Y100 original PID'):
                cid='AdaRES-Y99' if line.startswith('- AdaRES-Y99 ') else 'AdaRES-Y100';row=observations[cid];p=row['phase'];server='server1GPU0/350W' if cid=='AdaRES-Y99' else 'server2GPU0/280W'
                line='- '+cid+' original PID'+str(row['pid'])+'/'+server+', '+p['phase']+str(p['epoch'])+'/30 step'+str(p['step'])+'/'+str(p['count'])+'; exact original source/YAML/private runtime/power/increasing native counters '+evidence(row['evidence'])+'; original receipt/freeze intact.';updated.add(cid)
            elif line.startswith('- Fresh canonical occupation '):line='- Fresh canonical occupation '+'; '.join(evidence(r) for r in occupations.values())+'; both authorizedGPU0 busy/reserved by originalY99/Y100. Y101/Y102 wait for first native GPU tasks; original GPU1 frozen FIRST/cost remain user-held, never moved/substituted. Collector actual tick_attempt retained.';updated.add('occupation')
            elif line.startswith('- Next tick first observe '):line='- Next tick first observe Y99/Y100 and dispatch qualified waiting work. Y101 precedes Y102; do not replay passed CPU/version/plan tests. FIRST native run_y101_native.py then run_y102_native.py ONLY at HEADY100 after matching original Y99/server1 or Y100/server2 natural30/full finite exact checkpoint/canonical archive and fresh exclusive idle/unreserved authorizedGPU0. Changed HEAD/owner needs NEW tested same-byte binding; do not reuse old parent silently. Native PASS requires full original selected-device freeze/compatible original-run handoff/admission/source selection/dispatch/increasing counters same tick. Y99 terminal=finalize_y99_fixed30_y100.py; Y100 terminal=finalize_y100_fixed30.py after natural30/full original exact checks.';updated.add('next')
            lines.append(line)
        assert updated=={'token','backup','AdaRES-Y99','AdaRES-Y100','occupation','next'}
        at=lines.index('## Closed Or Failed Directions');lines[at:at]=[
            '- Y102 independent W2: add all original full-scene3D coordinates relative to all-valid point-mass scene centroid @zeroW3x256 to original static mask AFTER original batching, once/all seven readouts; +768/total164630405/noRNG. Complete original mask/projections/full inputs/initial queries/refinements/SCG/k/gamma/shared heads/hard gates/loss/selector/data/seed1999fixed30 unchanged; original padding/zero-mass branch preserved. Not Y77 sampled-position initial-query/Y69Y94 attention geometry/Y27Y31 mask-derived confidence; no new labels/bug premise/combination/tuning/reopening.',
            '- Y102 FIRST original CPU YAML/fullmodel/releasedbackbone/TRAIN collation PASS '+evidence(new['CPU_result'])+';162738points/8expressions. Independent scalarfloat32-64/centroid/projection/point-axis-scene permutations/translation/mass scale/paddedNaN/empty/zero-mass/scene-isolation/coordinate-mass gradients; seven-stage W2/RNG/weights/input-gradient anchors, causal initial-final masks/exact initial scores/bypass restoration and two finite synthetic updates. Original geometry/mass/batching/static7/projections/shared heads/gates/14 original visual parameter gradients verified. CPU is NOT native/efficacy.',
            '- Y102 exact7 tested additions retained '+new['retained_branch']+' commit'+new['code_commit']+'; '+evidence(new['source_manifest'])+'; parentHEADY100/mainindex/originalY99-Y100 and Y101 evidence preserved.10 actual-evidence+8 runtime guards PASS '+evidence(new['native_preparation'])+'; run_y102_native.py, both original runtime/data/mapping/350W280W GPU0 plans. NativeGPU/admission/launch unexecuted; Y101 goes first, changed HEAD/owner requires NEW tested binding.'
        ]
        lines+=['## Last Tick','- '+token+': canonical observe/dispatch existing qualified work first; no new physical terminal/operational issue. Original Y99/Y100 native counters increase; Y101 existing CPU/version/plans preserved without replay. Y102 FIRST CPU feasibility/isolated version/10+8 guards/unexecuted native plans completed; both authorizedGPU0 fresh busy/reserved.',
            '- New scientificIDs1/CPUscientificattempts1/CPUreplays0/nativeGPUtasks0/admissions0/launches0/archives0/FIRST0/cost0. Goal/original receipts/data/runtime/mappings/350W280W/seed1999fixed30/bans/GPU1+Y57 pauses/foreign jobs preserved. Closeout '+evidence(ref(out))+'; official finish/receipt/release authoritative.']
        assert len(lines)<120 and sum(r.startswith('## ') for r in lines)==7;memory.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
        notes=ROOT/('pipeline/reports/notes_'+token+'.json');assert not notes.exists()
        phases=[cid+' '+r['phase']['phase']+str(r['phase']['epoch'])+'/30 step'+str(r['phase']['step'])+'/'+str(r['phase']['count']) for cid,r in observations.items()]
        tick.save(notes,dict(actions=[
            '运行：'+'；'.join(phases)+'。原进程/源码/原生YAML/精确数据/私有环境/350W280W保持，原生步数递增；无新终态或派发故障，既有合格队列先检查，未重放历史终态工具。',
            '新准备：Y102独立W2将全部原场景三维坐标相对原点质量场景中心的偏移，经零初始化线性连接加入原静态掩码，保留原mask MLP、完整输入、原批处理与七次读出。首次原生YAML/完整CPU模型/发布backbone/原TRAIN加载通过，162738点8表达；标量float32/64/centroid/投影/置换/平移/质量缩放/填充NaN/空与零mass/场景隔离/坐标质量梯度、七阶段基线与输入梯度对齐、因果mask/初始scores精确/旁路恢复及两次有限synthetic decoder更新通过。CPU不是GPU原生检查或效果证明。',
            '七个实测科学文件保存隔离版本，生产HEAD/主索引/原运行/既有Y101及全部负证据保留；10项实际证据和8项运行计划检查通过。Y101准备不重跑，仍排在Y102之前。',
            '本轮新科学ID1/CPU科学尝试1/GPU原生0/准入0/新训练0/归档0/FIRST0/成本0。未加载任何活跃权重。'
        ],decisions=[
            '两张授权GPU0均实测busy/reserved。先Y101后Y102；首次原生检查均须匹配原Y99/Y100自然固定30轮、完整精确有限原检查点核验及规范归档，再验独占空闲GPU0。改变生产HEAD/原owner须新增同字节实测绑定，不静默复用旧parent；原生通过后同轮完整原设备协议/原运行兼容保护/准入/安全派发/步数核验。',
            '仅AdaRES；server2GPU1全部使用和Y57单独暂停保持，不迁移后评/成本、不重启受保护实验、不用server6、不改数据/功耗/调度、不干预外部任务。无新增用户处理事项。'
        ],evidence=[ref(out),ref(backup)]))
        assert all(ref(p)==r for p,r in before.items()) and git('ls-files','--stage','-z')==index
        return out,notes,len(lines)
    finally:lock.close()

def finish(token):
    lease(token,240)
    for _,cid,_,_ in RUNS:command(['scripts/y100_deployment.py','observe','--id',cid,'--token',token,'--label','release'])
    for name,_,_,_ in RUNS:command([SKILL+'/gpu_occupation.py','AdaRES_project','--token',token,'--server',name,'--label','release'])
    out,notes,lines=record(token);command([SKILL+'/foreground_tick.py','AdaRES_project','report','--token',token,'--notes',str(notes)]);command([SKILL+'/foreground_tick.py','AdaRES_project','finish','--token',token])
    receipt=tick.load(ROOT/('pipeline/tick_receipts/'+token+'.json'))
    assert receipt['status']=='FINALIZED' and receipt['postflight']=='PASS' and receipt['context_committed'] is True and receipt['lease_release']=='released'
    assert not (ROOT/'pipeline/headless/.foreground.lockdir').exists() and not (ROOT/'pipeline/headless/.tick_context_state.next.json').exists()
    pending=ROOT/'pipeline/archive_pending';assert not pending.exists() or not any(p.is_file() for p in pending.rglob('*'))
    assert ref('pipeline/goal.json')['sha256']==GOAL
    print(json.dumps(dict(closeout=ref(out),status=receipt['status'],postflight=receipt['postflight'],context_committed=True,lease_release=receipt['lease_release'],memory_lines=lines,observations={cid:r['phase'] for cid,r in tick.load(out)['observations'].items()},new_scientific_candidates=1,native_GPU_tasks=0,admissions=0,launches=0,archives=0)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);finish(p.parse_args().token)
