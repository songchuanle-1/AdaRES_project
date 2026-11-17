"""Close real Y101 CPU preparation while preserving original Y99/Y100 training."""
import argparse,ast,base64,json
from datetime import datetime
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from finish_y92_binding_tick import command,SKILL
from finish_y95_preparation_tick import fresh,GOAL,PAUSE
from y100_deployment import Y99,Y100,git
from prepare_y101_query_ffn_skip import files
from y101_contract import validate_cpu
from run_y101_native import runtime,validate_binding
from check_ef_execution import validate_frozen
import closure,gpu_occupation

RUNS=(('server-1','AdaRES-Y99',2980836,Y99),('server-2','AdaRES-Y100',19305,Y100))


def evidence(r):return r['path']+' SHA'+r['sha256']


def record(token):
    lease(token,180);writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        assert git('rev-parse','HEAD').decode().strip()==Y100
        before={p:ref(p) for p in ('pipeline/queue.json','pipeline/running.json','pipeline/goal.json','SERVERS.yaml')}
        index=git('ls-files','--stage','-z')
        assert before['pipeline/goal.json']['sha256']==GOAL and ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')['sha256']==PAUSE
        work=tick.load(ROOT/'pipeline/research_work.json');c={r['id']:r for r in work['candidates']}
        new=c['AdaRES-Y101'];original=c['AdaRES-Y100']
        assert new['stage']=='blocked' and new['blocker_kind']=='exclusive_gpu_slot' and new['resource_work_kind']=='native_smoke'
        assert not new.get('native_smoke_result') and not new.get('admission_receipt')
        cpu=tick.load(closure.artifact(ROOT,new['CPU_result']));validate_cpu(cpu)
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
        assert prep['token']==token and prep['CPU_result']==new['CPU_result']
        assert prep['guard_tests_passed']==10 and prep['runtime_guard_tests_passed']==8
        assert prep['same_callbacks_actually_executed_on_CPU'] and prep['original_Y99_Y100_preserved']
        assert not prep['native_executed'] and not prep['admitted'] and not prep['production_sync']
        assert 'Ran 10 tests' in closure.artifact(ROOT,prep['guard_tests']).read_text(encoding='utf-8')
        assert 'Ran 8 tests' in closure.artifact(ROOT,prep['runtime_guard_tests']).read_text(encoding='utf-8')
        servers=monitor.load_servers(str(ROOT))
        for name,key in (('server-1','native_smoke_plan_server1'),('server-2','native_smoke_plan_server2_gpu0')):
            plan=tick.load(closure.artifact(ROOT,new[key]));program=closure.artifact(ROOT,plan['program']).read_text(encoding='utf-8')
            f=ast.literal_eval(ast.parse(program).body[0].value)
            assert validate_frozen(new,plan,f)==7
            assert runtime(name,new,f,servers[name])==(350. if name=='server-1' else 280.)
            assert plan['execution_helper']==ref('scripts/run_y101_native.py')
        handoff=tick.load(closure.artifact(ROOT,original['terminal_handoff_plan']))
        assert handoff['token']==token and handoff['CPU_guard_tests_passed']==29
        assert handoff['full_terminal_verification_executed'] is False and not handoff['checkpoint_loaded'] and not handoff['archived']
        assert handoff['payload']['allowed_markers']==[Y100]
        location=tick.load(closure.artifact(ROOT,handoff['declared_location']))
        assert location['checkpoint_directory']=='20261111_085241' and not location['checkpoint_loaded'] and not location['remote_mutations']
        for d in handoff['dependencies']:closure.artifact(ROOT,d)
        queue=monitor.load_queue(str(ROOT));assert monitor.load_running(str(ROOT))=={'server-1':{'0':'AdaRES-Y99'},'server-2':{'0':'AdaRES-Y100'}}
        context=tick.load(ROOT/'pipeline/headless/.tick_context.json');view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json')
        assert not view['preflight']['operational_issues']
        observations={};occupations={}
        for name,cid,pid,commit in RUNS:
            task=next(r for r in queue['gpu_queues'][name] if r['id']==cid)
            assert (task['status'],task['launch_verified'],task['pid'],task['code_commit'],str(task['gpu']))==('running',True,pid,commit,'0')
            closure.admission(ROOT,task)
            entry=c[cid]['latest_training_observation'];v=tick.load(closure.artifact(ROOT,entry))
            assert v['token']==token and v['pid']==pid and v['native_progress_confirmed'] is True
            fresh(v,'observed_local');observations[cid]=dict(evidence=entry,phase=v['after']['latest'],pid=pid)
            proof=ref('pipeline/checks/gpu_occupation_'+name+'_release_'+token+'.json');fresh(tick.load(closure.artifact(ROOT,proof)),'observed_at')
            held=new if name=='server-1' else c['AdaRES-Y98']
            server_view=next(r for r in context['queue']['servers'] if r['name']==name)
            state=gpu_occupation.validated_state(ROOT,context,held,server_view,queue)
            assert state['busy_gpu_ids']==state['reserved_gpu_ids']==[0] and not state['idle_gpu_ids']
            occupations[name]=proof
        paused=next(r for r in queue['gpu_queues']['server-2'] if r['id']=='AdaRES-Y57')
        assert paused['status']=='paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']
        out=ROOT/('pipeline/checks/y101_preparation_closeout_'+token+'.json');assert not out.exists()
        tick.save(out,dict(complete=True,token=token,observed_at=datetime.now().astimezone().isoformat(),
            Y101_CPU=new['CPU_result'],Y101_source=new['source_manifest'],Y101_native_preparation=new['native_preparation'],
            Y100_terminal_preparation=original['terminal_handoff_plan'],observations=observations,occupations=occupations,
            new_scientific_candidates=1,CPU_scientific_attempts=1,scientific_CPU_replays=0,native_GPU_tasks=0,admissions=0,launches=0,archives=0,
            full_checkpoint_verification_executed=False,formal_FIRST_or_cost_executed=False,
            evidence=[ref(__file__),new['CPU_result'],new['CPU_program'],new['source_manifest'],new['native_preparation'],original['terminal_handoff_plan']]+[r['evidence'] for r in observations.values()]+list(occupations.values())))
        memory=ROOT/'pipeline/codex_memory.md';backup=ROOT/('pipeline/snapshots/codex_memory_before_'+token+'.md');assert not backup.exists()
        old_bytes=memory.read_bytes();backup.write_bytes(old_bytes);assert backup.read_bytes()==old_bytes
        old=old_bytes.decode('utf-8').splitlines();lines=[];updated=set()
        for line in old[:old.index('## Last Tick')]:
            if line.startswith('- Current token '):
                line='- Current token '+token+'; snapshot '+view['generated_at']+'; deadline '+view['deadline_at']+'; never extend.';updated.add('token')
            elif line.startswith('- Exact previous seven-section memory:'):
                line='- Exact previous seven-section memory: '+evidence(ref(backup))+'; complete historical routes/protocols/negative evidence/bans/dedicated recoveries byte-preserved; never reopening authority.';updated.add('backup')
            elif line.startswith('- AdaRES-Y99 original PID') or line.startswith('- AdaRES-Y100 original PID'):
                cid='AdaRES-Y99' if line.startswith('- AdaRES-Y99 ') else 'AdaRES-Y100';row=observations[cid];p=row['phase'];server='server1GPU0/350W' if cid=='AdaRES-Y99' else 'server2GPU0/280W'
                line='- '+cid+' original PID'+str(row['pid'])+'/'+server+', '+p['phase']+str(p['epoch'])+'/30 step'+str(p['step'])+'/'+str(p['count'])+'; exact original source/YAML/private runtime/power/native counter increase '+evidence(row['evidence'])+'; original receipt/freeze unchanged.';updated.add(cid)
            elif line.startswith('- Fresh canonical occupation '):
                line='- Fresh canonical occupation '+'; '.join(evidence(r) for r in occupations.values())+'; both authorizedGPU0 busy/reserved by originalY99/Y100. Y101 native waits; original GPU1 frozen FIRST/cost remain user-held, never moved/substituted. Collector actual tick_attempt retained.';updated.add('occupation')
            elif line.startswith('- Next tick first observe '):
                line='- Next tick first observe Y99/Y100, dispatch qualified waiting work. Do not replay Y101 CPU/version/plan tests or archived terminals. FIRST native run_y101_native.py ONLY at HEADY100 after matching original Y99/server1 or Y100/server2 natural30/full finite exact checkpoint/canonical archive plus fresh exclusive idle/unreserved authorizedGPU0; then full original selected-device freeze/admission/source selection/dispatch/increasing counters same tick. Changed HEAD/owner requires NEW tested binding. Y99 terminal=finalize_y99_fixed30_y100.py; Y100 terminal=finalize_y100_fixed30.py only after natural30/full exact terminal checks.';updated.add('next')
            lines.append(line)
        assert updated=={'token','backup','AdaRES-Y99','AdaRES-Y100','occupation','next'}
        at=lines.index('## Closed Or Failed Directions')
        lines[at:at]=[
            '- Y101 independent W2 shared full256 current-query linear bypass AFTER each complete original qqa_FFN_i including its own residual/norm, six calls; +65536/total164695173/noRNG. Original initial readout/full inputs/query generator/sampling/SCG/k/gamma/fusion norm/coarse-fine average/static7/shared confidence/gates/loss/selector/releasedbackbone/data/seed1999fixed30 intact. Not Y73 initial-minus-refined anchor/Y83 norm sharing/Y87 product/Y95 global context/Y96 initial seed; no combination/tuning/reopening.',
            '- Y101 FIRST original CPU YAML/fullmodel/releasedbackbone/TRAIN collation PASS '+evidence(new['CPU_result'])+';162738points/8expressions; scalarfloat32-64/noRNG/permutation/slot-local/full-input-gradient/seven-stage exact W2 anchors, initial readout exact/refined-mask causal intervention/exact bypass and two finite synthetic updates. All six complete FFNs/36 original FFN parameter gradients/twelve shared-skip consumers/static masks/shared heads/hard gates verified. CPU is NOT native/efficacy.',
            '- Y101 exact7 tested additions retained '+new['retained_branch']+' commit'+new['code_commit']+'; '+evidence(new['source_manifest'])+'; parentHEADY100/mainindex/production/old evidence preserved.10 actual-evidence+8 runtime guards PASS '+evidence(new['native_preparation'])+'; matching original owners Y99/server1,Y100/server2, native runner=run_y101_native.py; original two runtime/data/mapping/350W280W plans. No nativeGPU/admission/launch yet; do not replay passed preparation.'
        ]
        at=lines.index('## Open Questions And Risks')
        lines[at:at]=['- Y100 exact original directory20261111_085241 discovered from original save declaration, never inferred launch timestamp.29 lifecycle guards PASS '+evidence(original['terminal_handoff_plan'])+'; future finalize_y100_fixed30.py ONLY at HEADY100 after natural30/exact absent original PID/full Train-Val1..30/stable finite last-best-model-optimizer-scheduler and original source/private runtime/280W. No live checkpoints loaded or archived.']
        lines+=['## Last Tick','- '+token+': observed/dispatched existing qualified work first; snapshot new_terminalY98 is previous fully verified archive, NOT replayed. Original Y99/Y100 increasing native counters; Y100 exact original terminal29 checks; Y101 FIRST CPU feasibility/isolated version/10+8 guards/unexecuted native plans completed; both authorizedGPU0 fresh busy/reserved.',
            '- New scientificIDs1/CPUscientificattempts1/CPUreplays0/nativeGPUtasks0/admissions0/launches0/archives0/FIRST0/cost0. Goal/old receipts/data/runtime/mappings/350W280W/seed1999fixed30/bans/GPU1+Y57 pauses/foreign jobs preserved. Closeout '+evidence(ref(out))+'; official finish/receipt/release authoritative.']
        assert len(lines)<120 and sum(r.startswith('## ') for r in lines)==7
        memory.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
        phases=[cid+' '+r['phase']['phase']+str(r['phase']['epoch'])+'/30 step'+str(r['phase']['step'])+'/'+str(r['phase']['count']) for cid,r in observations.items()]
        notes=ROOT/('pipeline/reports/notes_'+token+'.json');assert not notes.exists()
        tick.save(notes,dict(actions=[
            '运行：'+'；'.join(phases)+'。原Y99/Y100进程、源码、原生YAML、私有环境与350W/280W不变，原生步数递增。快照列出的Y98为上轮完整核验归档，不重放终态工具。',
            '完成准备：Y100实际保存目录20261111_085241只读确认，29项精确终态保护检查通过；未加载活跃检查点、未归档、不停止训练。',
            '新准备：Y101独立W2在完整原查询FFN后加入同一当前查询的共享全通道线性旁路，保留全部原FFN/残差/归一化/六阶段/原初始读出。首次原生YAML/完整CPU模型/发布backbone/原TRAIN加载通过，162738点8表达；标量float32/64/置换/slot-locality/无RNG/七阶段基线与输入梯度对齐、初始输出精确/后续mask因果干预和旁路恢复、两次有限synthetic decoder更新通过。六个原FFN和36个原参数梯度、十二次旁路/原静态mask/共享head/hard-gates逐次核验。CPU不是GPU原生检查或效果证明。',
            '七个已测试科学文件保存隔离版本，生产HEAD/主索引/原运行及负证据保留；10项实际证据和8项运行计划检查通过。新科学ID1/CPU科学尝试1/GPU原生0/准入0/新训练0/归档0/FIRST0/成本0。'
        ],decisions=[
            '两张授权GPU0均实测busy/reserved，Y101等待匹配原Y99或Y100自然固定30轮、完整精确有限原检查点核验及规范归档，再核验空闲独占GPU0后做首次原生检查；通过后同轮保存完整原设备协议/准入/派发，不重跑已通过CPU或绕开资源保护。',
            '仅AdaRES；server2GPU1全部使用和Y57单独暂停保持；不迁移后评/成本、不重启受保护实验、不用server6、不改数据/功耗/调度、不干预外部任务。无新增用户处理事项。'
        ],evidence=[ref(out),ref(backup)]))
        assert all(ref(p)==r for p,r in before.items()) and git('ls-files','--stage','-z')==index
        return out,notes,len(lines)
    finally:writer.close()


def finish(token):
    lease(token,240)
    for _,cid,_,_ in RUNS:command(['scripts/y100_deployment.py','observe','--id',cid,'--token',token,'--label','release'])
    for name,_,_,_ in RUNS:command([SKILL+'/gpu_occupation.py','AdaRES_project','--token',token,'--server',name,'--label','release'])
    out,notes,lines=record(token)
    command([SKILL+'/foreground_tick.py','AdaRES_project','report','--token',token,'--notes',str(notes)])
    command([SKILL+'/foreground_tick.py','AdaRES_project','finish','--token',token])
    receipt=tick.load(ROOT/('pipeline/tick_receipts/'+token+'.json'))
    assert receipt['status']=='FINALIZED' and receipt['postflight']=='PASS' and receipt['context_committed'] is True and receipt['lease_release']=='released'
    assert not (ROOT/'pipeline/headless/.foreground.lockdir').exists() and not (ROOT/'pipeline/headless/.tick_context_state.next.json').exists()
    pending=ROOT/'pipeline/archive_pending';assert not pending.exists() or not any(p.is_file() for p in pending.rglob('*'))
    assert ref('pipeline/goal.json')['sha256']==GOAL
    print(json.dumps(dict(closeout=ref(out),status=receipt['status'],postflight=receipt['postflight'],context_committed=True,lease_release=receipt['lease_release'],memory_lines=lines,observations={cid:r['phase'] for cid,r in tick.load(out)['observations'].items()},new_scientific_candidates=1,native_GPU_tasks=0,admissions=0,launches=0,archives=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);finish(p.parse_args().token)
