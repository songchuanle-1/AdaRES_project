"""Close first Y103 CPU preparation and original Y101 lifecycle preparation."""
import argparse,ast,base64,json
from datetime import datetime
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from finish_y92_binding_tick import command,SKILL
from finish_y95_preparation_tick import fresh,GOAL,PAUSE
from y101_deployment import Y100,Y101,git
from prepare_y103_spatial_input_skip import files
from y103_contract import validate_cpu
from run_y103_native import runtime,validate_binding
from check_ef_execution import validate_frozen
import closure,gpu_occupation
RUNS=(('server-1','AdaRES-Y101',2263581,Y101,'AdaRES-Y103'),('server-2','AdaRES-Y100',19305,Y100,'AdaRES-Y102'))
def evidence(r):return r['path']+' SHA'+r['sha256']
def record(token):
    lease(token,180);lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:
        assert git('rev-parse','HEAD').decode().strip()==Y101
        before={p:ref(p) for p in ('pipeline/queue.json','pipeline/running.json','pipeline/goal.json','SERVERS.yaml')};index=git('ls-files','--stage','-z')
        assert before['pipeline/goal.json']['sha256']==GOAL and ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')['sha256']==PAUSE
        work=tick.load(ROOT/'pipeline/research_work.json');c={r['id']:r for r in work['candidates']};new=c['AdaRES-Y103'];earlier=c['AdaRES-Y102'];servers=monitor.load_servers(str(ROOT))
        for item in (new,earlier):assert item['stage']=='blocked' and item['blocker_kind']=='exclusive_gpu_slot' and item['resource_work_kind']=='native_smoke' and not item.get('native_smoke_result') and not item.get('admission_receipt')
        assert earlier['code_commit']=='0c6c844719c8466ddd8888d054fe3b7ee658e643'
        assert earlier['additive_version_binding']['sha256']=='d8ca575bb1833a6dacac242961268d533ec8555f56938cdaca7611bcc34e5d4c'
        closure.artifact(ROOT,earlier['additive_version_binding']);closure.artifact(ROOT,earlier['native_preparation']);closure.artifact(ROOT,earlier['CPU_result'])
        validate_cpu(tick.load(closure.artifact(ROOT,new['CPU_result'])))
        version=tick.load(closure.artifact(ROOT,new['source_manifest']));binding=tick.load(closure.artifact(ROOT,new['additive_version_binding']));validate_binding(binding,new,Y101)
        assert version['code_commit']==new['code_commit']=='960babebd812098efde2eb1b188e612028ae571e' and version['CPU_result']==new['CPU_result'] and len(version['exact_delta'])==7
        assert binding['supersedes']==new['source_manifest'] and binding['original_Y101_Y100_preserved'] is True
        snapshot=tick.load(closure.artifact(ROOT,version['source_snapshot']));payload=snapshot['CPU_tested_payload']
        assert payload==ast.literal_eval(ast.parse(closure.artifact(ROOT,new['CPU_program']).read_text(encoding='utf-8')).body[0].value)
        for p in files():
            row=payload['variant']['config_data'] if p.suffix=='.yaml' else payload['variant']['python_files'][p.name]
            assert p.read_bytes()==base64.b64decode(row['base64'])==git('show',new['code_commit']+':'+p.relative_to(ROOT).as_posix()) and ref(p)['sha256']==row['sha256']
        prep=tick.load(closure.artifact(ROOT,new['native_preparation']));assert prep['token']==token and prep['CPU_result']==new['CPU_result'] and prep['guard_tests_passed']==10 and prep['runtime_guard_tests_passed']==8
        assert prep['same_callbacks_actually_executed_on_CPU'] and prep['original_Y101_Y100_preserved'] and not prep['native_executed'] and not prep['admitted'] and not prep['production_sync']
        assert 'Ran 10 tests' in closure.artifact(ROOT,prep['guard_tests']).read_text(encoding='utf-8') and 'Ran 8 tests' in closure.artifact(ROOT,prep['runtime_guard_tests']).read_text(encoding='utf-8')
        for name,key in [('server-1','native_smoke_plan_server1'),('server-2','native_smoke_plan_server2_gpu0')]:
            plan=tick.load(closure.artifact(ROOT,new[key]));f=ast.literal_eval(ast.parse(closure.artifact(ROOT,plan['program']).read_text(encoding='utf-8')).body[0].value)
            assert validate_frozen(new,plan,f)==7 and runtime(name,new,f,servers[name])==(350. if name=='server-1' else 280.) and plan['execution_helper']==ref('scripts/run_y103_native.py')
        terminal=tick.load(closure.artifact(ROOT,c['AdaRES-Y101']['terminal_handoff_plan']))
        assert terminal['token']==token and terminal['CPU_guard_tests_passed']==30 and terminal['id']=='AdaRES-Y101' and not terminal['checkpoint_loaded'] and not terminal['archived']
        location=tick.load(closure.artifact(ROOT,terminal['declared_location']));assert location['checkpoint_directory']=='20261112_200238' and not location['checkpoint_loaded'] and not location['remote_mutations']
        for entry in terminal['dependencies']:closure.artifact(ROOT,entry)
        assert c['AdaRES-Y100']['terminal_handoff_plan']['sha256']=='5e07c3662e73d014f155f38b3a7cba7d0a5a95d61781e9e9638fb6430398ad1d'
        assert c['AdaRES-Y99']['terminal_evidence']['sha256']=='68b925d75e27cb0dfb232f0e8eb1909d08a41d2616d75d2034392a41a32add5f'
        queue=monitor.load_queue(str(ROOT));assert monitor.load_running(str(ROOT))=={'server-1':{'0':'AdaRES-Y101'},'server-2':{'0':'AdaRES-Y100'}}
        context=tick.load(ROOT/'pipeline/headless/.tick_context.json');view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json');assert not view['preflight']['operational_issues']
        observations={};occupations={}
        for name,cid,pid,commit,heldid in RUNS:
            task=next(r for r in queue['gpu_queues'][name] if r['id']==cid);assert (task['status'],task['launch_verified'],task['pid'],task['code_commit'],str(task['gpu']))==('running',True,pid,commit,'0');closure.admission(ROOT,task)
            entry=c[cid]['latest_training_observation'];v=tick.load(closure.artifact(ROOT,entry));assert v['token']==token and v['pid']==pid and v['native_progress_confirmed'] is True;fresh(v,'observed_local');observations[cid]=dict(evidence=entry,phase=v['after']['latest'],pid=pid)
            proof=ref('pipeline/checks/gpu_occupation_'+name+'_release_'+token+'.json');fresh(tick.load(closure.artifact(ROOT,proof)),'observed_at');sv=next(r for r in context['queue']['servers'] if r['name']==name);state=gpu_occupation.validated_state(ROOT,context,c[heldid],sv,queue)
            assert state['busy_gpu_ids']==state['reserved_gpu_ids']==[0] and not state['idle_gpu_ids'];occupations[name]=proof
        pause=next(r for r in queue['gpu_queues']['server-2'] if r['id']=='AdaRES-Y57');assert pause['status']=='paused' and pause['dispatch_hold'] and pause['resume_requires_user_authorization']
        assert work['tick_attempt']['tick_started_at']==view['generated_at'] and work['tick_attempt']['kind']=='resource_check' and work['tick_attempt']['result']=='blocked'
        failure=ref('pipeline/checks/y103_native_'+token+'_runtime_guards.log')
        out=ROOT/('pipeline/checks/y103_preparation_closeout_'+token+'.json');assert not out.exists()
        tick.save(out,dict(complete=True,token=token,observed_at=datetime.now().astimezone().isoformat(),Y103_CPU=new['CPU_result'],Y103_source=new['source_manifest'],Y103_corrected_binding=new['additive_version_binding'],Y103_native_preparation=new['native_preparation'],Y101_terminal_preparation=c['AdaRES-Y101']['terminal_handoff_plan'],preserved_Y102_binding=earlier['additive_version_binding'],preserved_runtime_schema_failure=failure,observations=observations,occupations=occupations,new_scientific_candidates=1,CPU_scientific_attempts=1,scientific_CPU_replays=0,native_GPU_tasks=0,admissions=0,launches=0,archives=0,checkpoint_loaded=False,formal_FIRST_or_cost_executed=False,evidence=[ref(__file__),new['CPU_result'],new['CPU_program'],new['source_manifest'],new['additive_version_binding'],new['native_preparation'],c['AdaRES-Y101']['terminal_handoff_plan'],earlier['additive_version_binding'],failure]+[r['evidence'] for r in observations.values()]+list(occupations.values())))
        memory=ROOT/'pipeline/codex_memory.md';backup=ROOT/('pipeline/snapshots/codex_memory_before_'+token+'.md');assert not backup.exists();oldbytes=memory.read_bytes();backup.write_bytes(oldbytes);assert backup.read_bytes()==oldbytes
        old=oldbytes.decode('utf-8').splitlines();lines=[];updated=set();phases={cid:r['phase']['phase']+str(r['phase']['epoch'])+'/30 step'+str(r['phase']['step'])+'/'+str(r['phase']['count']) for cid,r in observations.items()}
        for line in old[:old.index('## Last Tick')]:
            if line.startswith('- Current token '):line='- Current token '+token+'; snapshot '+view['generated_at']+'; deadline '+view['deadline_at']+'; never extend.';updated.add('token')
            elif line.startswith('- Exact previous seven-section memory:'):line='- Exact previous seven-section memory: '+evidence(ref(backup))+'; full sources/protocols/negative evidence/bans/dedicated recoveries byte-preserved; no reopening authority.';updated.add('backup')
            elif line.startswith('- AdaRES-Y101 original PID'):line='- AdaRES-Y101 original PID2263581/server1GPU0/350W, '+phases['AdaRES-Y101']+'; exact original source/YAML/private runtime/power/increasing native counters '+evidence(observations['AdaRES-Y101']['evidence'])+'; original receipt/freeze preserved.';updated.add('Y101')
            elif line.startswith('- AdaRES-Y100 original PID'):line='- AdaRES-Y100 original PID19305/server2GPU0/280W, '+phases['AdaRES-Y100']+'; exact original source/YAML/private runtime/power/increasing native counters '+evidence(observations['AdaRES-Y100']['evidence'])+'; no complete checkpoint claim while still running.';updated.add('Y100')
            elif line.startswith('- Fresh canonical occupation '):line='- Fresh canonical occupation '+'; '.join(evidence(r) for r in occupations.values())+'; both authorizedGPU0 physically busy/reserved by originalY101/Y100. Y102 then Y103 FIRST native tasks wait for exact matching original completion/archive/exclusive idle; collector actual resource_check tick_attempt retained.';updated.add('occupation')
            elif line.startswith('- Next tick first observe '):line='- Next tick first observe Y100/Y101 and dispatch qualified work. FIRST native order Y102 then Y103. At HEADY101 use run_y102_bound_native_y101.py / run_y103_native.py ONLY after matching original Y101/server1 or Y100/server2 naturally30/full finite exact checkpoint/canonical archive and fresh exclusive idle GPU0. Changed HEAD/owner needs NEW tested same-byte binding; do not replay passed CPU/version/plan tests. Native PASS continues same tick through complete original-device freeze/protected handoff/admission/source selection/dispatch/increasing counters. Y100 terminal=finalize_y100_fixed30_y101.py; Y101 terminal=finalize_y101_fixed30.py, exact original20261112_200238.';updated.add('next')
            if line=='## Closed Or Failed Directions':
                lines.extend([
                    '- Y103 independent W2 whole32 sparse-input linear bypass beside complete original input_proj BEFORE original sum_norm/visual memory; +8192/total164637829/noRNG. Complete spatial/appearance/mask MLPs/full features/static7/query generator/refinements/SCG/gates/shared confidence/targets/loss/selector/data/seed1999fixed30 intact. Distinct destination fromY99/static masks, source fromY100/appearance, Y102/centered geometry masks; no combination, subset scan, tuning or reopening.',
                    '- Y103 FIRST CPU YAML/fullmodel/releasedbackbone/TRAIN162738points/8expressions/scalarfloat32-64/noRNG/permutations/sign/row-local/input-and-weight gradients/seven-stage W2 anchor/causal initial-final masks/exact bypass/two finite synthetic updates PASS '+evidence(new['CPU_result'])+'; exact version '+evidence(new['source_manifest'])+' commit'+new['code_commit']+' '+new['retained_branch']+'. Shared future native callbacks ran on CPU with all14 original projection/mask parameter gradients and sum_norm/first visual memory/static7/shared-head/hard-gate identity checks. Generic component result keys retain legacy appearance labels for arbitrary11x7 projection controls; actual full32 raw-spatial input is independently verified in shared callbacks. CPU is NOT native or efficacy.',
                    '- Y10310 evidence+8 runtime guards PASS '+evidence(new['native_preparation'])+'; explicit originalY101-Y100 owner binding '+evidence(new['additive_version_binding'])+' parentHEADY101. Legacy keyword original_Y99_Y100_preserved remained in original version while exact saved owner rows already namedY101/PID2263581,Y100/PID19305; missing current-owner field caused original runtime failure '+evidence(failure)+'. Original version/programs/runner/failed log retained; v2 adds verified field after exact source/owner/receipt/index checks; native plans/bodies/science unchanged, no CPU/native replay. Y102 first; both native plans unexecuted.'])
            if line=='## Open Questions And Risks':lines.append('- Y101 exact original directory20261112_200238 discovered read-only from original save declaration, never launch timestamp;30 lifecycle guards PASS '+evidence(c['AdaRES-Y101']['terminal_handoff_plan'])+'; future finalize_y101_fixed30.py ONLY at HEADY101 after naturally30/exact absentPID/full Train-Val1..30/stable finite last-best-model-optimizer-scheduler and original source/private runtime/350W. No live weights loaded or archived.')
            lines.append(line)
        assert updated=={'token','backup','Y101','Y100','occupation','next'}
        lines+=['## Last Tick','- '+token+': canonical observe/dispatch qualified work first; originalY100/Y101 increasing native counters and exact runtime/source/350W280W preserved. Y99 appeared as snapshot terminal but its prior verified archive was NOT replayed. Y101 exact future terminal preparation/read-only actual directory+30 guards; Y103 FIRST CPU independent variant and exact retained7-file version/10 evidence+8 corrected runtime guards/plans; Y102 original same-byte current-owner binding preserved.',
            '- New scientificIDs1/CPUscientificattempts1/scientificCPUreplays0/nativeGPUtasks0/admissions0/launches0/archives0/FIRST0/cost0. Both authorizedGPU0 remain physically busy/reserved; actual resource-check evidence retained. Original scientific protocols/receipts/data/runtime/mappings/bans/GPU1+Y57 pauses/foreign jobs/scheduler/power intact. Closeout '+evidence(ref(out))+'; official finish/receipt/release authoritative.']
        assert len(lines)<120 and sum(r.startswith('## ') for r in lines)==7;memory.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
        notes=ROOT/('pipeline/reports/notes_'+token+'.json');assert not notes.exists();tick.save(notes,dict(actions=[
            '运行：'+'；'.join(cid+' '+p for cid,p in phases.items())+'。原PID/源码/原生YAML/精确数据/私有环境/350W280W与原生步数递增已验证；两张授权GPU0物理busy/reserved，未停止或挤占原训练。',
            '新准备：Y101真实原保存目录20261112_200238只读发现，30项生命周期守卫通过；仅准备未来精确终态工具，未加载活跃权重或归档。Y99的上一轮完整归档保留，不重做。',
            '新准备：独立W2变体Y103保留完整空间投影MLP并增加全32维线性旁路到原视觉记忆；首次CPU原生YAML/完整模型/原TRAIN数据加载、标量/梯度/W2锚定/因果旁路/两步合成解码器更新通过，7个测试文件隔离保存，10项证据和8项运行守卫通过。操作字段名未同步导致的原失败记录保留，v2按实际源码/owner/收据/索引补齐绑定，未重跑科学检查。',
            'Y102优先，Y102/Y103首次GPU原生检查均等待匹配原训练自然30轮、完整精确有限检查点与规范归档，再验独占空闲授权GPU0。CPU通过不等于原生或收益。新科学ID1、首次CPU实现检查1、科学重跑0、原生GPU检查0、准入0、新训练0、归档0、FIRST0、成本0。'
        ],decisions=['按本轮快照补齐两项有界待跑准备，不为凑额度重跑旧证据。资源仍忙，未执行GPU原生检查或准入；下一步Y102先于Y103，原生通过后同轮按完整原设备协议继续准入与门控派发。','仅AdaRES；server2GPU1全部使用和Y57单独暂停保持，不迁移冻结后评/成本、不重启受保护实验、不用server6、不改数据/功耗/调度、不干预外部任务。无新增用户处理事项。'],evidence=[ref(out),ref(backup)]))
        assert all(ref(p)==r for p,r in before.items()) and git('ls-files','--stage','-z')==index
        return out,notes,len(lines)
    finally:lock.close()
def finish(token):
    lease(token,300)
    for _,cid,_,_,_ in RUNS:command(['scripts/y101_deployment.py','observe','--id',cid,'--token',token,'--label','release'])
    for name,_,_,_,_ in RUNS:command([SKILL+'/gpu_occupation.py','AdaRES_project','--token',token,'--server',name,'--label','release'])
    out,notes,lines=record(token);command([SKILL+'/foreground_tick.py','AdaRES_project','report','--token',token,'--notes',str(notes)]);command([SKILL+'/foreground_tick.py','AdaRES_project','finish','--token',token])
    r=tick.load(ROOT/('pipeline/tick_receipts/'+token+'.json'));assert r['status']=='FINALIZED' and r['postflight']=='PASS' and r['context_committed'] is True and r['lease_release']=='released'
    assert not (ROOT/'pipeline/headless/.foreground.lockdir').exists() and not (ROOT/'pipeline/headless/.tick_context_state.next.json').exists()
    pending=ROOT/'pipeline/archive_pending';assert not pending.exists() or not any(p.is_file() for p in pending.rglob('*'))
    print(json.dumps(dict(closeout=ref(out),status=r['status'],postflight=r['postflight'],context_committed=True,lease_release=r['lease_release'],memory_lines=lines,observations={cid:v['phase'] for cid,v in tick.load(out)['observations'].items()},new_scientific_candidates=1,CPU_scientific_attempts=1,native_GPU_tasks=0,admissions=0,launches=0,archives=0)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);finish(p.parse_args().token)
