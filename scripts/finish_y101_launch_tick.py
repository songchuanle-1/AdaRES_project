"""Close exact Y99 archival, first native Y101 launch and unchanged Y102 rebinding."""
import argparse,json
from datetime import datetime
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from finish_y92_binding_tick import command,SKILL
from finish_y95_preparation_tick import fresh,GOAL,PAUSE
from y101_deployment import Y99,Y100,Y101,git,require_running_parent_handoff
from advance_y101 import prepared,validate_result,make_freeze
import closure,gpu_occupation
RUNS=(('server-1','AdaRES-Y101',Y101,2263581,'AdaRES-Y99'),('server-2','AdaRES-Y100',Y100,19305,'AdaRES-Y102'))
def evidence(r):return r['path']+' SHA'+r['sha256']
def record(token):
    lease(token,180);lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:
        assert git('rev-parse','HEAD').decode().strip()==Y101
        before={p:ref(p) for p in ('pipeline/queue.json','pipeline/running.json','pipeline/goal.json','SERVERS.yaml')};index=git('ls-files','--stage','-z')
        assert before['pipeline/goal.json']['sha256']==GOAL and ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')['sha256']==PAUSE
        assert ref('pipeline/admissions/AdaRES-Y99.json')['sha256']=='4d87ad8b3a6efe78be8dc9e8214bba9226160a22b649f64f8e1dd26d73988ede'
        assert ref('pipeline/admissions/AdaRES-Y100.json')['sha256']=='7091f206c21429c02349c73e68d5e2b06ca3adc9f585465450870ac2a0e019f2'
        work=tick.load(ROOT/'pipeline/research_work.json');c={r['id']:r for r in work['candidates']};queue=monitor.load_queue(str(ROOT));context=tick.load(ROOT/'pipeline/headless/.tick_context.json');view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json')
        assert monitor.load_running(str(ROOT))=={'server-1':{'0':'AdaRES-Y101'},'server-2':{'0':'AdaRES-Y100'}}
        terminal=tick.load(closure.artifact(ROOT,c['AdaRES-Y99']['terminal_evidence']))
        assert terminal['token']==token and terminal['id']=='AdaRES-Y99' and terminal['process_absent'] and not terminal['CUDA_initialized']
        assert terminal['completed_epochs']=={p:list(range(1,31)) for p in ('Train','Val')}
        assert [r['epoch'] for r in terminal['checkpoints']]==[30,23]
        assert all(r['model_finite'] and r['optimizer_finite'] and r['optimizer_state_count']==544 and r['scheduler_last_epoch']==r['epoch'] and '/20261111_180638/' in r['path'] for r in terminal['checkpoints'])
        assert c['AdaRES-Y99']['stage']=='blocked' and c['AdaRES-Y99']['scientific_status']=='fixed30_complete_frozen_review_pending'
        candidate,_,_,frozen=prepared(work);assert candidate['id']=='AdaRES-Y101'
        raw=tick.load(closure.artifact(ROOT,candidate['native_smoke_result']));validate_result(raw,candidate,frozen)
        assert raw['losses']==[4.615510940551758,6.345853805541992] and raw['peak_reserved_bytes']==6757023744
        receipt_ref=ref(candidate['admission_receipt']);receipt=tick.load(closure.artifact(ROOT,receipt_ref));smoke=tick.load(closure.artifact(ROOT,receipt['native_smoke']))
        assert smoke['command']=='python -B scripts/run_y101_native.py --server server-1 --token '+token
        freeze=tick.load(closure.artifact(ROOT,candidate['post_training_freeze']));parentref=c['AdaRES-Y99']['post_training_freeze'];expected=make_freeze(tick.load(closure.artifact(ROOT,parentref)),parentref)
        for k in ('training','evaluation','resource_condition','resource_authorization','parent_protocol'):assert freeze[k]==expected[k]
        require_running_parent_handoff(work,candidate);handoff=tick.load(closure.artifact(ROOT,c['AdaRES-Y100']['terminal_handoff_plan']))
        assert handoff['token']==token and handoff['CPU_guard_tests_passed']==29 and handoff['additive_guard_tests_passed']==8 and handoff['tested_additive_native_evidence']==candidate['native_smoke_result']
        assert handoff['payload']['allowed_markers']==[Y100,Y101] and not handoff['checkpoint_loaded'] and not handoff['full_terminal_verification_executed']
        guards=ref('pipeline/checks/adares-y101_admission_guards_'+token+'.log');assert 'Ran 11 tests' in closure.artifact(ROOT,guards).read_text(encoding='utf-8')
        dispatch=ref('pipeline/checks/adares-y101_dispatch_'+token+'.json');dv=tick.load(closure.artifact(ROOT,dispatch));assert dv['task']['status']=='running' and dv['task']['launch_verified'] is True and dv['task']['pid']==2263581
        y102=c['AdaRES-Y102'];binding=tick.load(closure.artifact(ROOT,y102['additive_version_binding']))
        assert binding['token']==token and binding['guard_tests_passed']==10 and binding['parent_commit']==binding['working_HEAD']==Y101
        assert binding['matching_current_owners']=={'server-1':'AdaRES-Y101','server-2':'AdaRES-Y100'} and binding['scientific_files_identical'] and binding['native_program_bodies_identical'] and binding['only_frozen_code_commit_metadata_changed']
        assert not y102.get('native_smoke_result') and not y102.get('admission_receipt') and y102['stage']=='blocked' and y102['blocker_kind']=='exclusive_gpu_slot'
        for entry in binding['evidence']:closure.artifact(ROOT,entry)
        observations={};occupations={}
        for name,cid,commit,pid,heldid in RUNS:
            task=next(r for r in queue['gpu_queues'][name] if r['id']==cid);assert (task['status'],task['launch_verified'],task['pid'],task['code_commit'],str(task['gpu']))==('running',True,pid,commit,'0');closure.admission(ROOT,task)
            entry=c[cid]['latest_training_observation'];v=tick.load(closure.artifact(ROOT,entry));fresh(v,'observed_local')
            assert v['token']==token and v['native_progress_confirmed'] is True and v['pid']==pid and v['source_marker'] in (commit,Y101)
            observations[cid]=dict(evidence=entry,phase=v['after']['latest'],pid=pid)
            occupation=ref('pipeline/checks/gpu_occupation_'+name+'_release_'+token+'.json');fresh(tick.load(closure.artifact(ROOT,occupation)),'observed_at')
            sv=next(r for r in context['queue']['servers'] if r['name']==name);state=gpu_occupation.validated_state(ROOT,context,c[heldid],sv,queue)
            assert state['busy_gpu_ids']==state['reserved_gpu_ids']==[0] and not state['idle_gpu_ids'];occupations[name]=occupation
        pause=next(r for r in queue['gpu_queues']['server-2'] if r['id']=='AdaRES-Y57');assert pause['status']=='paused' and pause['dispatch_hold'] and pause['resume_requires_user_authorization']
        negative=ref('pipeline/checks/y100_y101_terminal_binding_'+token+'_additive.log');failedobserver=ref('pipeline/checks/production_adares-y99_'+token+'_start.log')
        selection=ref('pipeline/checks/adares-y101_source_selection_'+token+'.json');helpers=ref('pipeline/checks/y101_launch_helpers_'+token+'.json')
        out=ROOT/('pipeline/checks/y101_launch_closeout_'+token+'.json');assert not out.exists()
        tick.save(out,dict(complete=True,token=token,observed_at=datetime.now().astimezone().isoformat(),observations=observations,occupations=occupations,original_Y99_terminal=c['AdaRES-Y99']['terminal_evidence'],original_Y100_handoff=c['AdaRES-Y100']['terminal_handoff_plan'],actual_Y101_native=candidate['native_smoke_result'],Y102_unchanged_binding=y102['additive_version_binding'],source_selection=selection,dispatch=dispatch,preserved_observer_failure=failedobserver,preserved_operational_test_failure=negative,new_scientific_candidates=0,scientific_CPU_replays=0,native_GPU_tasks=1,admissions=1,launches=1,archives=1,full_original_protocol_preserved=True,formal_FIRST_or_cost_executed=False,evidence=[ref(__file__),selection,dispatch,helpers,receipt_ref,candidate['post_training_freeze'],c['AdaRES-Y99']['terminal_evidence'],c['AdaRES-Y100']['terminal_handoff_plan'],y102['additive_version_binding'],guards,negative,failedobserver]+[r['evidence'] for r in observations.values()]+list(occupations.values())))
        memory=ROOT/'pipeline/codex_memory.md';backup=ROOT/('pipeline/snapshots/codex_memory_before_'+token+'.md');assert not backup.exists();oldbytes=memory.read_bytes();backup.write_bytes(oldbytes);assert backup.read_bytes()==oldbytes
        sections={};current=None
        for line in oldbytes.decode('utf-8').splitlines():
            if line.startswith('## '):current=line;sections[current]=[]
            elif current:sections[current].append(line)
        assert len(sections)==7
        phases={cid:r['phase']['phase']+str(r['phase']['epoch'])+'/30 step'+str(r['phase']['step'])+'/'+str(r['phase']['count']) for cid,r in observations.items()}
        active=[line for line in sections['## Active Hypotheses'] if line.startswith(('- Y100 ','- Y101 independent','- Y101 FIRST original','- Y102 independent','- Y102 FIRST original'))]
        lines=['# AdaRES_project Research Memory','## Goal And Phase','- Current token '+token+'; snapshot '+view['generated_at']+'; deadline '+view['deadline_at']+'; never extend.',
            '- Actual CODEX_THREAD_ID=owner01a07e32-78bf-7670-925b-370911ea48db; Pipeline-AdaRES_project Disabled; experiment_first enabled. PROJECT_ROOT=D:/project; PYTHONDONTWRITEBYTECODE=1; PYTHONUTF8=1; python -B; SSH keys only, no credential read.',
            '- Only server1GPU0 existing350W and server2GPU0 fixed280W, one experiment/card; ALL server2GPU1 use paused, Y57 separately paused epoch27. No server6/cross-project/scheduler/power/foreign-job changes.',
            '- Goal SHA'+GOAL+'; pause SHA'+PAUSE+'; original goal/receipts/protocols/data/mappings/bans byte-preserved.',
            '- Exact previous seven-section memory: '+evidence(ref(backup))+'; full historical sources, evidence, negative records and dedicated recoveries preserved through exact backup chain; never reopening authority.','## Champion And Baseline']+sections['## Champion And Baseline']+['## Active Hypotheses',
            '- Production HEADY101='+Y101+'; exact7 actual-native Y101 additions selected '+evidence(selection)+'; parentY100='+Y100+'; only server1 synchronized; original Y100 and all unrelated main-index/working files preserved.']+active+[
            '- Y101 original CPU/version/native plans retained without scientific replay; FIRST server1GPU0/350W actual native PASS '+evidence(candidate['native_smoke_result'])+'; losses'+str(raw['losses'])+' peak_reserved6757023744; all six complete original FFNs/current full256 query bypasses/36 original FFN parameter gradients/static7/shared heads/hard gates and two finite real TRAIN updates verified. No efficacy or full-data memory proof.',
            '- Y10111 admission guards PASS; full COMPLETE ORIGINAL Y99 server1GPU0/350W training/FIRST/subgroups/cost freeze '+evidence(candidate['post_training_freeze'])+'; receipt '+evidence(receipt_ref)+'; helpers '+evidence(helpers)+'; launch '+evidence(dispatch)+'. Original scientific source/CPU/native evidence retained; no formal FIRST/cost/promotion.',
            '- Y102 original exact7 scientific additions/CPU/native program bodies unchanged; NEW retained commit'+y102['code_commit']+' branch'+y102['retained_branch']+';10 operational guards PASS '+evidence(y102['additive_version_binding'])+'; parentHEADY101/current owners Y101server1-PID2263581,Y100server2-PID19305. Original branch/CPU/source/plans preserved in original_pre_y101_binding; only commit metadata changed. FIRST native unexecuted, new helper run_y102_bound_native_y101.py.','## Closed Or Failed Directions']+sections['## Closed Or Failed Directions']+['## Running And Queue',
            '- Current observer=y101_deployment.py observe --id AdaRES-Y100|AdaRES-Y101 --token CURRENT --label NEW; earlier observers reject HEADY101. Never live-observe/rearchive Y99/Y98/Y96 or earlier archives.',
            '- AdaRES-Y101 original PID2263581/server1GPU0/350W, '+phases['AdaRES-Y101']+'; launch '+evidence(dispatch)+'; exact source/YAML/private runtime/power/native increase '+evidence(observations['AdaRES-Y101']['evidence'])+'. Exact original checkpoint directory still needs read-only discovery; never infer launch timestamp.',
            '- AdaRES-Y100 original PID19305/server2GPU0/280W, '+phases['AdaRES-Y100']+'; original receipt/source/YAML/private runtime/power preserved; native increase '+evidence(observations['AdaRES-Y100']['evidence'])+'.',
            '- Y99 original20261111_180638 naturally fixed30 complete; exact PID2980836 absent/full Train-Val1..30/stable finite last30-best23/model/optimizer544/scheduler30-23 verified ONCE and canonically archived '+evidence(c['AdaRES-Y99']['terminal_evidence'])+'; trainingbest.5109262466430664 is NOT formal FIRST. Complete original server1/350W review waits onY101; no restart/migration/substitution/rearchive.',
            '- Y100 original20261111_085241; NEW actual-Y101-compatible29+8 terminal plan '+evidence(c['AdaRES-Y100']['terminal_handoff_plan'])+'; future finalize_y100_fixed30_y101.py ONLY at HEADY100/Y101 after natural30/exact absent original PID/full Train-Val1..30/stable finite exact last-best-model-optimizer-scheduler and original source/private runtime/280W. No live checkpoints loaded.',
            '- Y98 original20261110_080627 last30-best20/trainingbest.49940434098243713; Y96 original20261110_165702 last30-best23/.5081601738929749; Y95 original20261109_081912 last30-best20/.508540153503418. All earlier exact archives remain in backup chain, never restart or replay terminal helpers; trainingbest is NOT formal FIRST.',
            '- Fresh canonical occupation '+'; '.join(evidence(r) for r in occupations.values())+'; both authorizedGPU0 busy/reserved by originalY100/newY101. Original GPU1 full frozen FIRST/cost remain user-held, never moved/substituted.',
            '- Server1=songcl@192.168.1.175/dl02-MS-7B73, /data1/songcl/Project/AdaRES_project/AdaRES, private runtime/adares_project/bin/python+cuda113lib64, exact ../data/, only original two SERVERS mappings. Server2=songcl@192.168.1.163/dl01-MS-7C60, /bak_home/songcl/Project/AdaRES_project/AdaRES, /home/songcl/.conda/envs/adares_project/bin/python, exact /bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/.','## Open Questions And Risks',
            '- No formal FIRST/cost/promotion. Trainingbest/native feasibility/increasing counters do not prove efficacy. GPU1 all-use/Y57 separate pause remain; no new user action needed.',
            '- Y99 first live observation failed '+evidence(failedobserver)+'; exact original process subsequently absent/full original checkpoint proof and canonical archive passed. No restart or relaxed checkpoint verification.',
            '- Initial Y100-Y101 additive operational tests retained stale qualified deploy.Y99/Y100 attributes; failed draft/log preserved '+evidence(negative)+'; v2 correct originalY100/actual-nativeY101 attributes passed8. Lifecycle29 reused unchanged; no scientific/native replay or relaxed control.',
            '- Next tick first observe Y100/Y101 and dispatch qualified work. Y102 FIRST native uses run_y102_bound_native_y101.py ONLY at HEADY101 after matching original Y101/server1 or Y100/server2 naturally30/full finite exact checkpoint/canonical archive and fresh exclusive idle GPU0. Changed HEAD/owner needs NEW tested same-byte binding, no CPU/native/preparation replay. Native PASS must continue through full original-device freeze/admission/protected source selection/dispatch/increasing counters same tick. Prepare exact Y101 future terminal tools by read-only save-directory discovery; Y100 uses finalize_y100_fixed30_y101.py.','## Last Tick',
            '- '+token+': Y99 natural fixed30 full original finite terminal proof/canonical archive; Y101 FIRST native PASS, original Y100 compatible29+8 terminal binding with preserved operational draft failure; exact source selection/11 admission guards/full original server1 freeze/receipt/canonical dispatch/increasing counters. Original Y100 continues unchanged. Y102 exact same scientific bytes and native bodies bound to currentHEAD/owners,10 operational guards passed.',
            '- New scientificIDs0/CPUscientificreplays0/nativeGPUtasks1/admissions1/launches1/archives1/FIRST0/cost0. Goal/old receipts/data/runtime/mappings/seed1999fixed30/350W280W/bans/GPU1+Y57 pauses/foreign jobs intact. Closeout '+evidence(ref(out))+'; official finish/receipt/release authoritative.']
        assert len(lines)<120 and sum(r.startswith('## ') for r in lines)==7;memory.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
        notes=ROOT/('pipeline/reports/notes_'+token+'.json');assert not notes.exists()
        tick.save(notes,dict(actions=[
            '运行：'+'；'.join(cid+' '+p for cid,p in phases.items())+'。新Y101在server1GPU0/350W固定30轮运行，原Y100保留；精确源码/数据/私有环境/功耗和原生步数递增均已验证。',
            '完成：Y99自然30轮，原目录20261111_180638完整Train/Val1..30及稳定有限last30/best23/model/optimizer544/scheduler30-23通过精确核验并规范归档。训练best.510926不是正式FIRST，不据此宣称收益。',
            '本轮Y101首次真实TRAIN两步有限loss/backward及新旧路径检查通过；Y100兼容29+8、11项准入守卫、完整原server1冻结协议、收据、入队与门控派发完成。旧提交属性残留的操作测试失败原稿/日志保留，v2修正后通过，不重跑科学检查。',
            '新准备：Y102原7个科学文件与CPU/原生程序正文保持相同，仅绑定到当前HEADY101和原Y101/Y100资源归属，10项操作守卫通过；首次原生检查仍待空槽。新科学ID0、科学CPU重跑0、原生GPU检查1、准入1、新训练1、归档1、FIRST0、成本0。'
        ],decisions=[
            'Y99保留原产物和完整原设备后评协议；Y102等待匹配原owner自然30轮、完整精确有限检查点及规范归档，再验授权GPU0独占空闲；不挤占现有任务。准备Y101未来终态工具须只读发现原保存目录，不从启动时间推断。',
            '仅AdaRES，server2GPU1全部使用和Y57单独暂停保持，不迁移冻结后评/成本、不重启受保护实验、不用server6、不改数据/功耗/调度、不干预外部任务。无新增用户处理事项。'
        ],evidence=[ref(out),ref(backup)]))
        assert all(ref(p)==r for p,r in before.items()) and git('ls-files','--stage','-z')==index
        return out,notes,len(lines)
    finally:lock.close()
def finish(token):
    lease(token,300)
    for _,cid,_,_,_ in RUNS:command(['scripts/y101_deployment.py','observe','--id',cid,'--token',token,'--label','release'])
    for name,_,_,_,_ in RUNS:command([SKILL+'/gpu_occupation.py','AdaRES_project','--token',token,'--server',name,'--label','release'])
    out,notes,lines=record(token);command([SKILL+'/foreground_tick.py','AdaRES_project','report','--token',token,'--notes',str(notes)]);command([SKILL+'/foreground_tick.py','AdaRES_project','finish','--token',token])
    receipt=tick.load(ROOT/('pipeline/tick_receipts/'+token+'.json'));assert receipt['status']=='FINALIZED' and receipt['postflight']=='PASS' and receipt['context_committed'] is True and receipt['lease_release']=='released'
    assert not (ROOT/'pipeline/headless/.foreground.lockdir').exists() and not (ROOT/'pipeline/headless/.tick_context_state.next.json').exists()
    pending=ROOT/'pipeline/archive_pending';assert not pending.exists() or not any(p.is_file() for p in pending.rglob('*'))
    print(json.dumps(dict(closeout=ref(out),status=receipt['status'],postflight=receipt['postflight'],context_committed=True,lease_release=receipt['lease_release'],memory_lines=lines,observations={cid:r['phase'] for cid,r in tick.load(out)['observations'].items()},native_GPU_tasks=1,admissions=1,launches=1,archives=1)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);finish(p.parse_args().token)
