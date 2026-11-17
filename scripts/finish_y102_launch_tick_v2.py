"""Close verified Y100 archive, FIRST Y102 native/launch, and unchanged Y103 binding."""
import argparse,json
from datetime import datetime
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from finish_y92_binding_tick import command,SKILL
from finish_y95_preparation_tick import fresh,GOAL,PAUSE
from y102_observer_v2 import Y100,Y101,Y102,git,require_running_parent_handoff
from advance_y102 import prepared,validate_result,make_freeze
import closure,gpu_occupation
RUNS=(('server-1','AdaRES-Y101',Y101,2263581,'AdaRES-Y99'),('server-2','AdaRES-Y102',Y102,4747,'AdaRES-Y100'))
def evidence(r):return r['path']+' SHA'+r['sha256']

def record(token):
    lease(token,180);writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        assert git('rev-parse','HEAD').decode().strip()==Y102
        before={p:ref(p) for p in ('pipeline/queue.json','pipeline/running.json','pipeline/goal.json','SERVERS.yaml')};index=git('ls-files','--stage','-z')
        assert before['pipeline/goal.json']['sha256']==GOAL and ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')['sha256']==PAUSE
        assert ref('pipeline/admissions/AdaRES-Y100.json')['sha256']=='7091f206c21429c02349c73e68d5e2b06ca3adc9f585465450870ac2a0e019f2'
        assert ref('pipeline/admissions/AdaRES-Y101.json')['sha256']=='379a5b050624141f20345f614877f062f1a61309b55531fe977e023538fecd0c'
        work=tick.load(ROOT/'pipeline/research_work.json');c={r['id']:r for r in work['candidates']};q=monitor.load_queue(str(ROOT));context=tick.load(ROOT/'pipeline/headless/.tick_context.json');view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json')
        assert not view['preflight']['operational_issues']
        assert monitor.load_running(str(ROOT))=={'server-1':{'0':'AdaRES-Y101'},'server-2':{'0':'AdaRES-Y102'}}
        terminal=tick.load(closure.artifact(ROOT,c['AdaRES-Y100']['terminal_evidence']))
        assert terminal['token']==token and terminal['id']=='AdaRES-Y100' and terminal['process_absent'] and not terminal['CUDA_initialized']
        assert terminal['completed_epochs']=={p:list(range(1,31)) for p in ('Train','Val')}
        assert [r['epoch'] for r in terminal['checkpoints']]==[30,23]
        assert all(r['model_finite'] and r['optimizer_finite'] and r['optimizer_state_count']==544 and r['scheduler_last_epoch']==r['epoch'] and '/20261111_085241/' in r['path'] for r in terminal['checkpoints'])
        assert c['AdaRES-Y100']['stage']=='blocked' and c['AdaRES-Y100']['scientific_status']=='fixed30_complete_frozen_review_pending'
        candidate,_,_,frozen=prepared(work);assert candidate['id']=='AdaRES-Y102'
        raw=tick.load(closure.artifact(ROOT,candidate['native_smoke_result']));validate_result(raw,candidate,frozen)
        assert raw['losses']==[4.615511417388916,6.3116302490234375] and raw['peak_reserved_bytes']==6725566464
        receiptref=ref(candidate['admission_receipt']);receipt=tick.load(closure.artifact(ROOT,receiptref));smoke=tick.load(closure.artifact(ROOT,receipt['native_smoke']))
        assert smoke['command']=='python -B scripts/run_y102_bound_native_y101.py --server server-2 --token '+token
        freeze=tick.load(closure.artifact(ROOT,candidate['post_training_freeze']));parentref=c['AdaRES-Y100']['post_training_freeze'];expected=make_freeze(tick.load(closure.artifact(ROOT,parentref)),parentref)
        for k in ('training','evaluation','resource_condition','resource_authorization','parent_protocol'):assert freeze[k]==expected[k]
        require_running_parent_handoff(work,candidate);handoff=tick.load(closure.artifact(ROOT,c['AdaRES-Y101']['terminal_handoff_plan']))
        assert handoff['token']==token and handoff['CPU_guard_tests_passed']==30 and handoff['additive_guard_tests_passed']==8 and handoff['tested_additive_native_evidence']==candidate['native_smoke_result']
        assert handoff['payload']['allowed_markers']==[Y101,Y102] and not handoff['checkpoint_loaded'] and not handoff['full_terminal_verification_executed']
        guards=ref('pipeline/checks/adares-y102_admission_guards_'+token+'.log');assert 'Ran 11 tests' in closure.artifact(ROOT,guards).read_text(encoding='utf-8')
        dispatch=ref('pipeline/checks/adares-y102_dispatch_'+token+'.json');dv=tick.load(closure.artifact(ROOT,dispatch));assert dv['task']['status']=='running' and dv['task']['launch_verified'] is True and dv['task']['pid']==4747
        other=c['AdaRES-Y103'];binding=tick.load(closure.artifact(ROOT,other['additive_version_binding']))
        assert binding['token']==token and binding['guard_tests_passed']==10 and binding['parent_commit']==binding['working_HEAD']==Y102
        assert binding['matching_current_owners']=={'server-1':'AdaRES-Y101','server-2':'AdaRES-Y102'} and binding['scientific_files_identical'] and binding['native_program_bodies_identical'] and binding['only_frozen_code_commit_metadata_changed']
        assert other['code_commit']=='943ad702c59547aafd8b797f19c705f3661a8b9f' and not other.get('native_smoke_result') and not other.get('admission_receipt') and other['stage']=='blocked' and other['blocker_kind']=='exclusive_gpu_slot'
        for entry in binding['evidence']:closure.artifact(ROOT,entry)
        assert c['AdaRES-Y99']['terminal_evidence']['sha256']=='68b925d75e27cb0dfb232f0e8eb1909d08a41d2616d75d2034392a41a32add5f'
        observations={};occupations={}
        for name,cid,commit,pid,held in RUNS:
            task=next(r for r in q['gpu_queues'][name] if r['id']==cid);assert (task['status'],task['launch_verified'],task['pid'],task['code_commit'],str(task['gpu']))==('running',True,pid,commit,'0');closure.admission(ROOT,task)
            entry=c[cid]['latest_training_observation'];v=tick.load(closure.artifact(ROOT,entry));fresh(v,'observed_local')
            assert v['token']==token and v['native_progress_confirmed'] is True and v['pid']==pid and v['source_marker'] in (commit,Y102)
            observations[cid]=dict(evidence=entry,phase=v['after']['latest'],pid=pid)
            occupation=ref('pipeline/checks/gpu_occupation_'+name+'_release_'+token+'.json');fresh(tick.load(closure.artifact(ROOT,occupation)),'observed_at')
            server=next(r for r in context['queue']['servers'] if r['name']==name);state=gpu_occupation.validated_state(ROOT,context,c[held],server,q)
            assert state['busy_gpu_ids']==state['reserved_gpu_ids']==[0] and not state['idle_gpu_ids'];occupations[name]=occupation
        pause=next(r for r in q['gpu_queues']['server-2'] if r['id']=='AdaRES-Y57');assert pause['status']=='paused' and pause['dispatch_hold'] and pause['resume_requires_user_authorization']
        failedobserver=ref('pipeline/checks/production_adares-y100_'+token+'_start.log');selection=ref('pipeline/checks/adares-y102_source_selection_'+token+'.json');helpers=ref('pipeline/checks/y102_launch_helpers_'+token+'.json')
        out=ROOT/('pipeline/checks/y102_launch_closeout_'+token+'.json');assert not out.exists()
        tick.save(out,dict(complete=True,token=token,observed_at=datetime.now().astimezone().isoformat(),observations=observations,occupations=occupations,original_Y100_terminal=c['AdaRES-Y100']['terminal_evidence'],original_Y101_handoff=c['AdaRES-Y101']['terminal_handoff_plan'],actual_Y102_native=candidate['native_smoke_result'],Y103_unchanged_binding=other['additive_version_binding'],source_selection=selection,dispatch=dispatch,preserved_observer_failure=failedobserver,preserved_mapping_failure=ref('pipeline/checks/y102_observer_mapping_failure_'+token+'.json'),new_scientific_candidates=0,scientific_CPU_replays=0,native_GPU_tasks=1,admissions=1,launches=1,archives=1,full_original_protocol_preserved=True,formal_FIRST_or_cost_executed=False,evidence=[ref(__file__),selection,dispatch,helpers,receiptref,candidate['post_training_freeze'],c['AdaRES-Y100']['terminal_evidence'],c['AdaRES-Y101']['terminal_handoff_plan'],other['additive_version_binding'],guards,failedobserver,ref('pipeline/checks/y102_observer_mapping_failure_'+token+'.json')]+[r['evidence'] for r in observations.values()]+list(occupations.values())))
        memory=ROOT/'pipeline/codex_memory.md';backup=ROOT/('pipeline/snapshots/codex_memory_before_'+token+'.md');assert not backup.exists();oldbytes=memory.read_bytes();backup.write_bytes(oldbytes);assert backup.read_bytes()==oldbytes
        sections={};current=None
        for line in oldbytes.decode('utf-8').splitlines():
            if line.startswith('## '):current=line;sections[current]=[]
            elif current:sections[current].append(line)
        assert len(sections)==7
        phases={cid:r['phase']['phase']+str(r['phase']['epoch'])+'/30 step'+str(r['phase']['step'])+'/'+str(r['phase']['count']) for cid,r in observations.items()}
        active=[line for line in sections['## Active Hypotheses'] if line.startswith(('- Y100 ','- Y101 independent','- Y101 FIRST original','- Y102 independent','- Y102 FIRST original','- Y103 independent','- Y103 FIRST CPU'))]
        lines=['# AdaRES_project Research Memory','## Goal And Phase',
            '- Current token '+token+'; snapshot '+view['generated_at']+'; deadline '+view['deadline_at']+'; never extend.',
            '- Actual CODEX_THREAD_ID=owner01a07e32-78bf-7670-925b-370911ea48db; Pipeline-AdaRES_project Disabled; experiment_first enabled. PROJECT_ROOT=D:/project; PYTHONDONTWRITEBYTECODE=1; PYTHONUTF8=1; python -B; SSH keys only, no credential read.',
            '- Only server1GPU0 existing350W and server2GPU0 fixed280W, one experiment/card; ALL server2GPU1 use paused, Y57 separately paused epoch27. No server6/cross-project/scheduler/power/foreign-job changes.',
            '- Goal SHA'+GOAL+'; pause SHA'+PAUSE+'; original goal/receipts/protocols/data/mappings/bans byte-preserved.',
            '- Exact previous seven-section memory: '+evidence(ref(backup))+'; full historical sources, evidence, failures and dedicated recoveries preserved through exact backup chain; never reopening authority.',
            '## Champion And Baseline']+sections['## Champion And Baseline']+['## Active Hypotheses',
            '- Production HEADY102='+Y102+'; exact7 actual-native additions selected '+evidence(selection)+'; parentY101='+Y101+'; only server2 synchronized; original Y101 and all unrelated index/working files preserved.']+active+[
            '- Y102 FIRST actual server2GPU0/280W native PASS '+evidence(candidate['native_smoke_result'])+'; losses'+str(raw['losses'])+' peak_reserved6725566464; all full-scene3 coordinates/point masses/scene-centroid reference/original batching/projections/masks/static7/shared confidence/hard gates/14 original visual parameter gradients and two finite real TRAIN updates verified. Original CPU/retained files/native bodies not replayed. No efficacy/full-data memory proof.',
            '- Y10211 admission guards PASS; complete ORIGINAL Y100 server2GPU0/280W training/FIRST/subgroups/cost freeze '+evidence(candidate['post_training_freeze'])+'; receipt '+evidence(receiptref)+'; launch '+evidence(dispatch)+'. Original protocols preserved; no formal FIRST/cost/promotion.',
            '- Y103 original exact7 scientific files and CPU/native bodies unchanged; NEW commit'+other['code_commit']+' '+other['retained_branch']+';10 operational guards PASS '+evidence(other['additive_version_binding'])+'; actual-native parentHEADY102/current owners Y101server1-PID2263581,Y102server2-PID4747. Original branch/CPU/source/v2 metadata/plans preserved via original_pre_y102_binding and backup chain. FIRST native unexecuted, helper run_y103_bound_native_y102.py. Legacy missing-preservation-key failure and corrected v2 remain immutable; no scientific replay.',
            '## Closed Or Failed Directions']+sections['## Closed Or Failed Directions']+['## Running And Queue',
            '- Current observer=y102_observer_v2.py observe --id AdaRES-Y101|AdaRES-Y102 --token CURRENT --label NEW; earlier observers reject HEADY102. Never live-observe/rearchive Y100/Y99/Y98 or earlier archives.',
            '- AdaRES-Y101 original PID2263581/server1GPU0/350W, '+phases['AdaRES-Y101']+'; exact original source/YAML/private runtime/power/increasing native counters '+evidence(observations['AdaRES-Y101']['evidence'])+'; original receipt/freeze unchanged.',
            '- AdaRES-Y102 original PID4747/server2GPU0/280W, '+phases['AdaRES-Y102']+'; launch '+evidence(dispatch)+'; exact source/YAML/private runtime/power/increasing native counters '+evidence(observations['AdaRES-Y102']['evidence'])+'. Actual original checkpoint directory needs read-only discovery; never infer launch timestamp.',
            '- Y100 original20261111_085241 naturally fixed30 complete; exact PID19305 absent/full Train-Val1..30/stable finite last30-best23/model/optimizer544/scheduler30-23 verified ONCE and canonically archived '+evidence(c['AdaRES-Y100']['terminal_evidence'])+'; trainingbest.5164485573768616 is NOT formal FIRST or goal achievement. Complete original server2/280W review waits onY102; no restart/migration/substitution/rearchive.',
            '- Y101 exact original20261112_200238; NEW actual-Y102-compatible30+8 terminal plan '+evidence(c['AdaRES-Y101']['terminal_handoff_plan'])+'; future finalize_y101_fixed30_y102.py ONLY at HEADY101/Y102 after natural30/exact absent original PID/full Train-Val1..30/stable finite exact last-best-model-optimizer-scheduler and original source/private runtime/350W. No live checkpoints loaded.',
            '- Y99 original20261111_180638 last30-best23/.510926; Y98 original20261110_080627 last30-best20/.499404; Y96 original20261110_165702 last30-best23/.508160; Y95 original20261109_081912 last30-best20/.508540. Exact earlier archives and original evidence remain in backup chain; never restart or replay terminal helpers; trainingbest is NOT formal FIRST.',
            '- Fresh canonical occupation '+'; '.join(evidence(r) for r in occupations.values())+'; both authorizedGPU0 physically busy/reserved by originalY101/newY102. All original GPU1 frozen FIRST/cost remain user-held, never moved/substituted.',
            '- Server1=songcl@192.168.1.175/dl02-MS-7B73, /data1/songcl/Project/AdaRES_project/AdaRES, private runtime/adares_project/bin/python+cuda113lib64, exact ../data/, only original two SERVERS mappings. Server2=songcl@192.168.1.163/dl01-MS-7C60, /bak_home/songcl/Project/AdaRES_project/AdaRES, /home/songcl/.conda/envs/adares_project/bin/python, exact /bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/.',
            '## Open Questions And Risks',
            '- No formal FIRST/cost/promotion. Trainingbest/native feasibility/increasing counters do not prove efficacy. GPU1 all-use/Y57 separate pause remain; no new user action needed.',
            '- Y100 first live observation failed '+evidence(failedobserver)+'; exact original process subsequently absent/full original checkpoint proof/canonical archive passed. Failure preserved; no restart or relaxed checkpoint verification.',
            '- Original closeout observer had reversed server mapping, failed before SSH/source writes; original helpers/failure retained '+evidence(ref('pipeline/checks/y102_observer_mapping_failure_'+token+'.json'))+'; y102_observer_v2.py corrects only physical mapping and verifies actual current source/runtime/power/increasing counters. No native/archive/admission/scientific replay.',
            '- Next tick first observe Y101/Y102 and dispatch qualified work. Y103 FIRST native uses run_y103_bound_native_y102.py ONLY at HEADY102 after matching original Y101/server1 or Y102/server2 naturally30/full finite exact checkpoint/canonical archive and fresh exclusive idle GPU0. Changed HEAD/owner requires NEW tested same-byte binding, never CPU/native/preparation replay. Native PASS continues same tick through full original-device freeze/protected handoff/admission/source selection/dispatch/increasing counters. Prepare Y102 future terminal tools by read-only actual save-directory discovery; Y101 uses finalize_y101_fixed30_y102.py.',
            '## Last Tick',
            '- '+token+': observed/attempted existing work first; Y100 natural fixed30 full original finite proof/canonical archive; Y102 FIRST native PASS, exact original Y101 compatible30+8 lifecycle binding, source selection/11 admission guards/full original server2 freeze/receipt/canonical dispatch/increasing native counters. Y101 continues unchanged. Y103 same-byte scientific source/native bodies bound to actual currentHEAD/owners,10 operational guards passed.',
            '- New scientificIDs0/scientificCPUreplays0/nativeGPUtasks1/admissions1/launches1/archives1/FIRST0/cost0. Goal/old receipts/data/runtime/mappings/seed1999fixed30/350W280W/bans/GPU1+Y57 pauses/foreign jobs/scheduler intact. Closeout '+evidence(ref(out))+'; official finish/receipt/release authoritative.']
        assert len(lines)<120 and sum(r.startswith('## ') for r in lines)==7;memory.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
        notes=ROOT/('pipeline/reports/notes_'+token+'.json');assert not notes.exists()
        tick.save(notes,dict(actions=[
            '运行：'+'；'.join(cid+' '+p for cid,p in phases.items())+'。新Y102在server2GPU0/280W固定30轮运行，原Y101保留；精确源码/数据/私有环境/功耗和原生步数递增均已验证。',
            '完成：Y100自然30轮，原目录20261111_085241完整Train/Val1..30及稳定有限last30/best23/model/optimizer544/scheduler30-23通过精确核验并规范归档。活跃观察失败日志保留；训练best.516448557不是正式FIRST或目标达成，不据此宣称收益。',
            '收尾观察模板的服务器映射反转导致的失败及原工具已保留，v2仅纠正观察映射并以两条实际训练验证；不重跑科学检查/准入/归档。Y102首次真实TRAIN两步有限loss/backward及新旧路径检查通过；Y101兼容30+8、11项准入守卫、完整原server2/280W冻结协议、收据、入队与门控派发完成。原科学CPU/程序正文/数值控制保持，无重跑。',
            '新准备：Y103原7个科学文件与CPU/原生程序正文保持相同，仅绑定到当前HEADY102和原Y101/Y102资源归属，10项操作守卫通过；首次GPU原生检查仍待空槽。新科学ID0、科学CPU重跑0、原生GPU检查1、准入1、新训练1、归档1、FIRST0、成本0。'
        ],decisions=[
            'Y100保留原产物、收据和完整原设备后评协议；Y103等待匹配原owner自然30轮、完整精确有限检查点及规范归档，再验授权GPU0独占空闲；不挤占现有任务。Y102未来终态工具须只读发现原保存目录，不从启动时间推断。',
            '仅AdaRES；server2GPU1全部使用和Y57单独暂停保持，不迁移冻结后评/成本、不重启受保护实验、不用server6、不改数据/功耗/调度、不干预外部任务。无新增用户处理事项。'
        ],evidence=[ref(out),ref(backup)]))
        assert all(ref(p)==r for p,r in before.items()) and git('ls-files','--stage','-z')==index
        return out,notes,len(lines)
    finally:writer.close()

def finish(token):
    lease(token,300)
    for _,cid,_,_,_ in RUNS:command(['scripts/y102_observer_v2.py','observe','--id',cid,'--token',token,'--label','release'])
    for name,_,_,_,_ in RUNS:command([SKILL+'/gpu_occupation.py','AdaRES_project','--token',token,'--server',name,'--label','release'])
    out,notes,lines=record(token);command([SKILL+'/foreground_tick.py','AdaRES_project','report','--token',token,'--notes',str(notes)]);command([SKILL+'/foreground_tick.py','AdaRES_project','finish','--token',token])
    receipt=tick.load(ROOT/('pipeline/tick_receipts/'+token+'.json'));assert receipt['status']=='FINALIZED' and receipt['postflight']=='PASS' and receipt['context_committed'] is True and receipt['lease_release']=='released'
    assert not (ROOT/'pipeline/headless/.foreground.lockdir').exists() and not (ROOT/'pipeline/headless/.tick_context_state.next.json').exists()
    pending=ROOT/'pipeline/archive_pending';assert not pending.exists() or not any(p.is_file() for p in pending.rglob('*'))
    print(json.dumps(dict(closeout=ref(out),status=receipt['status'],postflight=receipt['postflight'],context_committed=True,lease_release=receipt['lease_release'],memory_lines=lines,observations={cid:r['phase'] for cid,r in tick.load(out)['observations'].items()},native_GPU_tasks=1,admissions=1,launches=1,archives=1)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);finish(p.parse_args().token)
