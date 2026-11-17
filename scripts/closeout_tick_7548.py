"""Persist the user's completed scoped launch; do not fabricate capacity work."""
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import closure
import foreground_tick as tick
import pipeline_monitor as monitor
TOKEN='75486f9263d34214b3bd2e1412c69639'
COMMIT='26ee1cec57c53ee2f94d6969d0bb637565b6ed59'
owner=tick.load(ROOT/'pipeline/headless/.foreground.lockdir/owner.json')
assert owner['token']==TOKEN and owner['thread_id']==os.environ['CODEX_THREAD_ID']
assert datetime.now(timezone.utc)<datetime.fromisoformat(owner['deadline_at'])
def ref(p):return dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest())
started_path='pipeline/analyses/y50_started_7548_'+TOKEN+'.json'
started=tick.load(ROOT/started_path)
assert started['complete'] and started['pid']==31539 and started['actual_single_GPU1']
steps=[closure.steps(sample['tail']) for sample in started['samples']]
assert all(s for s in steps) and max(s for _,s in steps[1])>max(s for _,s in steps[0])
progress=[dict(epoch=s[-1][0],step=s[-1][1]) for s in steps]
assert subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()==COMMIT
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    queue=monitor.load_queue(str(ROOT))
    task=next(t for t in queue['gpu_queues']['server-2'] if t['id']=='AdaRES-Y50')
    assert task['status']=='running' and task['launch_verified'] is True and task['pid']==31539
    closure.admission(ROOT,task)
    work=tick.load(ROOT/'pipeline/research_work.json')
    candidate=next(c for c in work['candidates'] if c['id']=='AdaRES-Y50')
    assert candidate['stage']=='training' and 'blocker_kind' not in candidate
    summary='pipeline/analyses/final_review_7548_'+TOKEN+'.json'
    assert not (ROOT/summary).exists()
    tick.save(ROOT/summary,dict(complete=True,recorded_at_local=datetime.now().astimezone().isoformat(),
        user_scope='Explicit direct-human authorization: ONLY tested Y50 seven-file commit and existing gated continuation.',
        authorization=ref('pipeline/authorizations/AdaRES-Y50_commit.json'),
        commit_binding=ref('pipeline/analyses/y50_commit_binding_7548_'+TOKEN+'.json'),
        code_commit=COMMIT,admission=ref('pipeline/admissions/AdaRES-Y50.json'),
        dispatch=ref('pipeline/analyses/y50_dispatch_7548_'+TOKEN+'.json'),independent_observation=ref(started_path),
        actual_native_progress=progress,authorized_GPU_rows=started['authorized_GPU_rows'],
        new_candidates=0,new_admissions=1,new_training_launches=1,unrelated_changes_preserved=True,
        source_protocol_unchanged=True,diagnostic_weights_not_deployed=True,
        scope_note='Y50 is actually training, never label it blocked to satisfy a capacity checker. The other authorized GPU has foreign PID19723; no invented candidate, widened scope or altered target.'))
    candidate['evidence'].extend([ref(started_path),ref(summary),ref('scripts/authorize_y50_7548.py'),ref('scripts/admit_dispatch_y50_7548.py')])
    candidate['next_action']='Continue SAME actual GPU1 PID31539 released-SPFormer/seed1999/fixed30 Y50; monitor steps/checkpoints and frozen terminal comparison. Commit authorization resolved by explicit human record; no diagnostic weight reuse, epoch tuning or blind restart.'
    work['design_cycle']=dict(stop_reason='external_resource_blocked',attempted_ids=['AdaRES-Y50'],
        evidence=[ref(summary),ref(started_path)],
        resume_condition='Monitor existing Y50 training. Any future separate capacity-gap work needs actual available resources and a defensible independent candidate; do not manufacture one for this scoped manual action.',
        detail='The explicitly requested Y50 commit/admission/launch is achieved with increasing native steps. GPU0 remains occupied by foreignPID19723; no second candidate requested or created. This is observed unavailable capacity, NOT a blocked scientific candidate or no-information-gain/exhaustion claim. If generic postflight cannot accept this honest disposition, preserve evidence and abort the tick without stopping training.')
    tick.save(ROOT/'pipeline/research_work.json',work)
    memory=f'''# AdaRES_project Research Memory
## Goal And Phase
- Currenttoken{TOKEN}; owner{owner['thread_id']}; snapshot2026-10-16T06:43:49.283459+00:00; immutabledeadline{owner['deadline_at']}; design. TickfinishPENDING, user-scopedcommit/launchalreadyexecuted.
- TargetScanRefer .515 unmet; historicalCSVY36 .5110/B1 .5020; no freshvalidation metric orPhase2. OnlyAdaRES/server2GPU0,1/oneexperiment-card/schedulerDisabled.
- Currentdirecthumanreply 'n那你授权啊' authorizes ONLY testedY50seven-filecommit and existing gatedcontinuation, not otherprojects/scheduler/goal/gate/protocolchanges.
## Champion And Baseline
- PriorfrozenFIRST Y48 .5034034117/W2 .5099093531/B1 .5021230823; Y48smallharmretained. Y50currentlytraining, trainedefficacyunknown.
- HistoricalexactY36 .510716624292 vs historicalB1 .502112417447 is anotherfrozencomparison, not freshsignificance.
## Active Hypotheses
- SAMEY50higher-orderjoint candidate-maskincidence now TRAINING, formerhumancommitblockresolved by pipeline/authorizations/AdaRES-Y50_commit.json; oldblockhistory/evidencepreserved.
- Exacttestedsevenfilescommitted{COMMIT}; no unrelatedstaged/worktreechangesincluded. Nativepackage/parent/goal/oldscientificevidencehashesverifiedagain, canonical additive sync verifiedsamefiles andadvancedmarker.
- Sixexistinggatespreserved: fixed32sourcecontrols/native128capturedexpressions, productionCPU/nativeTRAIN8/32all4gradients/cachedcorrespondence, released-inittools.train2steps/fullmodel+AdamW+PolyLRrestore, pairedresources, independentbytes/AST/init/RNG.
- Admissionpipeline/admissions/AdaRES-Y50.json bindsactualcommit/currentgoal/frozencomparison. Canonical_process_queueONLY; nativeprogressverified, notPIDonly.
- Actualserver2GPU1PID31539, log{task['log']}; independentobservation14:58:16-19+08 nativeTrain[1/30]158to165/4686. ReleasedSPFormer/seed1999/fixed30/native128/singleargmax/unchangedloss; no diagnosticweightsdeployed.
## Closed Or Failed Directions
- Y51public-logitcommon-mode-mismatchdiagnosisrejected: actualRESalreadysoftmaxforeground,65matchedmasksunchanged; precisioninstrumentation/importfailurepreserved, no calibrationvariant.
- Y49currentrev1realizationresource-rejected/implementation; originalTRAIN8/32zero+newTRAIN24/96zero; all6314newsubgraphsforests, no unsupportedrevision2. Mechanismnotuniversallyfalsified.
- Y46two-revisionrealizationresource-rejected; Y48currentOR-targetsmallsubgroupharmrejected/fixed30terminal; Y47/Y29/Y40closed. No silentreopen.
## Running And Queue
- Begin14:43:49+08 active0/running0/claim0/wait0/target2/admissionbudget2; lateronlyY50admittedandlaunched, queue/running/experimentupdatedbycanonicaldispatcher.
- Independent14:58:19+08 GPU0foreignPID19723/91%/22742MiBpreserved; GPU1Y50PID31539/61%/16370MiB. Thesearetime-stampedreadings, notconstantusageortrainingpeak.
- Taskgpu1/patience30/min_epochs30/fixed_epochs30; no secondcandidateortraininglaunched. Formalnewadmission1/training1/newcandidate0.
## Open Questions And Risks
- FulltrainedmIoU/Acc@.25/.5/activequeries/latency/peakmemory pending. Priorpairedcostsarenotaccuracy-resourceParetoor efficacyproof; noveltyunverified.
- FrozenY50/W2/B1full9508/141twiceEACH/FIRSTprimary; scene-bootstrap10000/1999,TRAINsizecuts1487/3325/subgroupCIupper<0harmreview; no validationselectionorautomaticPhase2.
- User-scopedonecandidateisrunning; fixedcapacitytarget2 includesforeign-occupiedGPU0. Neverfakerejected/blockedcandidateorchangegoal/contexttofitgenericcapacitypostflight. Ifcheckerrejectshonestscope, preserveandaborttick, nottraining.
## Last Tick
- Onecanonicalbegin; latesthumanexplicitY50requestresumesstableIDinstead ofcreatingplaceholderdesignbootstrapcandidate. No repeatedpreflight, no newtask/background/schedulerchange.
- Scopedauthrecordparentdirectorymissingfirstattemptfailedbeforeauth/candidatemutation; directorycreationfixed, no model/evidence/protocolchange. Laterexplicitseven-fileGitcommitandcanonicaldispatchsucceeded.
- Independenttrainedstep158to165, exactsourcecommit/GPU1ownership/no crashverified; native source/markerandforeignGPU0preserved. Evidence{summary}.
- TickfinishstillPENDING; ifpostflightcannotacceptfixedcapacity/scopedaction, recordexactfailureandabort/releasewithoutstoppingPID31539. No postreleaseprojectwrites.
- Receiptusageonlypartialbegin-to-receiptlocalwindow; cachedinput/reasoningoutputalreadyincluded, notbill; missingunknown. Nextautomationtimeunknown.
'''
    assert len(memory.splitlines())<120
    (ROOT/'pipeline/codex_memory.md').write_text(memory,encoding='utf-8')
    notes='pipeline/analyses/tick_notes_7548_'+TOKEN+'.json'
    tick.save(ROOT/notes,dict(actions=[
        '收到用户直接回复“n那你授权啊”，按前文明确范围记录仅Y50已测代码提交与既有门控续做授权，解除原提交授权阻塞；保留原阻塞历史，不扩大到其他项目、候选或协议。',
        '逐项复核Y50既有验收证据与七文件hash，限定提交26ee1cec57c53ee2f94d6969d0bb637565b6ed59；实际Git提交仅七个新增文件，其余暂存/工作区修改保留。',
        '旧六门槛、当前目标、真实commit和冻结评估绑定为新准入回执；原有canonical_process_queue检查空闲与增量同步后，仅向server2 GPU1派发Y50。released SPFormer/seed1999/fixed30/128/singleargmax/原loss不变，未部署诊断权重。',
        '14:58:16至14:58:19+08独立检查：真实Y50 PID31539在GPU1，已测源hash与marker精确一致；Train[1/30]第158步推进至165/4686，无崩溃标记。GPU0仍为外来PID19723/91%/22742MiB；GPU1为Y50/61%/16370MiB。',
        '本轮新候选0、正式准入1、真实训练启动1。科学状态training，用户要求的提交/启动已实施；tick执行状态须待finish/必要时abort核验，不能把PID或准备验收当训练后指标提升。',
        '通用目标容量为2，另GPU0外来占用，而本次用户仅要求已有Y50；不为满足通用容量检查捏造第二候选、假阻塞或改目标。若checker不接受诚实限定处置，保留证据abort并释放租约，已启动远端训练继续。',
        '收据用量在finish前未知；只统计begin至收据本地窗口，早先读取/晚到用量/最终回复不含；缓存已含输入、推理已含输出，不重复加、不当账单。'],
        decisions=[
        'Y50原提交授权阻塞已由本次明确人工授权解决，不再等待18:08重查或重复索要同一授权。该授权仅适用此次已测Y50，不是未来任意项目/代码的自动提交许可。',
        '只运行现有已准入Y50一个单卡实验，保留外来GPU0，不变更后台调度；固定30轮且保持原冻结评估/子组伤害规则。',
        '已验证递增原生训练步数后记training；不盲目重启，不改变优化器/epoch协议，不宣布尚未得到的mIoU收益。',
        '按真实训练与资源事实收尾，若通用容量校验与本次限定动作不一致，报告准确失败并释放，不放宽或绕过校验。']))
    print(json.dumps(dict(review=summary,notes=notes,stage='training',pid=task['pid'],progress=progress,finish='PENDING'),indent=2))
finally:lock.close()
