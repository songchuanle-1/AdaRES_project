"""Retain exact Y76 terminal closure, real Y79 server1 launch and CPU-only Y81."""
import argparse
import ast
import copy
from datetime import datetime, timedelta, timezone
import json
import subprocess
from prepare_y81_query_mask_refinement import ROOT, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
import closure
import gpu_occupation

TOKEN='b5d4f7646f114f71b5f59d796e2b1858'
HEAD='e0c7dba199fcf1a009941450d04aeb799418a7e6'


def candidate(work,cid):return next(row for row in work['candidates'] if row['id']==cid)


def holds():
    work=tick.load(ROOT/'pipeline/research_work.json');y76,y79=[candidate(work,cid) for cid in ('AdaRES-Y76','AdaRES-Y79')]
    assert y76['stage']=='validation' and y79['stage']=='training'
    assert monitor.load_experiment(str(ROOT),'AdaRES-Y76')['status']=='done_complete'
    proof=tick.load(closure.artifact(ROOT,y76['terminal_evidence']))
    assert proof['completed_epochs']=={phase:list(range(1,31)) for phase in ('Train','Val')} and proof['CPU_guard_tests_passed']==20
    freeze=tick.load(closure.artifact(ROOT,y76['post_training_freeze']));resource=freeze['resource_condition']
    assert (resource['server'],resource['gpu'],resource['power_limit_watts'])==('server-1',0,350.)
    assert monitor.load_running(str(ROOT))['server-1']['0']=='AdaRES-Y79'
    owner=tick.load(closure.artifact(ROOT,y79['latest_training_observation']))
    assert owner['token']==TOKEN and owner['native_progress_confirmed'] and owner['pid']==2218231
    out=ROOT/('pipeline/checks/y76_frozen_evaluation_hold_'+TOKEN+'.json');assert not out.exists()
    tick.save(out,dict(complete=True,token=TOKEN,id='AdaRES-Y76',original_terminal=y76['terminal_evidence'],original_freeze=y76['post_training_freeze'],
        frozen_resource=resource,slot_owner='AdaRES-Y79',owner_progress=y79['latest_training_observation'],frozen_bytes_changed=False,evaluation_executed=False,helper=ref(__file__)))
    y76.update(stage='blocked',blocked_from_stage='validation',blocker_kind='exclusive_gpu_slot',required_server='server-1',required_gpu=0,resource_work_kind='frozen_evaluation',
        blocker='Original Y76 fixed30 verified and canonically archived; original complete frozen FIRST/evaluation/cost requires exclusive original server1GPU0/350W, now occupied by independent Y79. Preserve that run.',
        retry_after=(datetime.now().astimezone()+timedelta(hours=1)).isoformat(),wake_paths=['pipeline/queue.json','pipeline/running.json','SERVERS.yaml'],
        unblock_condition='Fresh exclusive authorized server1GPU0/350W idle/no compute/no canonical reservation after complete original owner archive; preserve all frozen evaluation boundaries.',
        next_action='Preserve original last30/best21/model/optimizer544/scheduler30 and21, full log/freeze/receipt. Run only original full frozen FIRST/repeats/paired methods/bootstrap/subgroups/cost on exclusive original server1GPU0/350W. No restart or promotion from training best.')
    y76['evidence'].append(ref(out))
    y80=candidate(work,'AdaRES-Y80')
    y80['blocker']='All authorized cards run original independent Y77/Y78/Y79; actual native TRAIN task requires an exclusive matching idle/unreserved authorized GPU.'
    y80['next_action']='Do not replay completed CPU/model/TRAIN/version/native-plan/callback preparation. Preserve HEAD e0c and original Y77/Y78/Y79 plus paused Y57. Y80 precedes Y81. Only canonical matching owner full fixed30 archive plus fresh exclusive authorized server2GPU0/280W or GPU1/390W idle/no compute/no reservation permits its exact native YAML/TRAIN two-step task. Actual task PASS and original visual/query/padding/qia/next-lia/shared-head paths must pass before original complete evaluation/cost freeze, new receipt, exact 4e68 source contract, safe deployment and increasing steps. Server1 needs its own private-runtime/path/350W plan. No GPU PASS or efficacy from CPU checks.'
    tick.save(ROOT/'pipeline/research_work.json',work);print(json.dumps(dict(hold=ref(out),original_freeze_preserved=True)))


def save():
    work=tick.load(ROOT/'pipeline/research_work.json');get=lambda cid:candidate(work,cid)
    view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json');context=tick.load(ROOT/'pipeline/headless/.tick_context.json')
    queue,running=monitor.load_queue(str(ROOT)),monitor.load_running(str(ROOT));qref,rref=ref('pipeline/queue.json'),ref('pipeline/running.json')
    tasks={task['id']:task for rows in queue['gpu_queues'].values() for task in rows};observations={};phases={}
    for cid,name,gpu,pid in [('AdaRES-Y77','server-2','0',111334),('AdaRES-Y78','server-2','1',22192),('AdaRES-Y79','server-1','0',2218231)]:
        task=tasks[cid];assert task['status']=='running' and task['launch_verified'] and task['pid']==pid and running[name][gpu]==cid
        closure.admission(ROOT,task);evidence=get(cid)['latest_training_observation'];value=tick.load(closure.artifact(ROOT,evidence))
        assert value['token']==TOKEN and value['native_progress_confirmed'] and value['pid']==pid and evidence['path'].endswith('_final.json')
        observations[cid],phases[cid]=evidence,value['after']['latest']
    occupations={}
    for name in ('server-2','server-1'):
        evidence=ref('pipeline/checks/gpu_occupation_'+name+'_finish_'+TOKEN+'.json');value=tick.load(closure.artifact(ROOT,evidence))
        assert value['token']==TOKEN and (datetime.now(timezone.utc)-datetime.fromisoformat(value['observed_at'])).total_seconds()<300
        row=next(c for c in work['candidates'] if c.get('stage')=='blocked' and c.get('blocker_kind')=='exclusive_gpu_slot' and c.get('required_server')==name)
        server=next(s for s in context['queue']['servers'] if s['name']==name)
        assert not gpu_occupation.validated_state(ROOT,context,row,server,queue)['idle_gpu_ids'];occupations[name]=evidence
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()==HEAD
    assert ref('pipeline/goal.json')['sha256']=='4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a'
    assert tasks['AdaRES-Y57']['status']=='paused' and tasks['AdaRES-Y57']['dispatch_hold'] and tasks['AdaRES-Y57']['resume_requires_user_authorization']
    y76,y79,y81=[get(cid) for cid in ('AdaRES-Y76','AdaRES-Y79','AdaRES-Y81')]
    assert 'AdaRES-Y76' not in tasks and monitor.load_experiment(str(ROOT),'AdaRES-Y76')['status']=='done_complete'
    proof=tick.load(closure.artifact(ROOT,y76['terminal_evidence']));assert proof['completed_epochs']=={phase:list(range(1,31)) for phase in ('Train','Val')}
    closure.artifact(ROOT,y76['terminal_archive_binding']);closure.artifact(ROOT,y76['post_training_freeze'])
    raw=tick.load(closure.artifact(ROOT,y79['native_smoke']));assert raw['steps']==raw['optimizer_steps']==2 and raw['feedback_calls']==raw['qla_consumption_calls']==12 and raw['next_lla_consumption_calls']==10 and raw['shared_head_calls']==14
    assert raw['server']=='server-1' and raw['physical_gpu']==0 and raw['data_source']['config_data_root'].rstrip('/')=='/data1/songcl/Project/AdaRES_project/data'
    freeze=tick.load(closure.artifact(ROOT,y79['post_training_freeze']));parent=tick.load(closure.artifact(ROOT,freeze['parent_protocol']))
    restored=copy.deepcopy(freeze['evaluation']);restored['methods'][0]='AdaRES-Y64'
    assert restored==parent['evaluation'] and freeze['training']==parent['training'] and freeze['resource_condition']['power_limit_watts']==350.
    guards=ref('pipeline/checks/adares-y79_admission_guards_'+TOKEN+'.log');assert b'Ran 17 tests' in closure.artifact(ROOT,guards).read_bytes()
    for cid in ('AdaRES-Y80','AdaRES-Y81'):
        row=get(cid);assert row['stage']=='blocked' and row['blocker_kind']=='exclusive_gpu_slot' and row['implementation_test_passed']
        for field in ('source_manifest','CPU_native_observer_verification','native_preparation'):closure.artifact(ROOT,row[field])
        for field in ('native_smoke_plan_server2_gpu0','native_smoke_plan_server2_gpu1'):
            plan=tick.load(closure.artifact(ROOT,row[field]));frozen=ast.literal_eval(ast.parse(closure.artifact(ROOT,plan['program']).read_text(encoding='utf-8')).body[0].value)
            assert validate_frozen(row,plan,frozen)==7
    out=ROOT/('pipeline/checks/resource_handoff_'+TOKEN+'.json');assert not out.exists()
    tick.save(out,dict(complete=True,token=TOKEN,snapshot=view['generated_at'],observed_at=datetime.now().astimezone().isoformat(),original_live_training=observations,occupations=occupations,
        Y76_terminal=y76['terminal_evidence'],Y76_archive=y76['terminal_archive_binding'],Y76_original_freeze=y76['post_training_freeze'],
        Y76_frozen_evaluation_hold=ref('pipeline/checks/y76_frozen_evaluation_hold_'+TOKEN+'.json'),Y79_actual_native_smoke=y79['native_smoke'],Y79_receipt=ref('pipeline/admissions/AdaRES-Y79.json'),Y79_original_complete_freeze=y79['post_training_freeze'],
        Y79_source_selection=ref('pipeline/checks/adares-y79_source_selection_'+TOKEN+'.json'),Y79_dispatch=ref('pipeline/checks/adares-y79_dispatch_'+TOKEN+'.json'),Y79_admission_guards=guards,
        Y81_CPU_version=y81['source_manifest'],Y81_exact_CPU_native_callbacks=y81['CPU_native_observer_verification'],Y81_native_preparation=y81['native_preparation'],
        source_HEAD=HEAD,server1_source_marker=HEAD,server2_source_marker='5eee1e50c9e1d65bf113a4b8757bfc27d028a821',new_candidates=1,new_admissions=1,new_dispatches=1,archived=['AdaRES-Y76'],
        prepared_native_pending=['AdaRES-Y80','AdaRES-Y81'],Y57_pause_preserved=True,formal_metrics_updated=False,helper=ref(__file__)))
    for cid in ('AdaRES-Y76','AdaRES-Y77','AdaRES-Y78','AdaRES-Y79','AdaRES-Y80','AdaRES-Y81'):get(cid)['evidence'].append(ref(out))
    y81['evidence'].extend([ref('scripts/y81_observer_core.py'),ref('scripts/build_y81_probe_sources.py')])
    result=ref('pipeline/analyses/y81_native_observer_cpu_'+TOKEN+'.json')
    work['tick_attempt']=dict(tick_started_at=view['generated_at'],candidate_id='AdaRES-Y81',kind='implementation_test',result='passed',
        command='python -B scripts/run_experiment_first_probe.py y81_native_observer_cpu pipeline/checks/y81_native_observer_CPU_'+TOKEN+'_program.py --token '+TOKEN+' --seconds 480',
        observation='Thirteen exact native callbacks/classes executed on two synthetic CPU decoder optimizer steps after original native full model/pretrain/TRAIN loading. Original static mask/current refined query/padding and prediction/coarse-gate/shared-head consumers, zero/next-step response and finite original/new/input gradients and updates passed. No GPU/native sparse task/admission/efficacy for Y81.',evidence=[y81['CPU_native_observer_verification'],result])
    assert all(evidence in y81['evidence'] for evidence in work['tick_attempt']['evidence']);tick.save(ROOT/'pipeline/research_work.json',work)
    memory=ROOT/'pipeline/codex_memory.md';backup=ROOT/('pipeline/snapshots/codex_memory_before_'+TOKEN+'.md');assert not backup.exists();backup.write_bytes(memory.read_bytes())
    def e(value):return value['path']+' SHA'+value['sha256']
    old=memory.read_text(encoding='utf-8').splitlines();sections={};section=None
    for line in old:
        if line.startswith('## '):section=line;sections[section]=[]
        elif section:sections[section].append(line)
    lines=[
        '# AdaRES_project Research Memory','## Goal And Phase',
        '- 当前token '+TOKEN+'，snapshot'+view['generated_at']+'，deadline'+view['deadline_at']+'不延长；postflight只认官方同token收据。',
        '- 实际CODEX_THREAD_ID=owner=01a07e32-78bf-7670-925b-370911ea48db已核，Pipeline-AdaRES_project Disabled；experiment_first enabled，snapshot admission2/creation2。本轮新候选1/准入1/启动1/归档1：Y76完整fixed30归档、Y79实际server1 native PASS后训练、Y81 CPU准备；无正式FIRST/cost。',
        '- PROJECT_ROOT=D:/project，PYTHONDONTWRITEBYTECODE=1，PYTHONUTF8=1，python -B；SSH密钥可用，未读密码文件，不推定sudo。',
        '- 授权server1GPU0既有350W；server2GPU0保持280W/GPU1既有390W；一GPU一独立实验。Y57单独暂停；禁server6/跨项目/改调度/硬件功耗/驱逐/重启迁移受保护运行。',
        '- goal科学字节SHA4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a保持；最新AGENTS/SERVERS及server2_gpu1_20261029授权覆盖旧GPU1资源句，原receipt/科学goal不改。',
        '- 前轮完整记忆精确备份：'+e(ref(backup))+'；历史机制、禁用、专用恢复、全部原证据沿备份保持。',
        '## Champion And Baseline',*sections['## Champion And Baseline'],
        '## Active Hypotheses',
        '- Y77原query初始化geometry/mass乘积与Y78六次learned previous-mask-logit residual仍各自独立W2；原数据/全输入/现有2D/发布backbone/原采样/geometry/SCG/k/gamma/共享头-targets-loss-selector/seed1999/fixed30/native YAML和原freeze/receipt保持。原模型/CPU/native/终态与负结果全见备份，不重放。',
        '- Y79=adares_y79_query_language_feedback：六次language_next=language_ffn+clean((softmax(language_ffn@current_query.T/sqrt(d))@current_query)@W)，共享零W256x256/no RNG，+65536/总164695173，语言padding残差零；原stage0/static mask/全128query/全channels/未detach、所有原协议保持。保留commite0c7dba199fcf1a009941450d04aeb799418a7e6，parentY78/5eee；模型独立W2，不组合Y78。',
        '- Y79本轮server1GPU0/350W真实原native YAML/精确映射TRAIN两步PASS：loss4.615510940551758/6.303436279296875，四原场景16表达；六query/languageFFN/padding身份/未detach、六qla与五next-lla/七共享confidence-query_norm、独立float64公式/首零次响应/有限原新输入梯度及真实更新通过，峰6687817728B。17准入/资源/源版本负守卫通过；只覆盖两批，不是效益/全量显存保证。',
        '- Y79 actual smoke：'+e(y79['native_smoke'])+'；receipt：'+e(ref('pipeline/admissions/AdaRES-Y79.json'))+'；freeze：'+e(y79['post_training_freeze'])+'；Y64完整evaluation/training深复制核等。此前server1 CPU/8守卫与server2双计划保持原证据，但本轮已activate/run/admit/dispatch，勿重放。',
        '- Y76本轮专用finalize_y76_fixed30.py核20生命周期守卫、原源/YAML/receipt/freeze、完整Train/Val1..30及稳定finite last30/best21/model/optimizer544/scheduler30和21，canonical显式server1GPU0 done_complete归档。原目录20261101_210057产物保留；terminal：'+e(y76['terminal_evidence'])+'；archive：'+e(y76['terminal_archive_binding'])+'。训练best .5050052404403687不是FIRST。',
        '- Y80=adares_y80_query_visual_feedback：独立W2，六次visual_next=visual_ffn+clean((softmax(visual_ffn@current_query.T/sqrt(d))@current_query)@W)，零W65536/no RNG；原mask base不变，六qia/五next-lia与七共享头和全部原协议保持。CPU完整model/backbone/TRAIN与两合成优化步、11原native计划守卫/15冻结回调CPU执行已通过；无GPU任务PASS。commit4e68cd84d56d890222adc2bec8ab59cb0c8715b6，parentY79/e0c；version：'+e(get('AdaRES-Y80')['source_manifest'])+'。',
        '- Y81=adares_y81_query_mask_refinement：独立W2，六次mask_features_i=mask_base+clean((softmax(mask_base@current_refined_queries.T/sqrt(d))@current_refined_queries)@W)，共享零W256x256/no RNG，+65536/总164695173；原stage0不变，原static mask base逐次重用不累积，padding残差零，全部128query/allchannels/两输入未detach。不同于Y72 visual→mask、Y68语言mask调制、Y79语言流与Y80 visual流，不重开/组合旧路线，不宣称新信息/创新/效益。',
        '- Y81保持原initial prediction/sampling/geometry/coarse-fine equal fusion/原Boolean gate算法/SCG/k/gamma、七共享confidence头targets-loss-selector、全输入/现有2D/发布backbone/seed1999/fixed30/native YAML；新attention仅固定sqrt(d)，无temperature/threshold/loss/feature/epoch/validation扫描。',
        '- Y81 CPU完整原native model/backbone/TRAIN162738点8表达、float32/64独立逐标量softmax-context-projection、zero/RNG/parent weights/置换-padding-allpadding/同均值不同query分布/错配query、七阶段W2输出与输入梯度锚点、非零/refinement bypass通过。两合成CPU DEC原optimizer步各387有限梯度/10原头梯度/六原static mask-current refined query-padding身份、真实prediction消费/原hard gate与六coarse消费者/七共享头与query_norm、有限新梯度与更新通过。raw：'+e(ref('pipeline/analyses/y81_cpu_'+TOKEN+'.json'))+'。',
        '- Y81仅七精确测试A文件隔离index保留commit4157937d0c3cf344a44394ce15c95349aa94d50f，branch refs/heads/adares-y81-query-mask-refinement，parentY80/4e68；模型独立W2不组合Y80，主HEADe0c/index/queue/running保持。version：'+e(y81['source_manifest'])+'。',
        '- Y81双卡native两步计划/11证据拒绝守卫/Python3.8 AST/goal-源码YAML-version与stale/tamper拒绝通过；13冻结native函数/类与CPU wrapper AST相同且两合成CPU优化步实际执行通过。preparation：'+e(y81['native_preparation'])+'；verification：'+e(y81['CPU_native_observer_verification'])+'。无CUDA/native sparse task/GPU PASS/准入。',
        '- Y80优先Y81；二者blocked/validation/exclusive_gpu_slot/server2GPU0，GPU1备选；CPU/model/data/version/native-plan/callback准备已完成，不重放。仅canonical匹配owner完整归档+fresh独占idle/no compute/no reservation后原native两步PASS，才能原完整后评freeze/新receipt/精确source契约/安全派发并核计数；二者均未备server1方案。',
        '## Closed Or Failed Directions',*sections['## Closed Or Failed Directions'],
        '- Y81本轮实际CPU和完整原native回调CPU检查首次通过；未出现科学失败或模型修订。旧helper/失败程序/原日志保持不可变；不得修改旧hash-bound helper来支持新HEAD。',
        '## Running And Queue',
    ]
    for cid,resource,pid in [('AdaRES-Y77','server2GPU0/280W',111334),('AdaRES-Y78','server2GPU1/390W',22192),('AdaRES-Y79','server1GPU0/350W',2218231)]:
        phase=phases[cid];lines.append('- '+cid+' '+resource+' PID'+str(pid)+' 原source/YAML/receipt/marker/计算归属/功耗与native递增已核：'+phase['phase']+str(phase['epoch'])+'/30 step'+str(phase['step'])+'/'+str(phase['count'])+'；'+e(observations[cid])+'。')
    lines += [
        '- 主master/server1 marker e0c7dba199fcf1a009941450d04aeb799418a7e6；server2 marker仍5eee1e50c9e1d65bf113a4b8757bfc27d028a821。本轮仅选择七已保留且实际测试的Y79 A文件；server1从c384安全ff追加Y77/Y78/Y79共21文件，原server2两运行保持。现用scripts/y79_deployment.py observe --id AdaRES-Y77|AdaRES-Y78|AdaRES-Y79 --token NEW_TOKEN --label progress；旧y78及更早observer不兼容当前HEAD，勿调用。',
        '- 20项Y56/Y58..Y76已fixed30完成，等待各自原冻结完整FIRST/cost；原required server/GPU/功耗保持。Y76完整后评server1GPU0/350W由Y79占用；Y75完整后评server2GPU1/390W由Y78占用，不用训练best代替FIRST或重训。',
        '- 本轮fresh occupation server2：'+e(occupations['server-2'])+'；server1：'+e(occupations['server-1'])+'；三授权卡busy+reserved。finish前须<300秒，超时新label；未知/失败/缺设备/idle不能证明hold。',
        '- server1=songcl@192.168.1.175/dl02-MS-7B73，root=/data1/songcl/Project/AdaRES_project，private python=.../runtime/adares_project/bin/python，sitepackages=.../lib/python3.8/site-packages，pointnet=.../runtime/pointnet2-0.0.0-py3.8-linux-x86_64.egg，LD_LIBRARY_PATH=.../runtime/cuda113lib64，原精确data=.../data/，work=.../AdaRES_checkpoints；仅SERVERS两授权path mappings，原数据证明保持。',
        '- server2=songcl@192.168.1.163/dl01-MS-7C60，root=/bak_home/songcl/Project/AdaRES_project，code=.../AdaRES，python=/home/songcl/.conda/envs/adares_project/bin/python，原data=/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/；原环境/data保持。',
        '- Y57单独paused/dispatch_hold/resume_requires_user_authorization；原20261021_004739/last.pth epoch27/model/optimizer543/scheduler27/best .5028846859931946保留，只有单独明确Y57继续授权才能从28续至fixed30，不迁移/不从pretrain重启。',
        '## Open Questions And Risks',
        '- 无本轮正式全量FIRST/成本或科学胜出结论，单seed方差与server2历史掉卡/散热/Xid79根因仍未知。训练递增不证明硬件修复；两步native PASS只覆盖已测批次，CPU不作GPU或效益证明。',
        '- Y80/Y81未实际GPU任务，不能准入；既有19+Y76完整后评需要原资源。无新增项目常规操作授权缺口，不重复索权；保持Y57单独暂停。',
        '- Y77仍在训练，专用终态契约尚未准备：必须匹配原PID111334/server2GPU0/280W/原498e源与server2 marker5eee、当前HEADe0c和完整日志/检查点；不能套Y76或Y75 finalizer。',
        '- heartbeat原owner/原两小时间隔/通知保持；不建线程/新自动化/嵌套codex exec。旧resource wait v2只支持旧HEAD5eee与原Y76/Y77/Y78，不可直接用于新e0c和Y77/Y78/Y79；需要新绑定，保留旧helper。',
        '## Last Tick',
        '- '+TOKEN+' begin/design先查并派发waiting0；Y76已自然退出，核原terminal plan依赖/receipt/freeze，20生命周期守卫与完整fixed30 canonical显式server1GPU0归档。fresh idle/unreserved350W后Y79原server1 alternative guarded activate/run真实两步PASS、17准入守卫、七精确源码选择/原完整冻结后评、新receipt/安全server1-only派发PID2218231，native Train1实际递增。原Y77/Y78保持。',
        '- 按待跑准备缺口新增独立Y81，实际CPU model/backbone/TRAIN与独立公式/控制/两优化步、隔离保留版本、双GPU计划11守卫与13原native回调CPU执行通过；未Y81 GPU PASS。两待检Y80/Y81等授权独占空槽，优先Y80；无正式FIRST/cost。',
        '- 本轮新候选1/准入1/启动1/归档1；handoff：'+e(ref(out))+'；tick_attempt绑定Y81实际CPU检查。主HEAD/server1 marker e0c，server2仍5eee；新y79_deployment为后续精确observer，不改旧hash-bound helpers。',
        '- 下轮先y79_deployment核三原运行并派发现有任务。Y77接近后续fixed30时先准备独立完整server2GPU0/280W终态契约与守卫，原PID/源/YAML/receipt/freeze/日志/检查点全部核验后canonical显式GPU0归档；fresh idle/unreserved280W后Y80原native两步PASS→原完整freeze/新receipt/精确4e68版本source契约→安全派发。Y81随后；保留原数据/禁用/功耗/Y57单独暂停/调度。',
        '- 官方report/finish后只读核FINALIZED/PASS/context_committed/released及锁/pending/archive_pending；此记忆不预签结果，释放后不写。',
    ]
    assert len(lines)<120 and sum(line.startswith('## ') for line in lines)==7;memory.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    notes=ROOT/('pipeline/checks/foreground_notes_'+TOKEN+'.json');assert not notes.exists()
    tick.save(notes,dict(actions=[
        '完成：Y76自然fixed30，核原专用plan依赖/receipt/freeze、20生命周期守卫、完整Train/Val1..30与稳定finite last30/best21/model/optimizer544/scheduler30和21，canonical显式server1GPU0归档，原checkpoint全保留。训练best不作正式FIRST。',
        '运行：Y79在server1GPU0/350W实际原生YAML/精确映射TRAIN两步PASS，loss4.615510940551758/6.303436279296875、六原query/language-FFN/pad身份与六qla/五next-lla/七共享头和有限梯度更新通过；17准入守卫、原完整冻结后评、七精确源码选择、安全server1派发PID2218231后native Train1计数递增。Y77/Y78原运行保持。',
        '新准备：独立Y81查询到后续mask特征attention residual完成CPU原生模型/发布backbone/真实TRAIN和独立公式/zero-RNG/全部锚点/消费者/两合成CPU解码器优化步、七测试文件隔离版本、双GPU计划11守卫与13冻结native回调CPU执行；未GPU任务或准入。',
        '本轮新候选1/准入1/启动1/归档1，无正式FIRST/cost；20项fixed30模型等待原完整后评。已备份短记忆并保存新hash handoff；Y76原完整后评GPU0由Y79占用，原freeze字节不改。'],decisions=[
        '三授权GPU现由Y77/Y78/Y79独立训练占用，fresh occupation无空槽；Y80优先Y81等待匹配原native检查，不挤占现有运行或重放已通过CPU准备。',
        '后续用y79_deployment精确observer支持当前e0c HEAD，server1 marker e0c/server2仍5eee。Y77终态需独立server2GPU0/280W/原源与新版本链完整契约和守卫，不套其他ID finalizer。',
        '保留原数据、独立runtime与仅授权mapping、科学goal/旧receipt、seed1999固定30、全部禁用、server1GPU0 350W/server2GPU0 280W/GPU1 390W及Y57单独暂停；禁server6/跨项目/调度变更。当前无新增用户处理事项。']))
    assert qref==ref('pipeline/queue.json') and rref==ref('pipeline/running.json')
    print(json.dumps(dict(handoff=ref(out),notes=str(notes.relative_to(ROOT)),memory_lines=len(lines),counts=dict(new=1,admitted=1,launched=1,archived=1))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=('holds','save'));args=parser.parse_args();lease(TOKEN,180)
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:{'holds':holds,'save':save}[args.operation]()
    finally:lock.close()
