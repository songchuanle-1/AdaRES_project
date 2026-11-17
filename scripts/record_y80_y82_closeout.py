"""Persist this Y77 archive/Y80 launch/Y82 preparation with current resource evidence."""
import argparse
import ast
import copy
from datetime import datetime,timedelta,timezone
import json
import subprocess
import sys
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from check_ef_execution import validate_frozen
import closure
import gpu_occupation


def record(token):
    lease(token,180)
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:
        work=tick.load(ROOT/'pipeline/research_work.json')
        c=next(c for c in work['candidates'] if c['id']=='AdaRES-Y77')
        queue=monitor.load_queue(str(ROOT));running=monitor.load_running(str(ROOT))
        assert monitor.load_experiment(str(ROOT),c['id'])['status']=='done_complete'
        closure.artifact(ROOT,c['terminal_evidence']);closure.artifact(ROOT,c['terminal_archive_binding'])
        assert c['stage']=='validation' and running['server-2']['0']=='AdaRES-Y80'
        assert next(t for t in queue['gpu_queues']['server-2'] if t['id']=='AdaRES-Y80')['status']=='running'
        c.update(stage='blocked',blocked_from_stage='validation',blocker_kind='exclusive_gpu_slot',
            blocker='Original Y77 fixed30 and finite stable last/best/model/optimizer/scheduler were verified and canonically archived. Its original full frozen FIRST/cost comparison requires exclusive server-2 GPU0/280W, now occupied by verified independent Y80. Preserve both artifacts and training.',
            retry_after=(datetime.now().astimezone()+timedelta(hours=1)).isoformat(),
            unblock_condition='Fresh exclusive authorized original server-2 GPU0/280W, no compute process or canonical reservation; preserve original full frozen protocol.')
        tick.save(ROOT/'pipeline/research_work.json',work)
    finally:lock.close()
    for server in ('server-2','server-1'):
        subprocess.run([sys.executable,'-B','C:/Users/chuanle song/.codex/skills/songcl_research_pipeline/gpu_occupation.py',
            'AdaRES_project','--token',token,'--server',server,'--label','closeout'],cwd=ROOT,check=True,capture_output=True,timeout=45)
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:
        work=tick.load(ROOT/'pipeline/research_work.json');get=lambda cid:next(c for c in work['candidates'] if c['id']==cid)
        context=tick.load(ROOT/'pipeline/headless/.tick_context.json');view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json')
        queue=monitor.load_queue(str(ROOT));running=monitor.load_running(str(ROOT));qref,rref=ref('pipeline/queue.json'),ref('pipeline/running.json')
        assert closure.git(ROOT,'rev-parse','HEAD')=='4e68cd84d56d890222adc2bec8ab59cb0c8715b6'
        assert ref('pipeline/goal.json')['sha256']=='4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a'
        tasks={t['id']:t for rows in queue['gpu_queues'].values() for t in rows}
        observations={};phases={}
        for cid,name,gpu,pid in [('AdaRES-Y80','server-2','0',42226),('AdaRES-Y78','server-2','1',22192),('AdaRES-Y79','server-1','0',2218231)]:
            task=tasks[cid];assert task['status']=='running' and task['launch_verified'] and task['pid']==pid and running[name][gpu]==cid
            closure.admission(ROOT,task);evidence=get(cid)['latest_training_observation'];value=tick.load(closure.artifact(ROOT,evidence))
            assert value['token']==token and value['native_progress_confirmed'] and value['pid']==pid
            observations[cid],phases[cid]=evidence,value['after']['latest']
        paused=tasks['AdaRES-Y57'];assert paused['status']=='paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']
        done=get('AdaRES-Y77');terminal=tick.load(closure.artifact(ROOT,done['terminal_evidence']))
        archive=tick.load(closure.artifact(ROOT,done['terminal_archive_binding']))
        assert terminal['completed_epochs']=={p:list(range(1,31)) for p in ('Train','Val')} and terminal['CPU_guard_tests_passed']==20
        assert archive['only_exact_Y77_archived'] and archive['checkpoint_files_retained'] and not archive['full_frozen_evaluation_executed']
        occupations={}
        for name in ('server-2','server-1'):
            evidence=ref('pipeline/checks/gpu_occupation_'+name+'_closeout_'+token+'.json');value=tick.load(closure.artifact(ROOT,evidence))
            assert value['token']==token and (datetime.now(timezone.utc)-datetime.fromisoformat(value['observed_at'])).total_seconds()<300
            c=next(c for c in work['candidates'] if c.get('stage')=='blocked' and c.get('blocker_kind')=='exclusive_gpu_slot' and c.get('required_server')==name)
            server=next(s for s in context['queue']['servers'] if s['name']==name)
            assert not gpu_occupation.validated_state(ROOT,context,c,server,queue)['idle_gpu_ids'];occupations[name]=evidence
        retained={}
        for cid in ('AdaRES-Y81','AdaRES-Y82'):
            c=get(cid);assert c['stage']=='blocked' and c['blocker_kind']=='exclusive_gpu_slot' and c['implementation_test_passed']
            for field in ('source_manifest','native_preparation','CPU_native_observer_verification'):closure.artifact(ROOT,c[field])
            retained[cid]={}
            for field in ('native_smoke_plan_server2_gpu0','native_smoke_plan_server2_gpu1'):
                evidence=c[field];plan=tick.load(closure.artifact(ROOT,evidence));program=closure.artifact(ROOT,plan['program'])
                frozen=ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
                assert validate_frozen(c,plan,frozen)==7 and plan['native_executed'] is False;retained[cid][field]=evidence
        cpu82=get('AdaRES-Y82');version=tick.load(closure.artifact(ROOT,cpu82['source_manifest']));cpu=tick.load(closure.artifact(ROOT,version['CPU_result']))
        assert cpu['implementation_test_passed'] and cpu['synthetic_CPU_decoder_optimizer_steps']==2 and not cpu['native_smoke_passed'] and not cpu['CUDA_initialized']
        prep=tick.load(closure.artifact(ROOT,cpu82['native_preparation']));assert prep['guard_tests_passed']==11 and prep['exact_callback_functions_and_classes']==12 and prep['CPU_native_callbacks_actually_executed']
        out=ROOT/('pipeline/checks/y80_launch_y82_wait_'+token+'.json');assert not out.exists()
        tick.save(out,dict(complete=True,token=token,snapshot=view['generated_at'],original_live_training=observations,
            Y77_terminal=done['terminal_evidence'],Y77_archive=done['terminal_archive_binding'],occupations=occupations,
            Y80_admission=ref('pipeline/admissions/AdaRES-Y80.json'),Y80_native=get('AdaRES-Y80')['native_smoke'],
            retained_native_plans=retained,Y82_CPU=version['CPU_result'],Y82_version=cpu82['source_manifest'],Y82_native_preparation=cpu82['native_preparation'],
            new_candidates=1,new_admissions=1,new_launches=1,new_archives=1,formal_metrics_updated=False,Y57_pause_preserved=True,helper=ref(__file__)))
        for cid in ('AdaRES-Y77','AdaRES-Y78','AdaRES-Y79','AdaRES-Y80','AdaRES-Y81','AdaRES-Y82'):get(cid)['evidence'].append(ref(out))
        work['tick_attempt']=dict(tick_started_at=view['generated_at'],candidate_id='AdaRES-Y82',kind='implementation_test',result='passed',
            command='python -B scripts/run_experiment_first_probe.py y82_cpu pipeline/checks/y82_cpu_'+token+'_program.py --token '+token+' --seconds 480',
            observation='Recorded native model/pretrain/original TRAIN CPU collation, independent product controls and seven-stage W2 anchors, actual two synthetic CPU optimizer steps and twelve exact native observer callbacks passed. Retained source and dual native plans/eleven guards; no Y82 GPU task, admission or efficacy.',
            evidence=[version['CPU_result'],cpu82['source_manifest'],cpu82['native_preparation'],ref(out)])
        tick.save(ROOT/'pipeline/research_work.json',work)
        memory=ROOT/'pipeline/codex_memory.md';backup=ROOT/('pipeline/snapshots/codex_memory_before_'+token+'.md');assert not backup.exists();backup.write_bytes(memory.read_bytes())
        old=memory.read_text(encoding='utf-8').splitlines()
        def section(name):
            start=old.index('## '+name)+1;end=next((i for i in range(start,len(old)) if old[i].startswith('## ')),len(old));return old[start:end]
        def e(v):return v['path']+' SHA'+v['sha256']
        lines=['# AdaRES_project Research Memory','## Goal And Phase',
            '- 当前token '+token+'，snapshot'+view['generated_at']+'，deadline'+view['deadline_at']+'不延长；postflight只认同token官方收据。',
            '- 实际CODEX_THREAD_ID=owner=01a07e32-78bf-7670-925b-370911ea48db已核；Pipeline-AdaRES_project Disabled；experiment_first enabled，admission2/creation2。本轮新候选1/准入1/启动1/归档1，无正式FIRST/cost。',
            '- PROJECT_ROOT=D:/project，PYTHONDONTWRITEBYTECODE=1，PYTHONUTF8=1，python -B；SSH密钥有效，未读密码文件，不推定sudo。',
            '- 仅server1GPU0既有350W、server2GPU0保持280W/GPU1既有390W；一GPU一独立实验；Y57单独暂停，禁server6/跨项目/改调度/硬件功耗/驱逐/重启迁移受保护运行。',
            '- goal科学字节SHA4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a保持；最新AGENTS/SERVERS与server2_gpu1_20261029覆盖旧GPU1资源句，原receipt不改。',
            '- 前轮完整七节记忆精确备份：'+e(ref(backup))+'；历史机制/禁用/专用恢复与全部原证据沿备份保持。',
            '## Champion And Baseline']+section('Champion And Baseline')+[
            '## Active Hypotheses',
            '- Y80独立W2六次query-set→visual-FFN attention残差，零W256x256/no RNG/+65536，原static mask与六qia/五next-lia/七共享confidence-query_norm保持；本轮server2GPU0/280W真实原native YAML/TRAIN两步PASS，loss4.615510940551758/6.330480575561523，峰reserved7004487680B；仅两批，不是效益或全量显存保证。',
            '- Y80原生记录：'+e(get('AdaRES-Y80')['native_smoke'])+'；原始实际结果pipeline/analyses/y80_gpu0_smoke_'+token+'.json SHA7d7c5ca15d1c40e6a2d122f9760cd2e5de97de37284901b1d9fff24031f5ecdd；17准入/源链/资源拒绝守卫通过。',
            '- Y80 receipt：'+e(ref('pipeline/admissions/AdaRES-Y80.json'))+'；完整后评freeze：'+e(get('AdaRES-Y80')['post_training_freeze'])+'；原Y64 evaluation/training深复制核等，不缩减后评。',
            '- Y77本轮专用finalize_y77_fixed30.py核20守卫、原source/YAML/receipt/freeze、完整Train/Val1..30及稳定finite last30/best21/model/optimizer544/scheduler30和21后，canonical显式GPU0 done_complete归档；原目录20261101_175820/全部产物保留。训练best .5108122825622559不是FIRST。',
            '- Y77 terminal：'+e(done['terminal_evidence'])+'；archive：'+e(done['terminal_archive_binding'])+'；后评仍原server2GPU0/280W，现由Y80占用。不要重放terminal/prepare或重训。',
            '- Y81=adares_y81_query_mask_refinement；独立W2六次mask_base+query-set attention残差，零W65536/no RNG，原static base逐次重用不累积。既有完整CPU/model/data/标量/七阶段锚点/两优化步/11native计划守卫/13冻结回调CPU实际执行已通过；无GPU PASS。commit4157937d0c3cf344a44394ce15c95349aa94d50f；精确机制/全部原证据见前轮备份。',
            '- Y81 version：'+e(get('AdaRES-Y81')['source_manifest'])+'；native preparation：'+e(get('AdaRES-Y81')['native_preparation'])+'；不重放已通过CPU/prepare。',
            '- Y82=adares_y82_scale_product；独立W2六次(coarse+fine)/2+(coarse*fine)@W，共享零W256x256/no RNG/+65536/总164695173。两原branch全channel/128query未detach，初始prediction/static masks/Boolean gate算法/SCG/k/gamma/sampling/geometry/原全输入/2D/backbone/七共享头targets-loss-selector/seed1999/fixed30/native YAML保持。',
            '- Y82不同于Y70的query条件凸混合与Y71的3D/2D输入乘积位置，不组合或重开旧路线；不宣称新信息/创新/效益，无val扫描。原model/backbone/TRAIN162738点8表达、float32/64独立逐标量乘积投影、zero/RNG/全输入梯度/置换/branch swap/同均值不同乘积/错配/非零-bypass、七阶段W2与输入梯度锚点、两合成CPU优化步和原coarse/fine/qla/shared head/query_norm/hard gates检查通过。',
            '- Y82实际CPU：'+e(version['CPU_result'])+'；version：'+e(cpu82['source_manifest'])+'；commit5773768ffb87d7fea7432e85fd0da561556aa2fc，parentY81/415793，branch refs/heads/adares-y82-scale-product。仅七测试A文件隔离index保留，主HEAD/index/生产运行保持。',
            '- Y82双原native GPU计划：'+e(cpu82['native_preparation'])+'；11拒绝守卫/陈旧goal-源码篡改拒绝/Python3.8 AST/12原native回调与实际CPU回调AST一致通过。无CUDA/native sparse TRAIN任务/GPU PASS/准入。',
            '- Y81优先Y82；二者均blocked/validation/exclusive_gpu_slot/server2GPU0，GPU1原计划备选。先canonical匹配owner完整归档，再fresh独占idle/no compute/no reservation/原功耗与匹配冻结plan的native两步PASS，才能原完整后评freeze/receipt/精确source契约/安全派发并核计数；无server1计划。',
            '## Closed Or Failed Directions']+section('Closed Or Failed Directions')+[
            '- 本轮Y80实际GPU与17准入守卫、Y82实际CPU/11native计划守卫首次通过；旧hash-bound helper/失败probe/原日志均保持，不为新HEAD修改旧helper。',
            '## Running And Queue']
        for cid,name in [('AdaRES-Y80','server2GPU0/280W PID42226'),('AdaRES-Y78','server2GPU1/390W PID22192'),('AdaRES-Y79','server1GPU0/350W PID2218231')]:
            v=phases[cid];lines.append('- '+cid+' '+name+' source/YAML/receipt/marker/计算归属/功耗及native递增已核：'+v['phase']+str(v['epoch'])+'/30 step'+str(v['step'])+'/'+str(v['count'])+'；'+e(observations[cid])+'。')
        lines += [
            '- 主master/server2 marker4e68cd84d56d890222adc2bec8ab59cb0c8715b6；server1 marker仍e0c7dba199fcf1a009941450d04aeb799418a7e6。仅选择七已测试Y80 A文件；server2从5eee安全ff追加Y79/Y80共14文件，原Y78源码保持。现用scripts/y80_deployment.py observe --id AdaRES-Y78|AdaRES-Y79|AdaRES-Y80 --token NEW_TOKEN --label progress；旧y79及更早observer不支持新HEAD。',
            '- 21项Y56/Y58..Y77 fixed30已完成，等待各自原冻结完整FIRST/cost；保留原required server/GPU/功耗，不重训、不用训练best代替FIRST。',
            '- fresh occupation server2：'+e(occupations['server-2'])+'；server1：'+e(occupations['server-1'])+'；三卡busy+reserved。finish证据须<300秒，超时新label；未知/失败/缺设备/idle不能证明hold。']
        lines += [line for line in section('Running And Queue') if line.startswith('- server1=') or line.startswith('- server2=') or line.startswith('- Y57单独')]
        lines += ['## Open Questions And Risks',
            '- 原完整9508表达/141场景、候选+Y36/W2/B1各两次/FIRST/bootstrap10000/seed1999/TRAIN cuts1487-3325/原subgroup harm与同卡成本协议保持；本轮无正式科学胜出/效益或全量成本。',
            '- Y81/Y82未做实际GPU任务，不得准入；已有完成模型后评等待原资源。单seed方差与server2历史掉卡/Xid79根因仍未知，训练递增不证明硬件修复。',
            '- 无新增项目常规操作授权缺口，不重复索权；Y57原epoch27/model/optimizer543/scheduler27单独暂停，需其单独明确继续授权。',
            '- 下轮先y80_deployment核三原运行并派发现有任务；Y78自然fixed30终态须按原source/YAML/receipt/freeze/目录/完整Train-Val/稳定finite检查与当前精确source链建立专用契约后显式GPU1归档；不可直接套Y77 helper。随后Y81原GPU1/390W native计划可在fresh空槽激活；Y82随后。',
            '- 不调用旧record_prepared_y80_y81_wait.py（假设Y77仍跑/旧HEAD），本轮record_y80_y82_closeout.py为事件收尾且不可重放；未来按当前运行另记新的只读进度/资源复核。heartbeat原owner/频率/通知保持，不建线程/新自动化/嵌套codex exec。',
            '## Last Tick',
            '- '+token+' begin/design先查并尝试派发waiting0；Y77 exact PID退出、fixed30/20守卫/稳定finite全部checkpoint验证后canonical归档，Y80原native两步PASS/17守卫/测试源码选择/原完整后评freeze/准入/派发并核native递增。',
            '- 本轮新增Y82独立结构/CPU与原数据加载/两合成优化步/原生回调/版本保留/双GPU原native计划完成；新候选1/准入1/启动1/归档1，无FIRST/cost。准备待跑Y81/Y82，三授权卡忙。',
            '- handoff：'+e(ref(out))+'；tick_attempt绑定Y82本轮真实CPU检查。全部科学goal/原运行receipt/数据协议/禁用项/Y57单独暂停保持。',
            '- 官方report/finish后只读核FINALIZED/PASS/context_committed/released及无lock/pending/archive_pending；本记忆不预签postflight，释放后不写。']
        assert len(lines)<120 and sum(line.startswith('## ') for line in lines)==7
        memory.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
        notes=ROOT/('pipeline/checks/foreground_notes_'+token+'.json');assert not notes.exists()
        tick.save(notes,dict(actions=[
            '完成：Y77完整fixed30/20生命周期守卫/稳定finite last30和best21及模型优化器调度器核验后canonical显式GPU0归档；原产物保留。训练best不替代FIRST。',
            '运行：Y80原native YAML/TRAIN两步loss4.615510940551758/6.330480575561523和实际路径/有限梯度/真实更新通过，17守卫通过；七测试A文件选择和安全增量部署后server2GPU0/280W PID42226启动并核计数递增。Y78/Y79原运行/source/功耗保持。',
            '新准备：Y82独立粗细query乘积残差CPU原model/backbone/TRAIN与零/RNG/标量/七阶段锚点/原路径/两优化步首次通过；七测试文件版本保留，12原生回调与实际CPU回调一致、11拒绝守卫、双GPU原native计划完成。无Y82 GPU PASS或准入。',
            '新候选1/准入1/启动1/归档1，无正式FIRST/cost。Y81优先Y82等真实native GPU空槽，三卡fresh busy+reserved；七节记忆及备份保存。'],
            decisions=['保持原数据/私有环境/精确mapping/科学goal与原receipt/全部禁用/seed1999/fixed30和350W-280W-390W，不改调度/跨项目/使用server6。',
                '保留21个完成模型原冻结完整后评与全部checkpoint，不重训、不从训练best晋升。Y81/Y82不以CPU替代原生GPU任务；有fresh授权独占空槽后最近可执行native检查再准入派发。',
                'Y57原epoch27单独暂停，不因GPU1资源授权恢复。无新增用户操作事项。']))
        assert qref==ref('pipeline/queue.json') and rref==ref('pipeline/running.json')
        print(json.dumps(dict(handoff=ref(out),notes=str(notes.relative_to(ROOT)),memory_lines=len(lines),new_candidates=1,admissions=1,launches=1,archives=1)))
    finally:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);a=p.parse_args();record(a.token)
