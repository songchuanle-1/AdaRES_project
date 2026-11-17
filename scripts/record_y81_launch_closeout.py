"""Record this exact Y78 terminal/Y81 launch with unchanged Y82 preparation."""
import argparse
import ast
from datetime import datetime,timedelta,timezone
import json
import subprocess
import sys
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from check_ef_execution import validate_frozen
import closure
import gpu_occupation


def record(token):
    lease(token,180);lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:
        work=tick.load(ROOT/'pipeline/research_work.json');c=next(c for c in work['candidates'] if c['id']=='AdaRES-Y78')
        assert c['stage']=='validation' and c['required_server']=='server-2' and c['required_gpu']==1
        assert monitor.load_experiment(str(ROOT),c['id'])['status']=='done_complete'
        closure.artifact(ROOT,c['terminal_evidence']);closure.artifact(ROOT,c['terminal_archive_binding'])
        assert monitor.load_running(str(ROOT))['server-2']['1']=='AdaRES-Y81'
        c.update(stage='blocked',blocked_from_stage='validation',blocker_kind='exclusive_gpu_slot',
            blocker='Original Y78 fixed30 and finite stable last/best/model/optimizer/scheduler were verified and canonically archived. Original complete frozen FIRST/cost evaluation requires server-2 GPU1/390W now owned by verified independent Y81. Preserve artifacts and original training.',
            retry_after=(datetime.now().astimezone()+timedelta(hours=1)).isoformat(),
            unblock_condition='Fresh exclusive original authorized server-2 GPU1/390W with no compute or canonical reservation; execute original full frozen protocol.')
        tick.save(ROOT/'pipeline/research_work.json',work)
    finally:lock.close()
    for name in ('server-2','server-1'):
        subprocess.run([sys.executable,'-B','C:/Users/chuanle song/.codex/skills/songcl_research_pipeline/gpu_occupation.py',
            'AdaRES_project','--token',token,'--server',name,'--label','closeout'],cwd=ROOT,check=True,capture_output=True,timeout=45)
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:
        work=tick.load(ROOT/'pipeline/research_work.json');get=lambda cid:next(c for c in work['candidates'] if c['id']==cid)
        view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json');context=tick.load(ROOT/'pipeline/headless/.tick_context.json')
        queue=monitor.load_queue(str(ROOT));running=monitor.load_running(str(ROOT));qref,rref=ref('pipeline/queue.json'),ref('pipeline/running.json')
        tasks={t['id']:t for rows in queue['gpu_queues'].values() for t in rows}
        assert closure.git(ROOT,'rev-parse','HEAD')=='4157937d0c3cf344a44394ce15c95349aa94d50f'
        assert ref('pipeline/goal.json')['sha256']=='4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a'
        paused=tasks['AdaRES-Y57'];assert paused['status']=='paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']
        observations={};phases={}
        for cid,name,gpu in [('AdaRES-Y79','server-1','0'),('AdaRES-Y80','server-2','0'),('AdaRES-Y81','server-2','1')]:
            task=tasks[cid];assert task['status']=='running' and task['launch_verified'] and running[name][gpu]==cid
            closure.admission(ROOT,task);evidence=get(cid)['latest_training_observation'];value=tick.load(closure.artifact(ROOT,evidence))
            assert value['token']==token and value['native_progress_confirmed'] and value['pid']==task['pid']
            observations[cid],phases[cid]=evidence,value['after']['latest']
        done=get('AdaRES-Y78');terminal=tick.load(closure.artifact(ROOT,done['terminal_evidence']));binding=tick.load(closure.artifact(ROOT,done['terminal_archive_binding']))
        assert terminal['token']==token and terminal['completed_epochs']=={p:list(range(1,31)) for p in ('Train','Val')} and terminal['CPU_guard_tests_passed']==22
        assert binding['only_exact_Y78_archived'] and binding['checkpoint_files_retained'] and not binding['full_frozen_evaluation_executed']
        admitted=get('AdaRES-Y81');smoke=tick.load(closure.artifact(ROOT,admitted['native_smoke']));freeze=tick.load(closure.artifact(ROOT,admitted['post_training_freeze']))
        assert smoke['server']=='server-2' and smoke['physical_gpu']==1 and smoke['steps']==2 and smoke['native_data_loaded'] and smoke['backward_finite']
        assert freeze['resource_condition']['gpu']==1 and freeze['resource_condition']['power_limit_watts']==390.
        preparation=tick.load(closure.artifact(ROOT,admitted['GPU1_lifecycle_preparation']));assert preparation['guard_tests_passed']==7
        occupations={}
        for name in ('server-2','server-1'):
            evidence=ref('pipeline/checks/gpu_occupation_'+name+'_closeout_'+token+'.json');value=tick.load(closure.artifact(ROOT,evidence))
            assert value['token']==token and (datetime.now(timezone.utc)-datetime.fromisoformat(value['observed_at'])).total_seconds()<300
            c=next(c for c in work['candidates'] if c.get('stage')=='blocked' and c.get('blocker_kind')=='exclusive_gpu_slot' and c.get('required_server')==name)
            server=next(s for s in context['queue']['servers'] if s['name']==name)
            assert not gpu_occupation.validated_state(ROOT,context,c,server,queue)['idle_gpu_ids'];occupations[name]=evidence
        waiting=get('AdaRES-Y82');assert waiting['stage']=='blocked' and waiting['blocker_kind']=='exclusive_gpu_slot' and waiting['implementation_test_passed']
        for field in ('source_manifest','native_preparation','CPU_native_observer_verification'):closure.artifact(ROOT,waiting[field])
        plans={}
        for field in ('native_smoke_plan_server2_gpu0','native_smoke_plan_server2_gpu1'):
            evidence=waiting[field];plan=tick.load(closure.artifact(ROOT,evidence));program=closure.artifact(ROOT,plan['program'])
            frozen=ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
            assert validate_frozen(waiting,plan,frozen)==7 and not plan['native_executed'];plans[field]=evidence
        out=ROOT/('pipeline/checks/y81_launch_y82_wait_'+token+'.json');assert not out.exists()
        tick.save(out,dict(complete=True,token=token,snapshot=view['generated_at'],original_live_training=observations,
            Y78_terminal=done['terminal_evidence'],Y78_archive=done['terminal_archive_binding'],Y81_native=admitted['native_smoke'],
            Y81_receipt=ref('pipeline/admissions/AdaRES-Y81.json'),Y81_freeze=admitted['post_training_freeze'],Y81_lifecycle=admitted['GPU1_lifecycle_preparation'],
            retained_Y82_plans=plans,occupations=occupations,new_candidates=0,new_admissions=1,new_launches=1,new_archives=1,
            formal_metrics_updated=False,Y57_pause_preserved=True,helper=ref(__file__)))
        for cid in ('AdaRES-Y78','AdaRES-Y79','AdaRES-Y80','AdaRES-Y81','AdaRES-Y82'):get(cid)['evidence'].append(ref(out))
        work['tick_attempt']=dict(tick_started_at=view['generated_at'],candidate_id='AdaRES-Y81',kind='native_smoke',result='passed',command=smoke['command'],
            observation='Actual original native YAML/TRAIN two-step masks and current refined-query feedback, static-base/current-query/padding identities, prediction/hard-gate/shared-head consumers and finite new/original/input gradients/updates passed. Original complete protocol frozen; admitted and launched with increasing native steps. Efficacy pending.',
            evidence=[admitted['native_smoke'],admitted['post_training_freeze'],ref(out)])
        tick.save(ROOT/'pipeline/research_work.json',work)
        memory=ROOT/'pipeline/codex_memory.md';backup=ROOT/('pipeline/snapshots/codex_memory_before_'+token+'.md');assert not backup.exists();backup.write_bytes(memory.read_bytes())
        def e(v):return v['path']+' SHA'+v['sha256']
        lines=memory.read_text(encoding='utf-8').splitlines();lines=lines[:lines.index('## Last Tick')]
        for i,line in enumerate(lines):
            if line.startswith('- 当前token '):lines[i]='- 当前token '+token+'，snapshot'+view['generated_at']+'，deadline'+view['deadline_at']+'不延长；postflight只认同token官方收据。'
            elif line.startswith('- 实际CODEX_THREAD_ID=owner='):lines[i]='- 实际CODEX_THREAD_ID=owner=01a07e32-78bf-7670-925b-370911ea48db已核；Pipeline-AdaRES_project Disabled；experiment_first enabled，admission2/creation2。本轮新候选0/准入1/启动1/归档1，无正式FIRST/cost。'
            elif line.startswith('- 前轮完整七节记忆精确备份：'):lines[i]='- 前轮完整七节记忆精确备份：'+e(ref(backup))+'；历史科学/禁用/专用恢复与全部原证据沿备份保持。'
            elif line.startswith('- Y78专用终态准备：'):lines[i]='- Y78本轮专用finalize_y78_fixed30_v2.py核22生命周期守卫、原source/YAML/receipt/freeze/完整Train-Val1..30和稳定finite last30/best21/model/optimizer544/scheduler30-21，canonical显式GPU1 done_complete归档。原目录20261102_000333产物保留；训练best .5039719343185425不是FIRST。terminal：'+e(done['terminal_evidence'])+'；archive：'+e(done['terminal_archive_binding'])+'。不要重放prepare/终态或重训，完整后评仍原GPU1/390W。'
            elif line.startswith('- Y81=adares_'):lines[i]='- Y81=adares_y81_query_mask_refinement；独立W2六次mask_base+current refined-query-set attention残差，零W65536/no RNG/总164695173；原static base逐次重用不累积，初始prediction/全输入/2D/backbone/原hard gates-SCG-k-gamma/sampling/七共享confidence头targets-loss-selector/seed1999/fixed30/native YAML保持。此前CPU准备不重放，本轮实际原native GPU1/390W两步PASS。'
            elif line.startswith('- Y81 version：'):lines[i]='- Y81原native实测loss4.615511417388916/6.37349796295166、四原场景16表达、12refinement/prediction/original coarse-gate消费者与14shared-head/query_norm、float64参考/首零次响应/有限新-原-输入梯度与真实更新通过，峰reserved7067402240B。7预准入资源/source-chain守卫与17实际native准入守卫通过；两批不证明效益或全量显存上界。'
            elif line.startswith('- Y81优先Y82；'):lines[i]='- Y81已在GPU1独立训练；剩Y82已CPU/model/data/version/native-plan/callback准备、blocked/validation/exclusive_gpu_slot/server2GPU0，GPU1原计划备选，无server1计划。无需重放CPU，fresh匹配独占idle/no compute/no reservation/原功耗后原native两步PASS才能原完整后评freeze/receipt/精确source契约/安全派发和步数递增。'
            elif line.startswith('- Y78归档准备v1'):lines[i]=line.replace('，尚未执行归档；','，当时尚未执行归档；')+' 本轮仅v2完整真实归档成功，原v1未运行归档。'
            elif line.startswith('- AdaRES-Y') and ' server' in line:
                cid=line.split()[1];cid='AdaRES-Y81' if cid=='AdaRES-Y78' else cid
                if cid in observations:
                    t=tasks[cid];v=phases[cid];lines[i]='- '+cid+' '+t['server']+' GPU'+str(t['gpu'])+' PID'+str(t['pid'])+' source/YAML/receipt/marker/计算归属/原功耗和native递增核验通过：'+v['phase']+str(v['epoch'])+'/30 step'+str(v['step'])+'/'+str(v['count'])+'；'+e(observations[cid])+'。'
            elif line.startswith('- 主master/server2 marker'):lines[i]='- 主master/server2 marker4157937d0c3cf344a44394ce15c95349aa94d50f，server1仍e0c7dba199fcf1a009941450d04aeb799418a7e6；仅选择七已测试Y81 A文件，server2从4e68安全ff追加七文件，原Y79/Y80源码保持。现用scripts/y81_deployment.py observe --id AdaRES-Y79|AdaRES-Y80|AdaRES-Y81 --token NEW_TOKEN --label progress；旧y80及更早observer不支持新HEAD。'
            elif line.startswith('- 21项Y56/'):lines[i]=line.replace('21项Y56/Y58..Y77','22项Y56/Y58..Y78')
            elif line.startswith('- fresh occupation server2：'):lines[i]='- fresh occupation server2：'+e(occupations['server-2'])+'；server1：'+e(occupations['server-1'])+'；三卡busy+reserved。finish证据须<300秒，超时新label；未知/失败/缺设备/idle不能证明hold。'
            elif line.startswith('- Y81/Y82未做实际GPU'):lines[i]='- Y82未实际GPU任务，不得准入；22个完成模型完整后评仍需原冻结资源。单seed方差/server2历史掉卡Xid79根因仍未知，训练递增不证明硬件修复。'
            elif line.startswith('- 下轮先y80_deployment'):lines[i]='- 下轮先y81_deployment核三原运行并派发现有任务；Y82最近原native检查等待fresh授权独占匹配空槽。Y79后续自然fixed30终态须建立其原PID/source/YAML/receipt/freeze/原目录/完整Train-Val和稳定finite检查的专用契约再显式server1GPU0归档，不直接套其他ID helper。'
            elif line.startswith('- 不调用旧record_prepared'):lines[i]='- record_prepared_y81_y82_wait.py假设Y78仍训练/旧HEAD，本轮后不再适用；record_y81_launch_closeout.py为当前事件收尾，不重放。下一轮另以当前token保存真实资源/进度复核。保留当前heartbeat owner/频率/通知，不建线程/新自动化/嵌套codex exec。'
        lines.insert(lines.index('## Closed Or Failed Directions'),'- Y81本轮receipt：'+e(ref('pipeline/admissions/AdaRES-Y81.json'))+'；native smoke：'+e(admitted['native_smoke'])+'；原完整后评freeze：'+e(admitted['post_training_freeze'])+'。原Y64完整evaluation/training深复制核等；commit4157937d0c3cf344a44394ce15c95349aa94d50f精确七A文件选择，旧receipt和科学goal字节不改。')
        lines += ['## Last Tick',
            '- '+token+' begin/design先查并派发waiting0；Y78原PID完成后绑定v2/22守卫及原freeze-dependencies完成完整稳定fixed30核验和canonical显式GPU1归档，未影响其他任务/Y57暂停。',
            '- Y81本轮7资源/source生命周期守卫和实际原native GPU1/390W两步/17准入守卫通过；精确测试版本选择/原完整后评freeze/receipt/入队/安全派发，并确认native递增。新候选0/准入1/启动1/归档1；剩Y82原准备完整，未重放CPU，无正式FIRST/cost。',
            '- handoff：'+e(ref(out))+'；当前HEAD/server2 marker415793、server1仍e0c。Y79/Y80原训练保持，Y81接原GPU1独立训练，Y78全部原产物保留。',
            '- 后续先核三原运行/现有队列；空槽时Y82原native PASS后完整冻结后评/准入/派发。Y57继续单独暂停，所有原数据/mapping/私有环境/禁用/seed1999/fixed30与350W-280W-390W保持。',
            '- 官方report/finish后仅只读核FINALIZED/PASS/context_committed/released及无lock/pending/archive_pending；此记忆不预签postflight，释放后不写。']
        assert len(lines)<120 and sum(line.startswith('## ') for line in lines)==7;memory.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
        notes=ROOT/('pipeline/checks/foreground_notes_'+token+'.json');assert not notes.exists()
        tick.save(notes,dict(actions=[
            '完成：Y78专用v2/22生命周期守卫和完整fixed30日志/稳定finite last30-best21/模型优化器调度器核验后canonical显式GPU1归档；原产物保留，训练best不是FIRST。',
            '运行：Y81原native YAML/TRAIN两步loss4.615511417388916/6.37349796295166，真实refinement/原static base和query/padding/预测与hard-gate/shared-head消费/finite梯度与更新通过。7资源/source守卫和17实际native准入守卫通过，原完整后评freeze与receipt后安全派发，并核native递增；原Y79/Y80运行保持。',
            '本轮新候选0/准入1/启动1/归档1，无正式FIRST/cost。Y82原CPU/native计划完整性复核，不重放；三卡fresh busy+reserved，记忆及备份保存。'],
            decisions=['保持原完整后评/科学goal/receipt/精确原数据与mapping/私有环境/禁用项/seed1999-fixed30和350W-280W-390W，禁止跨项目/server6/调度变更。',
                '优先最近可执行Y82原native任务；有匹配授权fresh独占空槽才执行，CPU不得替代GPU PASS。完成模型只按原完整冻结协议复评，不重训、不从训练best晋升。',
                'Y57原epoch27单独暂停，GPU1通用授权不恢复它；无新增用户操作事项。']))
        assert qref==ref('pipeline/queue.json') and rref==ref('pipeline/running.json')
        print(json.dumps(dict(handoff=ref(out),notes=str(notes.relative_to(ROOT)),memory_lines=len(lines),new_candidates=0,admissions=1,launches=1,archives=1)))
    finally:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);record(p.parse_args().token)
