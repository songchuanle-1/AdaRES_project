"""Record current protected training and Y82/Y83 holds; optionally new Y83 preparation."""
import argparse
import ast
from datetime import datetime,timezone
import json
import subprocess
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from check_ef_execution import validate_frozen
import closure
import gpu_occupation


def record(token,new_preparation):
    lease(token,180);lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:
        work=tick.load(ROOT/'pipeline/research_work.json');get=lambda cid:next(c for c in work['candidates'] if c['id']==cid)
        view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json');context=tick.load(ROOT/'pipeline/headless/.tick_context.json')
        queue=monitor.load_queue(str(ROOT));running=monitor.load_running(str(ROOT));qref,rref=ref('pipeline/queue.json'),ref('pipeline/running.json')
        assert closure.git(ROOT,'rev-parse','HEAD')=='4157937d0c3cf344a44394ce15c95349aa94d50f'
        assert ref('pipeline/goal.json')['sha256']=='4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a'
        tasks={t['id']:t for rows in queue['gpu_queues'].values() for t in rows};observations={};phases={}
        for cid,name,gpu,pid in [('AdaRES-Y79','server-1','0',2218231),('AdaRES-Y80','server-2','0',42226),('AdaRES-Y81','server-2','1',120624)]:
            task=tasks[cid];assert task['status']=='running' and task['launch_verified'] and task['pid']==pid and running[name][gpu]==cid
            closure.admission(ROOT,task);evidence=get(cid)['latest_training_observation'];value=tick.load(closure.artifact(ROOT,evidence))
            assert value['token']==token and value['native_progress_confirmed'] and value['pid']==pid
            observations[cid],phases[cid]=evidence,value['after']['latest']
        pause=tasks['AdaRES-Y57'];assert pause['status']=='paused' and pause['dispatch_hold'] and pause['resume_requires_user_authorization']
        for cid in ('AdaRES-Y77','AdaRES-Y78'):
            assert cid not in tasks and monitor.load_experiment(str(ROOT),cid)['status']=='done_complete'
            for field in ('terminal_evidence','terminal_archive_binding','post_training_freeze'):closure.artifact(ROOT,get(cid)[field])
        occupations={}
        for name in ('server-2','server-1'):
            evidence=ref('pipeline/checks/gpu_occupation_'+name+'_finish_'+token+'.json');value=tick.load(closure.artifact(ROOT,evidence))
            assert value['token']==token and (datetime.now(timezone.utc)-datetime.fromisoformat(value['observed_at'])).total_seconds()<300
            c=next(c for c in work['candidates'] if c.get('stage')=='blocked' and c.get('blocker_kind')=='exclusive_gpu_slot' and c.get('required_server')==name)
            server=next(s for s in context['queue']['servers'] if s['name']==name)
            assert not gpu_occupation.validated_state(ROOT,context,c,server,queue)['idle_gpu_ids'];occupations[name]=evidence
        retained={}
        for cid in ('AdaRES-Y82','AdaRES-Y83'):
            c=get(cid);assert c['stage']=='blocked' and c['blocker_kind']=='exclusive_gpu_slot' and c['implementation_test_passed']
            for field in ('source_manifest','native_preparation','CPU_native_observer_verification'):closure.artifact(ROOT,c[field])
            retained[cid]={}
            for field in ('native_smoke_plan_server2_gpu0','native_smoke_plan_server2_gpu1'):
                evidence=c[field];plan=tick.load(closure.artifact(ROOT,evidence));program=closure.artifact(ROOT,plan['program'])
                frozen=ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
                assert validate_frozen(c,plan,frozen)==7 and not plan['native_executed'];retained[cid][field]=evidence
        c=get('AdaRES-Y83');version=tick.load(closure.artifact(ROOT,c['source_manifest']));cpu=tick.load(closure.artifact(ROOT,version['CPU_result']))
        prep=tick.load(closure.artifact(ROOT,c['native_preparation']))
        assert cpu['implementation_test_passed'] and cpu['synthetic_CPU_decoder_optimizer_steps']==2 and cpu['additional_parameters']==2560 and cpu['parameters']==164632197
        assert cpu['aggregate_shared_norm_gradient_anchor_close'] and not cpu['CUDA_initialized'] and not cpu['native_smoke_passed']
        assert prep['guard_tests_passed']==11 and prep['exact_callback_functions_and_classes']==14 and prep['CPU_native_callbacks_actually_executed']
        if new_preparation:assert prep['token']==token and token in version['CPU_result']['path']
        out=ROOT/('pipeline/checks/prepared_y82_y83_resource_'+token+'.json');assert not out.exists()
        tick.save(out,dict(complete=True,token=token,snapshot=view['generated_at'],original_live_training=observations,
            occupations=occupations,retained_native_plans=retained,Y83_CPU=version['CPU_result'],Y83_version=c['source_manifest'],Y83_native_preparation=c['native_preparation'],
            new_Y83_preparation=new_preparation,native_GPU_executed=False,new_candidates=int(new_preparation),new_admissions=0,new_launches=0,new_archives=0,
            formal_metrics_updated=False,Y57_pause_preserved=True,helper=ref(__file__)))
        for cid in ('AdaRES-Y79','AdaRES-Y80','AdaRES-Y81','AdaRES-Y82','AdaRES-Y83'):get(cid)['evidence'].append(ref(out))
        selected=get('AdaRES-Y83' if new_preparation else 'AdaRES-Y82')
        evidence=[ref(out),c['source_manifest'],c['native_preparation'],version['CPU_result']] if new_preparation else [ref(out),occupations['server-2']]
        for item in evidence:
            if item not in selected['evidence']:selected['evidence'].append(item)
        command='python -B scripts/run_experiment_first_probe.py y83_cpu pipeline/checks/y83_cpu_'+token+'_program.py --token '+token+' --seconds 480' if new_preparation else 'python -B scripts/record_prepared_y82_y83_wait.py --token '+token
        work['tick_attempt']=dict(tick_started_at=view['generated_at'],candidate_id=selected['id'],kind='implementation_test' if new_preparation else 'resource_check',
            result='passed' if new_preparation else 'blocked',command=command,
            observation='Actual original model/backbone/TRAIN CPU collation, private affine LayerNorm controls, seven-stage W2/input-gradient/aggregate shared-gradient anchors, two finite synthetic decoder optimizer steps passed. Fourteen exact native callbacks match the actually executed CPU callbacks; eleven plan rejection guards passed; tested source retained. No native GPU PASS/admission/efficacy.' if new_preparation else 'Fresh physical inventory/compute and canonical reservations plus retained native source/YAML/goal/version hashes verified; all authorized slots busy. No native task or efficacy.',evidence=evidence)
        tick.save(ROOT/'pipeline/research_work.json',work)
        memory=ROOT/'pipeline/codex_memory.md';backup=ROOT/('pipeline/snapshots/codex_memory_before_'+token+'.md');assert not backup.exists();backup.write_bytes(memory.read_bytes())
        def e(v):return v['path']+' SHA'+v['sha256']
        lines=memory.read_text(encoding='utf-8').splitlines();lines=lines[:lines.index('## Last Tick')]
        for i,line in enumerate(lines):
            if line.startswith('- 当前token '):lines[i]='- 当前token '+token+'，snapshot'+view['generated_at']+'，deadline'+view['deadline_at']+'不延长；postflight只认同token官方收据。'
            elif line.startswith('- 实际CODEX_THREAD_ID=owner='):lines[i]='- 实际CODEX_THREAD_ID=owner=01a07e32-78bf-7670-925b-370911ea48db已核；Pipeline-AdaRES_project Disabled；experiment_first enabled，admission2/creation2。本轮新候选'+str(int(new_preparation))+'/准入0/启动0/归档0；'+('Y83独立结构CPU与原数据/精确版本/双原native计划准备完成。' if new_preparation else '仅当前资源/进度与原准备完整性复核。')+'无正式FIRST/cost。'
            elif line.startswith('- 前轮完整七节记忆精确备份：'):lines[i]='- 前轮完整七节记忆精确备份：'+e(ref(backup))+'；历史科学/禁用/专用恢复与全部原证据沿备份保持。'
            elif line.startswith('- Y78本轮专用'):lines[i]=line.replace('本轮','在5d338轮',1)
            elif line.startswith('- Y81=adares_'):lines[i]=line.replace('本轮实际原native','在5d338轮实际原native')
            elif line.startswith('- Y81本轮receipt：'):lines[i]=line.replace('本轮','在5d338轮',1)
            elif line.startswith('- Y81已在GPU1独立训练；'):lines[i]='- Y82优先Y83；两者均已CPU/model/data/version/native-plan/callback准备、blocked/validation/exclusive_gpu_slot/server2GPU0，GPU1原计划备选，无server1计划。不要重放已通过CPU/prepare；fresh匹配独占idle/no compute/no reservation/原功耗后原native两步PASS才能原完整后评freeze/receipt/精确source契约/安全派发和步数递增。'
            elif line.startswith('- AdaRES-Y') and ' server' in line:
                cid=line.split()[1]
                if cid in observations:
                    v=phases[cid];t=tasks[cid];lines[i]='- '+cid+' '+t['server']+' GPU'+str(t['gpu'])+' PID'+str(t['pid'])+' source/YAML/receipt/marker/计算归属/原功耗和native递增核验通过：'+v['phase']+str(v['epoch'])+'/30 step'+str(v['step'])+'/'+str(v['count'])+'；'+e(observations[cid])+'。'
            elif line.startswith('- fresh occupation server2：'):lines[i]='- fresh occupation server2：'+e(occupations['server-2'])+'；server1：'+e(occupations['server-1'])+'；三卡busy+reserved。finish证据须<300秒，超时新label；未知/失败/缺设备/idle不能证明hold。'
            elif line.startswith('- Y82未实际GPU'):lines[i]='- Y82/Y83未实际GPU任务，不得准入；22个完成模型完整后评仍需原冻结资源。单seed方差/server2历史掉卡Xid79根因仍未知，训练递增不证明硬件修复。'
            elif line.startswith('- 下轮先y81_deployment'):lines[i]='- 下轮先y81_deployment核三原运行并派发现有任务；Y82优先Y83原native任务等fresh授权独占匹配空槽。Y79后续自然fixed30终态须建立其原PID/source/YAML/receipt/freeze/原目录/完整Train-Val与稳定finite检查的专用契约再显式server1GPU0归档，不套其他ID helper。'
            elif line.startswith('- record_prepared_y81_y82_wait.py'):lines[i]='- 旧record_prepared_y81_y82_wait.py和record_y81_launch_closeout.py不重放；当前record_prepared_y82_y83_wait.py支持HEAD415793与Y79/Y80/Y81运行、Y82/Y83等待。未来安静复核用新token不带--new-preparation，不重放CPU/prepare。保留heartbeat owner/频率/通知，不建线程/新自动化/嵌套codex exec。'
        marker='- Y83=adares_y83_stage_query_norm；'
        if not any(line.startswith(marker) for line in lines):
            at=lines.index('## Closed Or Failed Directions')
            lines[at:at]=[marker+'独立W2，仅将六轮已消费的norm_qqa(query_self+query_qla)共享仿射拆为stage0原norm_qqa及五个无RNG深拷贝refinement_query_norms；+2560/总164632197。保留原LayerNorm epsilon/统计/全128query-allchannels、共享out_norm和confidence头、初始prediction/static mask/Boolean gates/sampling/SCG/k/gamma/geometry/全输入/2D/backbone/targets-loss-selector/seed1999/fixed30/native YAML。',
                '- Y83不同于Y67的prediction_head out_norm拆分、Y60独立confidence头、Y76 confidence stage偏置，也不重开Y26丢弃qla LayerNorm重连；原qla路径保持。既有stage FFN可补偿，无新信息/创新/效益承诺，无val扫描。',
                '- Y83 CPU原model/backbone/TRAIN162738点8表达、float32/64独立逐标量affine LayerNorm/disjoint参数/no RNG/私有干预复原/兄弟梯度隔离/置换与平移、七阶段W2逐位与输入梯度/共享norm梯度总和锚点、六原self-language-sum和FFN消费者/七共享head-query_norm/原hard gates及两合成CPU优化步通过。实际CPU：'+e(version['CPU_result'])+'。',
                '- Y83 version：'+e(c['source_manifest'])+'；commit2818322e6049b2541f1590a4fa9ad6854b3c598d，parentY82/577376，branch refs/heads/adares-y83-stage-query-norm；仅七测试A文件隔离index保留，主HEAD415793/index/三训练保持。',
                '- Y83双原native GPU计划：'+e(c['native_preparation'])+'；11拒绝守卫/陈旧goal-篡改源码拒绝/Python3.8 AST/14原native回调与实际CPU回调完全相同通过，无CUDA/native sparse TRAIN任务/GPU PASS/准入。']
        lines += ['## Last Tick',
            '- '+token+' begin/design先查并尝试派发waiting0；快照new_terminal:Y78为前轮已完整归档，不重放。三原Y79/Y80/Y81训练进度递增，fresh三卡busy+reserved，无新训练完成/故障。',
            '- 本轮新候选'+str(int(new_preparation))+'/准入0/启动0/归档0；'+('Y83独立stage-private query-fusion normalization实际CPU/原数据/两优化步/源版本保留/双native计划11守卫和14真实回调完成。' if new_preparation else '仅当前资源/进度与已通过原准备哈希复核，不重放CPU或prepare。')+'Y82优先Y83等真实原native GPU任务，无正式FIRST/cost。',
            '- handoff：'+e(ref(out))+'；tick_attempt绑定'+selected['id']+'本轮实际'+('CPU结构检查。' if new_preparation else '资源复核。')+'主HEAD/server2 marker415793、server1仍e0c；当前observer为y81_deployment。',
            '- 后续先核原运行/现有队列，fresh独占匹配空槽后Y82原native PASS→完整后评freeze/receipt/精确source契约/安全派发及native递增，Y83随后。Y57单独暂停，原数据/mapping/私有环境/禁用/seed1999-fixed30/350W-280W-390W/调度保持。',
            '- 官方report/finish后仅只读核FINALIZED/PASS/context_committed/released及无lock/pending/archive_pending；此记忆不预签结果，释放后不写。']
        assert len(lines)<120 and sum(line.startswith('## ') for line in lines)==7;memory.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
        notes=ROOT/('pipeline/checks/foreground_notes_'+token+'.json');assert not notes.exists()
        status='；'.join(cid+' '+v['phase']+str(v['epoch'])+'/30 step'+str(v['step'])+'/'+str(v['count']) for cid,v in phases.items())
        tick.save(notes,dict(actions=['运行：'+status+'，原PID/source/YAML/receipt/marker/功耗与native计数递增核验通过。',
            '新准备：Y83独立stage-private query-fusion LayerNorm仿射拆分，原native model/backbone/TRAIN CPU加载、标量/七阶段W2及输入-共享梯度总和锚点、私有干预复原/两优化步通过；七测试文件版本保留，双native计划11拒绝守卫与14实际回调一致检查通过，无GPU PASS/准入。' if new_preparation else '仅当前资源/进度与Y82/Y83原native准备完整性复核，不重放CPU/prepare。',
            '新候选'+str(int(new_preparation))+'/准入0/启动0/归档0，无正式FIRST/cost；三卡fresh busy+reserved，Y82优先Y83等待原native GPU检查，记忆与备份保存。'],
            decisions=['保留原Y79/Y80/Y81训练，不挤占GPU；原Y82准备不重放，完成准备目标后按资源条件继续。',
                '原native任务必须使用匹配计划和fresh授权独占空槽；实测PASS后才按原完整冻结后评/receipt/精确source契约准入派发并核步数，CPU不得代替GPU检查或效益。',
                '保留科学goal/原receipt/数据协议/私有环境/精确mapping/禁用/seed1999-fixed30与350W-280W-390W；Y57单独暂停，禁server6/跨项目/调度变更，无新增用户操作。']))
        assert qref==ref('pipeline/queue.json') and rref==ref('pipeline/running.json')
        print(json.dumps(dict(handoff=ref(out),notes=str(notes.relative_to(ROOT)),memory_lines=len(lines),new_candidates=int(new_preparation),admissions=0,launches=0,archives=0)))
    finally:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);p.add_argument('--new-preparation',action='store_true');a=p.parse_args();record(a.token,a.new_preparation)
