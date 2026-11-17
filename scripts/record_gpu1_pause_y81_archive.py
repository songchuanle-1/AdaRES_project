"""Record actual CPU-only Y81 archival and current two-card resource wait."""
import argparse
import ast
from datetime import datetime, timedelta, timezone
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
import closure
import gpu_occupation


def record(token):
    lease(token, 180)
    writer = monitor.acquire_lock(str(ROOT)); assert writer is not None
    try:
        view = tick.load(ROOT/'pipeline/headless/.tick_model_view.json')
        context = tick.load(ROOT/'pipeline/headless/.tick_context.json')
        work = tick.load(ROOT/'pipeline/research_work.json')
        get = lambda cid: next(c for c in work['candidates'] if c['id']==cid)
        queue = monitor.load_queue(str(ROOT)); running = monitor.load_running(str(ROOT))
        qref, rref = ref('pipeline/queue.json'), ref('pipeline/running.json')
        tasks = {t['id']:t for rows in queue['gpu_queues'].values() for t in rows}
        assert set(tasks)=={'AdaRES-Y57','AdaRES-Y82','AdaRES-Y83'}
        assert running=={'server-2':{'0':'AdaRES-Y83'},'server-1':{'0':'AdaRES-Y82'}}
        assert tasks['AdaRES-Y57']['status']=='paused' and tasks['AdaRES-Y57']['resume_requires_user_authorization']
        assert closure.git(ROOT,'rev-parse','HEAD')=='2818322e6049b2541f1590a4fa9ad6854b3c598d'
        assert ref('pipeline/goal.json')['sha256']=='4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a'
        authority=ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')
        assert authority['sha256']=='3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45'
        grant=tick.load(closure.artifact(ROOT,authority))
        assert grant['resource_use_paused'] and grant['resource_resume_requires_user_authorization']
        servers=monitor.load_servers(str(ROOT))
        assert servers['server-1']['allowed_gpus']==servers['server-2']['allowed_gpus']==[0]
        assert servers['server-6']['allowed_gpus']==[]
        terminal=get('AdaRES-Y81')
        proof=tick.load(closure.artifact(ROOT,terminal['terminal_evidence']))
        binding=tick.load(closure.artifact(ROOT,terminal['terminal_archive_binding']))
        assert proof['token']==token and proof['CUDA_initialized'] is False
        assert proof['completed_epochs']=={p:list(range(1,31)) for p in ('Train','Val')}
        assert proof['checkpoints'][0]['epoch']==30 and proof['checkpoints'][1]['epoch']==21
        assert all(c['model_finite'] and c['optimizer_finite'] and c['optimizer_state_count']==544 for c in proof['checkpoints'])
        assert binding['only_exact_Y81_archived'] and binding['GPU1_use_paused'] and binding['CPU_only_terminal_verification']
        assert binding['checkpoint_files_retained'] and not binding['remote_mutations']
        experiment=monitor.load_experiment(str(ROOT),'AdaRES-Y81')
        assert experiment['status']=='done_complete' and str(experiment['gpu'])=='1'
        assert terminal['stage']=='blocked' and terminal['required_gpu']==1 and terminal['blocker_kind']=='human_authorization'
        assert not proof['full_frozen_evaluation_executed']
        observations, phases = {}, {}
        for cid,name,pid in [('AdaRES-Y82','server-1',1121817),('AdaRES-Y83','server-2',20849)]:
            task=tasks[cid]
            assert task['status']=='running' and task['launch_verified'] and task['pid']==pid
            assert task['server']==name and str(task['gpu'])=='0'
            closure.admission(ROOT,task)
            evidence=get(cid)['latest_training_observation']
            value=tick.load(closure.artifact(ROOT,evidence))
            assert value['token']==token and value['native_progress_confirmed'] and value['pid']==pid
            observations[cid],phases[cid]=evidence,value['after']['latest']
        occupations={}
        for name in ('server-2','server-1'):
            evidence=ref('pipeline/checks/gpu_occupation_'+name+'_finish_'+token+'.json')
            value=tick.load(closure.artifact(ROOT,evidence))
            assert value['token']==token
            assert 0<=(datetime.now(timezone.utc)-datetime.fromisoformat(value['observed_at'])).total_seconds()<300
            candidate=next(c for c in work['candidates'] if c.get('stage')=='blocked' and c.get('blocker_kind')=='exclusive_gpu_slot' and c.get('required_server')==name)
            server=next(s for s in context['queue']['servers'] if s['name']==name)
            state=gpu_occupation.validated_state(ROOT,context,candidate,server,queue)
            assert not state['idle_gpu_ids'] and state['busy_gpu_ids']==[0]
            occupations[name]=evidence
        plans={}
        for cid in ('AdaRES-Y84','AdaRES-Y85'):
            c=get(cid)
            assert c['stage']=='blocked' and c['blocker_kind']=='exclusive_gpu_slot' and c['required_gpu']==0
            assert c['implementation_test_passed'] and not c.get('admission_receipt')
            for field in ('source_manifest','native_preparation','CPU_native_observer_verification'):
                closure.artifact(ROOT,c[field])
            evidence=c['native_smoke_plan_server2_gpu0']
            plan=tick.load(closure.artifact(ROOT,evidence))
            program=closure.artifact(ROOT,plan['program'])
            frozen=ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
            assert validate_frozen(c,plan,frozen)==7 and not plan['native_executed']
            closure.artifact(ROOT,c['native_smoke_plan_server2_gpu1'])
            plans[cid]=dict(active_GPU0_plan=evidence,retained_but_not_authorized_GPU1_plan=c['native_smoke_plan_server2_gpu1'])
            c.update(resource_pause=authority,
                blocker='Only server-2 GPU0 is currently permitted for this exact native plan and is occupied by Y83; server-1 GPU0 runs Y82. GPU1 is explicitly user-paused.',
                unblock_condition='Fresh exclusive idle/unreserved authorized server-2 GPU0 at280W, exact original runtime/data/source/YAML and actual two-step native PASS; do not use GPU1.')
            if 'GPU1 paused' not in c['next_action']:
                c['next_action']+=' Latest resource authority: GPU1 paused until explicit user reauthorization; retain but never execute its old GPU1 plan. Y84 precedes Y85; no CPU preparation replay.'
        paused_evaluations=[]
        for c in work['candidates']:
            if c.get('stage')!='blocked' or c.get('required_server')!='server-2' or str(c.get('required_gpu'))!='1':
                continue
            assert c.get('post_training_freeze')
            closure.artifact(ROOT,c['post_training_freeze'])
            c.update(blocker_kind='human_authorization',resource_pause=authority,
                blocker='The user paused all AdaRES server-2 GPU1 use on 2026-11-04; the original GPU1/390W complete frozen evaluation/cost remains preserved and unexecuted.',
                retry_after=(datetime.now(timezone.utc)+timedelta(hours=4)).isoformat(),
                unblock_condition='Explicit user GPU1 reauthorization and fresh exclusive original server-2 GPU1/390W ownership; no migration, retraining or protocol substitution.',
                next_action='Preserve original complete fixed30 products, signed receipt and full frozen GPU1/390W FIRST/evaluation/cost. Wait for explicit user resource reauthorization; do not use another card or restart training.')
            if authority not in c['evidence']:c['evidence'].append(authority)
            paused_evaluations.append(c['id'])
        repair=ROOT/('pipeline/checks/y81_terminal_helper_repair_'+token+'.json')
        assert not repair.exists()
        tick.save(repair,dict(complete=True,token=token,scope='terminal helper identity metadata only',
            first_attempt_command='python -B scripts/prepare_y81_terminal_handoff.py --token '+token,
            first_attempt_observation='Exact-config guard rejected copied query_visual_feedback literal before checkpoint discovery, tests, loading or archival. Correct query_mask_refinement package and config; original model/YAML unchanged.',
            retained_initial_helpers=[ref('pipeline/snapshots/initial_generation_73fa7b70_finalize_y81_fixed30.py'),ref('pipeline/snapshots/initial_generation_73fa7b70_prepare_y81_terminal_handoff.py')],
            corrected_tests_passed=25,verified_terminal=terminal['terminal_evidence'],scientific_model_modified=False))
        out=ROOT/('pipeline/checks/gpu1_pause_y81_archive_'+token+'.json');assert not out.exists()
        tick.save(out,dict(complete=True,token=token,snapshot=view['generated_at'],
            Y81_terminal=terminal['terminal_evidence'],Y81_archive=terminal['terminal_archive_binding'],
            CPU_lifecycle_guards=25,terminal_helper_metadata_repair=ref(repair),
            resource_authority=authority,GPU1_pause_preserved=True,GPU1_frozen_evaluation_holds=paused_evaluations,
            occupations=occupations,original_live_training=observations,retained_prepared_plans=plans,
            new_candidates=0,new_admissions=0,new_launches=0,new_archives=1,
            native_GPU_tasks_executed=False,formal_metrics_updated=False,Y57_pause_preserved=True,helper=ref(__file__)))
        for cid in ('AdaRES-Y81','AdaRES-Y82','AdaRES-Y83','AdaRES-Y84','AdaRES-Y85'):
            get(cid)['evidence'].append(ref(out))
        get('AdaRES-Y81')['evidence'].append(ref(repair))
        work['tick_attempt']=dict(tick_started_at=view['generated_at'],candidate_id='AdaRES-Y84',kind='resource_check',result='blocked',
            command='python -B scripts/record_gpu1_pause_y81_archive.py --token '+token,
            observation='Fresh physical inventory and canonical reservations confirm both authorized GPU0 slots occupied by original Y82/Y83; GPU1 use explicitly paused. Y84/Y85 original tested source/native plans retained; no native GPU task or admission. Actual CPU-only Y81 fixed30 completion verified and archived separately.',
            evidence=[ref(out),occupations['server-2'],authority])
        tick.save(ROOT/'pipeline/research_work.json',work)
        memory=ROOT/'pipeline/codex_memory.md';backup=ROOT/('pipeline/snapshots/codex_memory_before_'+token+'.md')
        assert not backup.exists();backup.write_bytes(memory.read_bytes())
        def e(v):return v['path']+' SHA'+v['sha256']
        lines=memory.read_text(encoding='utf-8').splitlines();lines=lines[:lines.index('## Last Tick')]
        for i,line in enumerate(lines):
            if line.startswith('- 当前token '):lines[i]='- 当前token '+token+'，snapshot'+view['generated_at']+'，deadline'+view['deadline_at']+'不延长；postflight只认同token官方收据。'
            elif line.startswith('- 实际CODEX_THREAD_ID=owner='):lines[i]='- 实际CODEX_THREAD_ID=owner=01a07e32-78bf-7670-925b-370911ea48db已核；Pipeline-AdaRES_project Disabled；experiment_first enabled。本轮新候选/准入/启动0，Y81原fixed30 CPU核验归档1；无正式FIRST/cost。'
            elif line.startswith('- 仅server1GPU0既有'):lines[i]='- 仅server1GPU0既有350W、server2GPU0保持280W；GPU1全部项目使用由用户暂停至明确重新授权。每卡一独立实验；Y57单独暂停，禁server6/跨项目/改调度/硬件功耗/驱逐/重启迁移。'
            elif line.startswith('- goal科学字节SHA'):lines[i]='- goal科学字节SHA4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a保持；最新AGENTS/SERVERS与'+e(authority)+'覆盖旧GPU1许可；原receipt/科学协议不改。'
            elif line.startswith('- 前轮完整七节记忆精确备份：'):lines[i]='- 前轮完整七节记忆精确备份：'+e(ref(backup))+'；历史科学/禁用/专用恢复和原证据保持。'
            elif line.startswith('- AdaRES-Y81 server-2'):lines[i]='- Y81本轮原20261102_220631稳定finite last30/best21/model/optimizer544/scheduler30-21、完整Train-Val1..30及原source/YAML/receipt/freeze实核；canonical显式原GPU1 done_complete归档，原产物保留，CPU-only/CUDA未初始化，不使用GPU1。训练best .5078033804893494不是FIRST。terminal：'+e(terminal['terminal_evidence'])+'；archive：'+e(terminal['terminal_archive_binding'])+'。'
            elif line.startswith('- 2026-11-04手动核实GPU1'):lines[i]='- GPU1用户暂停保持；Y81现已完成精确CPU终态校验并归档，无活跃Y81映射。原GPU1冻结后评/成本继续human_authorization挂起，不迁移/不重训；外部任务不干预。'
            elif line.startswith('- AdaRES-Y') and ' server' in line:
                cid=line.split()[1]
                if cid in phases:
                    v=phases[cid];t=tasks[cid];lines[i]='- '+cid+' '+t['server']+' GPU0 PID'+str(t['pid'])+' 原source/YAML/receipt/marker/compute归属/功耗及native递增实核：'+v['phase']+str(v['epoch'])+'/30 step'+str(v['step'])+'/'+str(v['count'])+'；'+e(observations[cid])+'。'
            elif line.startswith('- 24项Y56/Y58..Y80'):lines[i]='- 25项Y56/Y58..Y81 fixed30完成，仍待各自原完整冻结FIRST/cost；原required server/GPU/功耗保持，不重训、不用训练best替代FIRST；GPU1后评暂停。'
            elif line.startswith('- fresh occupation server2：'):lines[i]='- fresh occupation server2GPU0：'+e(occupations['server-2'])+'；server1GPU0：'+e(occupations['server-1'])+'；两个授权GPU0 busy+reserved。finish须<300秒；GPU1物理idle不构成使用许可。'
            elif line.startswith('- 当前Y84优先Y85'):lines[i]='- 当前Y84优先Y85；CPU/model/原TRAIN/version/native计划保持，等原server2GPU0/280W独占空槽。旧GPU1备选计划仅保留不得执行，无server1计划；原native两步PASS后才准入/启动/核递增。'
            elif line.startswith('- Y84/Y85未实际GPU'):lines[i]='- Y84/Y85未实际GPU任务不得准入。25项完成模型仍待原完整后评；GPU1资源暂停不得换卡代替，训练递增不证明历史掉卡根因修复。'
            elif line.startswith('- 下轮先'):lines[i]='- 下轮先用y83_deployment核Y82/Y83，勿对已归档Y81使用live observer；Y84优先Y85等授权原GPU0 native检查。不重放Y81终态/CPU准备。'
            elif line.startswith('- 当前安静复核'):lines[i]='- 旧record_prepared_y84_y85_wait.py含三训练假设，已不适用。当前handoff='+e(ref(out))+'；后续安静收尾需新helper适配仅Y82/Y83和已完成Y81，勿重放本轮archive/repair记录。'
        status='；'.join(cid+' '+v['phase']+str(v['epoch'])+'/30 step'+str(v['step'])+'/'+str(v['count']) for cid,v in phases.items())
        lines+=['## Last Tick','- '+token+'：'+status+'，原两GPU0训练递增实核。',
            '- Y81完整fixed30 CPU-only校验、25生命周期守卫和显式原GPU1归档完成，原checkpoint/日志/receipt/freeze保持；训练best .5078033804893494不是FIRST。',
            '- 初次prepare因复制的错误package/config字面值被精确守卫拒绝，未加载checkpoint/未归档；原生成helper备份与修复记录保留，原科学模型未改。',
            '- GPU1最新暂停保持；6项原GPU1冻结后评记录为human_authorization，Y57单独暂停。两授权GPU0均busy+reserved，Y84/Y85等待，不重放原准备。',
            '- 本轮新候选/准入/启动0，归档1；handoff：'+e(ref(out))+'；无正式FIRST/cost。',
            '- 官方report/finish后只读核FINALIZED/PASS/context_committed/released及无owned lock/pending/archive_pending；不预签结果，释放后不写。']
        assert len(lines)<120 and sum(v.startswith('## ') for v in lines)==7
        memory.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
        notes=ROOT/('pipeline/checks/foreground_notes_'+token+'.json');assert not notes.exists()
        tick.save(notes,dict(actions=['运行：'+status+'，原source/YAML/receipt/marker/compute归属/功耗及native递增实核。',
            '完成：Y81完整Train-Val1..30、原20261102_220631稳定finite last30/best21/model/optimizer544/scheduler30-21、25生命周期守卫通过，CPU-only/CUDA未初始化，canonical显式GPU1归档；训练best .5078033804893494不是FIRST。',
            'Y81终态helper初次错误config被守卫拒绝，无终态副作用；修复精确package/config，原生成版本及记录保留。',
            '新准备：本轮无新增候选；Y84/Y85原准备保持，两授权GPU0均busy+reserved，原GPU1计划保留但禁执行。'],
            decisions=['遵守GPU1全部项目使用暂停，原GPU1/390W完整冻结后评继续挂起，不能换卡、重训、替换FIRST；Y57单独暂停。',
                'Y82/Y83继续，Y84优先Y85等原server2GPU0/280W独占空槽；原native两步PASS后才准入/派发/核递增。',
                '科学goal字节、原receipt/数据/私有运行环境/mapping/seed1999-fixed30/禁用和调度保持，无新增用户处理。']))
        assert qref==ref('pipeline/queue.json') and rref==ref('pipeline/running.json')
        print(json.dumps(dict(handoff=ref(out),notes=notes.relative_to(ROOT).as_posix(),memory_lines=len(lines),archives=1,new_admissions=0)))
    finally:writer.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);record(p.parse_args().token)
