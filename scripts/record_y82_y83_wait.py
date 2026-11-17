"""Quiet resource recheck for unchanged Y82/Y83 and prepared Y84/Y85."""
import argparse
import ast
from datetime import datetime, timezone
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
import closure
import gpu_occupation


def record(token):
    lease(token,180)
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:
        view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json')
        context=tick.load(ROOT/'pipeline/headless/.tick_context.json')
        work=tick.load(ROOT/'pipeline/research_work.json')
        get=lambda cid:next(c for c in work['candidates'] if c['id']==cid)
        queue=monitor.load_queue(str(ROOT));running=monitor.load_running(str(ROOT))
        qref,rref=ref('pipeline/queue.json'),ref('pipeline/running.json')
        tasks={t['id']:t for rows in queue['gpu_queues'].values() for t in rows}
        assert set(tasks)=={'AdaRES-Y57','AdaRES-Y82','AdaRES-Y83'}
        assert running=={'server-2':{'0':'AdaRES-Y83'},'server-1':{'0':'AdaRES-Y82'}}
        assert tasks['AdaRES-Y57']['status']=='paused' and tasks['AdaRES-Y57']['dispatch_hold'] and tasks['AdaRES-Y57']['resume_requires_user_authorization']
        assert closure.git(ROOT,'rev-parse','HEAD')=='2818322e6049b2541f1590a4fa9ad6854b3c598d'
        assert ref('pipeline/goal.json')['sha256']=='4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a'
        authority=ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')
        assert authority['sha256']=='3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45'
        servers=monitor.load_servers(str(ROOT))
        assert servers['server-1']['allowed_gpus']==servers['server-2']['allowed_gpus']==[0] and servers['server-6']['allowed_gpus']==[]
        observations,phases={},{}
        for cid,name,pid in [('AdaRES-Y82','server-1',1121817),('AdaRES-Y83','server-2',20849)]:
            task=tasks[cid]
            assert task['status']=='running' and task['launch_verified'] and task['pid']==pid and task['server']==name and str(task['gpu'])=='0'
            closure.admission(ROOT,task)
            evidence=get(cid)['latest_training_observation'];value=tick.load(closure.artifact(ROOT,evidence))
            assert value['token']==token and value['pid']==pid and value['native_progress_confirmed']
            observations[cid],phases[cid]=evidence,value['after']['latest']
        occupations={}
        for name in ('server-2','server-1'):
            evidence=ref('pipeline/checks/gpu_occupation_'+name+'_finish_'+token+'.json')
            value=tick.load(closure.artifact(ROOT,evidence));assert value['token']==token
            assert 0<=(datetime.now(timezone.utc)-datetime.fromisoformat(value['observed_at'])).total_seconds()<300
            candidate=next(c for c in work['candidates'] if c.get('stage')=='blocked' and c.get('blocker_kind')=='exclusive_gpu_slot' and c.get('required_server')==name)
            server=next(s for s in context['queue']['servers'] if s['name']==name)
            state=gpu_occupation.validated_state(ROOT,context,candidate,server,queue)
            assert not state['idle_gpu_ids'] and state['busy_gpu_ids']==[0]
            occupations[name]=evidence
        plans={}
        for cid in ('AdaRES-Y84','AdaRES-Y85'):
            c=get(cid)
            assert c['stage']=='blocked' and c['blocker_kind']=='exclusive_gpu_slot' and c['required_server']=='server-2' and c['required_gpu']==0
            assert c['implementation_test_passed'] and not c.get('admission_receipt') and c['resource_pause']==authority
            assert authority in c['evidence']
            for field in ('source_manifest','native_preparation','CPU_native_observer_verification'):
                closure.artifact(ROOT,c[field])
            plan=tick.load(closure.artifact(ROOT,c['native_smoke_plan_server2_gpu0']))
            program=closure.artifact(ROOT,plan['program'])
            frozen=ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
            assert validate_frozen(c,plan,frozen)==7 and not plan['native_executed']
            closure.artifact(ROOT,c['native_smoke_plan_server2_gpu1'])
            plans[cid]=dict(active_GPU0_plan=c['native_smoke_plan_server2_gpu0'],retained_but_unauthorized_GPU1_plan=c['native_smoke_plan_server2_gpu1'])
        done=get('AdaRES-Y81')
        assert done['stage']=='blocked' and done['blocker_kind']=='human_authorization' and done['required_gpu']==1
        proof=tick.load(closure.artifact(ROOT,done['terminal_evidence']))
        binding=tick.load(closure.artifact(ROOT,done['terminal_archive_binding']))
        assert proof['completed_epochs']=={p:list(range(1,31)) for p in ('Train','Val')}
        assert binding['only_exact_Y81_archived'] and binding['checkpoint_files_retained'] and binding['GPU1_use_paused']
        assert not proof['full_frozen_evaluation_executed']
        assert monitor.load_experiment(str(ROOT),'AdaRES-Y81')['status']=='done_complete'
        out=ROOT/('pipeline/checks/y82_y83_prepared_wait_'+token+'.json');assert not out.exists()
        tick.save(out,dict(complete=True,token=token,snapshot=view['generated_at'],resource_authority=authority,
            original_live_training=observations,occupations=occupations,retained_native_plans=plans,
            cached_Y81_terminal=done['terminal_evidence'],cached_Y81_archive=done['terminal_archive_binding'],
            Y81_terminal_verification_repeated=False,CPU_preparation_repeated=False,native_GPU_tasks_executed=False,
            new_candidates=0,new_admissions=0,new_launches=0,new_archives=0,formal_metrics_updated=False,
            GPU1_pause_preserved=True,Y57_pause_preserved=True,helper=ref(__file__)))
        for cid in ('AdaRES-Y84','AdaRES-Y85'):get(cid)['evidence'].append(ref(out))
        work['tick_attempt']=dict(tick_started_at=view['generated_at'],candidate_id='AdaRES-Y84',kind='resource_check',result='blocked',
            command='python -B scripts/record_y82_y83_wait.py --token '+token,
            observation='Fresh canonical physical inventory/compute and reservations verify both permitted GPU0 cards busy with original Y82/Y83. Y84/Y85 source/native plans remain unchanged; GPU1 user pause remains. No native task, admission, repeated CPU preparation or repeated Y81 terminal verification.',
            evidence=[ref(out),occupations['server-2']])
        assert all(v in get('AdaRES-Y84')['evidence'] for v in work['tick_attempt']['evidence'])
        tick.save(ROOT/'pipeline/research_work.json',work)
        memory=ROOT/'pipeline/codex_memory.md';backup=ROOT/('pipeline/snapshots/codex_memory_before_'+token+'.md')
        assert not backup.exists();backup.write_bytes(memory.read_bytes())
        def e(v):return v['path']+' SHA'+v['sha256']
        lines=memory.read_text(encoding='utf-8').splitlines();lines=lines[:lines.index('## Last Tick')]
        for i,line in enumerate(lines):
            if line.startswith('- 当前token '):lines[i]='- 当前token '+token+'，snapshot'+view['generated_at']+'，deadline'+view['deadline_at']+'不延长；postflight只认同token官方收据。'
            elif line.startswith('- 实际CODEX_THREAD_ID=owner='):lines[i]='- 实际CODEX_THREAD_ID=owner=01a07e32-78bf-7670-925b-370911ea48db已核；Pipeline-AdaRES_project Disabled；experiment_first enabled。本轮新候选/准入/启动/归档均0；无正式FIRST/cost。'
            elif line.startswith('- 前轮完整七节记忆精确备份：'):lines[i]='- 前轮完整七节记忆精确备份：'+e(ref(backup))+'；历史科学/禁用/专用恢复和原证据保持。'
            elif line.startswith('- Y81本轮原'):lines[i]=line.replace('- Y81本轮原','- Y81在73fa轮原',1)
            elif line.startswith('- AdaRES-Y') and ' server' in line:
                cid=line.split()[1]
                if cid in phases:
                    v=phases[cid];t=tasks[cid];lines[i]='- '+cid+' '+t['server']+' GPU0 PID'+str(t['pid'])+' 原source/YAML/receipt/marker/compute归属/功耗及native递增实核：'+v['phase']+str(v['epoch'])+'/30 step'+str(v['step'])+'/'+str(v['count'])+'；'+e(observations[cid])+'。'
            elif line.startswith('- fresh occupation server2GPU0：'):lines[i]='- fresh occupation server2GPU0：'+e(occupations['server-2'])+'；server1GPU0：'+e(occupations['server-1'])+'；两授权GPU0 busy+reserved。finish须<300秒，GPU1无使用许可。'
            elif line.startswith('- 旧record_prepared_y84_y85_wait.py'):lines[i]='- 当前安静复核使用scripts/record_y82_y83_wait.py --token NEW_TOKEN，仅适用于Y82/Y83仍原PID运行且HEAD281832、Y84/Y85仍等待GPU0。旧三训练wrapper和Y81归档wrapper不重放；当前handoff：'+e(ref(out))+'。'
        status='；'.join(cid+' '+v['phase']+str(v['epoch'])+'/30 step'+str(v['step'])+'/'+str(v['count']) for cid,v in phases.items())
        lines+=['## Last Tick','- '+token+'：'+status+'，原两GPU0训练递增与原功耗/source/YAML/receipt实核。',
            '- Y84/Y85原准备/版本/native计划哈希与最新两GPU0 busy+reserved复核；无新候选/准入/启动/归档，无正式FIRST/cost。',
            '- Y81原fixed30完整归档及训练best .5078033804893494保持，未重放CPU checkpoint加载/终态。GPU1后评与Y57单独暂停；不迁移、不重训。',
            '- 当前handoff：'+e(ref(out))+'；Y84优先Y85等原server2GPU0/280W独占空槽才做真实native两步。',
            '- 官方report/finish后只读核FINALIZED/PASS/context_committed/released与无owned lock/pending/archive_pending；不预签结果，释放后不写。']
        assert len(lines)<120 and sum(v.startswith('## ') for v in lines)==7
        memory.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
        notes=ROOT/('pipeline/checks/foreground_notes_'+token+'.json');assert not notes.exists()
        tick.save(notes,dict(actions=['运行：'+status+'，原source/YAML/receipt/marker/compute归属/功耗及native递增核验通过。',
            'Y84/Y85原准备、版本和GPU0 native计划保持；最新两授权GPU0 busy+reserved实核。',
            'Y81原完整归档保留，不重放；本轮无新候选/准入/启动/归档，无正式FIRST/cost。'],
            decisions=['GPU1全部项目使用继续用户暂停，原GPU1冻结后评/cost不迁移或替换；Y57单独暂停。',
                'Y84优先Y85等原server2GPU0/280W独占空槽后真实native两步PASS再准入/启动；不重放原CPU准备。',
                '原科学goal/receipt/数据/私有运行环境/mapping/seed1999-fixed30/禁用与调度保持；无新增用户处理。']))
        assert qref==ref('pipeline/queue.json') and rref==ref('pipeline/running.json')
        print(json.dumps(dict(handoff=ref(out),notes=notes.relative_to(ROOT).as_posix(),memory_lines=len(lines),new_admissions=0)))
    finally:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);record(p.parse_args().token)
