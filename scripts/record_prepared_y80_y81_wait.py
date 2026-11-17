"""Record current authorized resource holds and optional new Y77 terminal preparation."""
import argparse
import ast
from datetime import datetime, timezone
import json
import subprocess
from prepare_y81_query_mask_refinement import ROOT,lease,monitor,ref,tick
from check_ef_execution import validate_frozen
import closure
import gpu_occupation


def record(token,new_terminal_preparation):
    lease(token,180);lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:
        view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json');context=tick.load(ROOT/'pipeline/headless/.tick_context.json')
        work=tick.load(ROOT/'pipeline/research_work.json');get=lambda cid:next(c for c in work['candidates'] if c['id']==cid)
        queue,running=monitor.load_queue(str(ROOT)),monitor.load_running(str(ROOT));qref,rref=ref('pipeline/queue.json'),ref('pipeline/running.json')
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()=='e0c7dba199fcf1a009941450d04aeb799418a7e6'
        assert ref('pipeline/goal.json')['sha256']=='4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a'
        tasks={task['id']:task for rows in queue['gpu_queues'].values() for task in rows};observations={};phases={}
        for cid,name,gpu,pid in [('AdaRES-Y77','server-2','0',111334),('AdaRES-Y78','server-2','1',22192),('AdaRES-Y79','server-1','0',2218231)]:
            task=tasks[cid];assert task['status']=='running' and task['launch_verified'] and task['pid']==pid and running[name][gpu]==cid
            closure.admission(ROOT,task);evidence=get(cid)['latest_training_observation'];value=tick.load(closure.artifact(ROOT,evidence))
            assert value['token']==token and value['native_progress_confirmed'] and value['pid']==pid
            observations[cid],phases[cid]=evidence,value['after']['latest']
        assert tasks['AdaRES-Y57']['status']=='paused' and tasks['AdaRES-Y57']['dispatch_hold'] and tasks['AdaRES-Y57']['resume_requires_user_authorization']
        for cid in ('AdaRES-Y75','AdaRES-Y76'):
            assert cid not in tasks and monitor.load_experiment(str(ROOT),cid)['status']=='done_complete'
            closure.artifact(ROOT,get(cid)['terminal_evidence']);closure.artifact(ROOT,get(cid)['post_training_freeze'])
        occupations={}
        for name in ('server-2','server-1'):
            evidence=ref('pipeline/checks/gpu_occupation_'+name+'_finish_'+token+'.json');value=tick.load(closure.artifact(ROOT,evidence))
            assert value['token']==token and (datetime.now(timezone.utc)-datetime.fromisoformat(value['observed_at'])).total_seconds()<300
            row=next(c for c in work['candidates'] if c.get('stage')=='blocked' and c.get('blocker_kind')=='exclusive_gpu_slot' and c.get('required_server')==name)
            server=next(s for s in context['queue']['servers'] if s['name']==name)
            assert not gpu_occupation.validated_state(ROOT,context,row,server,queue)['idle_gpu_ids'];occupations[name]=evidence
        retained={}
        for cid in ('AdaRES-Y80','AdaRES-Y81'):
            row=get(cid);assert row['stage']=='blocked' and row['blocker_kind']=='exclusive_gpu_slot' and row['implementation_test_passed']
            for field in ('source_manifest','CPU_native_observer_verification','native_preparation'):closure.artifact(ROOT,row[field])
            retained[cid]={}
            for field in ('native_smoke_plan_server2_gpu0','native_smoke_plan_server2_gpu1'):
                evidence=row[field];plan=tick.load(closure.artifact(ROOT,evidence));program=closure.artifact(ROOT,plan['program'])
                frozen=ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
                assert validate_frozen(row,plan,frozen)==7 and plan['native_executed'] is False;retained[cid][field]=evidence
        terminal_ref=get('AdaRES-Y77')['terminal_handoff_plan'];terminal=tick.load(closure.artifact(ROOT,terminal_ref))
        assert terminal['id']=='AdaRES-Y77' and terminal['CPU_guard_tests_passed']==20 and not terminal['checkpoint_loaded'] and not terminal['archived'] and not terminal['full_terminal_verification_executed']
        assert terminal['admission']==ref('pipeline/admissions/AdaRES-Y77.json') and terminal['post_training_freeze']==get('AdaRES-Y77')['post_training_freeze']
        for evidence in terminal['dependencies']+[terminal['admission'],terminal['post_training_freeze']]:closure.artifact(ROOT,evidence)
        location=tick.load(closure.artifact(ROOT,terminal['declared_location']));assert not location['checkpoint_loaded'] and not location['remote_mutations']
        tests_ref=next(e for e in terminal['evidence'] if e['path'].endswith('_tests.log'))
        assert b'Ran 20 tests' in closure.artifact(ROOT,tests_ref).read_bytes()
        if new_terminal_preparation:assert terminal['token']==token
        out=ROOT/('pipeline/checks/prepared_y80_y81_resource_'+token+'.json');assert not out.exists()
        tick.save(out,dict(complete=True,token=token,snapshot=view['generated_at'],observed_at=datetime.now().astimezone().isoformat(),original_live_training=observations,
            occupations=occupations,retained_native_plans=retained,Y77_terminal_preparation=terminal_ref,Y77_terminal_dependencies_unchanged=True,
            new_Y77_terminal_preparation=new_terminal_preparation,checkpoint_loaded=False,archive_executed=False,native_GPU_executed=False,
            new_candidates=0,new_admissions=0,new_launches=0,new_archives=0,formal_metrics_updated=False,Y57_pause_preserved=True,helper=ref(__file__)))
        for cid in ('AdaRES-Y77','AdaRES-Y78','AdaRES-Y79','AdaRES-Y80','AdaRES-Y81'):get(cid)['evidence'].append(ref(out))
        selected=get('AdaRES-Y77' if new_terminal_preparation else 'AdaRES-Y80')
        if new_terminal_preparation:
            evidence=[terminal_ref,tests_ref,ref(out)]
            command='python -B scripts/prepare_y77_terminal_handoff.py --token '+token
            observation='Twenty CPU lifecycle rejection guards and actual read-only original source/YAML/receipt/freeze/exact declared checkpoint-directory checks passed for the original Y77 server2GPU0/280W run. Current PID/training and all other jobs preserved; no checkpoint load/full terminal validation/archive/GPU task/admission/efficacy.'
        else:
            evidence=[ref(out),occupations['server-2']]
            command='python -B scripts/record_prepared_y80_y81_wait.py --token '+token
            observation='Fresh canonical physical inventory/compute/reservations and exact retained native-plan source/YAML/goal/version hashes verified; no matching exclusive authorized slot. Original three jobs and all previously passed preparation preserved; no native GPU task/admission/efficacy.'
        for item in evidence:
            if item not in selected['evidence']:selected['evidence'].append(item)
        work['tick_attempt']=dict(tick_started_at=view['generated_at'],candidate_id=selected['id'],kind='implementation_test' if new_terminal_preparation else 'resource_check',result='passed' if new_terminal_preparation else 'blocked',command=command,observation=observation,evidence=evidence)
        assert all(item in selected['evidence'] for item in evidence);tick.save(ROOT/'pipeline/research_work.json',work)
        memory=ROOT/'pipeline/codex_memory.md';backup=ROOT/('pipeline/snapshots/codex_memory_before_'+token+'.md');assert not backup.exists();backup.write_bytes(memory.read_bytes())
        def e(value):return value['path']+' SHA'+value['sha256']
        lines=memory.read_text(encoding='utf-8').splitlines();lines=lines[:lines.index('## Last Tick')]
        for index,line in enumerate(lines):
            if line.startswith('- 当前token '):lines[index]='- 当前token '+token+'，snapshot'+view['generated_at']+'，deadline'+view['deadline_at']+'不延长；postflight只认官方同token收据。'
            elif line.startswith('- 实际CODEX_THREAD_ID=owner='):lines[index]='- 实际CODEX_THREAD_ID=owner=01a07e32-78bf-7670-925b-370911ea48db已核，Pipeline-AdaRES_project Disabled；experiment_first enabled，snapshot admission2/creation2。本轮新候选0/准入0/启动0/归档0；'+('Y77专用终态准备/20守卫与只读原目录检查完成；' if new_terminal_preparation else '仅当前进度/资源与原准备完整性复核；')+'无正式FIRST/cost。'
            elif line.startswith('- 前轮完整记忆精确备份：'):lines[index]='- 前轮完整记忆精确备份：'+e(ref(backup))+'；历史机制/禁用/专用恢复与全部原证据沿备份保持。'
            elif line.startswith('- Y79本轮server1'):lines[index]=line.replace('本轮','在b5d4轮',1)
            elif line.startswith('- Y76本轮专用'):lines[index]=line.replace('本轮','在b5d4轮',1)
            elif line.startswith('- Y81本轮实际'):lines[index]=line.replace('本轮','在b5d4轮',1)
            elif line.startswith('- 本轮fresh occupation server2：'):lines[index]='- 本轮fresh occupation server2：'+e(occupations['server-2'])+'；server1：'+e(occupations['server-1'])+'；三授权卡busy+reserved。finish前须<300秒，超时新label；未知/失败/缺设备/idle不能证明hold。'
            elif line.startswith('- Y77仍在训练，专用终态契约尚未准备：'):lines[index]='- Y77仍在原训练，独立专用终态准备已通过20守卫与原source/YAML/receipt/freeze/marker/声明目录只读检查：'+e(terminal_ref)+'；原PID111334/server2GPU0/280W/源498e、server2 marker5eee与HEADe0c绑定。未加载checkpoint/未提前full terminal验证/未归档；自然退出后先核plan.dependencies，再用其专用finalize_y77_fixed30.py，不套其他ID helper。'
            elif line.startswith('- heartbeat原owner/原两小时间隔/通知保持；'):lines[index]='- heartbeat原owner/原两小时间隔/通知保持，不建线程/新自动化/嵌套codex exec。旧resource wait v2只支持旧5eee与原Y76/Y77/Y78；现用record_prepared_y80_y81_wait.py支持e0c和Y77/Y78/Y79，绑定当前真实资源与可选专用终态准备，旧helper不改。'
            elif any(line.startswith('- '+cid+' server') for cid in observations):
                cid=line.split()[1];phase=phases[cid];resource={'AdaRES-Y77':'server2GPU0/280W PID111334','AdaRES-Y78':'server2GPU1/390W PID22192','AdaRES-Y79':'server1GPU0/350W PID2218231'}[cid]
                lines[index]='- '+cid+' '+resource+' 原source/YAML/receipt/marker/计算归属/功耗与native递增已核：'+phase['phase']+str(phase['epoch'])+'/30 step'+str(phase['step'])+'/'+str(phase['count'])+'；'+e(observations[cid])+'。'
        if not any(line.startswith('- Y77独立终态准备：') for line in lines):
            at=lines.index('## Active Hypotheses')+1
            lines.insert(at,'- Y77独立终态准备：'+e(terminal_ref)+'；20生命周期/原GPU0-280W-PID111334/原source498e与marker5eee-HEADe0c精确A边、完整epochs/finite模型优化器调度器/只删除原GPU0 map且保留Y78/Y79/Y57守卫通过；实际只读核六原源码/YAML/receipt/freeze/声明目录'+location['checkpoint_directory']+'，未加载checkpoint/未归档。未来先核plan.dependencies/原receipt/freeze，真实自然fixed30退出后仅用python -B scripts/finalize_y77_fixed30.py --token NEW_TOKEN；不要重放prepare。')
        lines += ['## Last Tick',
            '- '+token+' begin/design先查并尝试派发waiting0；snapshot中的旧Y76终态来自前轮完整归档，不重放。Y77/Y78/Y79原native进度递增，无新训练完成/故障；fresh三卡busy+reserved。',
            '- 本轮新候选0/准入0/启动0/归档0；'+('Y77专用完整终态准备20守卫及只读原source/YAML/receipt/freeze/marker/声明目录检查通过，仍在训练且未加载checkpoint。' if new_terminal_preparation else '只做新资源/进度和原native/终态准备哈希复核，不重放已通过CPU或prepare。')+'既有Y80/Y81仍等真实native GPU检查，优先Y80，无正式FIRST/cost。',
            '- handoff：'+e(ref(out))+'；tick_attempt绑定'+selected['id']+'本轮实际'+('终态准备检查。' if new_terminal_preparation else '资源检查。')+'主HEAD/server1 marker e0c，server2仍5eee；y79_deployment为当前精确observer，不改旧hash-bound helper。',
            '- 下轮先y79_deployment核三原运行并派发现有任务。Y77自然fixed30退出时先核其terminal_handoff_plan全部依赖/原receipt/freeze，再专用finalize_y77_fixed30.py完整Train/Val1..30、稳定finite last30/best/model/optimizer/scheduler和原源/日志/目录核验，canonical显式server2GPU0归档。fresh idle/unreserved280W后Y80原native两步PASS→原完整后评freeze/新receipt/精确4e68版本source契约→安全派发；Y81随后。保留数据/禁用/功耗/Y57单独暂停/原调度。',
            '- 官方report/finish后仅只读核FINALIZED/PASS/context_committed/released及锁/pending/archive_pending；此记忆不预签结果，释放后不写。']
        assert len(lines)<120 and sum(line.startswith('## ') for line in lines)==7;memory.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
        notes=ROOT/('pipeline/checks/foreground_notes_'+token+'.json');assert not notes.exists()
        status='；'.join(cid+' '+value['phase']+str(value['epoch'])+'/30 step'+str(value['step'])+'/'+str(value['count']) for cid,value in phases.items())
        actions=['运行：'+status+'，原PID/source/YAML/receipt/marker/功耗和native计数递增核验通过。',
            '新准备：Y77独立专用终态程序/20生命周期守卫与原六源码/YAML/receipt/freeze/生产marker/声明目录只读核验通过；保持原训练，未加载checkpoint、未提前完整终态验证或归档。' if new_terminal_preparation else '本轮仅当前资源和原native/终态准备哈希复核；未重放已通过CPU或prepare，无新终态或故障。',
            'fresh canonical三授权卡均busy+reserved，Y80/Y81四份原native计划源码/YAML/goal/version完整性复核有效；保存证据与七节记忆备份/更新。新候选0/准入0/启动0/归档0，无正式FIRST/cost。']
        tick.save(notes,dict(actions=actions,decisions=['保持Y77/Y78/Y79原训练，不挤占GPU或为额度重复设计；Y80优先Y81等待独占native检查。',
            'Y77真实自然fixed30退出后先核其专用plan依赖/原receipt/freeze，完整验证后canonical显式GPU0归档；fresh idle/unreserved280W后Y80实际native PASS再原完整后评冻结/准入/精确source契约/安全派发。',
            '保持原数据/独立环境/仅授权mapping、科学goal/receipt、全部禁用/seed1999固定30及350W-280W-390W；Y57单独暂停，禁止server6/跨项目/调度变更。无新增用户操作事项。']))
        assert qref==ref('pipeline/queue.json') and rref==ref('pipeline/running.json')
        print(json.dumps(dict(evidence=ref(out),notes=str(notes.relative_to(ROOT)),memory_lines=len(lines),new_terminal_preparation=new_terminal_preparation,new_candidates=0,admitted=0,launched=0,archived=0)))
    finally:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--token',required=True);parser.add_argument('--new-terminal-preparation',action='store_true');args=parser.parse_args();record(args.token,args.new_terminal_preparation)
