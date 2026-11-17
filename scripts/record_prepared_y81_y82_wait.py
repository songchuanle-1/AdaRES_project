"""Record current authorized resource holds and optional new Y78 terminal preparation."""
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
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()=='4e68cd84d56d890222adc2bec8ab59cb0c8715b6'
        assert ref('pipeline/goal.json')['sha256']=='4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a'
        tasks={task['id']:task for rows in queue['gpu_queues'].values() for task in rows};observations={};phases={}
        for cid,name,gpu,pid in [('AdaRES-Y80','server-2','0',42226),('AdaRES-Y78','server-2','1',22192),('AdaRES-Y79','server-1','0',2218231)]:
            task=tasks[cid];assert task['status']=='running' and task['launch_verified'] and task['pid']==pid and running[name][gpu]==cid
            closure.admission(ROOT,task);evidence=get(cid)['latest_training_observation'];value=tick.load(closure.artifact(ROOT,evidence))
            assert value['token']==token and value['native_progress_confirmed'] and value['pid']==pid
            observations[cid],phases[cid]=evidence,value['after']['latest']
        assert tasks['AdaRES-Y57']['status']=='paused' and tasks['AdaRES-Y57']['dispatch_hold'] and tasks['AdaRES-Y57']['resume_requires_user_authorization']
        for cid in ('AdaRES-Y76','AdaRES-Y77'):
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
        for cid in ('AdaRES-Y81','AdaRES-Y82'):
            row=get(cid);assert row['stage']=='blocked' and row['blocker_kind']=='exclusive_gpu_slot' and row['implementation_test_passed']
            for field in ('source_manifest','CPU_native_observer_verification','native_preparation'):closure.artifact(ROOT,row[field])
            retained[cid]={}
            for field in ('native_smoke_plan_server2_gpu0','native_smoke_plan_server2_gpu1'):
                evidence=row[field];plan=tick.load(closure.artifact(ROOT,evidence));program=closure.artifact(ROOT,plan['program'])
                frozen=ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
                assert validate_frozen(row,plan,frozen)==7 and plan['native_executed'] is False;retained[cid][field]=evidence
        terminal_ref=get('AdaRES-Y78')['terminal_handoff_plan'];terminal=tick.load(closure.artifact(ROOT,terminal_ref))
        assert terminal['id']=='AdaRES-Y78' and terminal['CPU_guard_tests_passed']==22 and not terminal['checkpoint_loaded'] and not terminal['archived'] and not terminal['full_terminal_verification_executed']
        assert terminal['admission']==ref('pipeline/admissions/AdaRES-Y78.json') and terminal['post_training_freeze']==get('AdaRES-Y78')['post_training_freeze']
        for evidence in terminal['dependencies']+[terminal['admission'],terminal['post_training_freeze']]:closure.artifact(ROOT,evidence)
        closure.artifact(ROOT,terminal['supersedes']);assert terminal['repair_reason'] and terminal['CPU_guard_tests_passed']==22
        location=tick.load(closure.artifact(ROOT,terminal['declared_location']));assert not location['checkpoint_loaded'] and not location['remote_mutations']
        tests_ref=next(e for e in terminal['evidence'] if e['path'].endswith('_tests.log'))
        assert b'Ran 22 tests' in closure.artifact(ROOT,tests_ref).read_bytes()
        if new_terminal_preparation:assert terminal['token']==token
        out=ROOT/('pipeline/checks/prepared_y81_y82_resource_'+token+'.json');assert not out.exists()
        tick.save(out,dict(complete=True,token=token,snapshot=view['generated_at'],observed_at=datetime.now().astimezone().isoformat(),original_live_training=observations,
            occupations=occupations,retained_native_plans=retained,Y78_terminal_preparation=terminal_ref,Y78_terminal_dependencies_unchanged=True,
            new_Y78_terminal_preparation=new_terminal_preparation,checkpoint_loaded=False,archive_executed=False,native_GPU_executed=False,
            new_candidates=0,new_admissions=0,new_launches=0,new_archives=0,formal_metrics_updated=False,Y57_pause_preserved=True,helper=ref(__file__)))
        for cid in ('AdaRES-Y78','AdaRES-Y79','AdaRES-Y80','AdaRES-Y81','AdaRES-Y82'):get(cid)['evidence'].append(ref(out))
        selected=get('AdaRES-Y78' if new_terminal_preparation else 'AdaRES-Y81')
        if new_terminal_preparation:
            evidence=[terminal_ref,tests_ref,ref(out)]
            command='python -B scripts/prepare_y78_terminal_handoff_v2.py --token '+token
            observation='Twenty-two CPU lifecycle rejection guards and actual read-only original source/YAML/receipt/freeze/exact declared checkpoint-directory checks passed for the original Y78 server2GPU1/390W run. Current PID/training and all other jobs preserved; no checkpoint load/full terminal validation/archive/GPU task/admission/efficacy.'
        else:
            evidence=[ref(out),occupations['server-2']]
            command='python -B scripts/record_prepared_y81_y82_wait.py --token '+token
            observation='Fresh canonical physical inventory/compute/reservations and exact retained native-plan source/YAML/goal/version hashes verified; no matching exclusive authorized slot. Original three jobs and all previously passed preparation preserved; no native GPU task/admission/efficacy.'
        for item in evidence:
            if item not in selected['evidence']:selected['evidence'].append(item)
        work['tick_attempt']=dict(tick_started_at=view['generated_at'],candidate_id=selected['id'],kind='implementation_test' if new_terminal_preparation else 'resource_check',result='passed' if new_terminal_preparation else 'blocked',command=command,observation=observation,evidence=evidence)
        assert all(item in selected['evidence'] for item in evidence);tick.save(ROOT/'pipeline/research_work.json',work)
        memory=ROOT/'pipeline/codex_memory.md';backup=ROOT/('pipeline/snapshots/codex_memory_before_'+token+'.md');assert not backup.exists();backup.write_bytes(memory.read_bytes())
        def e(value):return value['path']+' SHA'+value['sha256']
        lines=memory.read_text(encoding='utf-8').splitlines();lines=lines[:lines.index('## Last Tick')]
        for index,line in enumerate(lines):
            if line.startswith('- 当前token '):lines[index]='- 当前token '+token+'，snapshot'+view['generated_at']+'，deadline'+view['deadline_at']+'不延长；postflight只认同token官方收据。'
            elif line.startswith('- 实际CODEX_THREAD_ID=owner='):lines[index]='- 实际CODEX_THREAD_ID=owner=01a07e32-78bf-7670-925b-370911ea48db已核；Pipeline-AdaRES_project Disabled；experiment_first enabled，admission2/creation2。本轮新候选0/准入0/启动0/归档0；'+('Y78专用终态准备22守卫与只读原目录检查通过。' if new_terminal_preparation else '仅当前资源/进度与原准备完整性复核。')+'无正式FIRST/cost。'
            elif line.startswith('- 前轮完整七节记忆精确备份：'):lines[index]='- 前轮完整七节记忆精确备份：'+e(ref(backup))+'；历史机制/禁用/专用恢复与全部原证据沿备份保持。'
            elif line.startswith('- Y80独立W2'):lines[index]=line.replace('本轮server2','在bd938轮server2')
            elif line.startswith('- Y77本轮专用'):lines[index]=line.replace('本轮','在bd938轮',1)
            elif line.startswith('- 本轮Y80实际GPU'):lines[index]=line.replace('本轮','bd938轮',1)
            elif any(line.startswith('- '+cid+' server') for cid in observations):
                cid=line.split()[1];phase=phases[cid];resource={'AdaRES-Y78':'server2GPU1/390W PID22192','AdaRES-Y79':'server1GPU0/350W PID2218231','AdaRES-Y80':'server2GPU0/280W PID42226'}[cid]
                lines[index]='- '+cid+' '+resource+' 原source/YAML/receipt/marker/计算归属/功耗和native递增已核：'+phase['phase']+str(phase['epoch'])+'/30 step'+str(phase['step'])+'/'+str(phase['count'])+'；'+e(observations[cid])+'。'
            elif line.startswith('- fresh occupation server2：'):lines[index]='- fresh occupation server2：'+e(occupations['server-2'])+'；server1：'+e(occupations['server-1'])+'；三卡busy+reserved。finish证据须<300秒，超时新label；未知/失败/缺设备/idle不能证明hold。'
            elif line.startswith('- 下轮先y80_deployment'):lines[index]='- 下轮先y80_deployment核三原运行并派发现有任务；Y78自然fixed30终态先核其terminal_handoff_plan全部依赖/原receipt/freeze，再python -B scripts/finalize_y78_fixed30_v2.py --token NEW_TOKEN完整原目录/Train-Val1..30/稳定finite last30-best/model-optimizer-scheduler与精确source链核验，canonical显式GPU1归档；不可套Y77 helper。随后Y81原GPU1/390W native计划在fresh空槽激活，Y82随后。'
            elif line.startswith('- 不调用旧record_prepared'):lines[index]='- 不调用旧record_prepared_y80_y81_wait.py或事件型record_y80_y82_closeout.py；当前record_prepared_y81_y82_wait.py支持HEAD4e68与Y78/Y79/Y80运行、Y81/Y82等待、Y78可选新终态准备。未来安静复核使用新token且不带--new-terminal-preparation，不重放prepare/CPU。heartbeat原owner/频率/通知保持，不建线程/新自动化/嵌套codex exec。'
        marker='- Y78专用终态准备：'
        if not any(line.startswith(marker) for line in lines):
            at=lines.index('## Active Hypotheses')+1
            lines.insert(at,marker+e(terminal_ref)+'；22生命周期拒绝守卫、六原源码/YAML/receipt/freeze/生产marker和声明目录'+location['checkpoint_directory']+'只读核验通过；原PID22192/server2GPU1/390W/源5eee与当前HEAD4e68精确Y78→Y79→Y80 A边绑定。未加载checkpoint/未full terminal验证/未停止或归档。未来自然fixed30退出后先核plan.dependencies及原receipt/freeze，再专用finalize_y78_fixed30_v2.py；不要重放prepare。')
        repair_note='- Y78归档准备v1复查发现两个遗留GPU0字段（归档后断言/后评required_gpu），尚未执行归档；v1全部源码和原plan保留。v2从已核GPU1 task派生字段并新增两项终态后资源守卫，22检查通过；当前terminal_handoff_plan绑定v2及supersedes链，不改变科学模型。'
        if repair_note not in lines:lines.insert(lines.index('## Running And Queue'),repair_note)
        lines += ['## Last Tick',
            '- '+token+' begin/design先查并尝试派发waiting0；快照new_terminal:Y77是bd938轮既有归档，不重放。Y78/Y79/Y80原native进度递增，无新完成/故障，fresh三卡busy+reserved。',
            '- 本轮新候选0/准入0/启动0/归档0；'+('Y78独立专用终态准备/22守卫及原source/YAML/receipt/freeze/marker/声明目录只读核验首次通过；未加载checkpoint，仍原训练。' if new_terminal_preparation else '仅当前资源/进度与原native-终态准备哈希复核，不重放CPU/prepare。')+'Y81/Y82已准备，仍待真实native GPU任务；无FIRST/cost。',
            '- handoff：'+e(ref(out))+'；tick_attempt绑定'+selected['id']+'本轮实际'+('终态准备。' if new_terminal_preparation else '资源复核。')+'主HEAD/server2 marker4e68，server1仍e0c，当前observer为y80_deployment。',
            '- Y78真实自然fixed30退出后原专用完整核验/canonical显式GPU1归档，再fresh独占idle/no compute/no reservation/390W后Y81原native两步PASS→原完整后评freeze/receipt/精确source契约/安全派发及native递增；Y82随后，Y57单独暂停。',
            '- 官方report/finish后仅只读核FINALIZED/PASS/context_committed/released及无lock/pending/archive_pending；此记忆不预签结果，释放后不写。']
        assert len(lines)<120 and sum(line.startswith('## ') for line in lines)==7;memory.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
        notes=ROOT/('pipeline/checks/foreground_notes_'+token+'.json');assert not notes.exists()
        status='；'.join(cid+' '+value['phase']+str(value['epoch'])+'/30 step'+str(value['step'])+'/'+str(value['count']) for cid,value in phases.items())
        actions=['运行：'+status+'，原PID/source/YAML/receipt/marker/功耗和native计数递增核验通过。',
            '新准备：Y78独立专用终态程序/22生命周期守卫与原六源码/YAML/receipt/freeze/生产marker/声明目录只读核验通过；保持原训练，未加载checkpoint或提前full terminal验证/归档。' if new_terminal_preparation else '仅当前资源和原native/终态准备完整性复核，未重放CPU/prepare，无新完成/故障。',
            'fresh canonical三授权卡busy+reserved，Y81/Y82四份原native计划源码/YAML/goal/version完整性有效。新候选0/准入0/启动0/归档0，无FIRST/cost；记忆与备份保存。']
        tick.save(notes,dict(actions=actions,decisions=['保持原Y78/Y79/Y80训练；已有Y81/Y82两份CPU和native准备不为额度重复设计，按顺序等独占GPU空槽。',
            'Y78自然完整fixed30退出后先核专用plan.dependencies/原receipt/freeze，再完整稳定finite和source/原目录/Train-Val验证及显式GPU1 canonical归档；fresh独占390W后Y81真实native PASS再准入派发。',
            '保留原完整后评/精确数据及mapping/私有环境/科学goal和receipt/禁用/seed1999-fixed30及350W-280W-390W；Y57单独暂停，禁止server6/跨项目/调度变更，无新增用户操作。']))
        assert qref==ref('pipeline/queue.json') and rref==ref('pipeline/running.json')
        print(json.dumps(dict(handoff=ref(out),notes=str(notes.relative_to(ROOT)),memory_lines=len(lines),new_terminal_preparation=new_terminal_preparation,new_candidates=0,admitted=0,launched=0,archived=0)))
    finally:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--token',required=True);parser.add_argument('--new-terminal-preparation',action='store_true');args=parser.parse_args();record(args.token,args.new_terminal_preparation)
