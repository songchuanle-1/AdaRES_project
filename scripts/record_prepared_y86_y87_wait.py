"""Current Y84/Y85 observations and bounded Y86/Y87 preparation without GPU use."""
import argparse
import ast
from datetime import datetime, timezone
import json
import subprocess
import sys
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
from y85_deployment import Y84, Y85
import closure
import gpu_occupation


def record(token, new_preparation=False):
    lease(token, 240)
    for name in ('server-2', 'server-1'):
        response = subprocess.run([sys.executable, '-B',
            'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline/gpu_occupation.py',
            'AdaRES_project', '--token', token, '--server', name, '--label', 'finish'],
            cwd=ROOT, capture_output=True, timeout=45)
        assert response.returncode == 0, response.stderr.decode(errors='replace')
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        work = tick.load(ROOT / 'pipeline/research_work.json')
        get = lambda cid: next(candidate for candidate in work['candidates'] if candidate['id'] == cid)
        view = tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')
        context = tick.load(ROOT / 'pipeline/headless/.tick_context.json')
        queue, running = monitor.load_queue(str(ROOT)), monitor.load_running(str(ROOT))
        qref, rref = ref('pipeline/queue.json'), ref('pipeline/running.json')
        assert running == {'server-1': {'0': 'AdaRES-Y84'}, 'server-2': {'0': 'AdaRES-Y85'}}
        tasks = {task['id']: task for tasks in queue['gpu_queues'].values() for task in tasks}
        assert set(tasks) == {'AdaRES-Y57', 'AdaRES-Y84', 'AdaRES-Y85'}
        assert tasks['AdaRES-Y57']['status'] == 'paused' and tasks['AdaRES-Y57']['dispatch_hold']
        assert tasks['AdaRES-Y57']['resume_requires_user_authorization']
        assert closure.git(ROOT, 'rev-parse', 'HEAD') == Y85
        assert ref('pipeline/goal.json')['sha256'] == '4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a'
        authority = ref('pipeline/authorizations/server2_gpu1_pause_20261104.json')
        assert authority['sha256'] == '3715a2a5726e90c68e2e64ac22090fbcd5475e08b8edd6ab4dca6ed42fed5d45'
        servers = monitor.load_servers(str(ROOT))
        assert servers['server-1']['allowed_gpus'] == servers['server-2']['allowed_gpus'] == [0]
        assert servers['server-6']['allowed_gpus'] == []
        observations, phases = {}, {}
        for cid, name, commit, pid in [('AdaRES-Y84', 'server-1', Y84, 243952),
                                       ('AdaRES-Y85', 'server-2', Y85, 22090)]:
            task = tasks[cid]
            assert task['status'] == 'running' and task['launch_verified'] and task['pid'] == pid
            assert (task['server'], str(task['gpu']), task['code_commit']) == (name, '0', commit)
            closure.admission(ROOT, task)
            evidence = get(cid)['latest_training_observation']
            value = tick.load(closure.artifact(ROOT, evidence))
            assert value['token'] == token and value['pid'] == pid and value['native_progress_confirmed']
            observations[cid], phases[cid] = evidence, value['after']['latest']
        preparations = {}
        for cid, callbacks, parameters in [('AdaRES-Y86', 7, 164629639), ('AdaRES-Y87', 14, 164695173)]:
            candidate = get(cid)
            assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'exclusive_gpu_slot'
            assert (candidate['required_server'], candidate['required_gpu']) == ('server-2', 0)
            assert candidate['implementation_test_passed'] and not candidate.get('admission_receipt')
            assert candidate['resource_pause'] == authority
            assert not candidate.get('native_smoke_plan_server2_gpu1')
            version = tick.load(closure.artifact(ROOT, candidate['source_manifest']))
            cpu = tick.load(closure.artifact(ROOT, version['CPU_result']))
            preparation = tick.load(closure.artifact(ROOT, candidate['native_preparation']))
            assert cpu['complete'] and cpu['implementation_test_passed'] and cpu['parameters'] == parameters
            assert not cpu['CUDA_initialized'] and not cpu['native_smoke_passed'] and not cpu['native_task_backward']
            assert preparation['guard_tests_passed'] == 9
            assert preparation['exact_callback_functions_and_classes'] == callbacks
            assert preparation['CPU_native_callbacks_actually_executed'] and not preparation['native_executed']
            assert set(preparation['native_plans']) == {'0'}
            plan = tick.load(closure.artifact(ROOT, candidate['native_smoke_plan_server2_gpu0']))
            program = closure.artifact(ROOT, plan['program'])
            frozen = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
            assert validate_frozen(candidate, plan, frozen) == 7 and not plan['native_executed']
            assert (frozen['runtime_server'], frozen['runtime_physical_gpu'],
                    frozen['runtime_power_limit_watts']) == ('server-2', 0, 280.)
            preparations[cid] = dict(CPU_result=version['CPU_result'],
                version=candidate['source_manifest'], native_preparation=candidate['native_preparation'],
                GPU0_plan=candidate['native_smoke_plan_server2_gpu0'], code_commit=candidate['code_commit'],
                exact_executed_callbacks=callbacks, guards=9)
        assert closure.git(ROOT, 'rev-parse', preparations['AdaRES-Y87']['code_commit'] + '^') == preparations['AdaRES-Y86']['code_commit']
        occupations = {}
        for name in ('server-2', 'server-1'):
            evidence = ref('pipeline/checks/gpu_occupation_' + name + '_finish_' + token + '.json')
            value = tick.load(closure.artifact(ROOT, evidence))
            assert value['token'] == token
            assert 0 <= (datetime.now(timezone.utc) - datetime.fromisoformat(value['observed_at'])).total_seconds() < 300
            blocked = next(candidate for candidate in work['candidates']
                if candidate.get('stage') == 'blocked' and candidate.get('blocker_kind') == 'exclusive_gpu_slot'
                and candidate.get('required_server') == name)
            server_context = next(server for server in context['queue']['servers'] if server['name'] == name)
            state = gpu_occupation.validated_state(ROOT, context, blocked, server_context, queue)
            assert state['busy_gpu_ids'] == [0] and not state['idle_gpu_ids']
            occupations[name] = evidence
        if new_preparation:
            new = get('AdaRES-Y87')
            version87 = tick.load(closure.artifact(ROOT, new['source_manifest']))
            assert version87['CPU_result']['path'] == 'pipeline/analyses/y87_cpu_' + token + '.json'
            assert tick.load(closure.artifact(ROOT, new['native_preparation']))['token'] == token
        out = ROOT / ('pipeline/checks/y86_y87_prepared_wait_' + token + '.json')
        assert not out.exists()
        tick.save(out, dict(complete=True, token=token, snapshot=view['generated_at'],
            live_training=observations, occupations=occupations, retained_preparations=preparations,
            new_candidates=int(new_preparation), new_admissions=0, new_launches=0, new_archives=0,
            repeated_CPU_preparation=False, repeated_terminal_verification=False,
            native_GPU_executed=False, formal_metrics_updated=False, GPU1_pause_preserved=True,
            Y57_pause_preserved=True, helper=ref(__file__)))
        for cid in ('AdaRES-Y86', 'AdaRES-Y87'):
            get(cid)['evidence'].append(ref(out))
        if new_preparation:
            work['tick_attempt'] = dict(tick_started_at=view['generated_at'], candidate_id='AdaRES-Y87',
                kind='implementation_test', result='passed',
                command='python -B scripts/run_experiment_first_probe.py y87_cpu pipeline/checks/y87_cpu_' + token + '_program.py --token ' + token + ' --seconds 480',
                observation='Actual original CPU model/backbone/TRAIN loading, independent float32/64 sum-product/zero/RNG/permutation controls, seven-stage W2/input-gradient anchors and confidence heads, exact original self/language/norm/FFN consumers and two synthetic optimizer steps passed. Exact seven-file retained version and GPU0-only native plan/fourteen identical executed callbacks/nine guards complete. Both permitted GPU0 cards busy; no native GPU PASS/admission/efficacy.',
                evidence=[preparations['AdaRES-Y87']['CPU_result'], preparations['AdaRES-Y87']['version'],
                          preparations['AdaRES-Y87']['native_preparation'], ref(out)])
        else:
            work['tick_attempt'] = dict(tick_started_at=view['generated_at'], candidate_id='AdaRES-Y86',
                kind='resource_check', result='blocked',
                command='python -B scripts/record_prepared_y86_y87_wait.py --token ' + token,
                observation='Fresh canonical physical occupation and native ownership confirm both permitted GPU0 cards busy/reserved. Exact Y86/Y87 CPU versions and GPU0-only plans remain unchanged. No replay, GPU task, admission, archive or formal metric.',
                evidence=[ref(out), occupations['server-2']])
            if occupations['server-2'] not in get('AdaRES-Y86')['evidence']:
                get('AdaRES-Y86')['evidence'].append(occupations['server-2'])
        tick.save(ROOT / 'pipeline/research_work.json', work)
        memory = ROOT / 'pipeline/codex_memory.md'
        backup = ROOT / ('pipeline/snapshots/codex_memory_before_' + token + '.md')
        assert not backup.exists()
        backup.write_bytes(memory.read_bytes())
        old = memory.read_text(encoding='utf-8').splitlines()
        lines = old[:old.index('## Last Tick')]

        def e(evidence):
            return evidence['path'] + ' SHA' + evidence['sha256']

        def phase(cid):
            value = phases[cid]
            return cid + ' ' + value['phase'] + str(value['epoch']) + '/30 step' + str(value['step']) + '/' + str(value['count'])

        for index, line in enumerate(lines):
            if line.startswith('- Current token '):
                lines[index] = '- Current token ' + token + '; snapshot ' + view['generated_at'] + '; deadline ' + view['deadline_at'] + '; never extend.'
            elif line.startswith('- Exact previous seven-section memory:'):
                lines[index] = '- Exact previous seven-section memory: ' + e(ref(backup)) + '; all historical bans/negative evidence/special recovery requirements retained.'
            elif line.startswith('- AdaRES-Y84 ') or line.startswith('- AdaRES-Y85 '):
                cid = line.split()[1]
                lines[index] = '- ' + phase(cid) + '; ' + tasks[cid]['server'] + ' GPU0 PID' + str(tasks[cid]['pid']) + '; original source/YAML/receipt/marker/compute/power/native increase: ' + e(observations[cid])
            elif line.startswith('- Fresh occupation:'):
                lines[index] = '- Fresh occupation: ' + e(occupations['server-2']) + '; ' + e(occupations['server-1']) + '; both permitted GPU0 busy/reserved. Finish needs age<300s.'
            elif line.startswith('- Next tick observe Y84/Y85'):
                lines[index] = '- Next tick first observe Y84/Y85 and dispatch ready work; Y86 precedes Y87. Both complete CPU preparations/retained seven-file versions/GPU0-only native plans remain unadmitted. Only fresh authorized exclusive idle/unreserved matching GPU0 permits actual native PASS then original full freeze/receipt/safe deployment/increasing counters. No CPU/retain/plan or archived terminal replay.'
        if new_preparation:
            where = lines.index('## Closed Or Failed Directions')
            lines[where:where] = [
                '- Y87 self_language_product: independent W2, six original self+language+(self*language)@zeroW256x256 before SAME active norm_qqa and original stage FFNs; +65536/total164695173/no RNG. Original coarse/fine half-average, qla/SCG/sampling/inputs/2D/backbone/static masks/gates/shared confidence heads/targets/loss/selector/seed1999/fixed30 preserved. Not Y82 coarse/fine fusion/Y74 gate/Y75 initial query/Y83 norm untying; rejected Y26 norm reconnection not reopened.',
                '- Y87 actual CPU native full model/backbone/TRAIN162738 points8 expressions, float32/64 independent sum/product scalar/zero/RNG/input-gradient/permutation/swap/same-sum controls, seven-stage exact W2 and input-gradient anchors, original self/language and original norm/FFN consumers, nonzero intervention/bypass and two synthetic optimizer steps passed. No native sparse GPU task, admission or efficacy.',
                '- Y87 CPU: ' + e(preparations['AdaRES-Y87']['CPU_result']) + '; version: ' + e(preparations['AdaRES-Y87']['version']) + '; commit=' + preparations['AdaRES-Y87']['code_commit'] + ', parent=' + preparations['AdaRES-Y86']['code_commit'] + '; only seven tested A files retained using isolated index; main HEAD/index and production unchanged.',
                '- Y87 GPU0-only preparation: ' + e(preparations['AdaRES-Y87']['native_preparation']) + '; nine guards/fourteen exact actually CPU-executed callbacks/stale goal-source refusal/Python3.8 passed. No GPU1 plan; Y86 precedes Y87, no CPU/retain/plan replay.']
        status = '; '.join(phase(cid) for cid in phases)
        lines += ['## Last Tick',
            '- ' + token + ': ' + status + '; exact original native ownership/source/YAML/receipt/marker/power and increasing counters verified.',
            '- New candidates' + str(int(new_preparation)) + '/admissions0/launches0/archives0; Y86/Y87 GPU0-only readiness preserved; both authorized cards busy. No formal FIRST/cost or scientific promotion.',
            '- ' + ('New Y87 full CPU/version/native-plan preparation completed;14 exact callbacks/9 guards; no native GPU PASS. ' if new_preparation else 'Unchanged prepared reserve: no repeated CPU/native-plan/terminal checks. ') + 'Handoff: ' + e(ref(out)),
            '- GPU1 all project use and Y57 separate pause remain. Original27 fixed30 models/full frozen review resources/goal bytes/receipts/data/runtime/mappings/seed1999-fixed30/bans/schedule preserved.',
            '- Official report/finish then read only FINALIZED/PASS/context_committed/released and absent owned lock/pending/archive_pending; no preclaims or post-release writes.']
        assert len(lines) < 120 and sum(line.startswith('## ') for line in lines) == 7
        memory.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
        notes = ROOT / ('pipeline/checks/foreground_notes_' + token + '.json')
        assert not notes.exists()
        actions = ['运行：' + status + '，原source/YAML/receipt/marker/功耗及训练步数递增核实。',
            'Y86/Y87精确版本与GPU0-only原生计划保留，两授权GPU0均busy+reserved；无GPU检查/准入/启动/归档/正式FIRST或cost。']
        if new_preparation:
            actions.append('新准备：Y87独立self/language乘积投影保留原求和、LayerNorm和FFN路径，原native CPU model/backbone/TRAIN、float32/64标量/零RNG/七阶段W2及输入梯度/原消费者/两合成优化步通过；七测试A文件留版，14精确已执行回调与9原生计划拒绝守卫通过。尚未GPU检查/准入。')
        tick.save(notes, dict(actions=actions, decisions=[
            'Y86优先Y87，等原匹配GPU0自然空闲并解除canonical reservation后才做真实native两步；CPU不代替原生准入。',
            'GPU1全部使用继续暂停，原GPU1冻结后评/cost不迁移替换，Y57单独暂停；不重复索取常规操作授权。',
            '保持原goal字节/receipt/数据/私有环境/mapping/seed1999-fixed30/禁用/现有功耗及调度，无需用户处理。']))
        assert (qref, rref) == (ref('pipeline/queue.json'), ref('pipeline/running.json'))
        print(json.dumps(dict(handoff=ref(out), notes=notes.relative_to(ROOT).as_posix(),
                              memory_lines=len(lines), new_candidates=int(new_preparation), admissions=0)))
    finally:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    parser.add_argument('--new-preparation', action='store_true')
    args = parser.parse_args()
    record(args.token, args.new_preparation)
