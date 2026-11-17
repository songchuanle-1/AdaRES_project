"""Record current Y80/Y81/Y82 progress and the retained Y83/Y84 native plans."""
import argparse
import ast
from datetime import datetime, timezone
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
import closure
import gpu_occupation


def record(token, new_events):
    lease(token, 180)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        view = tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')
        context = tick.load(ROOT / 'pipeline/headless/.tick_context.json')
        work = tick.load(ROOT / 'pipeline/research_work.json')
        get = lambda cid: next(c for c in work['candidates'] if c['id'] == cid)
        queue = monitor.load_queue(str(ROOT)); running = monitor.load_running(str(ROOT))
        qref, rref = ref('pipeline/queue.json'), ref('pipeline/running.json')
        assert closure.git(ROOT, 'rev-parse', 'HEAD') == '5773768ffb87d7fea7432e85fd0da561556aa2fc'
        assert ref('pipeline/goal.json')['sha256'] == '4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a'
        tasks = {t['id']: t for rows in queue['gpu_queues'].values() for t in rows}
        observations, phases = {}, {}
        for cid, name, gpu, pid in [('AdaRES-Y80', 'server-2', '0', 42226),
                ('AdaRES-Y81', 'server-2', '1', 120624), ('AdaRES-Y82', 'server-1', '0', 1121817)]:
            task = tasks[cid]
            assert task['status'] == 'running' and task['launch_verified'] and task['pid'] == pid
            assert running[name][gpu] == cid
            closure.admission(ROOT, task)
            evidence = get(cid)['latest_training_observation']
            value = tick.load(closure.artifact(ROOT, evidence))
            assert value['token'] == token and value['native_progress_confirmed'] and value['pid'] == pid
            observations[cid], phases[cid] = evidence, value['after']['latest']
        pause = tasks['AdaRES-Y57']
        assert pause['status'] == 'paused' and pause['dispatch_hold'] and pause['resume_requires_user_authorization']
        done = get('AdaRES-Y79')
        assert 'AdaRES-Y79' not in tasks
        experiment = monitor.load_experiment(str(ROOT), 'AdaRES-Y79')
        assert experiment['status'] == 'done_complete' and str(experiment['gpu']) == '0'
        terminal = tick.load(closure.artifact(ROOT, done['terminal_evidence']))
        binding = tick.load(closure.artifact(ROOT, done['terminal_archive_binding']))
        assert terminal['completed_epochs'] == {p: list(range(1, 31)) for p in ('Train', 'Val')}
        assert terminal['checkpoints'][0]['epoch'] == 30
        assert binding['only_exact_Y79_archived'] and binding['checkpoint_files_retained']
        assert not binding['full_frozen_evaluation_executed']
        admitted = get('AdaRES-Y82')
        for field in ('native_smoke', 'post_training_freeze'):
            closure.artifact(ROOT, admitted[field])
        receipt = ref('pipeline/admissions/AdaRES-Y82.json')
        prep = tick.load(closure.artifact(ROOT, get('AdaRES-Y84')['native_preparation']))
        version = tick.load(closure.artifact(ROOT, get('AdaRES-Y84')['source_manifest']))
        cpu = tick.load(closure.artifact(ROOT, version['CPU_result']))
        assert cpu['implementation_test_passed'] and cpu['synthetic_CPU_decoder_optimizer_steps'] == 2
        assert cpu['parameters'] == 164695173 and cpu['additional_parameters'] == 65536
        assert not cpu['CUDA_initialized'] and not cpu['native_smoke_passed']
        assert prep['guard_tests_passed'] == 8 and prep['exact_callback_functions_and_classes'] == 6
        assert prep['CPU_native_callbacks_actually_executed'] and not prep['native_executed']
        if new_events:
            assert terminal['token'] == prep['token'] == token and token in admitted['native_smoke']['path']
        occupations = {}
        for name in ('server-2', 'server-1'):
            evidence = ref('pipeline/checks/gpu_occupation_' + name + '_finish_' + token + '.json')
            value = tick.load(closure.artifact(ROOT, evidence))
            assert value['token'] == token
            assert 0 <= (datetime.now(timezone.utc) - datetime.fromisoformat(value['observed_at'])).total_seconds() < 300
            candidate = next(c for c in work['candidates'] if c.get('stage') == 'blocked' and c.get('blocker_kind') == 'exclusive_gpu_slot' and c.get('required_server') == name)
            server = next(s for s in context['queue']['servers'] if s['name'] == name)
            assert not gpu_occupation.validated_state(ROOT, context, candidate, server, queue)['idle_gpu_ids']
            occupations[name] = evidence
        plans = {}
        for cid in ('AdaRES-Y83', 'AdaRES-Y84'):
            c = get(cid)
            assert c['stage'] == 'blocked' and c['blocker_kind'] == 'exclusive_gpu_slot' and c['implementation_test_passed']
            for field in ('source_manifest', 'native_preparation', 'CPU_native_observer_verification'):
                closure.artifact(ROOT, c[field])
            plans[cid] = {}
            for field in ('native_smoke_plan_server2_gpu0', 'native_smoke_plan_server2_gpu1'):
                evidence = c[field]; plan = tick.load(closure.artifact(ROOT, evidence))
                program = closure.artifact(ROOT, plan['program'])
                frozen = ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
                assert validate_frozen(c, plan, frozen) == 7 and not plan['native_executed']
                plans[cid][field] = evidence
        out = ROOT / ('pipeline/checks/prepared_y83_y84_resource_' + token + '.json')
        assert not out.exists()
        count = int(new_events)
        tick.save(out, dict(complete=True, token=token, snapshot=view['generated_at'],
            original_live_training=observations, occupations=occupations, retained_native_plans=plans,
            Y79_terminal=done['terminal_evidence'], Y79_archive=done['terminal_archive_binding'],
            Y82_native=admitted['native_smoke'], Y82_receipt=receipt, Y82_full_freeze=admitted['post_training_freeze'],
            Y84_CPU=version['CPU_result'], Y84_version=get('AdaRES-Y84')['source_manifest'],
            Y84_native_preparation=get('AdaRES-Y84')['native_preparation'],
            new_candidates=count, new_admissions=count, new_launches=count, new_archives=count,
            Y83_Y84_native_GPU_executed=False, formal_metrics_updated=False, Y57_pause_preserved=True,
            helper=ref(__file__)))
        for cid in ('AdaRES-Y79', 'AdaRES-Y80', 'AdaRES-Y81', 'AdaRES-Y82', 'AdaRES-Y83', 'AdaRES-Y84'):
            get(cid)['evidence'].append(ref(out))
        work['tick_attempt'] = dict(tick_started_at=view['generated_at'], candidate_id='AdaRES-Y84',
            kind='implementation_test' if new_events else 'resource_check', result='passed' if new_events else 'blocked',
            command='python -B scripts/record_prepared_y83_y84_wait.py --token ' + token + (' --new-events' if new_events else ''),
            observation=('Recorded actual Y84 CPU model/backbone/original TRAIN loading, scalar controls, seven-stage W2/input-gradient anchors, two synthetic CPU optimizer steps and exact six callbacks; eight native plan rejection guards passed. Y79 verified fixed30 archive and Y82 actual native admission/launch retained. No Y84 native GPU task or efficacy.' if new_events else 'Fresh physical inventory/compute and canonical reservations verified; all authorized slots busy; original Y83/Y84 source/YAML/version/goal and native plans retained. No native task or efficacy.'),
            evidence=[ref(out), version['CPU_result'], get('AdaRES-Y84')['native_preparation']] if new_events else [ref(out), occupations['server-2']])
        tick.save(ROOT / 'pipeline/research_work.json', work)
        memory = ROOT / 'pipeline/codex_memory.md'
        backup = ROOT / ('pipeline/snapshots/codex_memory_before_' + token + '.md')
        assert not backup.exists(); backup.write_bytes(memory.read_bytes())
        def e(v): return v['path'] + ' SHA' + v['sha256']
        lines = memory.read_text(encoding='utf-8').splitlines()
        lines = lines[:lines.index('## Last Tick')]
        for i, line in enumerate(lines):
            if line.startswith('- 当前token '): lines[i] = '- 当前token ' + token + '，snapshot' + view['generated_at'] + '，deadline' + view['deadline_at'] + '不延长；postflight只认同token官方收据。'
            elif line.startswith('- 实际CODEX_THREAD_ID=owner='): lines[i] = '- 实际CODEX_THREAD_ID=owner=01a07e32-78bf-7670-925b-370911ea48db已核；Pipeline-AdaRES_project Disabled；experiment_first enabled。本轮新候选/准入/启动/归档各' + str(count) + '；无正式FIRST/cost。'
            elif line.startswith('- 前轮完整七节记忆精确备份：'): lines[i] = '- 前轮完整七节记忆精确备份：' + e(ref(backup)) + '；历史科学/禁用/专用恢复与全部原证据沿备份保持。'
            elif line.startswith('- AdaRES-Y79 server-1'): lines[i] = '- Y79已自然fixed30 done_complete归档，原20261102_230148/last30与best15/model/optimizer544/scheduler30-15均稳定finite，25生命周期守卫；训练best .500257670879364不是FIRST。terminal：' + e(done['terminal_evidence']) + '；archive：' + e(done['terminal_archive_binding']) + '。'
            elif line.startswith('- AdaRES-Y') and ' server' in line:
                cid = line.split()[1]
                if cid in observations:
                    v = phases[cid]; t = tasks[cid]
                    lines[i] = '- ' + cid + ' ' + t['server'] + ' GPU' + str(t['gpu']) + ' PID' + str(t['pid']) + ' 原source/YAML/receipt/marker/计算归属/功耗和native递增核验通过：' + v['phase'] + str(v['epoch']) + '/30 step' + str(v['step']) + '/' + str(v['count']) + '；' + e(observations[cid]) + '。'
            elif line.startswith('- 主master/server2 marker'): lines[i] = '- 主master/server1 marker5773768ffb87d7fea7432e85fd0da561556aa2fc，server2仍4157937d0c3cf344a44394ce15c95349aa94d50f；精确选择七已测试Y82 A文件，server1从e0c按既有安全追加契约同步Y80/Y81/Y82，旧源码保持。当前scripts/y82_deployment.py observe支持Y80/Y81/Y82；旧hash-bound helper不改。'
            elif line.startswith('- 22项Y56/Y58..Y78'): lines[i] = '- 23项Y56/Y58..Y79 fixed30已完成，仍待各自原冻结完整FIRST/cost；保留原required server/GPU/功耗，不重训、不用训练best代替FIRST。'
            elif line.startswith('- fresh occupation server2：'): lines[i] = '- fresh occupation server2：' + e(occupations['server-2']) + '；server1：' + e(occupations['server-1']) + '；三卡busy+reserved。finish须<300秒，陈旧时新label刷新；未知/失败/缺设备/idle不能证明hold。'
            elif line.startswith('- Y82/Y83未实际GPU'): lines[i] = '- Y83/Y84未实际GPU任务，不得准入；Y82原生两步已通过并启动。23项完成模型完整后评仍需原冻结资源；训练递增不证明server2历史掉卡根因已修复。'
            elif line.startswith('- 下轮先'): lines[i] = '- 下轮先用y82_deployment核原Y80/Y81/Y82及现有队列；Y83优先Y84原native任务等fresh匹配独占空槽。Y79已完整归档，不重放终态；原Y83/Y84 CPU和计划准备不重放。'
            elif line.startswith('- 旧record_prepared_'): lines[i] = '- 当前安静复核用record_prepared_y83_y84_wait.py --token NEW_TOKEN，不带--new-events；该开关仅本轮首次Y79归档/Y82启动/Y84准备使用。旧closeout/helper不改不重放，保留owner/调度/通知。'
            elif line.startswith('- Y82优先Y83；'): lines[i] = '- 当前Y83优先Y84；均已CPU/model/原TRAIN/version/双原native计划准备，blocked/validation/exclusive_gpu_slot/server2GPU0，GPU1原计划备选，无server1计划。fresh原功耗/独占idle/no compute/no reservation后原native两步PASS才准入/派发/递增核验。'
        if not any(line.startswith('- AdaRES-Y82 server-1') for line in lines):
            v = phases['AdaRES-Y82']; lines.insert(lines.index('## Open Questions And Risks'), '- AdaRES-Y82 server-1 GPU0 PID1121817，原350W与source/YAML/receipt/marker/compute归属/native递增核验通过：' + v['phase'] + str(v['epoch']) + '/30 step' + str(v['step']) + '/' + str(v['count']) + '；' + e(observations['AdaRES-Y82']) + '。')
        if not any(line.startswith('- Y84=') for line in lines):
            at = lines.index('## Closed Or Failed Directions')
            lines[at:at] = [
                '- Y82在1fdaad轮原server1GPU0/350W YAML/TRAIN两步PASS，loss4.615511417388916/6.364487648010254，峰reserved6702497792B；六原coarse/fine身份、未detach乘积投影/qla消费者、七共享head/query_norm、有限梯度/更新及原hard gate/static mask保持；8映射守卫/17准入守卫通过。receipt：' + e(receipt) + '；native：' + e(admitted['native_smoke']) + '；原完整后评freeze：' + e(admitted['post_training_freeze']) + '。',
                '- Y84=adares_y84_visual_query_init；独立W2，仅初始q=q_generated+point_mass_weighted_mean(all valid original inst_feats)@W；零W256x256/no RNG/+65536/总164695173。原全256channel/128query、完整原视觉输入/2D、坐标/sampling/SCG/k/gamma/静态mask/hard gate/shared heads/targets-loss-selector/seed1999-fixed30/YAML保持；不组合Y75/Y77/Y80，不重开Y55或其他拒绝。无缺失信息/创新/效益承诺。',
                '- Y84 CPU原model/backbone/TRAIN162738点8表达、float32/64逐标量加权视觉均值投影/零-RNG/空支持-零质量/padding NaN/置换/质量响应与全输入梯度、七阶段W2逐位与输入梯度锚点、原generated-query/scene-feature/padding/mass身份与初始prediction消费者、七共享head-query_norm/原hard gates及两合成CPU优化步通过。CPU：' + e(version['CPU_result']) + '；version：' + e(get('AdaRES-Y84')['source_manifest']) + '；commit7bef2cd313b870583c970c56580be5a78097234d，parentY83/281832，branch refs/heads/adares-y84-visual-query-init，隔离index保留七A文件，主HEAD577/index/三训练不变。',
                '- Y84双原native计划：' + e(get('AdaRES-Y84')['native_preparation']) + '；8拒绝守卫/陈旧goal-源码篡改拒绝/Python3.8 AST及六精确CPU-native回调一致且CPU已实际执行。无CUDA/native sparse TRAIN任务/GPU PASS/准入。Y83先于Y84，无server1计划。']
        status = '；'.join(cid + ' ' + v['phase'] + str(v['epoch']) + '/30 step' + str(v['step']) + '/' + str(v['count']) for cid, v in phases.items())
        lines += ['## Last Tick', '- ' + token + ' begin/design先查并尝试派发已有waiting；' + status + '，原资源与步数递增均实核。',
            '- 本轮新候选/准入/启动/归档各' + str(count) + '；' + ('Y79完整fixed30归档、Y82原native PASS→完整冻结协议/receipt→安全追加同步/派发/递增核验，Y84独立CPU/版本/双native计划准备。' if new_events else '仅当前资源/进度与原准备完整性复核。') + '无正式FIRST/cost。',
            '- 当前资源/准备handoff：' + e(ref(out)) + '。Y83优先Y84等fresh授权匹配原native任务，不重放CPU准备。',
            '- 保留科学goal字节/原receipt/数据协议/私有环境/精确mapping/禁用/seed1999-fixed30与350W-280W-390W；Y57单独暂停，禁server6/跨项目/改调度。',
            '- 官方report/finish后仅只读核FINALIZED/PASS/context_committed/released及无foreground lock/pending/archive_pending；此记忆不预签结果，释放后不写。']
        assert len(lines) < 120 and sum(line.startswith('## ') for line in lines) == 7
        memory.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
        notes = ROOT / ('pipeline/checks/foreground_notes_' + token + '.json'); assert not notes.exists()
        tick.save(notes, dict(actions=['运行：' + status + '，source/YAML/receipt/marker/compute归属/原功耗及native递增核验通过。',
            '完成：Y79原fixed30/完整Train-Val1..30与稳定finite last30/best15/model/optimizer544/scheduler30-15实核，显式server1GPU0归档，原产物保留；best .500257670879364不是FIRST。' if new_events else '原Y79归档证据保留，不重放终态。',
            '新准备：Y82原native两步通过、完整冻结后评与receipt入队，精确七文件版本选择/安全追加同步/启动及递增已实核；Y84独立初始视觉场景上下文连接CPU/两优化步/版本保存/双native计划8守卫和6实际回调检查通过，Y83/Y84待GPU，无Y84 GPU PASS。' if new_events else 'Y83/Y84原准备完整性与fresh三卡busy+reserved复核，不重放CPU/prepare。',
            '无正式FIRST/cost更新；记忆和精确前版备份保存。'],
            decisions=['三卡均原独立实验，无额外可用授权空槽；Y83优先Y84原native检查，实测PASS后才准入/派发并验证递增。',
                '保留原完整9508表达/141场景及候选+Y36/W2/B1各两次/FIRST/bootstrap10000-seed1999/TRAIN cuts1487-3325/原subgroup harm与同卡cost，训练best不替代FIRST。',
                'Y57单独暂停；原科学goal/数据/私有运行环境/mapping/禁用/350W-280W-390W/调度不变，无新增用户操作。']))
        assert qref == ref('pipeline/queue.json') and rref == ref('pipeline/running.json')
        print(json.dumps(dict(handoff=ref(out), notes=notes.relative_to(ROOT).as_posix(), memory_lines=len(lines),
            new_candidates=count, admissions=count, launches=count, archives=count)))
    finally:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--token', required=True)
    parser.add_argument('--new-events', action='store_true'); args = parser.parse_args()
    record(args.token, args.new_events)
