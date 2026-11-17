"""Retain verified Y75 closure, Y78 launch and independent CPU-only Y80 preparation."""
import argparse
import ast
import copy
from datetime import datetime, timedelta, timezone
import json
import subprocess
from prepare_y80_query_visual_feedback import ROOT, lease, monitor, ref, tick
from check_ef_execution import validate_frozen
import closure
import gpu_occupation

TOKEN = 'eee3af894c7e4816a93bb3d1e7770040'
HEAD = '5eee1e50c9e1d65bf113a4b8757bfc27d028a821'


def get_candidate(work, cid):
    return next(c for c in work['candidates'] if c['id'] == cid)


def holds():
    work = tick.load(ROOT / 'pipeline/research_work.json')
    y75, y78 = [get_candidate(work, cid) for cid in ('AdaRES-Y75', 'AdaRES-Y78')]
    assert y75['stage'] == 'validation' and y78['stage'] == 'training'
    assert monitor.load_experiment(str(ROOT), 'AdaRES-Y75')['status'] == 'done_complete'
    proof = tick.load(closure.artifact(ROOT, y75['terminal_evidence']))
    assert proof['completed_epochs'] == {phase: list(range(1, 31)) for phase in ('Train', 'Val')}
    assert proof['CPU_guard_tests_passed'] == 20 and proof['checkpoints'][0]['epoch'] == 30
    freeze = tick.load(closure.artifact(ROOT, y75['post_training_freeze']))
    resource = freeze['resource_condition']
    assert (resource['server'], resource['gpu'], resource['power_limit_watts']) == ('server-2', 1, 390.)
    assert monitor.load_running(str(ROOT))['server-2']['1'] == 'AdaRES-Y78'
    observation = tick.load(closure.artifact(ROOT, y78['latest_training_observation']))
    assert observation['token'] == TOKEN and observation['native_progress_confirmed'] and observation['pid'] == 22192
    out = ROOT / ('pipeline/checks/y75_frozen_evaluation_hold_' + TOKEN + '.json')
    assert not out.exists()
    tick.save(out, dict(complete=True, token=TOKEN, id='AdaRES-Y75', original_terminal=y75['terminal_evidence'],
        original_freeze=y75['post_training_freeze'], frozen_resource=resource, slot_owner='AdaRES-Y78', owner_progress=y78['latest_training_observation'],
        correction='Bind required_gpu metadata to original immutable GPU1/390W evaluation condition; terminal helper left0 while its next_action and original freeze require1.',
        frozen_bytes_changed=False, evaluation_executed=False, helper=ref(__file__)))
    y75.update(stage='blocked', blocked_from_stage='validation', blocker_kind='exclusive_gpu_slot',
        required_server='server-2', required_gpu=1, resource_work_kind='frozen_evaluation',
        blocker='Original fixed30 verified and canonically archived. Original complete frozen FIRST/evaluation/cost requires server-2 GPU1/390W, now occupied by original independent Y78; preserve that run.',
        retry_after=(datetime.now().astimezone() + timedelta(hours=1)).isoformat(), wake_paths=['pipeline/queue.json', 'pipeline/running.json', 'SERVERS.yaml'],
        unblock_condition='Fresh authorized exclusive server-2 GPU1/390W idle/no compute/no canonical reservation after full owner terminal archive; preserve all original evaluation boundaries.',
        next_action='Preserve original last30/best23/model/optimizer544/scheduler30 and23, immutable freeze/receipt/full native log. Run only the original full frozen FIRST/repeats/paired methods/bootstrap/subgroup/cost on exclusive original GPU1/390W. No restart or promotion from training best.')
    y75['evidence'].append(ref(out))
    y79 = get_candidate(work, 'AdaRES-Y79')
    y79['blocker'] = 'All authorized cards run original independent Y76/Y77/Y78; actual native TRAIN task requires an exclusive matching idle/unreserved GPU.'
    y79['next_action'] = 'Do not replay completed CPU/model/TRAIN/version/native-plan/callback preparation. Preserve HEAD5eee and original Y76/Y77/Y78 plus paused Y57. Y79 precedes Y80. Only canonical slot-owner fixed30 archive plus fresh matching server2GPU0/280W or GPU1/390W exclusive idle/no compute/no reservation permits its exact frozen native YAML/tools.train TRAIN two-step task. Actual finite task loss/backward/optimizer and original query/language/padding/qla/next-lla/shared-head paths must PASS before original complete evaluation/cost freeze, new receipt/admission, exact e0c source contract, safe production selection/sync, canonical dispatch and increasing steps. Server1 needs a separate private-runtime/path/350W plan. No GPU PASS or efficacy from CPU checks.'
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(hold=ref(out), original_freeze_preserved=True)))


def save():
    view = tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')
    context = tick.load(ROOT / 'pipeline/headless/.tick_context.json')
    work = tick.load(ROOT / 'pipeline/research_work.json')
    get = lambda cid: get_candidate(work, cid)
    queue = monitor.load_queue(str(ROOT))
    running = monitor.load_running(str(ROOT))
    qref, rref = ref('pipeline/queue.json'), ref('pipeline/running.json')
    tasks = {t['id']: t for ts in queue['gpu_queues'].values() for t in ts}
    observations, phases = {}, {}
    for cid, server, gpu, pid in [('AdaRES-Y76', 'server-1', '0', 3489002), ('AdaRES-Y77', 'server-2', '0', 111334), ('AdaRES-Y78', 'server-2', '1', 22192)]:
        task = tasks[cid]
        assert task['status'] == 'running' and task['launch_verified'] and task['pid'] == pid and running[server][gpu] == cid
        closure.admission(ROOT, task)
        evidence = get(cid)['latest_training_observation']
        value = tick.load(closure.artifact(ROOT, evidence))
        assert value['token'] == TOKEN and value['pid'] == pid and value['native_progress_confirmed'] and evidence['path'].endswith('_final.json')
        observations[cid], phases[cid] = evidence, value['after']['latest']
    occupations = {}
    for name in ('server-2', 'server-1'):
        evidence = ref('pipeline/checks/gpu_occupation_' + name + '_finish_' + TOKEN + '.json')
        value = tick.load(closure.artifact(ROOT, evidence))
        assert value['token'] == TOKEN and (datetime.now(timezone.utc) - datetime.fromisoformat(value['observed_at'])).total_seconds() < 300
        candidate = next(c for c in work['candidates'] if c.get('stage') == 'blocked' and c.get('blocker_kind') == 'exclusive_gpu_slot' and c.get('required_server') == name)
        server = next(s for s in context['queue']['servers'] if s['name'] == name)
        assert not gpu_occupation.validated_state(ROOT, context, candidate, server, queue)['idle_gpu_ids']
        occupations[name] = evidence
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip() == HEAD
    assert ref('pipeline/goal.json')['sha256'] == '4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a'
    paused = tasks['AdaRES-Y57']
    assert paused['status'] == 'paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']
    y75, y78, y80 = [get(cid) for cid in ('AdaRES-Y75', 'AdaRES-Y78', 'AdaRES-Y80')]
    assert 'AdaRES-Y75' not in tasks and y75['stage'] == 'blocked' and y75['required_gpu'] == 1
    assert monitor.load_experiment(str(ROOT), 'AdaRES-Y75')['status'] == 'done_complete'
    terminal = tick.load(closure.artifact(ROOT, y75['terminal_evidence']))
    assert terminal['completed_epochs'] == {phase: list(range(1, 31)) for phase in ('Train', 'Val')}
    closure.artifact(ROOT, y75['terminal_archive_binding'])
    raw = tick.load(closure.artifact(ROOT, y78['native_smoke']))
    assert raw['steps'] == raw['optimizer_steps'] == 2 and raw['residual_calls'] == raw['hard_gate_consumption_calls'] == 12 and raw['shared_head_calls'] == 14
    freeze = tick.load(closure.artifact(ROOT, y78['post_training_freeze']))
    parent = tick.load(closure.artifact(ROOT, freeze['parent_protocol']))
    restored = copy.deepcopy(freeze['evaluation'])
    restored['methods'][0] = 'AdaRES-Y64'
    assert restored == parent['evaluation'] and freeze['training'] == parent['training']
    for cid in ('AdaRES-Y79', 'AdaRES-Y80'):
        candidate = get(cid)
        assert candidate['stage'] == 'blocked' and candidate['implementation_test_passed'] and candidate['blocker_kind'] == 'exclusive_gpu_slot'
        for field in ('source_manifest', 'CPU_native_observer_verification', 'native_preparation'):
            closure.artifact(ROOT, candidate[field])
        for field in ('native_smoke_plan_server2_gpu0', 'native_smoke_plan_server2_gpu1'):
            plan = tick.load(closure.artifact(ROOT, candidate[field]))
            frozen = ast.literal_eval(ast.parse(closure.artifact(ROOT, plan['program']).read_text(encoding='utf-8')).body[0].value)
            assert validate_frozen(candidate, plan, frozen) == 7
    out = ROOT / ('pipeline/checks/resource_handoff_' + TOKEN + '.json')
    assert not out.exists()
    tick.save(out, dict(complete=True, token=TOKEN, snapshot=view['generated_at'], observed_at=datetime.now().astimezone().isoformat(),
        original_live_training=observations, occupations=occupations, Y75_terminal=y75['terminal_evidence'], Y75_archive=y75['terminal_archive_binding'],
        Y75_original_freeze=y75['post_training_freeze'], Y75_original_evaluation_resource=ref('pipeline/checks/y75_frozen_evaluation_hold_' + TOKEN + '.json'),
        Y78_native_smoke=y78['native_smoke'], Y78_receipt=ref('pipeline/admissions/AdaRES-Y78.json'), Y78_freeze=y78['post_training_freeze'],
        Y78_source_selection=ref('pipeline/checks/adares-y78_source_selection_' + TOKEN + '.json'), Y78_dispatch=ref('pipeline/checks/adares-y78_dispatch_' + TOKEN + '.json'),
        Y80_CPU_version=y80['source_manifest'], Y80_CPU_callback_verification=y80['CPU_native_observer_verification'], Y80_native_preparation=y80['native_preparation'],
        source_HEAD=HEAD, server2_source_marker=HEAD, server1_source_marker='c3849ead0b8e4f9320d3e353d561016b3b495ce1',
        new_candidates=1, new_admissions=1, new_dispatches=1, archived=['AdaRES-Y75'], prepared_native_pending=['AdaRES-Y79', 'AdaRES-Y80'],
        Y57_pause_preserved=True, original_complete_evaluation_preserved=True, formal_metrics_updated=False, helper=ref(__file__)))
    for cid in ('AdaRES-Y75', 'AdaRES-Y76', 'AdaRES-Y77', 'AdaRES-Y78', 'AdaRES-Y79', 'AdaRES-Y80'):
        get(cid)['evidence'].append(ref(out))
    result = ref('pipeline/analyses/y80_native_observer_cpu_' + TOKEN + '.json')
    work['tick_attempt'] = dict(tick_started_at=view['generated_at'], candidate_id='AdaRES-Y80', kind='implementation_test', result='passed',
        command='python -B scripts/run_experiment_first_probe.py y80_native_observer_cpu pipeline/checks/y80_native_observer_CPU_' + TOKEN + '_program.py --token ' + TOKEN + ' --seconds 480',
        observation='Exact fifteen frozen native functions/classes executed on two synthetic CPU decoder optimizer steps after original native full model/pretrain/TRAIN loading; finite original/new/input gradients and real updates. Native GPU task/admission/efficacy remain untested for Y80.',
        evidence=[y80['CPU_native_observer_verification'], result])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    memory = ROOT / 'pipeline/codex_memory.md'
    backup = ROOT / ('pipeline/snapshots/codex_memory_before_' + TOKEN + '.md')
    assert not backup.exists()
    backup.write_bytes(memory.read_bytes())
    def e(value):
        return value['path'] + ' SHA' + value['sha256']
    lines = memory.read_text(encoding='utf-8').splitlines()
    lines = lines[:lines.index('## Last Tick')]
    updated = []
    for line in lines:
        if line.startswith('- 当前token '):
            line = '- 当前token ' + TOKEN + '，snapshot' + view['generated_at'] + '，deadline' + view['deadline_at'] + '不延长；postflight仅认官方同token收据。'
        elif line.startswith('- 实际CODEX_THREAD_ID=owner='):
            line = '- 实际CODEX_THREAD_ID=owner=01a07e32-78bf-7670-925b-370911ea48db已核；Pipeline-AdaRES_project Disabled。experiment_first enabled；snapshot admission2/creation2；本轮新增候选1/准入1/启动1/归档1：Y75完整fixed30归档、Y78实际native PASS后训练、Y80 CPU准备。无正式FIRST/cost更新。'
        elif line.startswith('- 前轮完整记忆精确备份：'):
            line = '- 前轮完整记忆精确备份：' + e(ref(backup)) + '；历史机制/禁用/专用恢复与原证据沿备份保留。'
        elif line.startswith('- Y78 version：'):
            line = line.replace('未执行GPU native任务，不能入队。', '本轮实际GPU1原native PASS后准入训练，见新增原始证据。')
        elif line.startswith('- Y78优先Y79；'):
            line = '- Y78本轮已实际GPU1 native PASS并训练；Y79优先Y80，二者blocked/validation/exclusive_gpu_slot/server2GPU0，GPU1备选。保留完整CPU/model/data/version/plans/callback；canonical匹配owner完整归档+fresh idle/no compute/no reservation后原native两步PASS再原完整freeze/新准入/精确source契约/安全派发。server1需另备private-runtime/path/350W方案。'
        elif line.startswith('- Y75原专用终态准备：'):
            line = '- Y75本轮专用finalize_y75_fixed30.py核20生命周期守卫、原源/YAML/receipt/freeze与稳定finite last30/best23/model/optimizer544/scheduler30和23、完整Train/Val1..30，显式GPU1 canonical done_complete归档；原目录20261101_020954全部产物保留。terminal：' + e(y75['terminal_evidence']) + '；archive：' + e(y75['terminal_archive_binding']) + '。训练best .5044635534286499不是FIRST。'
        elif line.startswith('- AdaRES-Y75 server'):
            continue
        elif line.startswith('- Y78/Y79实际原生GPU任务尚未执行'):
            line = '- Y79/Y80实际原生GPU任务尚未执行，不能准入；原完整复评/cost仍需原资源。不存在新的项目内常规操作授权缺口，Y57继续单独暂停。'
        elif any(line.startswith('- ' + cid + ' server') for cid in observations):
            cid = line.split()[1]
            phase = phases[cid]
            resource = {'AdaRES-Y76': 'server1GPU0/350W PID3489002', 'AdaRES-Y77': 'server2GPU0/280W PID111334', 'AdaRES-Y78': 'server2GPU1/390W PID22192'}[cid]
            line = '- ' + cid + ' ' + resource + ' 原源码/YAML/receipt/marker/计算归属/功耗与native递增核验通过：' + phase['phase'] + str(phase['epoch']) + '/30 step' + str(phase['step']) + '/' + str(phase['count']) + '；' + e(observations[cid]) + '。'
        elif line.startswith('- 主master/server2 marker'):
            line = '- 主master/server2 marker5eee1e50c9e1d65bf113a4b8757bfc27d028a821，server1仍c3849ead0b8e4f9320d3e353d561016b3b495ce1。本轮仅选择并安全追加七已保留且实际测试Y78 A文件，其他index/工作文件/原Y76/Y77运行保持。现用scripts/y78_deployment.py observe --id AdaRES-Y76|AdaRES-Y77|AdaRES-Y78 --token NEW_TOKEN --label progress；旧y77及更早observer不兼容当前HEAD，勿调用；旧签名helpers保持。'
        elif line.startswith('- 原18项'):
            line = '- 19项Y56/Y58..Y75已fixed30完成并等待各自原完整FIRST/cost，保留required server/GPU与原freeze。Y75原freeze要求server2GPU1/390W，当前Y78占用；已把候选required_gpu元数据由终态helper遗留0校正回原1，不改原freeze。'
        elif line.startswith('- 本轮fresh canonical server2 occupation：') or line.startswith('- fresh canonical server2 occupation：'):
            line = '- 本轮fresh occupation server2：' + e(occupations['server-2']) + '；server1：' + e(occupations['server-1']) + '；三授权卡busy+reserved。finish前须<300秒，超时新label，未知/失败/缺设备/idle不能证明hold。'
        updated.append(line)
        if line.startswith('## Active Hypotheses'):
            updated += [
                '- Y78本轮真实server2GPU1/390W原生YAML/tools.train/TRAIN两步PASS：loss4.615510940551758/6.302280902862549，原四场景16表达、134280/350887点；每步540有限梯度/10原头梯度/7共享confidence-query_norm/6残差与原coarse gate消费者、query/raw/previous/padding身份与previous未detach、独立公式/首零次响应/真实新更新通过，峰6855589888B。不是全量显存或效益保证。',
                '- Y78 smoke：' + e(y78['native_smoke']) + '；17准入/资源/源版本负守卫通过；Y64完整evaluation/training深复制核等。receipt：' + e(ref('pipeline/admissions/AdaRES-Y78.json')) + '；freeze：' + e(y78['post_training_freeze']) + '。',
                '- Y80=adares_y80_query_visual_feedback：独立W2，六次visual_next=visual_ffn+clean((softmax(visual_ffn@current_query.T/sqrt(d))@current_query)@W)，共享零W256x256/no RNG，+65536/总164695173。原ila_ffn输出后、coarse/fine前；visual padding残差置0、全部128query/visual channel/未detach，static mask及原stage0/sampling/geometry/Boolean gate/SCG/k/gamma/七共享头/所有数据-输入-2D-loss/seed1999/fixed30/YAML保持。与Y79改语言流、Y72改mask特征不同，不重开拒绝路线，不宣称新信息/创新/效益。',
                '- Y80实际CPU原生完整model/backbone/TRAIN162738点8表达、float32/64独立标量softmax-context-projection/zero-RNG/置换-padding-allpadding/同query均值不同分布/错query配对、全部七阶段W2输出与输入梯度锚点、六原query/visualFFN/padding身份、六qia与五next-lia及七原共享头消费者、非零/bypass通过；两合成CPU DEC原optimizer步各387有限梯度/10头梯度/有限新梯度和更新。raw：' + e(ref('pipeline/analyses/y80_cpu_' + TOKEN + '.json')) + '。',
                '- Y80七精确测试A文件隔离index保留commit4e68cd84d56d890222adc2bec8ab59cb0c8715b6，branch refs/heads/adares-y80-query-visual-feedback，parentY79/e0c；模型独立W2不组合Y79。主HEAD5eee/index/queue/running保持；version：' + e(y80['source_manifest']) + '。',
                '- Y80原native两步双卡计划/11证据守卫/goal-七源码YAML-version与stale/tamper拒绝通过；15冻结native函数/类与CPU wrapper AST相同且两合成CPU优化步实际执行通过。preparation：' + e(y80['native_preparation']) + '；verification：' + e(y80['CPU_native_observer_verification']) + '。无CUDA/native sparse task/GPU PASS/准入，原生GPU检查等待Y79之后。',
            ]
        if line.startswith('## Closed Or Failed Directions'):
            updated.append('- Y80实际CPU及原native回调首次执行通过；元数据核验helper把15函数数目的宽文本替换误及162738点为152738，初次assert失败；attempt2仅修元数据预期，原模型/计划/实测结果不变，失败helper/repair证据保留于CPU verification。')
    if not any(line.startswith('- AdaRES-Y78 server') for line in updated):
        index = updated.index('## Running And Queue') + 1
        phase = phases['AdaRES-Y78']
        updated.insert(index, '- AdaRES-Y78 server2GPU1/390W PID22192 原source/YAML/receipt/功耗/native步数已核：Train' + str(phase['epoch']) + '/30 step' + str(phase['step']) + '/' + str(phase['count']) + '；' + e(observations['AdaRES-Y78']) + '。')
    updated += [
        '## Last Tick',
        '- ' + TOKEN + ' begin/design先查并派发waiting0，Y75在Val30自然结束；核原专用plan依赖/receipt/freeze，完整fixed30 canonical显式GPU1归档。fresh idle/unreserved390W后Y78精确原native两步PASS、七已测试A文件source选择/17准入守卫/原完整冻结后评、新receipt/安全server2-only派发PID22192，native Train1实际递增；原Y76/Y77保持。',
        '- 按待跑准备缺口新增独立Y80，实际CPU model/backbone/TRAIN与控制/两优化步、隔离保留版本、双卡计划11守卫/15原native回调CPU执行通过；无Y80 GPU PASS。两待检查Y79/Y80等授权空槽，优先Y79。Y75原完整后评GPU1等Y78资源，无正式FIRST/cost。',
        '- 本轮新候选1/准入1/启动1/归档1；handoff：' + e(ref(out)) + '；tick_attempt绑定Y80本轮实际CPU结果。主HEAD/server2 marker5eee，server1仍c384；新y78_deployment observer为后续唯一精确现用版本，不改旧hash-bound helpers。',
        '- 下轮先用y78_deployment核三原运行并派发现有任务。Y76当前20/30：若后续自然终态，须先准备其独立完整server1/private-runtime/path/350W终态契约与生命周期守卫，不套用Y75；完整归档后优先推进已有Y79原native GPU计划，server1需要单独精确适配。保留原数据协议/禁用/功耗/Y57单独暂停/原调度。',
        '- 官方report/finish后仅只读核FINALIZED/PASS/context_committed/released及foreground锁/pending/archive_pending；此记忆不预签结果，释放后不写。',
    ]
    assert len(updated) < 120 and sum(line.startswith('## ') for line in updated) == 7
    memory.write_text('\n'.join(updated) + '\n', encoding='utf-8', newline='\n')
    notes = ROOT / ('pipeline/checks/foreground_notes_' + TOKEN + '.json')
    assert not notes.exists()
    tick.save(notes, dict(actions=[
        '完成：Y75自然fixed30，原专用plan依赖/receipt/freeze、20生命周期守卫、完整Train/Val1..30与稳定finite last30/best23/model/optimizer544/scheduler30和23通过，显式GPU1 canonical归档，保留全部原checkpoint。训练best不作FIRST。',
        '运行：Y78在server2GPU1/390W实际原生YAML/TRAIN两步PASS，loss4.615510940551758/6.302280902862549，四原场景16表达；六残差/原coarse gates/七共享头与全部梯度更新核验通过，17准入守卫/原完整后评冻结/七精确源码版本选择/安全派发PID22192，native Train1步数递增。Y76 Train20、Y77 Train9原运行保持。',
        '新准备：独立Y80查询到视觉流反馈完成CPU原生模型/发布backbone/真实TRAIN及公式/zero-RNG/全部锚点/消费者/两合成CPU优化步；七测试文件隔离保留版本，双GPU检查计划11守卫与15冻结native回调CPU执行通过。Y80未执行GPU任务或准入。',
        '本轮新候选1/准入1/启动1/归档1，无正式FIRST/cost；19项完整fixed30后评保持原冻结协议。已备份短记忆并保存哈希handoff；Y80元数据wrapper点数常量笔误已修并保留原失败，模型与实测结果保持。'], decisions=[
        '三授权GPU现由原Y76/Y77及新Y78独立训练占用，fresh canonical occupation无空槽；Y79优先Y80等待各自原native检查，不挤占运行，不重复CPU准备。',
        'Y75原完整冻结后评/cost要求GPU1/390W；候选required_gpu元数据校正回原freeze的1，不改原冻结字节/receipt。效益待完整FIRST，不由训练日志或两步检查宣称。',
        '后续统一用y78_deployment精确observer支持新5eee HEAD，server1仍c384。Y76终态需独立server1环境/路径/350W完整契约与守卫，不套其他ID helper。',
        '保持原数据、seed1999/fixed30、全部禁用项、server2GPU0 280W/GPU1既有390W、server1GPU0既有350W，Y57单独暂停；禁server6/跨项目/调度变更。当前无新增用户处理事项。']))
    assert qref == ref('pipeline/queue.json') and rref == ref('pipeline/running.json')
    print(json.dumps(dict(handoff=ref(out), notes=str(notes.relative_to(ROOT)), memory_lines=len(updated), counts=dict(new=1, admitted=1, launched=1, archived=1))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('holds', 'save'))
    args = parser.parse_args()
    lease(TOKEN, 180)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        {'holds': holds, 'save': save}[args.operation]()
    finally:
        lock.close()
