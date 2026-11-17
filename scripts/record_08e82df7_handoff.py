"""Persist bounded Y79 preparation and unchanged live-job handoff before finish."""
from datetime import datetime, timezone
import json
import os
import subprocess
from prepare_y79_query_language_feedback import ROOT, lease, monitor, ref, tick
import closure
import gpu_occupation

TOKEN = '08e82df71e3743dcb86c0eec4d39460c'
HEAD = '498e721ea05e96aeb6f948d9cf8cd7a91b65ae22'


def main():
    lease(TOKEN, 180)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        assert os.environ['CODEX_THREAD_ID'] == '01a07e32-78bf-7670-925b-370911ea48db'
        context = tick.load(ROOT / 'pipeline/headless/.tick_context.json')
        view = tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')
        work = tick.load(ROOT / 'pipeline/research_work.json')
        get = lambda cid: next(c for c in work['candidates'] if c['id'] == cid)
        queue = monitor.load_queue(str(ROOT))
        running = monitor.load_running(str(ROOT))
        qref, rref = ref('pipeline/queue.json'), ref('pipeline/running.json')
        tasks = {t['id']: t for ts in queue['gpu_queues'].values() for t in ts}
        observations = {}
        for cid, name, gpu, pid in [('AdaRES-Y75', 'server-2', '1', 65454), ('AdaRES-Y76', 'server-1', '0', 3489002), ('AdaRES-Y77', 'server-2', '0', 111334)]:
            task = tasks[cid]
            assert task['status'] == 'running' and task['launch_verified'] and task['pid'] == pid and running[name][gpu] == cid
            closure.admission(ROOT, task)
            evidence = get(cid)['latest_training_observation']
            value = tick.load(closure.artifact(ROOT, evidence))
            assert value['token'] == TOKEN and value['pid'] == pid and value['native_progress_confirmed'] and evidence['path'].endswith('_final.json')
            observations[cid] = evidence
        occupations = {}
        for name in ('server-2', 'server-1'):
            evidence = ref('pipeline/checks/gpu_occupation_' + name + '_finish_' + TOKEN + '.json')
            value = tick.load(closure.artifact(ROOT, evidence))
            assert value['token'] == TOKEN and (datetime.now(timezone.utc) - datetime.fromisoformat(value['observed_at'])).total_seconds() < 300
            candidate = next(c for c in work['candidates'] if c.get('stage') == 'blocked' and c.get('blocker_kind') == 'exclusive_gpu_slot' and c.get('required_server') == name)
            server = next(s for s in context['queue']['servers'] if s['name'] == name)
            assert not gpu_occupation.validated_state(ROOT, context, candidate, server, queue)['idle_gpu_ids']
            occupations[name] = evidence
        paused = tasks['AdaRES-Y57']
        assert paused['status'] == 'paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']
        assert ref('pipeline/goal.json')['sha256'] == '4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a'
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip() == HEAD
        for cid in ('AdaRES-Y78', 'AdaRES-Y79'):
            candidate = get(cid)
            assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'exclusive_gpu_slot' and candidate['implementation_test_passed']
            for field in ('source_manifest', 'CPU_native_observer_verification', 'native_preparation', 'native_smoke_plan_server2_gpu0', 'native_smoke_plan_server2_gpu1'):
                closure.artifact(ROOT, candidate[field])
            candidate['blocker'] = 'All authorized cards run original independent Y75/Y76/Y77; preserve them. Actual native TRAIN task requires exclusive matching idle/unreserved authorized GPU.'
        terminal_ref = ref('pipeline/checks/y75_terminal_preparation_909e7705ed74484fbc9bb4d16c7dbb59.json')
        assert terminal_ref['sha256'] == '1e2ec654157b569a6fda192170e42737b7f80e6b216e14424b9ee1e82b624b33'
        terminal = tick.load(closure.artifact(ROOT, terminal_ref))
        for dependency in terminal['dependencies']:
            closure.artifact(ROOT, dependency)
        assert terminal['admission'] == ref('pipeline/admissions/AdaRES-Y75.json') and terminal['post_training_freeze'] == get('AdaRES-Y75')['post_training_freeze']
        closure.artifact(ROOT, terminal['post_training_freeze'])
        assert not terminal['checkpoint_loaded'] and not terminal['archived']
        assert work['tick_attempt']['tick_started_at'] == view['generated_at'] and work['tick_attempt']['candidate_id'] == 'AdaRES-Y79'
        y79 = get('AdaRES-Y79')
        out = ROOT / ('pipeline/checks/resource_handoff_' + TOKEN + '.json')
        assert not out.exists()
        tick.save(out, dict(complete=True, token=TOKEN, snapshot=view['generated_at'], observed_at=datetime.now().astimezone().isoformat(),
            original_live_training=observations, occupation=occupations, source_HEAD=HEAD,
            new_candidates=1, new_admissions=0, new_dispatches=0, archived=[], prepared_native_pending=['AdaRES-Y78', 'AdaRES-Y79'],
            Y79_CPU_version=y79['source_manifest'], Y79_CPU_native_callback_verification=y79['CPU_native_observer_verification'],
            Y79_native_preparation=y79['native_preparation'], Y75_original_terminal_preparation=terminal_ref, terminal_dependencies_unchanged=True,
            Y57_pause_preserved=True, queue_and_running_preserved=True, formal_FIRST_metrics_updated=False,
            conclusion='Y79 CPU/full native model/TRAIN and exact callback preparation completed; no native GPU task, admission or training efficacy. All three original training jobs advance; Y78 then Y79 wait for exclusive resources.'))
        for cid in ('AdaRES-Y75', 'AdaRES-Y76', 'AdaRES-Y77', 'AdaRES-Y78', 'AdaRES-Y79'):
            get(cid)['evidence'].append(ref(out))
        tick.save(ROOT / 'pipeline/research_work.json', work)
        memory = ROOT / 'pipeline/codex_memory.md'
        backup = ROOT / ('pipeline/snapshots/codex_memory_before_' + TOKEN + '.md')
        assert not backup.exists()
        backup.write_bytes(memory.read_bytes())
        def e(value):
            return value['path'] + ' SHA' + value['sha256']
        phases = {cid: tick.load(ROOT / value['path'])['after']['latest'] for cid, value in observations.items()}
        lines = [
            '# AdaRES_project Research Memory',
            '## Goal And Phase',
            '- 当前token ' + TOKEN + '，snapshot' + view['generated_at'] + '，deadline' + view['deadline_at'] + '不延长；postflight仅认官方同token收据。',
            '- 实际CODEX_THREAD_ID=owner=01a07e32-78bf-7670-925b-370911ea48db已核；Pipeline-AdaRES_project Disabled。experiment_first enabled；snapshot admission2/creation2，本轮新增候选1/准入0/启动0/归档0。Y74为前轮完成，不重复归档；本轮无正式FIRST/cost更新。',
            '- 授权server1 GPU0既有350W；server2 GPU0保持280W/GPU1既有390W；每卡一独立实验。Y57实验暂停保留；禁server6/跨项目/改调度/改硬件功耗/驱逐/重启迁移受保护运行。',
            '- goal科学字节SHA4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a不变；最新AGENTS/SERVERS及pipeline/authorizations/server2_gpu1_20261029.json覆盖旧GPU1资源句；原收据/科学goal不改。',
            '- PROJECT_ROOT=D:/project，PYTHONDONTWRITEBYTECODE=1，PYTHONUTF8=1，python -B；密钥SSH可用，未读密码文件，不推定sudo。',
            '- 前轮完整记忆精确备份：' + e(ref(backup)) + '；历史机制/禁用/专用恢复/所有原native与终态证据沿备份保留。',
            '## Champion And Baseline',
            '- 目标mIoU .5150；历史CSV Y36 .5110。冻结FIRST：Y36 .510725438294、W2 .510107995189、B1 .502221764356、Y52 .509708428366、Y54 .506342952986。本轮无正式全量FIRST/成本更新；训练best不替代FIRST或晋升。',
            '- 原后评协议：9508表达/141场景、候选+Y36/W2/B1各两次、FIRST主结果、scene bootstrap10000/seed1999、TRAIN cuts1487/3325、原subgroup harm/mIoU/Acc25/50/activequeries/latency/memory和同卡cost。Y54大目标对Y36伤害CI[-.02678985,-.00070265]与Y52总体跨0保留；单seed不估seed方差。',
            '## Active Hypotheses',
            '- Y75原query初始化语言乘积，Y76原共享confidence的七stage偏置，Y77原query初始化geometry/mass乘积；各自独立W2，原数据/采样/2D/targets/loss/selector/seed1999/fixed30/YAML及原native PASS/freeze/receipt保持。不要重放activate/probe/admit/dispatch。',
            '- Y78=adares_y78_mask_logit_residual：六次raw_current_masks+tanh(current_query@w)*clean(previous_adjusted_masks)，共享零w[256,1]/no RNG，+256/总164629893。上一mask未detach，原初始预测/static mask/原Boolean gate算法/SCG/k/gamma/Q128/七共享头与所有输入loss保持；不是旧最终confidence历史读出/软gate，不宣称创新或效益。',
            '- Y78 CPU原生完整模型/发布backbone/原TRAIN162738点8表达、独立float32/64公式/zero/RNG/置换/padding/全部stage与输入梯度W2锚点、非零响应/bypass、两合成CPU DEC原optimizer有限梯度/真实更新通过。模型commit5eee1e50c9e1d65bf113a4b8757bfc27d028a821，parentY77/498e；独立W2，未组合Y77。',
            '- Y78 version：' + e(get('AdaRES-Y78')['source_manifest']) + '；native双卡计划/11守卫与14原回调CPU执行已完成，不重放。CPU verification：' + e(get('AdaRES-Y78')['CPU_native_observer_verification']) + '。未执行GPU native任务，不能入队。',
            '- Y79=adares_y79_query_language_feedback：原language-FFN输出后六次language_next=language_ffn+clean((softmax(language_ffn@current_query.T/sqrt(d))@current_query)@W)，共享零W[256,256]/no RNG，+65536/总164695173。language padded残差行置0；全部128query/全部channels/两输入未detach。与Y74 language→query gate、Y75 pooled-language初始query不同，不重开旧confidence/history/padding路线，不声称新信息/创新/效益。',
            '- Y79保留原initial prediction/static masks/sampling-geometry/所有输入与现有2D/发布backbone/6refine/原hard Boolean gates/SCG/k/gamma/七共享confidence头targets-loss-selector/Q128/seed1999/fixed30/native YAML；sqrt(d)为新attention标准固定scale，无temperature sweep。',
            '- Y79实际CPU native YAML/full model/pretrain/TRAIN162738点8表达、独立float32/64逐标量softmax/context/projection、zero输出与输入梯度/RNG/parent weights、query-token-batch-channel置换/padding/all-padding/wrong-query-batch/同均值不同分布、全部七阶段W2输出与输入梯度锚点、六原query/language-FFN/pad身份、六qla/五next-lla/七共享头与query_norm消费、非零效果/bypass通过。',
            '- Y79两合成CPU DEC原完整optimizer步每步387有限梯度/10原头梯度、两输入与新W有限梯度和真实更新通过；不是native sparse TRAIN任务。raw：' + e(ref('pipeline/analyses/y79_cpu_' + TOKEN + '.json')) + '。',
            '- Y79七精确测试A文件以隔离index保留commite0c7dba199fcf1a009941450d04aeb799418a7e6，branch refs/heads/adares-y79-query-language-feedback，parentY78/5eee；模型独立W2，不组合Y78。主HEAD/index/queue/running不变。version：' + e(y79['source_manifest']) + '。',
            '- Y79原生YAML/tools.train/TRAIN两步双GPU0-280W/GPU1-390W计划冻结，11证据负守卫/Python3.8 AST/goal-源码YAML-保留版本绑定与stale-goal/tampered-source拒绝通过。preparation：' + e(y79['native_preparation']) + '。未执行GPU。',
            '- Y79冻结native16函数/类与CPU wrapper AST完全相同且实际执行两合成CPU DEC optimizer步，六query/language/pad身份/未detach/float64公式/真实qla与next-lla/七共享头消费者、首零次响应/有限梯度及更新通过。verification：' + e(y79['CPU_native_observer_verification']) + '；明确无CUDA/native sparse task/GPU PASS/准入/效益结论。',
            '- Y78优先Y79；均blocked/validation/exclusive_gpu_slot/server2GPU0，GPU1备选。不要重放已完成CPU/model/data/version/plans/callback；canonical slot owner完整fixed30归档+fresh匹配idle/no compute/no reservation后，精确native两步PASS再原完整freeze/新准入/明确新增版本source契约/安全派发并核原生步数。server1须另备精确private runtime/path/350W方案，不能套server2程序。',
            '- Y75原专用终态准备：' + e(terminal_ref) + '；本轮核全部dependencies及原receipt/freeze哈希未变。原PID65454/GPU1/390W/声明目录20261101_020954，20生命周期守卫已通过；尚未checkpoint加载/完整终态验证/归档。自然fixed30退出后先核plan.dependencies/原receipt/freeze，再python -B scripts/finalize_y75_fixed30.py --token NEW_TOKEN；不重放prepare，不用别ID finalizer。',
            '## Closed Or Failed Directions',
            '- 保留Y53/Y55/Y50/Y51/Y49rev1/Y46/Y48等拒绝及goal bans；F1否决mask-entropy，禁val扫描entropy/margin/occupancy/threshold/temperature/loss/feature subsets/epoch数，不改名重开。旧Y22/Y23与全部专用恢复见AGENTS和精确备份。',
            '- Y78原两CPU wrapper失败.py/.log保持：float64置换逐位比较1.11e-16舍入在原128eps范围；decoder-only缺原text_encoder优化器group，attempt3完整model绑定通过。模型与YAML未改，revision0；不是机制否决或GPU PASS。Y79本轮CPU/model/原生回调检查首次即通过。',
            '- 所有已完成fixed30模型/原checkpoint/日志/freeze/receipt保留，不重训、不重放终态helper。原server1私有torch/CUDA11.3库、server2掉卡与专用恢复要求均不变。',
            '## Running And Queue',
        ]
        for cid, resource, pid in [('AdaRES-Y75', 'server2GPU1/390W', 65454), ('AdaRES-Y76', 'server1GPU0/350W', 3489002), ('AdaRES-Y77', 'server2GPU0/280W', 111334)]:
            phase = phases[cid]
            lines.append('- ' + cid + ' ' + resource + ' PID' + str(pid) + ' 本轮原源码/YAML/receipt/marker/计算归属与功耗核验通过；实际' + phase['phase'] + str(phase['epoch']) + '/30 step' + str(phase['step']) + '/' + str(phase['count']) + '且递增；evidence：' + e(observations[cid]) + '。')
        lines += [
            '- 主master/server2 marker498e721ea05e96aeb6f948d9cf8cd7a91b65ae22，server1 markerc3849ead0b8e4f9320d3e353d561016b3b495ce1。现用scripts/y77_deployment.py observe --id AdaRES-Y75|AdaRES-Y76|AdaRES-Y77 --token NEW_TOKEN --label progress；旧y76 observer与当前HEAD不兼容。部署Y78/Y79须另注册精确5eee/e0c契约，不改旧hash-bound helper。',
            '- 原18项Y56/Y58..Y74等待各自冻结完整FIRST/cost；保留required server/GPU，训练best不替代FIRST。Y74前轮已done_complete归档，不重复处理snapshot new_terminal。',
            '- fresh canonical server2 occupation：' + e(occupations['server-2']) + '；server1：' + e(occupations['server-1']) + '；三卡busy+reserved。finish前证据须<300秒，超时新label保留原证据；未知/失败/缺设备/idle不能证明hold。',
            '- server2=songcl@192.168.1.163/dl01-MS-7C60，root=/bak_home/songcl/Project/AdaRES_project，code=.../AdaRES，python=/home/songcl/.conda/envs/adares_project/bin/python，data=/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/。既有环境与原data保持。',
            '- server1=songcl@192.168.1.175/dl02-MS-7B73，root=/data1/songcl/Project/AdaRES_project，private python=.../runtime/adares_project/bin/python，sitepackages=.../lib/python3.8/site-packages，pointnet=.../runtime/pointnet2-0.0.0-py3.8-linux-x86_64.egg，LD_LIBRARY_PATH=.../runtime/cuda113lib64，data=.../data/，work=.../AdaRES_checkpoints；精确数据证明与仅SERVERS两path mappings保持。',
            '- Y57 paused/dispatch_hold/resume_requires_user_authorization，原20261021_004739/last.pth epoch27/model/optimizer543/scheduler27/best .5028846859931946保持；仅单独明确Y57继续授权后从28续至30，不迁移/不从pretrain重启。',
            '## Open Questions And Risks',
            '- 本轮无正式FIRST/完整成本或科学胜出结论；seed方差与server2掉卡/散热/Xid79根因仍未知；训练递增不是硬件修复证明。两步native PASS仅覆盖已测批次，CPU不是GPU验证。',
            '- Y78/Y79实际原生GPU任务尚未执行，不能准入；原全量复评/cost仍需原资源。不存在新的项目内常规操作授权缺口，不能重复索权或越过Y57明确暂停。',
            '- heartbeat保持原owner/原2小时频率，不改自动化/通知/调度，不建线程/嵌套codex exec；已签helper/负结果不可重写。',
            '## Last Tick',
            '- ' + TOKEN + ' begin/design先查进展并派发现有waiting0；三原实验核验与native步数增长。按快照待跑缺口新增独立Y79，实际原生YAML/model/backbone/TRAIN与CPU公式/控制/两optimizer步、七测试文件隔离版本、双卡native计划11守卫及16原回调CPU执行完成；未GPU检查/未准入/未启动/未归档。',
            '- handoff：' + e(ref(out)) + '；当前两待检查Y78/Y79均等授权独占资源，未宣称效益。Y75终态准备原依赖/receipt/freeze保持；本轮tick_attempt绑定Y79真实CPU结果。',
            '- 下轮先用y77_deployment精确observer核三运行并派发现有任务。Y75若自然fixed30退出，先核专用plan依赖/原receipt/freeze，再完整last30/best/model/optimizer/scheduler/full-log核验并canonical显式GPU1归档；fresh idle/unreserved390W后Y78原native两步PASS→原完整freeze/新准入/5eee源契约→安全派发。Y79随后；保持数据/禁用/功耗/Y57暂停/原调度。',
            '- 官方report/finish后只读核FINALIZED/PASS/context_committed/released及锁/pending/archive_pending。此记忆不预签结果，释放后不写。',
        ]
        assert len(lines) < 120 and sum(line.startswith('## ') for line in lines) == 7
        memory.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
        notes = ROOT / ('pipeline/checks/foreground_notes_' + TOKEN + '.json')
        assert not notes.exists()
        tick.save(notes, dict(actions=[
            '运行：原Y75/server2GPU1、Y76/server1GPU0、Y77/server2GPU0实际训练计数递增，PID/原源码YAML/收据/版本/功耗保持；最新Train25/30、15/30、3/30。',
            '新准备：新增独立Y79查询集合到语言反馈；原生YAML完整模型/发布backbone/真实TRAIN加载、公式和全部锚点控制及两合成CPU解码器优化步通过，七测试文件隔离保留版本，双GPU原生计划11守卫与16冻结回调CPU执行通过。',
            '完成：本轮新候选1/准入0/启动0/归档0，无正式FIRST/cost结果；GPU检查未执行，CPU成功不作native任务PASS或效益证明。',
            '已更新精确记忆备份、七节短记忆与哈希handoff；Y75原专用终态计划全部依赖/收据/freeze保持，未加载checkpoint或提前归档。'], decisions=[
            'fresh canonical occupation确认三授权卡均busy+reserved，Y78优先Y79等待匹配独占空槽；不挤占现有运行，不重复已通过的CPU和计划准备。',
            'Y75自然fixed30结束后按其专用完整终态程序canonical显式GPU1归档，再在fresh idle/unreserved390W上执行Y78精确原native两步；实际PASS后当轮原完整冻结后评协议/新准入/精确source契约/安全派发。',
            '保持科学goal与原收据、原数据协议/独立环境/部署映射、server2GPU0 280W与GPU1 390W、server1GPU0 350W；Y57独立实验继续暂停，server6禁止，不跨项目或修改调度。',
            '需处理：当前无新增用户操作事项；Y57继续暂停，无须重复确认已有项目内操作授权。']))
        assert qref == ref('pipeline/queue.json') and rref == ref('pipeline/running.json')
        print(json.dumps(dict(handoff=ref(out), notes=str(notes.relative_to(ROOT)), memory_lines=len(lines), counts=dict(new=1, admitted=0, launched=0, archived=0))))
    finally:
        lock.close()


if __name__ == '__main__':
    main()
