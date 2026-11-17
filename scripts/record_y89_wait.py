"""Close this bounded preparation with real tests and fresh physical occupation."""
import argparse
import ast
from datetime import datetime, timezone
import json
import subprocess
import sys
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from y89_contract import validate_cpu
from y88_deployment import Y88
import closure
import gpu_occupation


def record(token):
    work = tick.load(ROOT/'pipeline/research_work.json')
    candidate = next(candidate for candidate in work['candidates'] if candidate['id']=='AdaRES-Y89')
    assert candidate['stage']=='blocked' and candidate['blocker_kind']=='exclusive_gpu_slot'
    assert candidate['required_server']=='server-2' and candidate['required_gpu']==0
    assert not candidate.get('admission_receipt') and not candidate.get('native_smoke_result')
    cpu = tick.load(closure.artifact(ROOT,candidate['CPU_result']))
    validate_cpu(cpu)
    assert closure.git(ROOT,'rev-parse','HEAD')==Y88
    plans = [tick.load(closure.artifact(ROOT,candidate[key]))
             for key in ('native_smoke_plan_server1','native_smoke_plan_server2_gpu0')]
    assert {plan['server'] for plan in plans}=={'server-1','server-2'} and all(plan['gpu']==0 for plan in plans)
    source = tick.load(closure.artifact(ROOT,candidate['source_manifest']))
    assert source['main_index_unchanged'] and len(source['exact_delta'])==7
    assert all(row[0]=='A' for row in source['exact_delta'])
    assert closure.git(ROOT,'rev-parse',candidate['code_commit']+'^')==Y88
    tests = ROOT/('pipeline/checks/y89_final_preparation_guards_'+token+'.log')
    out = ROOT/('pipeline/checks/y89_preparation_closeout_'+token+'.json')
    notes = ROOT/('pipeline/reports/notes_'+token+'.json')
    backup = ROOT/('pipeline/snapshots/codex_memory_before_'+token+'.md')
    assert not any(path.exists() for path in (tests,out,notes,backup))
    response = subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','scripts',
        '-p','test_y89_*.py','-v'],cwd=ROOT,capture_output=True,timeout=60)
    tests.write_bytes(response.stdout+response.stderr)
    assert response.returncode==0 and b'Ran 16 tests' in tests.read_bytes()
    queue = monitor.load_queue(str(ROOT))
    running = monitor.load_running(str(ROOT))
    active, occupations = {}, {}
    now = datetime.now(timezone.utc)
    for name,cid,pid in (('server-1','AdaRES-Y88',3527278),('server-2','AdaRES-Y87',26281)):
        task = next(task for task in queue['gpu_queues'][name] if task['id']==cid)
        assert task['status']=='running' and task['launch_verified'] and task['pid']==pid and str(task['gpu'])=='0'
        assert running[name]['0']==cid
        closure.admission(ROOT,task)
        current = next(current for current in work['candidates'] if current['id']==cid)
        value = tick.load(closure.artifact(ROOT,current['latest_training_observation']))
        assert value['native_progress_confirmed'] is True
        active[cid] = dict(server=name,pid=pid,phase=value['after']['latest'],
            evidence=current['latest_training_observation'])
        path = ROOT/('pipeline/checks/gpu_occupation_'+name+'_finish_'+token+'.json')
        occupation = tick.load(path)
        # The canonical schema owns availability; assert its exact current binding.
        assert occupation['token']==token and occupation['scope']['server']==name
        assert occupation['complete'] is True and occupation['read_only'] is True
        assert occupation['scope']['allowed_gpus']==[0]
        assert 0 <= (now-datetime.fromisoformat(occupation['observed_at'])).total_seconds() < 300
        state = gpu_occupation._raw_state(occupation,{0},queue,name)
        assert state['busy_gpu_ids']==[0] and not state['idle_gpu_ids']
        assert state['reserved_gpu_ids']==[0]
        assert pid in state['compute_pids'][state['inventory'][0]['uuid']]
        occupations[name] = ref(path)
    paused = next(task for task in queue['gpu_queues']['server-2'] if task['id']=='AdaRES-Y57')
    assert paused['status']=='paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']
    assert ref('pipeline/goal.json')['sha256']=='4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a'
    repair = tick.load(closure.artifact(ROOT,candidate['observer_reference_repair']))
    assert not repair['model_and_YAML_revision'] and repair['original_failure_preserved']
    diagnosis = tick.load(closure.artifact(ROOT,repair['diagnosis']))
    assert diagnosis['original_guard_repeated_unchanged'] and not diagnosis['original_guard_passed']
    assert diagnosis['same_dtype_reference_exact'] and diagnosis['same_dtype_max_error']==0.
    for plan in plans:
        frozen = ast.literal_eval(ast.parse(closure.artifact(ROOT,plan['program']).read_text(encoding='utf-8')).body[0].value)
        assert frozen['variant']['id']==candidate['id'] and frozen['runtime_physical_gpu']==0
    backup.write_bytes((ROOT/'pipeline/codex_memory.md').read_bytes())
    tick.save(out,dict(complete=True,id=candidate['id'],token=token,observed_at=now.isoformat(),
        CPU_result=candidate['CPU_result'],CPU_version=candidate['source_manifest'],
        native_preparation=candidate['native_preparation'],observer_reference_repair=candidate['observer_reference_repair'],
        guard_tests=ref(tests),guard_tests_passed=16,active=active,occupations=occupations,
        new_candidates=1,admissions=0,launches=0,archives=0,native_GPU_executed=False,
        formal_frozen_FIRST_updated=False,original_negative_evidence_preserved=True,
        production_HEAD_preserved=Y88,GPU1_paused=True,Y57_separately_paused=True,
        previous_memory=ref(backup),helper=ref(__file__)))
    candidate['evidence'].extend([ref(out),ref(tests),ref(__file__),ref('scripts/run_y89_native.py'),
        ref('scripts/test_y89_runtime_plan.py')] + list(occupations.values()))
    work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT/'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id=candidate['id'],kind='implementation_test',result='passed',
        command='python -B scripts/run_experiment_first_probe.py y89_cpu_reference2 '+candidate['CPU_program']['path']+' --token '+token+' --seconds 480',
        observation='Actual complete CPU native YAML/model/backbone/TRAIN loading and two synthetic decoder steps passed after evidence-backed observer dtype correction. Original failure and unchanged tolerance retained. Exact seven tested files and matching GPU0 native plans stored;16 final guards passed. Both GPU0 cards reserved by confirmed progressing Y87/Y88, no native GPU/admission/efficacy claim.',
        evidence=[candidate['CPU_result'],ref(out),ref(tests)])
    tick.save(ROOT/'pipeline/research_work.json',work)
    tick.save(notes,dict(actions=[
        '核验当前实际会话归属、前台模式和Disabled后台调度；只使用AdaRES，科学goal字节、原收据、数据/私有环境/功耗/暂停均保持。',
        '先观察原Y87/Y88，实际源码/YAML/收据/功耗和原生计数增长均通过；最新Y87='+str(active['AdaRES-Y87']['phase'])+'，Y88='+str(active['AdaRES-Y88']['phase'])+'。预检已派发现有合格队列，无就绪waiting。',
        '新增独立Y89：只把七个置信度readout的evidence_norm affine拆分，原共享MLP/query_norm和全部原竞争证据/数据/损失保持。新增9228参数，不与Y60/Y67/Y83组合或重开已拒绝路线。',
        '首次CPU检查因参考计算提前转float64而失败；原守卫复现诊断确认同float32参考逐位一致。保留原程序/日志和失败结论，只修参考运算类型，原容差/LayerNorm双精度/正响应/梯度/consumer守卫未改，模型/YAML未改。',
        '修正后真实CPU原生YAML/完整模型/发布backbone/原TRAIN加载、标量/无RNG/私有梯度/七阶段W2/输入与聚合norm梯度锚点、最终置信度私有干预和两步合成更新通过。',
        '七个实际测试文件已独立保存版本；server1GPU0/350W与server2GPU0/280W原生检查计划一致，16项准备/资源/原终态前置守卫通过，不占GPU、不改HEAD/原运行/其他index。',
        '两台canonical物理占用证据新鲜，获准GPU0均由原Y87/Y88占用。Y89等待原匹配owner终态精确核验归档后空槽，当前未执行GPU原生检查、未准入/新启动。'],
        decisions=[
        '冻结正式FIRST仍Y36=0.510725438294，目标0.5150；本轮无正式FIRST/Acc/成本新结果或晋升，CPU合成loss不当任务效果。',
        '一次有证据的独立准备已完成；保留原完整冻结协议、29个fixed30后评和全部禁用项。GPU1原后评路径仍暂停不迁移，Y57单独暂停不恢复。',
        'Y86保留原numerical_control_failure及不重放/不弱化守卫条件；本轮不以计时器当新证据。',
        '下轮先观察和派发现有任务；Y89只在真实独占空闲且原owner已精确checkpoint核验归档时运行匹配原生计划，通过后同轮准入与派发。']))
    print(json.dumps(dict(closeout=ref(out),notes=ref(notes),previous_memory=ref(backup),
        tests=16,active=active,native_GPU_executed=False)))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token',required=True)
    args = parser.parse_args()
    lease(args.token,180)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        record(args.token)
    finally:
        writer.close()
