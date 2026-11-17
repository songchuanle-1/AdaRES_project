"""Persist actual bounded Y90 preparation without changing paused or running work."""
import argparse
import ast
from datetime import datetime,timezone
import json
import subprocess
import sys
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from y90_contract import validate_cpu
from y88_deployment import Y88
import closure
import gpu_occupation


def record(token):
    work = tick.load(ROOT/'pipeline/research_work.json')
    candidate = next(candidate for candidate in work['candidates'] if candidate['id']=='AdaRES-Y90')
    assert candidate['stage']=='blocked' and candidate['blocker_kind']=='exclusive_gpu_slot'
    assert candidate['required_server']=='server-2' and candidate['required_gpu']==0
    assert not candidate.get('admission_receipt') and not candidate.get('native_smoke_result')
    cpu = tick.load(closure.artifact(ROOT,candidate['CPU_result']))
    validate_cpu(cpu)
    assert closure.git(ROOT,'rev-parse','HEAD')==Y88
    source = tick.load(closure.artifact(ROOT,candidate['source_manifest']))
    assert source['main_index_unchanged'] and source['earlier_Y89_preserved']
    assert source['no_untested_parent_candidate_included']
    assert len(source['exact_delta'])==7 and all(row[0]=='A' for row in source['exact_delta'])
    assert closure.git(ROOT,'rev-parse',candidate['code_commit']+'^')==Y88
    prior = next(item for item in work['candidates'] if item['id']=='AdaRES-Y89')
    assert prior['code_commit']=='5f1c4bde4b682fabd84fdc84a4341ba2c845fe71'
    assert prior['stage']=='blocked' and not prior.get('admission_receipt') and not prior.get('native_smoke_result')
    closure.artifact(ROOT,prior['source_manifest'])
    for key in ('native_smoke_plan_server1','native_smoke_plan_server2_gpu0'):
        closure.artifact(ROOT,prior[key])
    plans = [tick.load(closure.artifact(ROOT,candidate[key]))
             for key in ('native_smoke_plan_server1','native_smoke_plan_server2_gpu0')]
    assert {plan['server'] for plan in plans}=={'server-1','server-2'} and all(plan['gpu']==0 for plan in plans)
    tests = ROOT/('pipeline/checks/y90_final_preparation_guards_'+token+'.log')
    out = ROOT/('pipeline/checks/y90_preparation_closeout_'+token+'.json')
    notes = ROOT/('pipeline/reports/notes_'+token+'.json')
    backup = ROOT/('pipeline/snapshots/codex_memory_before_'+token+'.md')
    assert not any(path.exists() for path in (tests,out,notes,backup))
    response = subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','scripts',
        '-p','test_y90_*.py','-v'],cwd=ROOT,capture_output=True,timeout=60)
    tests.write_bytes(response.stdout+response.stderr)
    assert response.returncode==0 and b'Ran 16 tests' in tests.read_bytes()
    queue = monitor.load_queue(str(ROOT))
    running = monitor.load_running(str(ROOT))
    active,occupations = {},{}
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
        assert occupation['token']==token and occupation['scope']['server']==name
        assert occupation['complete'] is True and occupation['read_only'] is True
        assert occupation['scope']['allowed_gpus']==[0]
        assert 0<=(now-datetime.fromisoformat(occupation['observed_at'])).total_seconds()<300
        state = gpu_occupation._raw_state(occupation,{0},queue,name)
        assert state['busy_gpu_ids']==[0] and not state['idle_gpu_ids']
        assert state['reserved_gpu_ids']==[0]
        assert pid in state['compute_pids'][state['inventory'][0]['uuid']]
        occupations[name] = ref(path)
    paused = next(task for task in queue['gpu_queues']['server-2'] if task['id']=='AdaRES-Y57')
    assert paused['status']=='paused' and paused['dispatch_hold'] and paused['resume_requires_user_authorization']
    assert ref('pipeline/goal.json')['sha256']=='4b0d4443a0c38a297a2b9022f4756f6992cf956d7a89374990c0e0fc60224f5a'
    for plan in plans:
        frozen = ast.literal_eval(ast.parse(closure.artifact(ROOT,plan['program']).read_text(encoding='utf-8')).body[0].value)
        assert frozen['variant']['id']==candidate['id'] and frozen['runtime_physical_gpu']==0
    backup.write_bytes((ROOT/'pipeline/codex_memory.md').read_bytes())
    tick.save(out,dict(complete=True,id=candidate['id'],token=token,observed_at=now.isoformat(),
        CPU_result=candidate['CPU_result'],CPU_version=candidate['source_manifest'],
        native_preparation=candidate['native_preparation'],guard_tests=ref(tests),guard_tests_passed=16,
        active=active,occupations=occupations,earlier_Y89=prior['source_manifest'],
        earlier_Y89_preserved=True,new_candidates=1,admissions=0,launches=0,archives=0,
        native_GPU_executed=False,formal_frozen_FIRST_updated=False,model_revision=False,
        production_HEAD_preserved=Y88,GPU1_paused=True,Y57_separately_paused=True,
        previous_memory=ref(backup),helper=ref(__file__)))
    candidate['evidence'].extend([ref(out),ref(tests),ref(__file__),ref('scripts/run_y90_native.py'),
        ref('scripts/test_y90_runtime_plan.py'),ref('scripts/create_y90_runtime_helpers.py')]+list(occupations.values()))
    work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT/'pipeline/headless/.tick_model_view.json')['generated_at'],
        candidate_id=candidate['id'],kind='implementation_test',result='passed',
        command='python -B scripts/run_experiment_first_probe.py y90_cpu '+candidate['CPU_program']['path']+' --token '+token+' --seconds 480',
        observation='Actual original CPU YAML/model/backbone/TRAIN loading, scalar bilinear/bounded/directed/no-RNG/permutation controls, seven-stage W2/input-gradient anchors, confidence-only intervention/bypass and fourteen exact overlap/affinity/context/shared-head consumers/two synthetic updates passed first execution. Exact seven-file version, matching GPU0-only plans and16 final guards passed. Both GPUs busy; no native GPU/admission/efficacy claim.',
        evidence=[candidate['CPU_result'],ref(out),ref(tests)])
    tick.save(ROOT/'pipeline/research_work.json',work)
    tick.save(notes,dict(actions=[
        '核验实际所属会话、foreground模式、Disabled后台调度与最新GPU1暂停；只处理AdaRES，科学goal字节、原运行收据、数据/私有环境/功耗保持。',
        '先观察原Y87/Y88，源码/YAML/收据/功耗和原生步数增长通过；最新Y87='+str(active['AdaRES-Y87']['phase'])+'，Y88='+str(active['AdaRES-Y88']['phase'])+'。既有合格队列预检已尝试派发，无就绪waiting。',
        '原Y89七文件版本和两GPU0计划保持，不重放CPU、原生检查或恢复暂停候选。新增独立Y90，只对原重叠竞争权重乘有界查询对兼容性，不改原overlap/max、原掩码/查询/目标/损失/selector。',
        'Y90首次实际CPU原生YAML/完整模型/发布backbone/原TRAIN162738点8表达加载，float32/64标量双线性/有界有向/无RNG/置换控制通过。',
        '七阶段W2输出与输入梯度锚点、仅置信度干预/旁路恢复、十四次原重叠/兼容性/上下文/共享head消费者，以及两步有限梯度和真实参数更新通过；仅CPU合成decoder，不冒充GPU任务。',
        '仅七个实测Y90文件从当前Y88独立保存，生产HEAD/index/原运行和未准入Y89保持。两套获准GPU0原生检查计划、16项准备/运行资源守卫通过，尚未执行GPU原生或准入。',
        'canonical物理证据确认server1GPU0被Y88、server2GPU0被Y87占用，Y89/Y90等待真空槽，不挤占、不改GPU1/硬件/调度。'],
        decisions=[
        '正式冻结FIRST仍Y36=0.510725438294，目标0.5150；无新的正式FIRST/Acc/成本或晋升，CPU损失/正响应/有限梯度不代表训练收益。',
        '本轮完成一次实际独立实现测试及保存，不为凑额度强行建第二变体。Y89先于Y90；两者都需真实独占授权GPU0原生TRAIN两步通过，才可同轮准入与派发。',
        '保持原29个fixed30完整后评协议与原设备要求；GPU1原后评仍暂停不迁移，Y57实验单独暂停，Y86原numerical_control_failure不重放或弱化守卫。',
        '若生产HEAD在Y90执行前已推进，只做保持七个实测文件逐字节相同的显式加法版本绑定，不混合机制、不reset、不带未通过候选部署。']))
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
