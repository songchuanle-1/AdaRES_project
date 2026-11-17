"""Prepare exact Y76 native checks while all three authorized GPUs train."""
import argparse
import ast
import copy
from datetime import datetime,timedelta
import json
import subprocess
import sys
from prepare_y76_stage_confidence_conditioning import ROOT,CID,files,lease,monitor,ref,tick
from save_y76_cpu_version import git,HEAD,BRANCH
from check_ef_execution import validate_frozen
import closure

def save(token):
    work=tick.load(ROOT/'pipeline/research_work.json');c=next(c for c in work['candidates'] if c['id']==CID)
    assert c['stage']=='implementation' and c['implementation_test_passed'] and not c.get('native_preparation')
    version=tick.load(closure.artifact(ROOT,c['source_manifest']));bundle=tick.load(closure.artifact(ROOT,version['source_snapshot']))
    payload=bundle['CPU_tested_payload'];cpu=tick.load(closure.artifact(ROOT,version['CPU_result']))
    assert cpu['complete'] and cpu['implementation_test_passed'] and not cpu['CUDA_initialized']
    assert git('rev-parse','HEAD').decode().strip()==HEAD and git('rev-parse',BRANCH).decode().strip()==c['code_commit']==version['code_commit']
    qref,rref=ref('pipeline/queue.json'),ref('pipeline/running.json')
    body=ROOT/'scripts/probe_y76_native_smoke.py';ast.parse(body.read_text(encoding='utf-8'),feature_version=(3,8))
    base=ROOT/('pipeline/checks/y76_native_'+token);tests=base.with_name(base.name+'_guards.log');out=base.with_suffix('.json')
    assert not tests.exists() and not out.exists()
    tested=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','scripts','-p','test_y76_native_smoke_plan.py','-v'],cwd=ROOT,capture_output=True,timeout=60)
    tests.write_bytes(tested.stdout+tested.stderr);assert tested.returncode==0 and b'Ran 11 tests' in tests.read_bytes()
    grant=ref('pipeline/authorizations/server2_gpu1_20261029.json');closure.artifact(ROOT,grant)
    plans={};programs=[]
    for gpu,power in ((0,280.),(1,390.)):
        frozen=dict(variant=payload['variant'],code_commit=c['code_commit'],goal_sha256=payload['goal_sha256'],
            train_source_lf_sha256=payload['train_source_lf_sha256'],runtime_server='server-2',runtime_physical_gpu=gpu,runtime_power_limit_watts=power,resource_authorization=grant)
        plan=dict(complete=True,id=CID,config=c['config'],profile='experiment_first',code_commit=c['code_commit'],goal_sha256=frozen['goal_sha256'],
            server='server-2',gpu=gpu,runtime_power_limit_watts=power,resource_authorization=grant,working_HEAD_preserved=HEAD,retained_branch=BRANCH,
            planned_command='python -B scripts/run_experiment_first_probe.py y76_gpu'+str(gpu)+'_smoke '+base.relative_to(ROOT).as_posix()+'_gpu'+str(gpu)+'_program.py --token <CURRENT_TOKEN> --seconds 600 --gpu --gpu-id '+str(gpu),
            required_resource='Fresh exclusive authorized idle/unreserved server-2 GPU'+str(gpu)+' after canonical terminal archive; exact host/private runtime/original data/unchanged power. Preserve live tasks, Y57 pause and original full evaluation protocol.',
            checks='Native YAML/tools.train/TRAIN two finite losses/backward/optimizer; seven original head-query and integer stage identities and actual query-norm consumers, static mask/padding identities, seven shared confidence heads, independent scalar stage-offset formula, zero first/next-step response, native modes/finite original-query/new-offset gradients with all seven rows updated/real updates/peak memory.',
            native_executed=False,admitted=False,native_PASS_required=True,guard_tests_passed=11,Python38_compile=True)
        assert validate_frozen(c,plan,frozen)==7
        for field in ('goal','source'):
            bad=copy.deepcopy(frozen)
            if field=='goal':bad['goal_sha256']='0'*64
            else:bad['variant']['python_files']['dec.py']['base64']='Y2hhbmdlZA=='
            rejected=False
            try:validate_frozen(c,plan,bad)
            except AssertionError:rejected=True
            assert rejected
        program=base.with_name(base.name+'_gpu'+str(gpu)+'_program.py');path=base.with_name(base.name+'_gpu'+str(gpu)+'_plan.json')
        assert not program.exists() and not path.exists()
        program.write_text('FROZEN='+repr(frozen)+'\n'+body.read_text(encoding='utf-8'),encoding='utf-8',newline='\n');ast.parse(program.read_text(encoding='utf-8'),feature_version=(3,8))
        plan['program']=ref(program);tick.save(path,plan);plans[str(gpu)]=ref(path);programs.append(ref(program))
    assert git('rev-parse','HEAD').decode().strip()==HEAD and ref('pipeline/queue.json')==qref and ref('pipeline/running.json')==rref
    tick.save(out,dict(complete=True,id=CID,token=token,code_commit=c['code_commit'],CPU_version=c['source_manifest'],CPU_result=version['CPU_result'],
        native_plans=plans,native_body=ref(body),guard_tests=ref(tests),guard_tests_passed=11,stale_goal_refused=True,tampered_source_refused=True,
        same_scientific_body=True,working_HEAD=HEAD,queue_and_running_preserved=True,native_executed=False,admitted=False,production_sync=False,
        evidence=[ref(__file__),ref('scripts/test_y76_native_smoke_plan.py')]))
    c.update(stage='blocked',blocked_from_stage='validation',blocker_kind='exclusive_gpu_slot',required_server='server-2',required_gpu=0,resource_work_kind='native_smoke',
        native_smoke_plan=plans['0'],native_smoke_plan_server2_gpu0=plans['0'],native_smoke_plan_server2_gpu1=plans['1'],native_preparation=ref(out),
        blocker='All authorized cards run original independent Y72/Y73/Y74; preserve them. Native TRAIN two-step task check requires an exclusive idle/unreserved authorized GPU.',
        retry_after=(datetime.now().astimezone()+timedelta(hours=1)).isoformat(),wake_paths=['pipeline/queue.json','pipeline/running.json','SERVERS.yaml'],
        unblock_condition='Canonical terminal archive plus fresh exclusive authorized physical idle/no compute/no reservation; exact matching frozen plan and original data/runtime/power.',
        next_action='Do not replay CPU/version/native-plan preparation. Preserve active HEAD3c, retained Y75/Y76 branches; prioritize already-prepared Y75; exercise exact Y76 native observation callbacks on CPU before using a future GPU slot if not yet done. Only a fresh authorized exclusive idle/unreserved matching GPU permits the native two-step YAML/TRAIN task. Actual finite task loss/backward/optimizer and all intended paths must PASS before original complete frozen evaluation/cost freeze, receipt/admission, safe production selection/rebinding live contracts, canonical gated dispatch and increasing native steps. No GPU PASS or efficacy from CPU checks. Y57 remains separately paused.')
    c['evidence'].extend([ref(out),ref(tests),ref(body),ref(__file__),ref('scripts/test_y76_native_smoke_plan.py')]+list(plans.values())+programs)
    work['tick_attempt']=dict(tick_started_at=tick.load(ROOT/'pipeline/headless/.tick_model_view.json')['generated_at'],candidate_id=CID,kind='implementation_test',result='passed',
        command='python -B scripts/save_y76_native_plans.py --token '+token,observation='Eleven native-evidence rejection controls passed; exact retained-version dual GPU plans prepared, stale goal/tampered source refused. No actual GPU task steps, admission or efficacy.',evidence=[ref(out),ref(tests)])
    tick.save(ROOT/'pipeline/research_work.json',work);print(json.dumps(dict(version=ref(out),plans=plans,tests=11,native_executed=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);a=p.parse_args();lease(a.token,240)
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:save(a.token)
    finally:lock.close()
