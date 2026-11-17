"""Add the explicitly verified current-owner field missing from the retained schema."""
import argparse,ast,base64,copy,json,subprocess,sys
from datetime import datetime
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from y101_deployment import Y100,Y101,git,contract
from prepare_y103_spatial_input_skip import files
from y103_contract import validate_cpu
from check_ef_execution import validate_frozen
import closure
def complete(token):
    lease(token,300);writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        assert git('rev-parse','HEAD').decode().strip()==Y101
        before={p:ref(p) for p in ('pipeline/queue.json','pipeline/running.json','pipeline/goal.json','SERVERS.yaml')};index=git('ls-files','--stage','-z')
        w=tick.load(ROOT/'pipeline/research_work.json');prior=copy.deepcopy(w);c=next(r for r in w['candidates'] if r['id']=='AdaRES-Y103')
        assert c['stage']=='blocked' and c['blocker_kind']=='exclusive_gpu_slot' and not c.get('native_preparation') and not c.get('native_smoke_result') and not c.get('admission_receipt')
        oldref=c['additive_version_binding'];old=tick.load(closure.artifact(ROOT,oldref));assert oldref==c['source_manifest']
        assert old['main_index_unchanged'] and old['CPU_tested_scientific_files_exact'] and old['parent_commit']==old['working_HEAD']==Y101
        assert old['matching_current_owners']=={'server-1':'AdaRES-Y101','server-2':'AdaRES-Y100'}
        assert 'original_Y101_Y100_preserved' not in old and old['original_Y99_Y100_preserved'] is True
        assert [(r['id'],r['pid'],r['code_commit']) for r in old['original_owner_rows']]==[('AdaRES-Y101',2263581,Y101),('AdaRES-Y100',19305,Y100)]
        q=monitor.load_queue(str(ROOT))
        for name,cid,pid,commit in [('server-1','AdaRES-Y101',2263581,Y101),('server-2','AdaRES-Y100',19305,Y100)]:
            task=next(r for r in q['gpu_queues'][name] if r['id']==cid);assert (task['status'],task['pid'],task['code_commit'],task['launch_verified'])==('running',pid,commit,True);closure.admission(ROOT,task);contract(commit,Y101,cid)
        validate_cpu(tick.load(closure.artifact(ROOT,c['CPU_result'])))
        snapshot=tick.load(closure.artifact(ROOT,old['source_snapshot']));payload=snapshot['CPU_tested_payload']
        assert payload==ast.literal_eval(ast.parse(closure.artifact(ROOT,c['CPU_program']).read_text(encoding='utf-8')).body[0].value)
        for p in files():
            encoded=payload['variant']['config_data'] if p.suffix=='.yaml' else payload['variant']['python_files'][p.name]
            assert p.read_bytes()==base64.b64decode(encoded['base64'])==git('show',c['code_commit']+':'+p.relative_to(ROOT).as_posix()) and ref(p)['sha256']==encoded['sha256']
        failure=ref('pipeline/checks/y103_native_'+token+'_runtime_guards.log');assert "KeyError: 'original_Y101_Y100_preserved'" in closure.artifact(ROOT,failure).read_text(encoding='utf-8')
        binding=ROOT/('pipeline/checks/y103_current_owner_binding_'+token+'_v2.json');assert not binding.exists()
        fixed=copy.deepcopy(old);fixed.update(token=token,observed_at=datetime.now().astimezone().isoformat(),supersedes=oldref,original_Y101_Y100_preserved=True,operational_metadata_correction='Original retained owner rows already name exact original Y101/PID2263581 and Y100/PID19305. The template saved its legacy preservation keyword; add the current-owner field after explicit source/receipt/owner/index verification. Preserve original version/failed plans and raw log; no scientific payload, native program body, version or control changed.',correction_evidence=[ref(__file__),failure,oldref,c['CPU_result'],c['CPU_program']])
        tick.save(binding,fixed);c['pre_runtime_metadata_binding']=oldref;c['additive_version_binding']=ref(binding);c['evidence'].extend([ref(binding),failure,ref(__file__)])
        tick.save(ROOT/'pipeline/research_work.json',w)
        log=ROOT/('pipeline/checks/y103_native_'+token+'_runtime_guards_v2.log');assert not log.exists()
        r=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','scripts','-p','test_y103_runtime_plan.py','-v'],cwd=ROOT,capture_output=True,timeout=60);log.write_bytes(r.stdout+r.stderr)
        assert r.returncode==0 and b'Ran 8 tests' in log.read_bytes() and b'skipped' not in log.read_bytes(),str(log)
        plans={name:c[key] for name,key in [('server-1','native_smoke_plan_server1'),('server-2','native_smoke_plan_server2_gpu0')]}
        programs=[]
        for name,entry in plans.items():
            plan=tick.load(closure.artifact(ROOT,entry));f=ast.literal_eval(ast.parse(closure.artifact(ROOT,plan['program']).read_text(encoding='utf-8')).body[0].value);assert validate_frozen(c,plan,f)==7 and plan['execution_helper']==ref('scripts/run_y103_native.py');programs.append(plan['program'])
        out=ROOT/('pipeline/checks/y103_native_'+token+'_v2.json');assert not out.exists()
        evidence=[ref(__file__),ref(binding),oldref,c['CPU_result'],c['CPU_program'],c['CPU_guard_tests'],ref(log),failure,c['prepared_native_body'],c['observer_core'],ref('scripts/run_y103_native.py'),ref('scripts/test_y103_runtime_plan.py'),ref('scripts/retain_y103_cpu_native.py')]+list(plans.values())+programs
        tick.save(out,dict(complete=True,id=c['id'],token=token,code_commit=c['code_commit'],CPU_version=c['source_manifest'],current_owner_binding=ref(binding),CPU_result=c['CPU_result'],CPU_program=c['CPU_program'],native_plans=plans,guard_tests=c['CPU_guard_tests'],guard_tests_passed=10,runtime_guard_tests=ref(log),runtime_guard_tests_passed=8,native_observer_body=c['prepared_native_body'],native_observer_callbacks_executed_on_GPU=False,same_callbacks_actually_executed_on_CPU=True,original_Y101_Y100_preserved=True,native_executed=False,admitted=False,production_sync=False,scientific_replay=False,evidence=evidence))
        c['native_preparation']=ref(out);c['evidence'].extend([ref(out)]+evidence)
        w['tick_attempt']=dict(tick_started_at=tick.load(ROOT/'pipeline/headless/.tick_model_view.json')['generated_at'],candidate_id=c['id'],kind='implementation_test',result='passed',command='python -B scripts/run_experiment_first_probe.py y103_cpu '+c['CPU_program']['path']+' --token '+token+' --seconds 600',observation='FIRST real CPU YAML/fullmodel/releasedbackbone/TRAIN collation and independent scalar/gradient/W2-anchor/causal masks/two finite synthetic updates passed;10 source/evidence+8 runtime guards pass after explicit current-owner metadata repair. Raw failed schema log/version/plans retained; no scientific/native replay, admission or GPU execution.',evidence=[c['CPU_result'],c['source_manifest'],ref(binding),ref(out),ref(log),failure])
        assert [r for r in w['candidates'] if r['id']!=c['id']]==[r for r in prior['candidates'] if r['id']!=c['id']]
        tick.save(ROOT/'pipeline/research_work.json',w);assert all(ref(p)==r for p,r in before.items()) and git('ls-files','--stage','-z')==index
        print(json.dumps(dict(binding=ref(binding),preparation=ref(out),tests=10,runtime_guards=8,native_executed=False,scientific_replay=False)))
    finally:writer.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);complete(p.parse_args().token)
