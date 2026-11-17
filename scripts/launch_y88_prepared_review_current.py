"""Launch the already prepared Y88 review with current-token metadata, retaining its original binding."""
import argparse,ast,copy,json
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from y88_frozen_review_job import module
import closure

def launch(token):
    lease(token,600);w=tick.load(ROOT/'pipeline/research_work.json');by={r['id']:r for r in w['candidates']};c=by['AdaRES-Y88'];done=by['AdaRES-Y90']
    assert done['stage']=='rejected' and done['cost_resource_recovery_status']=='complete_guarded_cost_and_scientific_review'
    result=tick.load(closure.artifact(ROOT,done['complete_scientific_review']));assert result['full_cost_and_postaudit_complete'] and result['resource_exclusivity_verified_from_guard']
    assert c['stage']=='validation' and not c.get('frozen_review_job') and not c.get('frozen_review_launch_intent') and not c.get('frozen_review_launch_binding')
    original=copy.deepcopy(c['frozen_review_execution_binding']);binding=tick.load(closure.artifact(ROOT,original));assert binding['CPU_identity_and_launch_guards_passed']==11 and binding['source_files_verified']==26 and not binding['GPU_used']
    old=closure.artifact(ROOT,binding['program']).read_text(encoding='utf-8');p=ast.literal_eval(ast.parse(old).body[0].value);assert p['id']=='AdaRES-Y88' and p['gpu']==0 and p['power']==350. and p['code_commit']==c['code_commit']
    fresh=copy.deepcopy(p);fresh.update(token=token,folder=p['folder'].rsplit('/',1)[0]+'/'+token)
    assert {k for k in p if p[k]!=fresh[k]}=={'token','folder'}
    code='P='+repr(fresh)+'\n'+old.split('\n',1)[1];ast.parse(code,feature_version=(3,8));assert code.split('\n',1)[1]==old.split('\n',1)[1]
    path=ROOT/('pipeline/checks/y88_review_current_launch_'+token+'.py');manifest=path.with_suffix('.json');assert not path.exists() and not manifest.exists();path.write_text(code,encoding='utf-8',newline='\n')
    new=copy.deepcopy(binding);new.update(token=token,original_preparation=original,original_prepared_program=binding['program'],program=ref(path),only_payload_metadata_changed=['token','folder'],worker_body_and_all_scientific_payload_byte_identical=True,scientific_CPU_native_terminal_tests_not_replayed=True,operational_guards_reused_not_replayed=11,evidence=binding['evidence']+[ref(__file__),ref(path),original,done['complete_scientific_review']]);tick.save(manifest,new)
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:
        w=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in w['candidates'] if r['id']=='AdaRES-Y88');assert not c.get('frozen_review_launch_binding');c['frozen_review_launch_binding']=ref(manifest);c['evidence'].extend([ref(manifest),ref(__file__)]);tick.save(ROOT/'pipeline/research_work.json',w)
    finally:lock.close()
    namespace=module(token);base=(ROOT/'scripts/y99_frozen_review_job.py').read_text(encoding='utf-8');tree=ast.parse(base);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='launch');source='\n'.join(base.splitlines()[fn.lineno-1:fn.end_lineno])+'\n'
    for a,b in [('AdaRES-Y99','AdaRES-Y88'),('y99_','y88_'),('2980836','3527278')]:source=source.replace(a,b)
    needle="c['stage']=='blocked' and c['blocker_kind']=='exclusive_gpu_slot' and";assert source.count(needle)==1;source=source.replace(needle,"c['stage']=='validation' and")
    needle="assert c['stage']=='blocked' and not c.get('frozen_review_job')";assert source.count(needle)==1;source=source.replace(needle,"assert c['stage']=='validation' and not c.get('frozen_review_job')")
    assert source.count("c['frozen_review_execution_binding']")==2;source=source.replace("c['frozen_review_execution_binding']","c['frozen_review_launch_binding']")
    namespace['__file__']=__file__;exec(compile(ast.parse(source),__file__,'exec'),namespace);namespace['launch'](token)
    print(json.dumps(dict(current_launch_binding=ref(manifest),original_preparation_preserved=original,original_worker_body_unchanged=True,prior_scientific_tests_replayed=0,new_training=False)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--token',required=True);launch(parser.parse_args().token)
