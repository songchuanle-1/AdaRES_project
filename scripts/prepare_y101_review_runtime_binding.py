"""Check the original Y101 review imports/config bindings on CPU, without model or data execution."""
import argparse,ast,json,shlex,subprocess
from datetime import datetime
from pathlib import PurePosixPath
import yaml
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from y102_deployment import git,Y101,Y102,adapt
import closure

BODY=r'''import hashlib,json,os,pathlib,subprocess,sys
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and os.environ['PYTHONDONTWRITEBYTECODE']=='1'
assert sys.executable==P['python'] and sys.flags.no_site==1 and sys.dont_write_bytecode
os.chdir(P['code']);sys.path[:0]=P['python_paths']
root=pathlib.Path(P['code']).resolve();config=pathlib.Path(P['config'])
assert config.is_file() and hashlib.sha256(config.read_bytes()).hexdigest()==P['config_sha256']
import torch
assert not torch.cuda.is_initialized()
blocked=[]
def no_load(*args,**kwargs):
    blocked.append('checkpoint_load');raise RuntimeError('NO_CHECKPOINT_LOAD_IN_IMPORT_CHECK')
def no_cuda(*args,**kwargs):
    blocked.append('CUDA_init');raise RuntimeError('NO_CUDA_INIT_IN_IMPORT_CHECK')
torch.load=no_load;torch.cuda._lazy_init=no_cuda
import gorilla
from ipdn.dataset import build_dataset,build_dataloader
from ipdn.utils import get_root_logger
import train
from pointnet2 import pointnet2_utils
import spconv.pytorch as spconv
assert pathlib.Path(train.__file__).resolve()==root/'tools/train.py'
assert pathlib.Path(torch.__file__).resolve().is_relative_to(pathlib.Path(P['site_packages']).resolve()) if hasattr(pathlib.Path('.'),'is_relative_to') else str(pathlib.Path(torch.__file__).resolve()).startswith(P['site_packages']+'/')
assert str(pathlib.Path(gorilla.__file__).resolve()).startswith(P['site_packages']+'/')
assert P['pointnet_egg']+'/' in pointnet2_utils.__file__
if config.stem=='adares_b1_standard_res':
    registered=gorilla.LOSSES.obj_map.get('Criterion')
    if registered is not None:
        assert registered.__module__=='ipdn.model_paper5.adares_b3_mask_evidence.loss'
        gorilla.LOSSES.obj_map.pop('Criterion')
cls=train.resolve_model_class(str(config));assert cls.__module__==P['expected_module']
module=sys.modules[cls.__module__];assert str(pathlib.Path(module.__file__).resolve()).startswith(str(root)+'/')
cfg=gorilla.Config.fromfile(str(config))
for key in ('model','data','train','work_dir'):assert cfg[key]==P['expected_config'][key],key
namespace={'__name__':'y101_import_check_only','__file__':'tested_resource_adapter_not_deployed.py'}
exec(compile(P['adapter_source'],'tested_resource_adapter_not_deployed.py','exec'),namespace)
assert callable(namespace['main']) and namespace['main'].__module__=='y101_import_check_only'
assert not torch.cuda.is_initialized() and not blocked
print(json.dumps(dict(complete=True,id=P['id'],executable=sys.executable,no_site=True,config=str(config),config_sha256=P['config_sha256'],resolved_model=cls.__module__+'.'+cls.__name__,module_file=module.__file__,module_SHA=hashlib.sha256(pathlib.Path(module.__file__).read_bytes()).hexdigest(),torch_module=torch.__file__,gorilla_module=gorilla.__file__,pointnet_module=pointnet2_utils.__file__,sparseconv_module=spconv.__file__,native_train_module=train.__file__,native_yaml_semantics_verified=True,adapter_definitions_imported=True,evaluator_main_called=False,model_created=False,dataset_created=False,checkpoint_loaded=False,CUDA_initialized=False,real_GPU_queries=False,scientific_or_native_test_replayed=False,remote_file_mutations=False)))
'''

def prepare(token):
    lease(token,600);assert git('rev-parse','HEAD').decode().strip()==Y102
    immutable={p:ref(p) for p in ('pipeline/goal.json','SERVERS.yaml','pipeline/queue.json','pipeline/running.json','pipeline/admissions/AdaRES-Y101.json','pipeline/admissions/AdaRES-Y102.json')};index=git('ls-files','--stage','-z')
    view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json');assert view['queue']['candidate_budget']==view['research']['creation_budget']==view['research']['reserve_budget']==0
    work=tick.load(ROOT/'pipeline/research_work.json');candidate=next(r for r in work['candidates'] if r['id']=='AdaRES-Y101');assert candidate['stage']=='training' and not candidate.get('frozen_review_runtime_binding')
    adapter=tick.load(closure.artifact(ROOT,candidate['frozen_review_resource_adapter']));assert adapter['tests_passed']==14 and not adapter['evaluation_executed']
    adapter_path=closure.artifact(ROOT,adapter['adapted_evaluator']);adapter_source=adapter_path.read_text(encoding='utf-8')
    staging=tick.load(closure.artifact(ROOT,candidate['frozen_review_reference_stage']));assert staging['all_three_reference_bytes_match_frozen_SHA']
    freeze=tick.load(closure.artifact(ROOT,candidate['post_training_freeze']));assert freeze['resource_condition']==adapter['resource_condition']
    server=monitor.load_servers(str(ROOT))['server-1'];closure.validate_adares_server(ROOT,server);assert server['allowed_gpus']==[0]
    native=adapt(yaml.safe_load(git('show',Y101+':AdaRES/'+candidate['config'])),server)
    rows=[dict(id='AdaRES-Y101',config=server['remote_code_root']+'/'+candidate['config'],expected_config=native,config_sha256=None)]
    for row,config_ref in zip(staging['references'],staging['adapted_native_YAML']):
        path=closure.artifact(ROOT,config_ref);expected=yaml.safe_load(path.read_text(encoding='utf-8'))
        assert expected==adapt(yaml.safe_load(git('show',Y101+':AdaRES/'+row['config'])),server)
        rows.append(dict(id=row['id'],config=staging['remote_adapted_config_root']+'/'+path.name,expected_config=expected,config_sha256=config_ref['sha256']))
    assert [r['id'] for r in rows]==freeze['evaluation']['methods']
    observation=tick.load(closure.artifact(ROOT,candidate['latest_training_observation']));assert observation['native_YAML_semantics_verified']
    prefix='PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 LD_LIBRARY_PATH='+shlex.quote(server['runtime_library_path'])+' '+shlex.quote(server['python'])+' -B -S -'
    results=[];evidence=[]
    for row in rows:
        lease(token,300)
        paths=[server['remote_code_root'],server['remote_code_root']+'/tools',server['remote_code_root']+'/ipdn/lib',server['site_packages'],server['pointnet_egg']]
        payload=dict(row,python=server['python'],code=server['remote_code_root'],site_packages=server['site_packages'],pointnet_egg=server['pointnet_egg'],python_paths=paths,expected_module='ipdn.model_paper5.'+PurePosixPath(row['config']).stem.removesuffix('_res')+'.ipdn',adapter_source=adapter_source)
        base='pipeline/checks/y101_review_import_'+token+'_'+row['id'].lower();program=ROOT/(base+'.py');log=ROOT/(base+'.log');result_path=ROOT/(base+'.json')
        assert not any(p.exists() for p in (program,log,result_path))
        # The active native YAML was path-adapted at launch; verify its bytes against the frozen original parsed content before importing.
        body=BODY
        if row['id']=='AdaRES-Y101':
            body=body.replace("assert config.is_file() and hashlib.sha256(config.read_bytes()).hexdigest()==P['config_sha256']", "assert config.is_file()\nP['config_sha256']=hashlib.sha256(config.read_bytes()).hexdigest()")
        text='P='+repr(payload)+'\n'+body;ast.parse(text,feature_version=(3,8));program.write_text(text,encoding='utf-8',newline='\n')
        response=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',server['user']+'@'+server['host'],prefix],input=text.encode(),capture_output=True,timeout=120)
        log.write_bytes(response.stdout+response.stderr)
        assert response.returncode==0,'Preserve failed import evidence; do not deploy, evaluate or change the environment blindly'
        value=json.loads(response.stdout.decode().splitlines()[-1]);assert value['complete'] and value['native_yaml_semantics_verified'] and not value['CUDA_initialized'] and not value['checkpoint_loaded']
        tick.save(result_path,value);results.append(ref(result_path));evidence.extend([ref(program),ref(log),ref(result_path)])
        print(json.dumps(dict(import_binding=row['id'],passed=True,model_created=False,checkpoint_loaded=False,CUDA_initialized=False)),flush=True)
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        assert all(ref(p)==r for p,r in immutable.items()) and git('ls-files','--stage','-z')==index
        out=ROOT/('pipeline/checks/y101_review_runtime_'+token+'.json');assert not out.exists()
        tick.save(out,dict(complete=True,token=token,id='AdaRES-Y101',observed_at=datetime.now().astimezone().isoformat(),original_resource=freeze['resource_condition'],methods=freeze['evaluation']['methods'],isolated_CPU_imports_passed=4,private_python_and_site_packages_verified=True,private_pointnet_egg_and_sparseconv_imported=True,original_native_train_resolver_verified=True,all_four_native_YAML_semantics_verified=True,adapter_definitions_imported_without_main=True,launch_python_paths=paths,launch_python=server['python'],launch_runtime_library_path=server['runtime_library_path'],actual_command_prefix=prefix,results=results,model_created=False,dataset_created=False,checkpoint_loaded=False,CUDA_initialized=False,real_GPU_queries=False,adapter_deployed=False,evaluation_executed=False,evaluation_ready=False,remaining='Await original natural Y101 fixed30/exact finite terminal proof/canonical archive, fresh exclusive originalserver1GPU0/350W, final guarded executable launch and full paired FIRST/bootstrap/subgroup/cost. Runtime import PASS is not GPU/data/effectiveness or target achievement.',evidence=[ref(__file__),candidate['frozen_review_resource_adapter'],candidate['frozen_review_reference_stage'],candidate['post_training_freeze'],candidate['latest_training_observation']]+evidence))
        work=tick.load(ROOT/'pipeline/research_work.json');candidate=next(r for r in work['candidates'] if r['id']=='AdaRES-Y101');assert candidate['stage']=='training';candidate['frozen_review_runtime_binding']=ref(out);candidate['evidence'].extend([ref(out),ref(__file__)]);tick.save(ROOT/'pipeline/research_work.json',work)
        print(json.dumps(dict(runtime_binding=ref(out),methods=4,GPU_used=False,evaluation_executed=False)))
    finally:writer.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);prepare(p.parse_args().token)
