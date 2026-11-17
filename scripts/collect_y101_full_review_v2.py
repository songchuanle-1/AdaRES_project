"""Collect immutable original review captures; apply the already-corrected frozen analysis without GPU replay."""
import argparse,ast,hashlib,json,math,statistics,subprocess
from datetime import datetime
from pathlib import PurePosixPath
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from y101_frozen_review_job import remote
from analyze_y27_frozen_terminal import comparison
from analyze_frozen_full import keyed
import sys
sys.path.insert(0,str(ROOT/'AdaRES/tools'))
from evaluate_adares_frozen_full import summarize
import closure

BODY=r'''import hashlib,json,pathlib,subprocess
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
folder=pathlib.Path(Q['folder']);assert str(folder.resolve()).startswith('/data1/songcl/Project/AdaRES_project/pipeline/review_jobs/AdaRES-Y101/')
state=json.loads((folder/'state.json').read_text());assert state['status']=='complete' and state['completed_methods']==Q['ids']
launch=json.loads((folder/'launch.json').read_text());assert launch['pid']==Q['pid'] and launch['token']==Q['job_token']
assert hashlib.sha256((folder/'worker.py').read_bytes()).hexdigest()==Q['program_sha256']
rows=[]
for name in Q['names']:
    path=folder/name;before=path.stat();sha=hashlib.sha256(path.read_bytes()).hexdigest();after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    rows.append(dict(name=name,path=str(path),bytes=before.st_size,sha256=sha))
print(json.dumps(dict(complete=True,id='AdaRES-Y101',state=state,launch=launch,files=rows,original_training_absent=not pathlib.Path('/proc/2263581').exists(),GPU_used=False,checkpoint_loaded=False,remote_mutations=False)))
'''

def collect(token):
    lease(token,600);work=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in work['candidates'] if r['id']=='AdaRES-Y101');assert c['stage']=='validation' and not c.get('full_frozen_evaluation_result')
    job=c['frozen_review_job'];execution=closure.artifact(ROOT,job['program']);p=ast.literal_eval(ast.parse(execution.read_text(encoding='utf-8')).body[0].value)
    assert p['token']==job['token'] and p['folder']==job['folder'] and p['old_pid']==2263581 and p['gpu']==0 and p['power']==350.
    server=monitor.load_servers(str(ROOT))['server-1'];closure.validate_adares_server(ROOT,server);assert server['allowed_gpus']==[0]
    immutable={v:ref(v) for v in ('pipeline/goal.json','SERVERS.yaml','pipeline/queue.json','pipeline/running.json','pipeline/admissions/AdaRES-Y101.json','pipeline/admissions/AdaRES-Y102.json')}
    names=[r['id'].lower()+'.json' for r in p['methods']]+['paired_result.json']+[r['id'].lower()+'.log' for r in p['methods']]+['worker.log']
    root=ROOT/('pipeline/analyses/y101_full_review_'+token);assert not root.exists();root.mkdir()
    metadata=root/'remote_manifest.json';v=remote(server,'Q='+repr(dict(folder=job['folder'],pid=job['pid'],job_token=job['token'],program_sha256=ref(execution)['sha256'],ids=[r['id'] for r in p['methods']],names=names))+'\n'+BODY,metadata);assert v['complete'] and v['original_training_absent']
    for row in v['files']:
        target=root/row['name'];assert not target.exists() and target.resolve().is_relative_to(root.resolve())
        subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=10',server['user']+'@'+server['host']+':'+row['path'],str(target)],cwd=ROOT,check=True,timeout=90)
        assert target.stat().st_size==row['bytes'] and ref(target)['sha256']==row['sha256']
    captures={};summaries={};repeats={}
    for row in p['methods']:
        capture=tick.load(root/(row['id'].lower()+'.json'));assert capture['complete'] and capture['completed_passes']==2 and capture['seed']==1999 and capture['size_cuts_from_training']==[1487,3325]
        assert capture['config']==row['config'] and capture['config_sha256']==row['config_sha256'] and capture['checkpoint']==row['checkpoint'] and capture['checkpoint_sha256']==row['checkpoint_sha256']
        assert capture['active_queries']==[128] and capture['selected_queries']==1
        for passes in capture['passes']:
            rows=passes['rows'];assert len(rows)==9508 and len(keyed(rows))==9508 and len({r['scan_id'] for r in rows})==141
            assert all(math.isfinite(r['piou']) and 0<=r['piou']<=1 and math.isfinite(r['spiou']) and 0<=r['spiou']<=1 for r in rows)
            assert all(r['size']==('small' if r['target_points']<=1487 else 'medium' if r['target_points']<=3325 else 'large') for r in rows)
            actual=summarize(rows)
            for key,value in actual.items():assert value==passes['summary'][key]
        first,second=[r['rows'] for r in capture['passes']];assert list(keyed(first))==list(keyed(second))
        maximum=max(abs(a['piou']-b['piou']) for a,b in zip(first,second));assert maximum==capture['repeat_max_iou_difference']
        summaries[row['id']]=capture['passes'][0]['summary'];repeats[row['id']]=dict(max_expression_difference=maximum,miou_difference=capture['repeat_miou_difference']);captures[row['id']]=capture
    assert captures['AdaRES-Y101']['parameters']==164695173 and captures['AdaRES-W2']['parameters']==164629637
    paired={name:comparison(captures['AdaRES-Y101'],captures[name]) for name in ('AdaRES-Y36','AdaRES-W2','AdaRES-B1')}
    harm=[dict(reference=name,group=group,delta=result['paired_miou_difference'],ci95=result['scene_clustered_ci95']) for name in ('AdaRES-Y36','AdaRES-W2') for group,result in paired[name].items() if group!='all' and result.get('frozen_material_harm_ci_upper_below_zero')]
    original_worker=tick.load(root/'paired_result.json');assert original_worker['complete'] and original_worker['full_passes']==8
    for name,result in paired.items():
        for group,values in result.items():
            if 'scene_clustered_ci95' in values:assert values['scene_clustered_ci95']==original_worker['comparisons'][name][group]['scene_clustered_ci95']
    # The original corrected mean-absolute-repeat calculation is authoritative; keep the worker's historical abs-mean diagnostic unchanged.
    out=root/'formal_comparison.json';assert not out.exists()
    tick.save(out,dict(complete=True,id='AdaRES-Y101',token=token,observed_at=datetime.now().astimezone().isoformat(),full_passes_verified=8,methods=list(captures),split='Frozen ScanRefer val9508/141',primary='FIRST',summary=summaries,repeats=repeats,comparisons=paired,frozen_material_subgroup_harm=harm,target=.515,target_reached=summaries['AdaRES-Y101']['miou']>=.515,original_resource=p['freeze']['resource_condition'],bootstrap_draws=10000,seed=1999,TRAIN_size_cuts=[1487,3325],corrected_original_repeat_noise='mean absolute per-expression repeat difference; frozen Y23/Y27 correction reused unchanged',worker_historical_repeat_noise_retained=True,worker_CI_values_identical=True,native_GPU_passes_replayed=False,new_predictions=False,formal_complete_cost_review=False,scientific_disposition_pending_cost_and_postaudit=True,cost_from_full_capture={name:dict(parameters=value['parameters'],warm_first_chunk_seconds=value['warm_first_chunk_seconds'],peak_allocated_gib=value['peak_allocated_gib'],peak_reserved_gib=value['peak_reserved_gib']) for name,value in captures.items()},evidence=[ref(__file__),ref('scripts/analyze_frozen_full.py'),ref('scripts/analyze_y23_frozen_terminal.py'),ref('scripts/analyze_y27_frozen_terminal.py'),ref(metadata),ref(root/'paired_result.json'),c['post_training_freeze'],c['terminal_evidence'],job['program']]+[ref(root/(r['id'].lower()+'.json')) for r in p['methods']]))
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        assert all(ref(path)==value for path,value in immutable.items())
        work=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in work['candidates'] if r['id']=='AdaRES-Y101');assert c['stage']=='validation' and not c.get('full_frozen_evaluation_result')
        c['full_frozen_evaluation_result']=ref(out);c['frozen_review_capture_directory']=root.relative_to(ROOT).as_posix();c['scientific_status']='full_frozen_evaluation_verified_cost_postaudit_pending';c['evidence'].extend([ref(out),ref(metadata),ref(metadata.with_suffix('.log')),ref(__file__)]+[ref(root/name) for name in names]);c['next_action']='Original fixed four-method eight full passes verified and FIRST/scene-paired subgroup outcomes recorded. Preserve all raw outputs and corrected original repeat-noise analysis; do not rerun predictions/terminal/training or auto-promote. Complete only the original server1GPU0/350W frozen TRAIN8+largest-pair serial/native memory-only stress cost and read-only source/checkpoint postaudit before final scientific disposition; GPU1/Y57 holds unchanged.';tick.save(ROOT/'pipeline/research_work.json',work)
    finally:writer.close()
    print(json.dumps(dict(comparison=ref(out),summaries=summaries,harm=harm,target_reached=summaries['AdaRES-Y101']['miou']>=.515,full_passes=8,cost_pending=True,native_predictions_replayed=False)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--token',required=True);collect(parser.parse_args().token)
