"""Verify completed frozen Y101 cost/postaudit and record its full inconclusive scientific result."""
import argparse,ast,json,math,statistics,subprocess
from datetime import datetime
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from y101_frozen_review_job import remote
import closure

BODY=r'''import hashlib,json,pathlib,subprocess
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
folder=pathlib.Path(Q['folder']);state=json.loads((folder/'state.json').read_text());assert state['status']=='complete' and state['native_cost_batches_done']==24 and state['pid']==Q['pid']
assert hashlib.sha256((folder/'worker.py').read_bytes()).hexdigest()==Q['program_sha256']
files=[]
for name in ('trained_cost.json','postaudit.json','worker.log','state.json','launch.json'):
 p=folder/name;a=p.stat();sha=hashlib.sha256(p.read_bytes()).hexdigest();b=p.stat();assert (a.st_size,a.st_mtime_ns)==(b.st_size,b.st_mtime_ns);files.append(dict(name=name,path=str(p),bytes=a.st_size,sha256=sha))
print(json.dumps(dict(complete=True,id='AdaRES-Y101',state=state,files=files,checkpoint_loaded=False,GPU_used=False,remote_mutations=False)))
'''

def complete(token):
    lease(token,600);w=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in w['candidates'] if r['id']=='AdaRES-Y101');assert c['stage']=='validation' and not c.get('complete_scientific_review')
    formal_ref=c['full_frozen_evaluation_result'];formal=tick.load(closure.artifact(ROOT,formal_ref));assert formal['complete'] and formal['full_passes_verified']==8
    job=c['frozen_trained_cost_job'];program=closure.artifact(ROOT,job['program']);p=ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
    assert p['gpu']==0 and p['power']==350. and p['token']==job['token']
    immutable={path:ref(path) for path in ('pipeline/goal.json','SERVERS.yaml','pipeline/queue.json','pipeline/running.json','pipeline/experiments.csv','pipeline/admissions/AdaRES-Y101.json','pipeline/admissions/AdaRES-Y102.json')}
    server=monitor.load_servers(str(ROOT))['server-1'];closure.validate_adares_server(ROOT,server)
    root=ROOT/('pipeline/analyses/y101_trained_cost_'+token);assert not root.exists();root.mkdir();meta=root/'remote_manifest.json';value=remote(server,'Q='+repr(dict(folder=job['folder'],pid=job['pid'],program_sha256=ref(program)['sha256']))+'\n'+BODY,meta)
    for row in value['files']:
        path=root/row['name'];assert not path.exists() and path.resolve().is_relative_to(root.resolve());subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=10',server['user']+'@'+server['host']+':'+row['path'],str(path)],check=True,timeout=90);assert path.stat().st_size==row['bytes'] and ref(path)['sha256']==row['sha256']
    cost=tick.load(root/'trained_cost.json');audit=tick.load(root/'postaudit.json')
    assert cost['complete'] and cost['id']=='AdaRES-Y101' and cost['parameters']==dict(candidate=164695173,parent=164629637) and cost['available_queries']==128 and cost['selected_queries']==1
    assert len(cost['manifest_indices'])==10 and cost['manifest_indices'][-2:]==[728,6780]
    for name in ('candidate','parent'):
        assert len(cost['rows'][name])==10 and len(cost['training_stress'][name])==2
        assert [r['dataset_index'] for r in cost['rows'][name]]==cost['manifest_indices']
        assert cost['rows'][name][-2]['scan_ids']==['scene0054_00'] and cost['rows'][name][-1]['scan_ids']==['scene0497_00']
        for row in cost['rows'][name]+cost['training_stress'][name]:
            assert len(row['readings'])==3 and row['points']>0
            for r in row['readings']:
                assert all(math.isfinite(r[k]) and r[k]>=0 for k in ('seconds','peak_allocated_gib','peak_reserved_gib'))
                assert r['peak_allocated_gib']<=r['peak_reserved_gib'] and r['peak_reserved_gib']<=24
                if 'loss' in r:assert math.isfinite(r['loss'])
    assert [(r['scan_ids'],r['points']) for r in cost['rows']['candidate']]==[(r['scan_ids'],r['points']) for r in cost['rows']['parent']]
    assert [(r['scan_ids'],r['points']) for r in cost['training_stress']['candidate']]==[(r['scan_ids'],r['points']) for r in cost['training_stress']['parent']]
    assert audit['complete'] and audit['cost_sha256']==ref(root/'trained_cost.json')['sha256'] and audit['native_cost_batches_done']==24
    assert audit['source_raw_bytes_preserved'] and audit['original_training_absent'] and audit['memory_only_stress_updates'] and not audit['training_restarted'] and not audit['original_checkpoints_overwritten']
    assert {r['path']:r['sha256'] for r in audit['checkpoints']}=={r['path']:r['sha256'] for r in p['checkpoint_audit']}
    assert audit['original_resource']==dict(server='server-1',gpu=0,power_limit_watts=350.)
    harm=formal['frozen_material_subgroup_harm'];first=formal['summary']['AdaRES-Y101']['miou'];assert not harm and first<.515
    assert all(v['all']['scene_clustered_ci95'][0]<=0<=v['all']['scene_clustered_ci95'][1] for v in formal['comparisons'].values())
    review=root/'complete_scientific_review.json';assert not review.exists()
    tick.save(review,dict(complete=True,id='AdaRES-Y101',token=token,observed_at=datetime.now().astimezone().isoformat(),disposition='complete_inconclusive',FIRST_miou=first,target=.515,target_reached=False,full_native_passes=8,paired_scene_bootstrap_draws=10000,seed=1999,subgroup_material_harm=False,all_overall_comparison_CIs_include_zero=True,confirmed_improvement=False,mechanism_falsified=False,promotion=False,training_protocol='Original released backbone/seed1999/fixed30; best24 original checkpoint, no restart or epoch tuning',cost_summary=cost['summary'],parameters=cost['parameters'],parameter_delta=cost['parameters']['candidate']-cost['parameters']['parent'],serial_scene_pairs_verified=True,native_memory_only_stress_batches=4,native_memory_only_updates_including_warmups=20,full_cost_and_postaudit_complete=True,original_resource=audit['original_resource'],source_and_original_checkpoint_bytes_preserved=True,limitations=['One trained seed/checkpoint per method; native repeats quantify observed prediction noise, not seed variance or a universal bound.','Point estimates below Y36/W2 and all overall paired CIs cross zero; no demonstrated improvement or equivalence, and no prespecified subgroup harm.','Frozen TRAIN8 plus two largest-pair scenes; serial warm2/read3 and separate batch2 memory-only native stress; not a full-data memory bound.','Sequential candidate/parent order and descriptive p95 do not establish causal hardware effects or Pareto dominance.'],evidence=[ref(__file__),formal_ref,c['post_training_freeze'],c['terminal_evidence'],c['terminal_archive_binding'],c['frozen_trained_cost_plan'],job['program'],job['launch'],ref(meta),ref(root/'trained_cost.json'),ref(root/'postaudit.json'),ref(root/'worker.log')]))
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        assert all(ref(path)==r for path,r in immutable.items())
        w=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in w['candidates'] if r['id']=='AdaRES-Y101');assert c['stage']=='validation' and not c.get('complete_scientific_review')
        c.update(stage='complete',scientific_status='complete_inconclusive',full_trained_cost_result=ref(root/'trained_cost.json'),post_measurement_audit=ref(root/'postaudit.json'),complete_scientific_review=ref(review),completed_at=datetime.now().astimezone().isoformat(),next_action='Original fixed30, all eight native frozen passes, corrected paired/subgroup analysis, trained serial/native stress cost and source/checkpoint postaudit COMPLETE. Retain inconclusive result and all negatives; no promotion, mechanism falsification, restart, tuning or repeated GPU/terminal tests. Preserve original receipts/goal and old GPU1/Y57 pauses.')
        c['frozen_trained_cost_job']['complete']=True;c['frozen_trained_cost_job']['native_progress_confirmed']=True;c['evidence'].extend([ref(review),ref(meta),ref(root/'trained_cost.json'),ref(root/'postaudit.json'),ref(root/'worker.log'),ref(__file__)])
        tick.save(ROOT/'pipeline/research_work.json',w)
        exp_path=ROOT/'pipeline/experiments/AdaRES-Y101.json';backup=ROOT/('pipeline/snapshots/AdaRES-Y101_experiment_before_scientific_'+token+'.json');assert not backup.exists();backup.write_bytes(exp_path.read_bytes())
        experiment=monitor.load_experiment(str(ROOT),'AdaRES-Y101');assert experiment['status']=='done_complete';experiment.update(formal_FIRST_miou=first,complete_scientific_review=ref(review),full_frozen_evaluation_result=formal_ref,full_trained_cost_result=ref(root/'trained_cost.json'),post_measurement_audit=ref(root/'postaudit.json'),scientific_disposition='complete_inconclusive');monitor.save_experiment(str(ROOT),experiment)
        assert all(ref(path)==r for path,r in immutable.items())
    finally:writer.close()
    print(json.dumps(dict(review=ref(review),disposition='complete_inconclusive',FIRST_miou=first,target_reached=False,subgroup_harm=False,cost_summary=cost['summary'],parameters=cost['parameters'],source_and_checkpoint_bytes_preserved=True,promotion=False)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--token',required=True);complete(parser.parse_args().token)
