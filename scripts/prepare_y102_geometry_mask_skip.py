"""Prepare independent all-scene centered-coordinate features beside native masks."""
import argparse,ast,base64,hashlib,json
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from prepare_y96_seed_query_skip import PARENT,NAMES
from prepare_y98_appearance_mask_skip import once
from y100_deployment import Y100,git
CID='AdaRES-Y102';PACKAGE='adares_y102_geometry_mask_skip'
MODULE='''class GeometryMaskSkip(nn.Module):
    """Retain native static masks and add all three original scene-relative coordinates."""
    def __init__(self,d_model):
        super().__init__()
        self.weight=nn.Parameter(torch.zeros(3,d_model))

    def forward(self,mask_base,scene_pos,padding,point_mass):
        weights=point_mass.masked_fill(padding,0.0)
        clean=scene_pos.masked_fill(padding.unsqueeze(-1),0.0)
        totals=weights.sum(1,keepdim=True)
        center=(clean*weights.unsqueeze(-1)).sum(1)/totals.clamp_min(1.0)
        relative=(clean-center.unsqueeze(1)).masked_fill(padding.unsqueeze(-1)|(totals<=0).unsqueeze(-1),0.0)
        return mask_base+relative@self.weight


'''
EDITS=(('class DEC(nn.Module):',MODULE+'class DEC(nn.Module):'),
    ('        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)','        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)\n        self.geometry_mask_skip = GeometryMaskSkip(d_model)'),
    ('        sp_pos, _ = self.get_batches(sp_pos, batch_offsets)','        sp_pos, _ = self.get_batches(sp_pos, batch_offsets)\n        mask_feats = self.geometry_mask_skip(mask_feats, sp_pos, batch_mask, point_mass)'))

def files():return [ROOT/'AdaRES/ipdn/model_paper5'/PACKAGE/n for n in NAMES]+[ROOT/'AdaRES/configs_song/model_paper5'/(PACKAGE+'_res.yaml')]

def bodies():
    prefix,m,_=(ROOT/'scripts/probe_y83_cpu.py').read_text(encoding='utf-8').partition('parameters=sum(');assert m
    _,m,a=(ROOT/'scripts/probe_y84_cpu.py').read_text(encoding='utf-8').partition('kwargs = dict(cfg.model.dec');assert m
    a,m,_=('kwargs = dict(cfg.model.dec'+a).partition('parent.zero_grad(set_to_none=True);');assert m;a=a.replace('visual_query_init.','geometry_mask_skip.')
    core=(ROOT/'scripts/y102_observer_core.py').read_text(encoding='utf-8')
    tail=(ROOT/'scripts/probe_y96_cpu_tail.py').read_text(encoding='utf-8').replace('seed_query_skip','geometry_mask_skip')
    tail=once(tail,'    changed = decoder(**inputs)',"    changed = decoder(**inputs)\n    assert torch.equal(parts(changed)[0]['scores'], parts(baseline)[0]['scores'])")
    tail=tail.replace('lambda generated, seed: generated','lambda base, pos, padding, mass: base')
    tail=once(tail,'        value = decoder(**feed)',"        state['raw_spatial_expected'] = feed['x']\n        state['raw_appearance_expected'] = feed['feats_2d']\n        state['raw_coordinates_expected'] = feed['sp_pos']\n        state['raw_mass_expected'] = feed['point_mass']\n        state['batch_offsets_expected'] = feed['batch_offsets']\n        value = decoder(**feed)")
    for x,y in (('additional_parameters=65536','additional_parameters=768'),('original_sampled_seed_and_generator_consumers_verified','original_geometry_mass_and_static_mask_consumers_verified'),('initial_and_first_coarse_consumers_verified','all_seven_static_mask_consumers_verified'),('nonzero_seed_skip_changes_initial_and_final_masks','nonzero_geometry_skip_changes_initial_and_final_masks'),('CPU_initialization_paths','CPU_geometry_mask_paths')):tail=tail.replace(x,y)
    cpu=prefix+(ROOT/'scripts/probe_y102_cpu_controls.py').read_text(encoding='utf-8')+'\n'+a+'\n'+tail.replace('# SHARED_OBSERVER_CORE',core)
    native,m,_=(ROOT/'scripts/probe_y83_native_smoke.py').read_text(encoding='utf-8').partition('losses=[];');assert m
    native=native.replace("'AdaRES-Y83'",repr(CID)).replace('164632197','164630405')
    native+='\nlosses,gradient_checks,batches,modes,step_rows,path_rows,state=[],[],[],[],[],[],{}\nhead=model.dec.indi_embedding;residual=model.dec.geometry_mask_skip;original_prediction=model.dec.prediction_head\n'
    tail=(ROOT/'scripts/probe_y96_native_tail.py').read_text(encoding='utf-8')
    tail=once(tail,'    reset_state()\n',"    reset_state()\n    state['raw_spatial_expected'] = args[0]\n    state['raw_appearance_expected'] = args[6]\n    state['raw_coordinates_expected'] = args[5]\n    state['raw_mass_expected'] = args[7]\n    state['batch_offsets_expected'] = args[2]\n")
    tail=tail.replace('initialization_paths=path_rows, initialization_calls=2','geometry_mask_paths=path_rows, geometry_mask_calls=2').replace('generator_layer_calls=10','original_visual_projection_and_mask_calls=2').replace('static_mask_features_preserved=True','original_base_mask_and_static_reuse_preserved=True')
    native+=core+'\n'+tail
    for text in (cpu,native):ast.parse(text,feature_version=(3,8));assert text.count(core)==1
    return cpu,native

def prepare(token):
    lease(token,800);v=tick.load(ROOT/'pipeline/headless/.tick_model_view.json');assert v['experiment_first']['enabled'] and v['research']['creation_budget']>=1
    assert git('rev-parse','HEAD').decode().strip()==Y100
    w=tick.load(ROOT/'pipeline/research_work.json');assert not any(r['id']==CID for r in w['candidates'])
    before={p:ref(p) for p in ('pipeline/queue.json','pipeline/running.json','pipeline/goal.json','SERVERS.yaml')};index=git('ls-files','--stage','-z')
    q=monitor.load_queue(str(ROOT))
    for name,cid,pid in (('server-1','AdaRES-Y99',2980836),('server-2','AdaRES-Y100',19305)):
        t=next(r for r in q['gpu_queues'][name] if r['id']==cid);assert t['status']=='running' and t['pid']==pid and t['launch_verified']
    earlier=next(r for r in w['candidates'] if r['id']=='AdaRES-Y101');assert earlier['stage']=='blocked' and earlier['blocker_kind']=='exclusive_gpu_slot' and not earlier.get('native_smoke_result')
    paths=files();parent=paths[0].parent.parent/PARENT;assert not paths[0].parent.exists() and not any(p.exists() for p in paths)
    text=(parent/'dec.py').read_text(encoding='utf-8')
    for a,b in EDITS:text=once(text,a,b)
    ast.parse(text,feature_version=(3,8));cpu,native=bodies();paths[0].parent.mkdir()
    for p in paths[:-1]:
        raw=text.encode() if p.name=='dec.py' else (parent/p.name).read_bytes()
        if p.name=='__init__.py':raw=b'# AdaRES-Y102: original scene-centered coordinates directly beside static masks.\nfrom .ipdn import IPDN\n'
        p.write_bytes(raw)
    paths[-1].write_bytes((ROOT/'AdaRES/configs_song/model_paper5'/(PARENT+'_res.yaml')).read_bytes());assert list((ROOT/'AdaRES/configs_song').rglob(paths[-1].name))==[paths[-1]]
    def encoded(p):return dict(base64=base64.b64encode(p.read_bytes()).decode(),sha256=ref(p)['sha256'])
    payload=dict(variant=dict(id=CID,package=PACKAGE,config=paths[-1].relative_to(ROOT/'AdaRES').as_posix(),python_files={p.name:encoded(p) for p in paths[:-1]},config_data=encoded(paths[-1])),parent_config='configs_song/model_paper5/'+PARENT+'_res.yaml',goal_sha256=ref('pipeline/goal.json')['sha256'],parent_dec_lf_sha256=hashlib.sha256((parent/'dec.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),train_source_lf_sha256=hashlib.sha256((ROOT/'AdaRES/tools/train.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest())
    program=ROOT/('pipeline/checks/y102_cpu_'+token+'_program.py');nb=ROOT/('pipeline/checks/y102_native_'+token+'_body.py');history=ROOT/('pipeline/checks/y102_targeted_history_'+token+'.json');assert not any(p.exists() for p in (program,nb,history))
    program.write_text('PAYLOAD='+repr(payload)+'\n'+cpu,encoding='utf-8',newline='\n');nb.write_text(native,encoding='utf-8',newline='\n')
    ids={'AdaRES-Y27','AdaRES-Y31','AdaRES-Y53','AdaRES-Y55','AdaRES-Y69','AdaRES-Y77','AdaRES-Y85','AdaRES-Y94','AdaRES-Y98','AdaRES-Y99'}
    tick.save(history,dict(complete=True,scope='Targeted source/input/destination comparison; not novelty proof.',matches=[{k:r.get(k) for k in ('id','stage','hypothesis','matched_control')} for r in w['candidates'] if r['id'] in ids],parent_source=ref(parent/'dec.py'),distinction='Y102 routes all original full-scene3D centered coordinates into original static mask embeddings AFTER original batching. Y77 routes only sampled original query positions into initial queries; Y69/Y94 act on attention logits; Y27/Y31 mask-derived geometry acts on confidence. No new geometry labels or changed data/padding/SCG; no combination or reopening.'))
    w['candidates'].append(dict(id=CID,profile='experiment_first',stage='implementation',comparison_baseline='AdaRES-W2',implementation_package=PACKAGE,config=payload['variant']['config'],
        hypothesis='Original scene coordinates influence native sampling and SCG attention, while static masks are projected from sparse features. A zero-initialized direct projection of all original superpoint coordinates relative to the point-mass scene centroid into static mask embeddings may ease spatial mask fitting. Existing sparse features and queries already carry geometry; no missing information, defect, novelty or benefit is presumed.',
        matched_control='Independent W2: batched_static=x_mask(x)_batched+(all_scene_pos-all_valid_point_mass_scene_center)@zeroW3x256 once, preserving original padding/zero-mass no-op and SAME seven mask consumers. +768/total164630405/noRNG. All three original coordinates/all valid masses/all original sparse and appearance features/all points remain; no feature selection. Original initial query/sampling/query positions/SCG/k/gamma/refinements/coarse-fine average/norms/shared confidence/hard-gate algorithm/targets/loss/selector/releasedbackbone/data/seed1999fixed30/native YAML intact. No threshold/temperature/loss/epoch/validation scan.',
        historical_scope='Not Y77 sampled-coordinate initial-query connection, Y69/Y94 attention geometry, Y27/Y31 mask-derived confidence, Y98 appearance-to-mask or Y99 sparse-to-mask. Original Y53/Y55 leakage rejections/numerical holds and all bans retained; no new labels, premise of bugs, combination/reopening/retry or tolerance relaxation.',prepared_native_body=ref(nb),observer_core=ref('scripts/y102_observer_core.py'),next_action='Run FIRST original CPU YAML/fullmodel/releasedbackbone/TRAIN collation and scalarfloat32/64 scene centroid/projection/permutation/translation/mass scale/paddedNaN/empty/scene-isolation/coordinate-mass gradients, seven-stage W2/input-gradient anchors, causal masks/exact initial scores/bypass restoration and two finite synthetic updates with exact original projection/mass/static7 consumers. Y101 precedes Y102; no GPU until matching original completion/full exact checkpoint/canonical archive and fresh authorized idle GPU0; actual native PASS required for admission.',evidence=[ref(p) for p in paths]+[ref(program),ref(nb),ref(history),ref(__file__),ref('scripts/y102_observer_core.py'),ref('scripts/probe_y102_cpu_controls.py')]))
    tick.save(ROOT/'pipeline/research_work.json',w);assert all(ref(p)==r for p,r in before.items()) and git('ls-files','--stage','-z')==index
    print(json.dumps(dict(candidate=CID,program=ref(program),native_body=ref(nb),CPU_executed=False,native_executed=False,admitted=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);a=p.parse_args();lease(a.token,800);lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:prepare(a.token)
    finally:lock.close()
