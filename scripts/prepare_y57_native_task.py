"""Prepare source-bound Y57 batch2 diagnostics without touching training."""
import argparse
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, 'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import authorize_prepared_candidates as scoped
import foreground_tick as tick
import pipeline_monitor as monitor

BASE_PROGRAM = 'pipeline/analyses/y56_native_batch2_222ed2a6e80c43f6b1de2d050be97fa0_program.py'
BASE_BODY = 'scripts/probe_y56_native_batch2_incidence.py'
CAPTURE = 'pipeline/analyses/y46_native_capture_d179_d179d824a64943e0b06bfa90de0caf52.json'
COMMON = r'''
import base64
import zlib
from torch_scatter import scatter_mean

def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def pair_targets(batch, expected):
    targets = []
    expression = 0
    for scene_index, length in enumerate(batch['scenes_len']):
        start, end = [int(v) for v in batch['batch_offsets'][scene_index:scene_index+2]]
        selected = (batch['superpoints'] >= start) & (batch['superpoints'] < end)
        membership = batch['superpoints'][selected].long() - start
        total = torch.bincount(membership, minlength=end-start)
        coordinates = scatter_mean(batch['coords_float'][selected], membership, dim=0)
        for _ in range(int(length)):
            row = expected[expression]
            assert batch['scan_ids'][scene_index] == row['scene']
            assert str(batch['ann_ids'][expression]) == row['ann_id']
            assert batch['object_idss'][expression] == row['object_ids']
            gt = batch['gt_pmasks'][expression].reshape(-1).float()
            assert gt.numel() == membership.numel()
            positive = torch.bincount(membership, weights=(gt > .5).float(), minlength=len(total))
            for value, key in ((gt, 'reconstructed_GT_sha256'), (membership, 'membership_sha256'),
                               (total, 'total_counts_sha256'), (positive, 'positive_counts_sha256')):
                assert tensor_hash(value) == row[key], 'Pair binding mismatch: ' + key
            packed = FROZEN['coordinate_reference'][row['scene']]
            raw = zlib.decompress(base64.b64decode(packed['zlib_base64'], validate=True))
            assert hashlib.sha256(raw).hexdigest() == packed['sha256']
            reference = torch.from_numpy(np.frombuffer(raw, dtype=np.dtype(packed['dtype'])).reshape(packed['shape']).copy())
            error = float((coordinates - reference).abs().max())
            bound = 32 * torch.finfo(coordinates.dtype).eps * max(1., float(coordinates.abs().max()))
            assert error <= bound
            raw_coordinates_sha = tensor_hash(batch['coords_float'][selected])
            if 'raw_point_coordinates_sha256' in row:
                assert raw_coordinates_sha == row['raw_point_coordinates_sha256']
            metadata = {key: row[key] for key in ('scene', 'ann_id', 'object_ids', 'reconstructed_GT_sha256',
                'membership_sha256', 'total_counts_sha256', 'positive_counts_sha256')}
            metadata.update(raw_point_coordinates_sha256=raw_coordinates_sha, coordinate_error=error,
                            coordinate_bound=bound, single_to_pair_labels_exact=True)
            targets.append(dict(total=total, positive=positive, metadata=metadata))
            expression += 1
    assert expression == len(expected)
    return targets
'''
CPU = r'''
import os
assert P['physical_gpu'] is None and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import copy
import hashlib
import json
from pathlib import Path
import gorilla
import numpy as np
import torch
import yaml
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
torch.set_num_threads(1)
root = Path(P['root'])
assert yaml.safe_load((root/FROZEN['config_path']).read_text()) == FROZEN['config_values']
for row in FROZEN['sources']:
    assert hashlib.sha256((root/row['path']).read_bytes().replace(b'\r\n',b'\n')).hexdigest() == row['lf_sha256']
cfg = gorilla.Config.fromfile(str(root/FROZEN['config_path']))
dataset = build_dataset(cfg.data.train.copy(),get_root_logger())
'''
CPU_LOOP = r'''
rows=[]
for pair in range(0,8,2):
    scenes=FROZEN['scenes'][pair:pair+2]
    items=[]
    for scene in scenes:
        gorilla.set_random_seed(1999)
        items.append(dataset[scene['dataset_index']])
    batch=dataset.collate_fn(copy.deepcopy(items))
    repeated=dataset.collate_fn(copy.deepcopy(items))
    expected=[row for scene in scenes for row in FROZEN['labels'] if row['scene']==scene['scan_id']]
    left,right=pair_targets(batch,expected),pair_targets(repeated,expected)
    assert [row['metadata'] for row in left] == [row['metadata'] for row in right]
    rows.extend(row['metadata'] for row in left)
assert len(rows)==32 and not torch.cuda.is_initialized()
result=dict(complete=True,id='AdaRES-Y57',CPU_native_batch2_dataflow=True,
    pairs=4,expressions=32,all_GT_mass_membership_geometry_bindings_exact=True,
    max_coordinate_error=max(row['coordinate_error'] for row in rows),
    rows=rows,gpu_tests=0,full_native_forward=False,full_native_backward=False,
    training_launches=0,admission=False,
    limitation='Exact current single-to-pair point-GT/count/membership/raw geometry binding only; no sparse forward, causal peer intervention, model benefit or retroactive old capture provenance.')
Path(P['output']).write_text(json.dumps(result),encoding='utf-8')
'''
OBSERVER = r'''
def observe(module,inputs,output):
    query,masks,padding,mass=inputs
    assert query.shape==(8,128,256) and output.shape==(8,128,2)
    layer=len(calls)
    for index,(scene,ann) in enumerate(expression_ids):
        valid=~padding[index]
        local_masks=masks[index,:,valid]
        local_mass=mass[index,valid]
        target=native_targets[index]
        total=target['total'].to(query.device,dtype=query.dtype)
        positive=target['positive'].to(query.device,dtype=query.dtype)
        assert torch.equal(local_mass,total) and len(total)==int(valid.sum())
        assert scene==target['metadata']['scene'] and ann==target['metadata']['ann_id']
        probabilities=local_masks.sigmoid()
        weighted=probabilities*local_mass.unsqueeze(0)
        areas=weighted.sum(-1)
        intersection=weighted@probabilities.T
        overlap=intersection/(areas[:,None]+areas[None,:]-intersection).clamp_min(1e-6)
        overlap.fill_diagonal_(0)
        competition=overlap/overlap.sum(-1,keepdim=True).clamp_min(1e-6)
        normalized=module.query_norm(query[index])
        context=competition@normalized
        diagonal_variance=(competition@normalized.square()-context.square()).clamp_min(0)
        dispersion=diagonal_variance.square().sum(-1).sqrt()
        binary=(probabilities>=.5).to(query.dtype)
        numerator=(binary*positive.unsqueeze(0)).sum(-1)
        quality=numerator/((binary*total.unsqueeze(0)).sum(-1)+positive.sum()-numerator+1e-6)
        confidence=output[index].softmax(-1)[:,1]
        selected=int(confidence.argmax())
        records.append(dict(pair=pair//2,layer=layer,scene=scene,ann_id=ann,
            object_ids=target['metadata']['object_ids'],GT_binding=target['metadata'],
            confidence=confidence.detach().cpu().tolist(),point_quality=quality.detach().cpu().tolist(),
            dispersion=dispersion.detach().cpu().tolist(),native_argmax_query=selected,
            selected_point_quality=float(quality[selected]),oracle_point_quality=float(quality.max())))
    calls.append(layer)
'''
NATIVE_RESULT = r'''
result=dict(complete=True,id='AdaRES-Y57',native_TRAIN_batch2=True,actual_sparse_FPS=True,
    fixed_scenes=8,expressions=32,indicator_calls=28,expression_layer_records=224,
    primary_layer=6,checkpoint_sha256=FROZEN['checkpoint_sha256'],
    parameters_unchanged=True,buffers_restored=True,rows=records,
    elapsed_seconds=time.perf_counter()-started,peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
    optimizer_steps=0,weights_saved=False,model_or_config_changes=False,training_launches=0,admission=False,
    full_native_backward=False,
    limitation='Fresh source/checkpoint/current32GT-bound all7layer observational capture only. Final layer6 primary; other layers exploratory, no layer selection. Associations are not conditional incremental evidence, reachable counterfactuals, training benefit, native backward/resources gate or novelty.')
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--token',required=True)
    parser.add_argument('operation',choices=('pair_prepare','pair_run','native_prepare'))
    args=parser.parse_args()
    scoped.lease(args.token)
    base='pipeline/analyses/y57_pair_binding_'+args.token
    pair_plan,pair_program,pair_result=(base+suffix for suffix in ('_prereg.json','_program.py','_result.json'))
    if args.operation=='pair_run':
        plan=tick.load(ROOT/pair_plan)
        assert scoped.ref(pair_program)==plan['program']
        command=[sys.executable,'-B',str(ROOT/'scripts/run_adares_authorized_probe.py'),
                 'y57_pair_binding',str(ROOT/pair_program),'--token',args.token,'--seconds','300']
        response=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',timeout=380)
        writer=monitor.acquire_lock(str(ROOT));assert writer is not None
        try:
            scoped.lease(args.token)
            execution=base+'_execution.json'
            assert not (ROOT/execution).exists()
            tick.save(ROOT/execution,dict(complete=response.returncode==0,observed_at=datetime.now().astimezone().isoformat(),
                program=plan['program'],returncode=response.returncode,stdout=response.stdout,stderr=response.stderr))
            work=tick.load(ROOT/'pipeline/research_work.json')
            candidate=next(c for c in work['candidates'] if c['id']=='AdaRES-Y57')
            candidate['evidence'].append(scoped.ref(execution))
            actual='pipeline/analyses/y57_pair_binding_'+args.token+'.json'
            if response.returncode==0:
                measured=tick.load(ROOT/actual)
                tick.save(ROOT/pair_result,measured)
                candidate['evidence'].append(scoped.ref(pair_result))
                candidate['next_action']='Review exact CPU single-to-pair GT/count/geometry binding, then freeze the fresh native batch2 all7layer observational task capture. No model or admission.'
            tick.save(ROOT/'pipeline/research_work.json',work)
        finally:writer.close()
        print(response.stdout)
        if response.returncode:
            print(response.stderr)
            raise ValueError('Pair CPU test failed; preserve original logs and plan')
        return
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        scoped.lease(args.token)
        work=tick.load(ROOT/'pipeline/research_work.json')
        candidate=next(c for c in work['candidates'] if c['id']=='AdaRES-Y57')
        assert candidate['stage'] in ('design','validation')
        if args.operation=='pair_prepare':
            result_path='pipeline/analyses/y57_gt_binding_result_'+args.token+'.json'
            result=tick.load(ROOT/result_path)
            assert result['complete'] and result['all_bindings_passed'] and result['expressions']==32
            old=ast.parse((ROOT/BASE_PROGRAM).read_text(encoding='utf-8')).body[0]
            frozen=ast.literal_eval(old.value)
            frozen['config_path']='configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml'
            frozen['labels']=[{key:row[key] for key in ('scene','ann_id','object_ids','reconstructed_GT_sha256',
                'membership_sha256','total_counts_sha256','positive_counts_sha256')} for row in result['rows']]
            cached=tick.load(ROOT/CAPTURE)
            frozen['coordinate_reference']={row['scan_id']:row['expressions'][0]['packed_native_inputs']['coordinates'] for row in cached['rows']}
            program='FROZEN='+repr(frozen)+'\n'+CPU+'\n'+COMMON+'\n'+CPU_LOOP
            ast.parse(program,feature_version=(3,8));compile(program,pair_program,'exec')
            assert not (ROOT/pair_program).exists() and not (ROOT/pair_plan).exists()
            (ROOT/pair_program).write_text(program,encoding='utf-8',newline='\n')
            tick.save(ROOT/pair_plan,dict(complete=True,id='AdaRES-Y57',preregistered_at=datetime.now().astimezone().isoformat(),
                program=scoped.ref(pair_program),current_GT=scoped.ref(result_path),base_native=scoped.ref(BASE_PROGRAM),
                fixed_test='Same8scenes/32expressions,4CPU native batch2 collates and repeat; exact point-GT/membership/total/positive hashes to single reconstruction,32eps centroid bound, record raw point-coordinate hash. No new quality association or sampling.',frozen=frozen))
            candidate['evidence'].extend([scoped.ref(pair_plan),scoped.ref(pair_program)])
            candidate['next_action']='Run the same frozen32expression CPU native-batch2 GT binding immediately; do not infer full-model forward or native backward.'
            tick.save(ROOT/'pipeline/research_work.json',work)
            print(json.dumps(dict(plan=scoped.ref(pair_plan),program=scoped.ref(pair_program))))
            return
        plan=tick.load(ROOT/pair_plan)
        measured=tick.load(ROOT/pair_result)
        assert measured['complete'] and measured['all_GT_mass_membership_geometry_bindings_exact'] and measured['expressions']==32
        frozen=dict(plan['frozen']);frozen['labels']=measured['rows']
        tree=ast.parse((ROOT/BASE_BODY).read_text(encoding='utf-8'))
        class Transform(ast.NodeTransformer):
            def visit_FunctionDef(self,node):
                return ast.parse(OBSERVER).body[0] if node.name=='observe' else self.generic_visit(node)
            def visit_Assign(self,node):
                targets={target.id for target in node.targets if isinstance(target,ast.Name)}
                if 'records' in targets:return [*ast.parse(COMMON).body,node]
                if 'result' in targets:return ast.parse(NATIVE_RESULT).body
                if 'batch' in targets:
                    extra=ast.parse("native_targets=pair_targets(batch,[row for scene in FROZEN['scenes'][pair:pair+2] for row in FROZEN['labels'] if row['scene']==scene['scan_id']])").body
                    return [node,*extra]
                if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='register_forward_pre_hook':
                    node.value.func.attr='register_forward_hook'
                return self.generic_visit(node)
        tree=ast.fix_missing_locations(Transform().visit(tree))
        native_path='pipeline/analyses/y57_native_task_'+args.token+'_program.py'
        native_plan='pipeline/analyses/y57_native_task_'+args.token+'_prereg.json'
        program='FROZEN='+repr(frozen)+'\n'+ast.unparse(tree)+'\n'
        ast.parse(program,feature_version=(3,8));compile(program,native_path,'exec')
        denied=[]
        for gpu in (None,2):
            try:exec(compile(program,native_path,'exec'),dict(P={'physical_gpu':gpu}))
            except AssertionError:denied.append(gpu)
            else:raise ValueError('Native diagnostic GPU guard did not reject')
        assert not (ROOT/native_path).exists() and not (ROOT/native_plan).exists()
        (ROOT/native_path).write_text(program,encoding='utf-8',newline='\n')
        tick.save(ROOT/native_plan,dict(complete=True,id='AdaRES-Y57',preregistered_at=datetime.now().astimezone().isoformat(),
            program=scoped.ref(native_path),source_manager=scoped.ref('scripts/prepare_y57_native_task.py'),
            inherited_native_safety=scoped.ref(BASE_PROGRAM),CPU_pair_binding=scoped.ref(pair_result),
            observation='Same8TRAIN scenes/32expressions/Q128;4native batch2 forwards,all7indicator layers/224records. Source/checkpoint/GT/count/raw geometry bindings, BN buffers restored and parameters unchanged; no optimizer or saved weights.',
            primary='Final layer6 descriptor/confidence/actual point-quality associations; layers0-5 exhaustive exploratory. No layer, query, direction or threshold selection.',
            limitation='Observational provenance/task-association capture only; does not establish conditional incremental evidence, reachable counterfactuals, harm, trained benefit, native backward/resource gate or novelty.',
            syntax_python38=True,compile_pass=True,missing_or_unauthorized_GPU_entry_rejected=(denied==[None,2]),
            actual_GPU_execution=False,full_native_backward=False,training_launches=0,admission=False))
        candidate['evidence'].extend([scoped.ref(native_path),scoped.ref(native_plan)])
        candidate['next_action']='Execute frozen Y57 native-task all7layer batch2 capture only after current authorized exclusive idle GPU/source/checkpoint/GT checks; observational evidence, not a model or admission. No training may be interrupted.'
        tick.save(ROOT/'pipeline/research_work.json',work)
        print(json.dumps(dict(program=scoped.ref(native_path),plan=scoped.ref(native_plan),native_executed=False,entry_guards_passed=True)))
    finally:writer.close()


if __name__=='__main__':main()
