"""Prepare one independent W2 full-appearance bypass around its visual-memory MLP."""
import argparse
import ast
import base64
import hashlib
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from prepare_y96_seed_query_skip import PARENT, NAMES
from prepare_y98_appearance_mask_skip import once
from y101_deployment import Y101, git
CID = 'AdaRES-Y103'
PACKAGE = 'adares_y103_spatial_input_skip'
MODULE = 'class SpatialInputSkip(nn.Module):\n    """Keep complete native spatial MLP; all sparse channels bypass it in parallel."""\n    def __init__(self,input_dim,d_model):\n        super().__init__()\n        self.weight=nn.Parameter(torch.zeros(input_dim,d_model))\n    def forward(self,projected,spatial):\n        return projected+spatial@self.weight\n\n\n'
EDITS = (('class DEC(nn.Module):', 'class SpatialInputSkip(nn.Module):\n    """Keep complete native spatial MLP; all sparse channels bypass it in parallel."""\n    def __init__(self,input_dim,d_model):\n        super().__init__()\n        self.weight=nn.Parameter(torch.zeros(input_dim,d_model))\n    def forward(self,projected,spatial):\n        return projected+spatial@self.weight\n\n\nclass DEC(nn.Module):'), ('        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)', '        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)\n        self.spatial_input_skip = SpatialInputSkip(self.input_proj[0].in_features, d_model)'), ('        inst_feats = self.input_proj(x)', '        inst_feats = self.spatial_input_skip(self.input_proj(x), x)'))

def files():
    return [ROOT / 'AdaRES/ipdn/model_paper5' / PACKAGE / n for n in NAMES] + [ROOT / 'AdaRES/configs_song/model_paper5' / (PACKAGE + '_res.yaml')]

def bodies():
    from prepare_y100_appearance_input_skip import bodies as original_bodies
    from create_y103_preparation import sub
    cpu, native = original_bodies()
    original_core = (ROOT / 'scripts/y100_observer_core.py').read_text(encoding='utf-8')
    new_core = (ROOT / 'scripts/y103_observer_core.py').read_text(encoding='utf-8')
    changes = {'AppearanceInputSkip': 'SpatialInputSkip', 'appearance_input_skip': 'spatial_input_skip', 'appearance_input': 'spatial_input', '164891781': '164637829', '262144': '8192', "'AdaRES-Y100'": repr(CID), 'original_appearance_projection_and_fusion_consumers_verified': 'original_spatial_projection_and_fusion_consumers_verified'}
    result = []
    for body in (cpu, native):
        assert body.count(original_core) == 1
        body = body.replace(original_core, '# SHARED_NEW_SPATIAL_CORE')
        body = sub(body, changes).replace('# SHARED_NEW_SPATIAL_CORE', new_core)
        ast.parse(body, feature_version=(3, 8))
        assert body.count(new_core) == 1
        result.append(body)
    return tuple(result)

def prepare(token):
    lease(token, 800)
    view = tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')
    assert view['experiment_first']['enabled'] and view['research']['creation_budget'] >= 1
    assert git('rev-parse', 'HEAD').decode().strip() == Y101
    work = tick.load(ROOT / 'pipeline/research_work.json')
    assert not any((r['id'] == CID for r in work['candidates']))
    before = {p: ref(p) for p in ('pipeline/queue.json', 'pipeline/running.json', 'pipeline/goal.json', 'SERVERS.yaml')}
    index = git('ls-files', '--stage', '-z')
    queue = monitor.load_queue(str(ROOT))
    for name, cid, pid in (('server-1', 'AdaRES-Y101', 2263581), ('server-2', 'AdaRES-Y100', 19305)):
        task = next((r for r in queue['gpu_queues'][name] if r['id'] == cid))
        assert task['status'] == 'running' and task['pid'] == pid and task['launch_verified']
    paths = files()
    parent = paths[0].parent.parent / PARENT
    assert not paths[0].parent.exists() and (not any((p.exists() for p in paths)))
    source = (parent / 'dec.py').read_text(encoding='utf-8')
    for old, new in EDITS:
        source = once(source, old, new)
    ast.parse(source, feature_version=(3, 8))
    cpu, native = bodies()
    paths[0].parent.mkdir()
    for path in paths[:-1]:
        raw = source.encode() if path.name == 'dec.py' else (parent / path.name).read_bytes()
        if path.name == '__init__.py':
            raw = b'# AdaRES-Y100: full raw appearance bypass beside the original visual-memory MLP.\nfrom .ipdn import IPDN\n'
        path.write_bytes(raw)
    paths[-1].write_bytes((ROOT / 'AdaRES/configs_song/model_paper5' / (PARENT + '_res.yaml')).read_bytes())
    assert list((ROOT / 'AdaRES/configs_song').rglob(paths[-1].name)) == [paths[-1]]

    def encoded(path):
        return dict(base64=base64.b64encode(path.read_bytes()).decode(), sha256=ref(path)['sha256'])
    payload = dict(variant=dict(id=CID, package=PACKAGE, config=paths[-1].relative_to(ROOT / 'AdaRES').as_posix(), python_files={p.name: encoded(p) for p in paths[:-1]}, config_data=encoded(paths[-1])), parent_config='configs_song/model_paper5/' + PARENT + '_res.yaml', goal_sha256=ref('pipeline/goal.json')['sha256'], parent_dec_lf_sha256=hashlib.sha256((parent / 'dec.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest(), train_source_lf_sha256=hashlib.sha256((ROOT / 'AdaRES/tools/train.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest())
    program = ROOT / ('pipeline/checks/y103_cpu_' + token + '_program.py')
    native_path = ROOT / ('pipeline/checks/y103_native_' + token + '_body.py')
    history = ROOT / ('pipeline/checks/y103_targeted_history_' + token + '.json')
    assert not any((p.exists() for p in (program, native_path, history)))
    program.write_text('PAYLOAD=' + repr(payload) + '\n' + cpu, encoding='utf-8', newline='\n')
    native_path.write_text(native, encoding='utf-8', newline='\n')
    ids = {'AdaRES-Y68', 'AdaRES-Y71', 'AdaRES-Y72', 'AdaRES-Y80', 'AdaRES-Y84', 'AdaRES-Y96', 'AdaRES-Y97', 'AdaRES-Y98', 'AdaRES-Y99'}
    tick.save(history, dict(complete=True, scope='Targeted input/destination comparison, not a novelty proof.', matches=[{k: r.get(k) for k in ('id', 'stage', 'hypothesis', 'matched_control')} for r in work['candidates'] if r['id'] in ids], parent_source=ref(parent / 'dec.py'), distinction='Y103 routes all32 original sparse features beside complete input_proj BEFORE original sum_norm/visual memory. Y99 spatial-to-static-mask, Y100 appearance-to-memory, Y102 centered geometry-to-mask and Y71 projected product have different sources or destinations. No features removed, no combination, tuning or reopening.'))
    work['candidates'].append(dict(id=CID, profile='experiment_first', stage='implementation', comparison_baseline='AdaRES-W2', implementation_package=PACKAGE, config=payload['variant']['config'], hypothesis='Whole original32D sparse features enter visual memory through the complete original input_proj MLP. An independent zero-initialized full32 linear bypass beside that same MLP may ease spatial visual-memory fitting before the same original sum_norm. Original MLP and all modalities remain; no missing information, bug, novelty or benefit is presumed.', matched_control='Independent W2: spatial=input_proj(all32 sparse)+all32 sparse@zeroW32x256 BEFORE SAME sum_norm(spatial+original full1024 appearance), +8192/total164637829/noRNG. Preserve complete original spatial/appearance/mask projections, all raw channels/points, static7/query generator/refinements/sampling/SCG/k/gamma/shared heads/hard gates/targets/loss/selector/backbone/data/seed1999fixed30/native YAML. No feature subset, threshold, temperature, loss, epoch or validation scan.', historical_scope='Y103 connects whole original spatial features to visual memory before sum_norm. Y99 connects spatial features directly to static masks; Y100 bypasses appearance projection; Y102 connects centered scene coordinates to masks; Y71 multiplies projected modalities. Independent W2, no combination, tuning, reopened rejected route or relaxed numerical control.', prepared_native_body=ref(native_path), observer_core=ref('scripts/y103_observer_core.py'), next_action='Run first exact original CPU YAML/fullmodel/releasedbackbone/TRAIN collation and scalar float32/64 no-RNG/permutation/sign/row-local controls, seven-stage W2 anchors/input gradients, causal bypass and two finite synthetic decoder updates; native GPU TRAIN still requires authorized idle exclusive GPU0 and actual two-step native PASS before admission.', evidence=[ref(p) for p in paths] + [ref(program), ref(native_path), ref(history), ref(__file__), ref('scripts/y103_observer_core.py')]))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    assert all((ref(p) == r for p, r in before.items())) and git('ls-files', '--stage', '-z') == index
    print(json.dumps(dict(candidate=CID, program=ref(program), native_body=ref(native_path), CPU_executed=False, native_executed=False, admitted=False)))
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--token', required=True)
    token = p.parse_args().token
    lease(token, 800)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        prepare(token)
    finally:
        writer.close()
