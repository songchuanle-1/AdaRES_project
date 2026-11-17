"""Prepare independent W2 visual-side mask readout without combining other routes."""
import argparse
import ast
import base64
import hashlib
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick

CID = 'AdaRES-Y92'
PACKAGE = 'adares_y92_mask_feature_quadratic'
PARENT = 'adares_w2_pointmass_competition'
NAMES = ('__init__.py', 'backbone.py', 'dec.py', 'ipdn.py', 'loss.py', 'sample_model.py')
MODULE = '''class MaskFeatureQuadratic(nn.Module):
    """One shared zero-anchored visual-side residual before static mask batching."""

    def __init__(self, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(d_model, d_model))

    def forward(self, features):
        return features + features.square() @ self.weight


'''
EDITS = (
    ('class DEC(nn.Module):', MODULE + 'class DEC(nn.Module):'),
    ('        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)',
     '        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)\n'
     '        self.mask_feature_quadratic = MaskFeatureQuadratic(d_model)'),
    ('        mask_feats = self.x_mask(x)',
     '        mask_feats = self.mask_feature_quadratic(self.x_mask(x))'))


def files():
    return [ROOT / 'AdaRES/ipdn/model_paper5' / PACKAGE / name for name in NAMES] + [
        ROOT / 'AdaRES/configs_song/model_paper5' / (PACKAGE + '_res.yaml')]


def bodies():
    preamble = (ROOT / 'scripts/probe_y85_cpu.py').read_text(encoding='utf-8').partition('parameters=sum(')[0]
    controls = (ROOT / 'scripts/probe_y88_cpu_controls.py').read_text(encoding='utf-8')
    controls = controls.replace('MaskQueryQuadratic', 'MaskFeatureQuadratic')
    controls = controls.replace('mask_query_quadratic', 'mask_feature_quadratic')
    controls = controls.replace('query_and_channel_permutations', 'point_and_channel_permutations')
    core = (ROOT / 'scripts/y92_observer_core.py').read_text(encoding='utf-8')
    cpu_tail = (ROOT / 'scripts/probe_y88_cpu_tail.py').read_text(encoding='utf-8')
    cpu_tail = cpu_tail.replace('    model.dec.prediction_head = original_prediction',
        '    model.dec.get_batches = original_batch\n    model.dec.prediction_head = original_prediction')
    cpu_tail = cpu_tail.replace('static_mask_and_original_hard_gate_preserved',
                              'shared_visual_features_and_original_hard_gate_preserved')
    cpu = preamble + controls + '\noriginal_batch = decoder.get_batches\n' + core + '\n' + cpu_tail
    native = (ROOT / 'scripts/probe_y83_native_smoke.py').read_text(encoding='utf-8').partition('losses=[];')[0]
    native = native.replace("'AdaRES-Y83'", "'AdaRES-Y92'").replace('164632197', '164695173')
    native += '\nlosses,gradient_checks,batches,modes,step_rows,path_rows,state=[],[],[],[],[],[],{}\n'
    native += 'head=model.dec.indi_embedding;projection=model.dec.mask_feature_quadratic\n'
    native += 'original_prediction=model.dec.prediction_head;original_batch=model.dec.get_batches\n'
    tail = (ROOT / 'scripts/probe_y88_native_tail.py').read_text(encoding='utf-8')
    tail = tail.replace('    model.dec.prediction_head = original_prediction',
        '    model.dec.get_batches = original_batch\n    model.dec.prediction_head = original_prediction')
    tail = tail.replace('projection_calls=14', 'projection_calls=2')
    tail = tail.replace('static_mask_features_preserved=True', 'shared_transformed_visual_mask_features_consumed=True')
    native += core + '\n' + tail
    for source in (cpu, native):
        ast.parse(source, feature_version=(3, 8))
    return cpu, native


def prepare(token):
    view = tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')
    assert view['experiment_first']['enabled'] and view['research']['creation_budget'] >= 1
    work = tick.load(ROOT / 'pipeline/research_work.json')
    assert not any(row['id'] == CID for row in work['candidates'])
    paths = files()
    parent = paths[0].parent.parent / PARENT
    assert not paths[0].parent.exists() and not any(path.exists() for path in paths)
    source = (parent / 'dec.py').read_text(encoding='utf-8')
    for old, new in EDITS:
        assert source.count(old) == 1
        source = source.replace(old, new)
    paths[0].parent.mkdir()
    for path in paths[:-1]:
        raw = source.encode() if path.name == 'dec.py' else (parent / path.name).read_bytes()
        if path.name == '__init__.py':
            raw = b'# AdaRES-Y92: one visual-side quadratic static mask-feature readout.\nfrom .ipdn import IPDN\n'
        path.write_bytes(raw)
        ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 8))
    paths[-1].write_bytes((ROOT / 'AdaRES/configs_song/model_paper5' / (PARENT + '_res.yaml')).read_bytes())

    def encoded(path):
        return dict(base64=base64.b64encode(path.read_bytes()).decode(), sha256=ref(path)['sha256'])

    payload = dict(variant=dict(id=CID, package=PACKAGE, config=paths[-1].relative_to(ROOT / 'AdaRES').as_posix(),
        python_files={path.name: encoded(path) for path in paths[:-1]}, config_data=encoded(paths[-1])),
        parent_config='configs_song/model_paper5/' + PARENT + '_res.yaml',
        goal_sha256=ref('pipeline/goal.json')['sha256'],
        parent_dec_lf_sha256=hashlib.sha256((parent / 'dec.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        train_source_lf_sha256=hashlib.sha256((ROOT / 'AdaRES/tools/train.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest())
    base = ROOT / ('pipeline/checks/y92_' + token)
    program = base.with_name('y92_cpu_' + token + '_program.py')
    native_path = base.with_name('y92_native_' + token + '_body.py')
    history = base.with_name('y92_targeted_history_' + token + '.json')
    assert not any(path.exists() for path in (program, native_path, history))
    cpu, native = bodies()
    program.write_text('PAYLOAD=' + repr(payload) + '\n' + cpu, encoding='utf-8', newline='\n')
    native_path.write_text(native, encoding='utf-8', newline='\n')
    matches = [dict(id=row['id'], stage=row['stage'], hypothesis=row.get('hypothesis', ''),
                   historical_scope=row.get('historical_scope', ''))
        for row in work['candidates'] if row['id'] in
        ('AdaRES-Y63', 'AdaRES-Y68', 'AdaRES-Y72', 'AdaRES-Y81', 'AdaRES-Y85', 'AdaRES-Y88',
         'AdaRES-Y89', 'AdaRES-Y90', 'AdaRES-Y91')]
    tick.save(history, dict(complete=True, scope='Targeted consumer comparison, not novelty or efficacy proof.',
                           matches=matches, parent=ref(parent / 'dec.py')))
    work['candidates'].append(dict(id=CID, profile='experiment_first', stage='implementation',
        hypothesis='Original static visual mask embeddings are used by a dot product with normalized queries. '
            'A shared zero-initialized projection of squared visual channels before original batching may ease '
            'visual-side nonlinear mask fitting. The original x_mask MLP already has nonlinear capacity; '
            'no defect, missing information, novelty or trained benefit is presumed.',
        matched_control='Independent W2; features_mask=x_mask+x_mask.square()@zeroW256x256 once before '
            'the SAME original batching and seven mask dots. One W/no RNG/+65536. Query normalization, '
            'score and shared confidence consumers stay original. Original all channels/points/data/2D/'
            'backbone/sampling/SCG/refinement/point masses/gate algorithm/targets/loss/selector/seed1999/'
            'fixed30 and native YAML unchanged. Exact zero output/RNG/weight/seven-stage anchors and input '
            'gradients; nonzero initial mask response/unchanged initial scores and exact bypass required.',
        historical_scope='Y88 applies a quadratic residual to each normalized query only at seven mask '
            'readouts; Y92 transforms the static visual feature side once before original batching, not '
            'queries. Y85 adds affine LayerNorm there; Y68 language-modulates static features; Y72/Y81 '
            'change per-stage features. Y90 weights confidence peers and Y91 gates confidence hidden states. '
            'None combined, replayed or reopened; no rejected/banned route renamed.',
        comparison_baseline='AdaRES-W2', implementation_package=PACKAGE, config=payload['variant']['config'],
        prepared_native_body=ref(native_path), observer_core=ref('scripts/y92_observer_core.py'),
        next_action='Run original CPU YAML/full model/backbone/TRAIN collation and exact scalar/RNG/'
            'permutation/seven-stage anchors/actual consumer/gradient/two synthetic update controls. '
            'Retain only tested seven files and freeze original native GPU0 plans. CPU is not native '
            'task backward, admission or efficacy; preserve Y89/Y90 and both user pauses.',
        evidence=[ref(path) for path in paths] + [ref(program), ref(native_path), ref(history),
            ref('scripts/y92_observer_core.py'), ref(__file__)]))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(candidate=CID, program=ref(program), native_body=ref(native_path))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 180)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        prepare(args.token)
    finally:
        writer.close()
