"""Prepare one independent W2 mask-only nonlinear query readout."""
import argparse
import ast
import base64
import hashlib
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick

CID = 'AdaRES-Y88'
PACKAGE = 'adares_y88_mask_query_quadratic'
PARENT = 'adares_w2_pointmass_competition'
NAMES = ('__init__.py', 'backbone.py', 'dec.py', 'ipdn.py', 'loss.py', 'sample_model.py')
MODULE = '''class MaskQueryQuadratic(nn.Module):
    """A zero-anchored mask-only quadratic residual; other query consumers stay original."""

    def __init__(self, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(d_model, d_model))

    def forward(self, query):
        return query + query.square() @ self.weight


'''


def files():
    return [ROOT / 'AdaRES/ipdn/model_paper5' / PACKAGE / name for name in NAMES] + [
        ROOT / 'AdaRES/configs_song/model_paper5' / (PACKAGE + '_res.yaml')]


def bodies():
    preamble = (ROOT / 'scripts/probe_y85_cpu.py').read_text(encoding='utf-8').partition('parameters=sum(')[0]
    core = (ROOT / 'scripts/y88_observer_core.py').read_text(encoding='utf-8')
    cpu = preamble + (ROOT / 'scripts/probe_y88_cpu_controls.py').read_text(encoding='utf-8')
    cpu += '\n' + core + '\n' + (ROOT / 'scripts/probe_y88_cpu_tail.py').read_text(encoding='utf-8')
    native = (ROOT / 'scripts/probe_y83_native_smoke.py').read_text(encoding='utf-8').partition('losses=[];')[0]
    native = native.replace("'AdaRES-Y83'", "'AdaRES-Y88'").replace('164632197', '164695173')
    native += '\nlosses,gradient_checks,batches,modes,step_rows,path_rows,state=[],[],[],[],[],[],{}\n'
    native += 'head=model.dec.indi_embedding;projection=model.dec.mask_query_quadratic;original_prediction=model.dec.prediction_head\n'
    native += core + '\n' + (ROOT / 'scripts/probe_y88_native_tail.py').read_text(encoding='utf-8')
    for text in (cpu, native):
        ast.parse(text, feature_version=(3, 8))
    return cpu, native


def prepare(token):
    view = tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')
    assert view['experiment_first']['enabled'] and view['research']['creation_budget'] >= 1
    work = tick.load(ROOT / 'pipeline/research_work.json')
    assert not any(candidate['id'] == CID for candidate in work['candidates'])
    paths = files()
    parent = paths[0].parent.parent / PARENT
    assert not paths[0].parent.exists() and not any(path.exists() for path in paths)
    source = (parent / 'dec.py').read_text(encoding='utf-8')
    edits = [('class DEC(nn.Module):', MODULE + 'class DEC(nn.Module):'),
        ('        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)',
         '        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)\n        self.mask_query_quadratic = MaskQueryQuadratic(d_model)'),
        ('        pred_masks, attn_masks = self.get_mask(query, mask_feats, batch_mask)',
         '        pred_masks, attn_masks = self.get_mask(self.mask_query_quadratic(query), mask_feats, batch_mask)')]
    for old, new in edits:
        assert source.count(old) == 1
        source = source.replace(old, new)
    paths[0].parent.mkdir()
    for path in paths[:-1]:
        raw = source.encode() if path.name == 'dec.py' else (parent / path.name).read_bytes()
        if path.name == '__init__.py':
            raw = b'# AdaRES-Y88: mask-only quadratic normalized-query readout.\nfrom .ipdn import IPDN\n'
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
    program = ROOT / ('pipeline/checks/y88_cpu_' + token + '_program.py')
    native_path = ROOT / ('pipeline/checks/y88_native_' + token + '_body.py')
    history = ROOT / ('pipeline/checks/y88_targeted_history_' + token + '.json')
    assert not any(path.exists() for path in (program, native_path, history))
    cpu, native = bodies()
    program.write_text('PAYLOAD=' + repr(payload) + '\n' + cpu, encoding='utf-8', newline='\n')
    native_path.write_text(native, encoding='utf-8', newline='\n')
    matches = [dict(id=candidate['id'], stage=candidate['stage'], hypothesis=candidate.get('hypothesis', ''))
        for candidate in work['candidates']
        if candidate['id'] in ('AdaRES-Y63', 'AdaRES-Y67', 'AdaRES-Y72', 'AdaRES-Y78', 'AdaRES-Y81',
                              'AdaRES-Y82', 'AdaRES-Y83', 'AdaRES-Y85', 'AdaRES-Y87')]
    tick.save(history, dict(complete=True, scope='Targeted consumer comparison, not novelty proof.',
                           matches=matches, parent=ref(parent / 'dec.py')))
    work['candidates'].append(dict(id=CID, profile='experiment_first', stage='implementation',
        hypothesis='Original masks are a dot product between a normalized query and static visual features. A shared zero-initialized projection of squared normalized-query channels before only that dot product may ease nonlinear mask fitting. The query generator already represents nonlinear functions; no defect, missing information, novelty or benefit is presumed.',
        matched_control='Independent W2; q_mask=q_norm+(q_norm squared)@zeroW256x256 at seven original mask readouts. Original out_score receives q_norm and the original shared indicator receives original raw query plus intended masks. One W/no RNG/+65536. Keep all original channels, points, data, heads, norms, initial/recurrent query generation, coarse/fine average, self/language sum, 2D, backbone, coordinates, SCG, gates algorithm, targets/loss/selector, seed1999/fixed30 and original YAML. Zero anchors exact; nonzero mask response, original consumers and input gradients tested.',
        historical_scope='Y63 is a bilinear confidence readout; Y82 and Y87 change recurrent fusion. Y85 normalizes visual mask features. This is a mask-only nonlinear transform after the existing shared out_norm, not those consumers or a combination. No previously rejected mechanism reopened.',
        comparison_baseline='AdaRES-W2', implementation_package=PACKAGE, config=payload['variant']['config'],
        prepared_native_body=ref(native_path),
        next_action='Execute actual CPU original model/backbone/TRAIN loading and scalar/RNG/permutation/anchor/consumer controls with two synthetic updates. Retain exactly tested files and prepare native server1GPU0/350W YAML/TRAIN task; wait for true exclusive slot, then actual native PASS, admission and launch. CPU checks do not establish task efficacy.',
        evidence=[ref(path) for path in paths] + [ref(program), ref(native_path), ref(history),
            ref('scripts/y88_observer_core.py'), ref('scripts/probe_y88_cpu_controls.py'),
            ref('scripts/probe_y88_cpu_tail.py'), ref('scripts/probe_y88_native_tail.py'), ref(__file__)]))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(candidate=CID, program=ref(program), native_body=ref(native_path))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 180)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        prepare(args.token)
    finally:
        lock.close()
