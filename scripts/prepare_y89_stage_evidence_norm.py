"""Prepare one independent partial-sharing confidence normalization experiment."""
import argparse
import ast
import base64
import hashlib
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick

CID = 'AdaRES-Y89'
PACKAGE = 'adares_y89_stage_evidence_norm'
PARENT = 'adares_w2_pointmass_competition'
NAMES = ('__init__.py', 'backbone.py', 'dec.py', 'ipdn.py', 'loss.py', 'sample_model.py')
ORIGINAL_HEAD = '''class QueryCompetitionIndicator(nn.Module):
    """Predict confidence from a query and its overlapping competitors."""

    def __init__(self, d_model):
'''
PRIVATE_HEAD = ORIGINAL_HEAD.replace('def __init__(self, d_model):', 'def __init__(self, d_model, num_stages=7):')
EDITS = [
    (ORIGINAL_HEAD, PRIVATE_HEAD),
    ('        self.evidence_norm = nn.LayerNorm(3 * d_model + 1)',
     '        self.evidence_norm = nn.LayerNorm(3 * d_model + 1)\n        self.refinement_evidence_norms = nn.ModuleList([copy.deepcopy(self.evidence_norm) for _ in range(num_stages - 1)])'),
    ('    def forward(self, query, pred_masks, batch_mask, point_mass):',
     '    def forward(self, query, pred_masks, batch_mask, point_mass, stage_index=0):'),
    ('        return self.head(self.evidence_norm(evidence))',
     '        norm = self.evidence_norm if stage_index == 0 else self.refinement_evidence_norms[stage_index - 1]\n        return self.head(norm(evidence))'),
    ('        self.indi_embedding = QueryCompetitionIndicator(d_model)',
     '        self.indi_embedding = QueryCompetitionIndicator(d_model, num_stages=self.num_layer + 1)'),
    ('        pred_indis = self.indi_embedding(query, pred_masks, batch_mask, point_mass)\n',
     '        pred_indis = self.indi_embedding(query, pred_masks, batch_mask, point_mass, 0)\n'),
    ('            pred_indis = self.indi_embedding(query, pred_masks, batch_mask, point_mass)\n',
     '            pred_indis = self.indi_embedding(query, pred_masks, batch_mask, point_mass, i + 1)\n'),
]


def files():
    return [ROOT / 'AdaRES/ipdn/model_paper5' / PACKAGE / name for name in NAMES] + [
        ROOT / 'AdaRES/configs_song/model_paper5' / (PACKAGE + '_res.yaml')]


def bodies():
    source = (ROOT / 'scripts/probe_y83_cpu.py').read_text(encoding='utf-8')
    before, anchor, _ = source.partition('parameters=sum(')
    assert anchor
    core = (ROOT / 'scripts/y89_observer_core.py').read_text(encoding='utf-8')
    cpu = before + (ROOT / 'scripts/probe_y89_cpu_controls.py').read_text(encoding='utf-8')
    cpu += '\n' + core + '\n' + (ROOT / 'scripts/probe_y89_cpu_tail.py').read_text(encoding='utf-8')
    native, anchor, _ = (ROOT / 'scripts/probe_y83_native_smoke.py').read_text(encoding='utf-8').partition('losses=[];')
    assert anchor
    native = native.replace("'AdaRES-Y83'", "'AdaRES-Y89'").replace('164632197', '164638865')
    native += '\nlosses,gradient_checks,batches,modes,step_rows,path_rows,state=[],[],[],[],[],[],{}\n'
    native += 'head=model.dec.indi_embedding;norms=[head.evidence_norm]+list(head.refinement_evidence_norms);original_prediction=model.dec.prediction_head\n'
    native += core + '\n' + (ROOT / 'scripts/probe_y89_native_tail.py').read_text(encoding='utf-8')
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
    source = 'import copy\n' + (parent / 'dec.py').read_text(encoding='utf-8')
    # Match complete lines so the initial readout cannot consume the recurrent edit.
    for old, new in EDITS:
        if old.startswith('        pred_indis'):
            old, new = '\n' + old, '\n' + new
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    paths[0].parent.mkdir()
    for path in paths[:-1]:
        raw = source.encode() if path.name == 'dec.py' else (parent / path.name).read_bytes()
        if path.name == '__init__.py':
            raw = b'# AdaRES-Y89: stage-private evidence affines with original shared confidence MLP.\nfrom .ipdn import IPDN\n'
        path.write_bytes(raw)
        ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 8))
    paths[-1].write_bytes((ROOT / 'AdaRES/configs_song/model_paper5' / (PARENT + '_res.yaml')).read_bytes())
    def encoded(path):
        return dict(base64=base64.b64encode(path.read_bytes()).decode(), sha256=ref(path)['sha256'])
    payload = dict(variant=dict(id=CID, package=PACKAGE, config=paths[-1].relative_to(ROOT / 'AdaRES').as_posix(),
        python_files={path.name: encoded(path) for path in paths[:-1]}, config_data=encoded(paths[-1])),
        parent_config='configs_song/model_paper5/' + PARENT + '_res.yaml', goal_sha256=ref('pipeline/goal.json')['sha256'],
        parent_dec_lf_sha256=hashlib.sha256((parent / 'dec.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        train_source_lf_sha256=hashlib.sha256((ROOT / 'AdaRES/tools/train.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest())
    program = ROOT / ('pipeline/checks/y89_cpu_' + token + '_program.py')
    native_path = ROOT / ('pipeline/checks/y89_native_' + token + '_body.py')
    history = ROOT / ('pipeline/checks/y89_targeted_history_' + token + '.json')
    assert not any(path.exists() for path in (program, native_path, history))
    cpu, native = bodies()
    program.write_text('PAYLOAD=' + repr(payload) + '\n' + cpu, encoding='utf-8', newline='\n')
    native_path.write_text(native, encoding='utf-8', newline='\n')
    matches = [dict(id=candidate['id'], stage=candidate['stage'], hypothesis=candidate.get('hypothesis', ''),
        matched_control=candidate.get('matched_control', '')) for candidate in work['candidates']
        if candidate['id'] in ('AdaRES-Y26', 'AdaRES-Y53', 'AdaRES-Y55', 'AdaRES-Y60', 'AdaRES-Y67',
                              'AdaRES-Y76', 'AdaRES-Y83')]
    tick.save(history, dict(complete=True, scope='Targeted partial-sharing and consumer comparison, not novelty proof.',
                           matches=matches, parent_source=ref(parent / 'dec.py')))
    work['candidates'].append(dict(id=CID, profile='experiment_first', stage='implementation',
        hypothesis='The same evidence LayerNorm affine transform calibrates focal query, peer context, difference and maximum overlap at all seven confidence readouts. Independent stage affine copies with the SAME shared confidence MLP and query_norm may ease stage-specific evidence fitting. Existing queries/head can compensate; no missing information, defect, novelty or trained benefit is presumed.',
        matched_control='Independent W2. Original evidence_norm for initial confidence; six no-RNG deep copies for refinement readouts, +9228/total164638865. Keep original LayerNorm statistics/epsilon/dtype/full769 evidence dimensions, one shared MLP and query_norm, all point-mass competition, all original queries/masks/scores, seven targets/losses/selector, data/fullinputs/2D/releasedbackbone/sampling/SCG/hardgate algorithm/seed1999/fixed30/native YAML. All original outputs and input/aggregate norm gradients anchor; final private intervention changes only final confidence and restores.',
        historical_scope='Y60 unties the entire indicator including MLP/query_norm; Y89 changes only evidence-normalization affine sharing, retaining the exact original shared MLP/query_norm. Y67 changes mask/score out_norm; Y83 changes recurrent query fusion norm. None is combined, validation-selected or reopened. Rejected Y26 discarded qla norm and Y53/Y55 data premises remain untouched.',
        comparison_baseline='AdaRES-W2', implementation_package=PACKAGE, config=payload['variant']['config'],
        prepared_native_body=ref(native_path),
        next_action='Execute original CPU native YAML/model/releasedbackbone/TRAIN loading, scalar/disjoint/no-RNG/private-gradient and seven-stage/input/aggregate-norm anchors; exact competition/private-norm/shared-MLP consumers and two synthetic optimizer updates. Retain seven tested files and native GPU0-only plans. Native TRAIN PASS still required for admission; preserve Y87/Y88, GPU1 and Y57 pauses.',
        evidence=[ref(path) for path in paths] + [ref(program), ref(native_path), ref(history),
            ref('scripts/y89_observer_core.py'), ref('scripts/probe_y89_cpu_controls.py'),
            ref('scripts/probe_y89_cpu_tail.py'), ref('scripts/probe_y89_native_tail.py'), ref(__file__)]))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(candidate=CID, program=ref(program), native_body=ref(native_path), native_executed=False)))


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
