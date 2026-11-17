"""Prepare an independent all-stage readout shortcut for already available max overlap."""
import argparse
import ast
import base64
import hashlib
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick

CID = 'AdaRES-Y86'
PACKAGE = 'adares_y86_overlap_readout_skip'
PARENT = 'adares_w2_pointmass_competition'
NAMES = ('__init__.py', 'backbone.py', 'dec.py', 'ipdn.py', 'loss.py', 'sample_model.py')
COMPONENT = '''class OverlapReadout(nn.Module):
    """Two learned output coefficients for an existing scalar; no new mask statistic."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2))

    def forward(self, logits, max_overlap):
        return logits + max_overlap * self.weight


'''


def files():
    return [ROOT / 'AdaRES/ipdn/model_paper5' / PACKAGE / name for name in NAMES] + [
        ROOT / 'AdaRES/configs_song/model_paper5' / (PACKAGE + '_res.yaml')]


def prepare(token):
    view = tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')
    assert view['experiment_first']['enabled'] and view['research']['creation_budget'] >= 1
    work = tick.load(ROOT / 'pipeline/research_work.json')
    assert not any(candidate['id'] == CID for candidate in work['candidates'])
    paths = files()
    parent = paths[0].parent.parent / PARENT
    assert not paths[0].parent.exists() and not any(path.exists() for path in paths)
    source = (parent / 'dec.py').read_text(encoding='utf-8')
    edits = [('class QueryCompetitionIndicator(nn.Module):', COMPONENT + 'class QueryCompetitionIndicator(nn.Module):'),
        ('            nn.Linear(d_model, 2),\n        )',
         '            nn.Linear(d_model, 2),\n        )\n        self.overlap_readout = OverlapReadout()'),
        ('        return self.head(self.evidence_norm(evidence))',
         '        return self.overlap_readout(self.head(self.evidence_norm(evidence)), max_overlap)')]
    for old, new in edits:
        assert source.count(old) == 1
        source = source.replace(old, new)
    paths[0].parent.mkdir()
    for path in paths[:-1]:
        raw = source.encode() if path.name == 'dec.py' else (parent / path.name).read_bytes()
        if path.name == '__init__.py':
            raw = b'# AdaRES-Y86: direct readout of existing maximum peer overlap.\nfrom .ipdn import IPDN\n'
        path.write_bytes(raw)
        ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 8))
    paths[-1].write_bytes((ROOT / 'AdaRES/configs_song/model_paper5' / (PARENT + '_res.yaml')).read_bytes())
    assert list((ROOT / 'AdaRES/configs_song').rglob(paths[-1].name)) == [paths[-1]]

    def encoded(path):
        return dict(base64=base64.b64encode(path.read_bytes()).decode(), sha256=ref(path)['sha256'])

    payload = dict(variant=dict(id=CID, package=PACKAGE, config=paths[-1].relative_to(ROOT / 'AdaRES').as_posix(),
        python_files={path.name: encoded(path) for path in paths[:-1]}, config_data=encoded(paths[-1])),
        parent_config='configs_song/model_paper5/' + PARENT + '_res.yaml',
        goal_sha256=ref('pipeline/goal.json')['sha256'],
        parent_dec_lf_sha256=hashlib.sha256((parent / 'dec.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        train_source_lf_sha256=hashlib.sha256((ROOT / 'AdaRES/tools/train.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest())
    prefix = (ROOT / 'scripts/probe_y85_cpu.py').read_text(encoding='utf-8').partition('parameters=sum(')
    assert prefix[1] and 'parameters=sum(' not in prefix[0]
    core = ROOT / 'scripts/y86_observer_core.py'
    body = prefix[0] + (ROOT / 'scripts/probe_y86_cpu_body.py').read_text(encoding='utf-8') + '\n' + core.read_text(
        encoding='utf-8') + '\n' + (ROOT / 'scripts/probe_y86_cpu_tail.py').read_text(encoding='utf-8')
    program = ROOT / ('pipeline/checks/y86_cpu_' + token + '_program.py')
    history = ROOT / ('pipeline/checks/y86_targeted_history_' + token + '.json')
    assert not program.exists() and not history.exists()
    program.write_text('PAYLOAD=' + repr(payload) + '\n' + body, encoding='utf-8', newline='\n')
    ast.parse(program.read_text(encoding='utf-8'), feature_version=(3, 8))
    matches = [dict(id=candidate['id'], stage=candidate['stage'], hypothesis=candidate.get('hypothesis', ''),
                    matched_control=candidate.get('matched_control', ''))
        for candidate in work['candidates'] if candidate['id'] in
        ('AdaRES-Y34', 'AdaRES-Y47', 'AdaRES-Y50', 'AdaRES-Y60', 'AdaRES-Y62', 'AdaRES-Y63', 'AdaRES-Y76')]
    tick.save(history, dict(complete=True, scope='Existing scalar output shortcut versus rejected new-graph-information claims; not novelty proof.',
        matches=matches, parent_source=ref(parent / 'dec.py')))
    work['candidates'].append(dict(id=CID, profile='experiment_first', stage='implementation',
        hypothesis='The existing maximum peer-overlap scalar enters a joint evidence LayerNorm and nonlinear confidence head. A separate zero-initialized two-output linear shortcut of that SAME scalar at all seven readouts may ease confidence fitting. The head already receives this information; no missing information, defect, novelty or trained benefit is presumed.',
        matched_control='Independent W2 except shared confidence logits += existing_max_overlap * learned_zero_vector2. Exactly two new parameters, no RNG; no new mask statistic, graph, data or selector. Keep original all128 queries/all channels, overlap formula/diagonal/padding/mass, full inputs/2D/backbone/coordinates/sampling/SCG/k/gamma, original shared head/norms/masks/gates/targets/loss/seed1999/fixed30/native YAML. Initial all-stage outputs and input gradients must match W2. Nonzero intervention changes only confidence and restores exactly.',
        historical_scope='Rejected Y34/Y47 sought new final graph-correspondence information; this uses the already consumed exact max-overlap scalar without a new graph statistic or final-only head. Y62 adds density; Y63 adds vector bilinear interactions; Y76 adds stage input. None is reopened or combined. This is an architectural output parameterization comparison, not the rejected information claim.',
        comparison_baseline='AdaRES-W2', implementation_package=PACKAGE, config=payload['variant']['config'],
        next_action='Run full CPU native model/pretrain/original TRAIN collation, independent float32/64 scalar/zero/RNG/permutation/gradient controls, seven-stage W2 anchors, confidence-only intervention/restoration, original max-overlap/readout consumer identities and two synthetic optimizer steps. Only later actual native GPU TRAIN PASS may admit. Preserve Y84/Y85 and both user pauses.',
        evidence=[ref(path) for path in paths] + [ref(program), ref(history), ref(core),
            ref('scripts/probe_y86_cpu_body.py'), ref('scripts/probe_y86_cpu_tail.py'), ref(__file__)]))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(candidate=CID, program=ref(program), native_GPU_executed=False)))


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
