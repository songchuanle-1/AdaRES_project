"""Prepare one independent full-evidence connection inside the W2 confidence MLP."""
import argparse
import ast
import base64
import hashlib
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick

CID = 'AdaRES-Y93'
PACKAGE = 'adares_y93_evidence_hidden_skip'
PARENT = 'adares_w2_pointmass_competition'
NAMES = ('__init__.py', 'backbone.py', 'dec.py', 'ipdn.py', 'loss.py', 'sample_model.py')
MODULE = '''class EvidenceHiddenSkip(nn.Module):
    """Add the original full normalized evidence to the second hidden state."""

    def __init__(self, evidence_dim, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(evidence_dim, d_model))

    def forward(self, evidence, second):
        return second + evidence @ self.weight


'''
EDITS = [
    ('class QueryCompetitionIndicator(nn.Module):', MODULE + 'class QueryCompetitionIndicator(nn.Module):'),
    ('        self.evidence_norm = nn.LayerNorm(3 * d_model + 1)',
     '        self.evidence_norm = nn.LayerNorm(3 * d_model + 1)\n'
     '        self.evidence_skip = EvidenceHiddenSkip(3 * d_model + 1, d_model)'),
    ('        return self.head(self.evidence_norm(evidence))',
     '''        normalized = self.evidence_norm(evidence)
        first = self.head[1](self.head[0](normalized))
        second = self.head[3](self.head[2](first))
        return self.head[4](self.evidence_skip(normalized, second))'''),
]


def files():
    return [ROOT / 'AdaRES/ipdn/model_paper5' / PACKAGE / name for name in NAMES] + [
        ROOT / 'AdaRES/configs_song/model_paper5' / (PACKAGE + '_res.yaml')]


def bodies():
    cpu, anchor, _ = (ROOT / 'scripts/probe_y83_cpu.py').read_text(encoding='utf-8').partition('parameters=sum(')
    assert anchor
    core = (ROOT / 'scripts/y93_observer_core.py').read_text(encoding='utf-8')
    cpu += (ROOT / 'scripts/probe_y93_cpu_controls.py').read_text(encoding='utf-8')
    cpu += '\n' + core + '\n' + (ROOT / 'scripts/probe_y93_cpu_tail.py').read_text(encoding='utf-8')
    native, anchor, _ = (ROOT / 'scripts/probe_y83_native_smoke.py').read_text(encoding='utf-8').partition('losses=[];')
    assert anchor and native.count("'AdaRES-Y83'") == 1 and native.count('164632197') == 1
    native = native.replace("'AdaRES-Y83'", "'AdaRES-Y93'").replace('164632197', '164826501')
    native += '\nlosses,gradient_checks,batches,modes,step_rows,path_rows,state=[],[],[],[],[],[],{}\n'
    native += 'head=model.dec.indi_embedding;residual=head.evidence_skip;original_prediction=model.dec.prediction_head\n'
    native += core + '\n' + (ROOT / 'scripts/probe_y93_native_tail.py').read_text(encoding='utf-8')
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
    for old, new in EDITS:
        assert source.count(old) == 1
        source = source.replace(old, new)
    paths[0].parent.mkdir()
    for path in paths[:-1]:
        raw = source.encode() if path.name == 'dec.py' else (parent / path.name).read_bytes()
        if path.name == '__init__.py':
            raw = b'# AdaRES-Y93: full normalized-evidence residual into the second confidence hidden state.\nfrom .ipdn import IPDN\n'
        path.write_bytes(raw)
        ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 8))
    paths[-1].write_bytes((ROOT / 'AdaRES/configs_song/model_paper5' / (PARENT + '_res.yaml')).read_bytes())

    def encoded(path):
        return dict(base64=base64.b64encode(path.read_bytes()).decode(), sha256=ref(path)['sha256'])

    payload = dict(
        variant=dict(id=CID, package=PACKAGE, config=paths[-1].relative_to(ROOT / 'AdaRES').as_posix(),
                     python_files={path.name: encoded(path) for path in paths[:-1]}, config_data=encoded(paths[-1])),
        parent_config='configs_song/model_paper5/' + PARENT + '_res.yaml',
        goal_sha256=ref('pipeline/goal.json')['sha256'],
        parent_dec_lf_sha256=hashlib.sha256((parent / 'dec.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        train_source_lf_sha256=hashlib.sha256((ROOT / 'AdaRES/tools/train.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest())
    cpu, native = bodies()
    program = ROOT / ('pipeline/checks/y93_cpu_' + token + '_program.py')
    native_path = ROOT / ('pipeline/checks/y93_native_' + token + '_body.py')
    history = ROOT / ('pipeline/checks/y93_targeted_history_' + token + '.json')
    assert not any(path.exists() for path in (program, native_path, history))
    program.write_text('PAYLOAD=' + repr(payload) + '\n' + cpu, encoding='utf-8', newline='\n')
    native_path.write_text(native, encoding='utf-8', newline='\n')
    matches = [dict(id=candidate['id'], stage=candidate['stage'], hypothesis=candidate.get('hypothesis', ''),
                   matched_control=candidate.get('matched_control', ''))
        for candidate in work['candidates'] if candidate['id'] in (
            'AdaRES-Y26', 'AdaRES-Y60', 'AdaRES-Y63', 'AdaRES-Y76', 'AdaRES-Y86',
            'AdaRES-Y89', 'AdaRES-Y90', 'AdaRES-Y91', 'AdaRES-Y92')]
    tick.save(history, dict(complete=True, scope='Targeted full-evidence/hidden connection comparison, not novelty proof.',
                           matches=matches, parent_source=ref(parent / 'dec.py')))
    work['candidates'].append(dict(
        id=CID, profile='experiment_first', stage='implementation',
        hypothesis='The original normalized full competition evidence reaches final confidence through two GELU hidden states. A learned residual from that same evidence into the second hidden state may ease confidence fitting while retaining the nonlinear path. The MLP already has nonlinear capacity; no missing information, defect, novelty or trained benefit is presumed.',
        matched_control='Independent W2. E=evidence_norm(original full769 evidence); h1=GELU(L1(E)); h2=GELU(L2(h1)); logits=L3(h2+E@zeroW769x256). Shared/no RNG/+196864/total164826501. Retain original five MLP layers, both norms, all769 dimensions, seven-stage sharing, original query/mask/mass/overlap/context/evidence and all original masks/scores/data/fullinputs/2D/backbone/sampling/SCG/Boolean gates/targets-loss-selector/seed1999/fixed30/native YAML. Zero anchors and input gradients, full evidence/second-hidden consumers, confidence-only intervention/exact bypass and two finite synthetic updates required; no feature/threshold/weight/validation scan.',
        historical_scope='Y91 multiplicatively conditions second hidden on first hidden; Y93 adds full normalized evidence to second hidden and retains original nonlinear path. Y86 is a held scalar max-overlap-to-logit shortcut, not this full769-to-hidden learned connection. Y63 bilinear outer readout, Y89 normalization sharing and Y90 peer weights are absent; Y92 visual mask feature transform is not combined. No paused, rejected or banned route is reopened; no missing-information/novelty/benefit claim.',
        comparison_baseline='AdaRES-W2', implementation_package=PACKAGE, config=payload['variant']['config'],
        prepared_native_body=ref(native_path), observer_core=ref('scripts/y93_observer_core.py'),
        next_action='Run actual original CPU native YAML/full model/releasedbackbone/TRAIN loading and scalar/no-RNG/permutation/all-stage W2/input-gradient/causal-bypass/consumer controls with two synthetic updates. Only after actual CPU PASS retain exact files and prepare authorized GPU0 native programs. Do not displace Y90/Y91, use GPU1 or call CPU checks native/efficacy. Existing Y92 precedes new Y93.',
        evidence=[ref(path) for path in paths] + [ref(program), ref(native_path), ref(history), ref(__file__),
            ref('scripts/y93_observer_core.py'), ref('scripts/probe_y93_cpu_controls.py'),
            ref('scripts/probe_y93_cpu_tail.py'), ref('scripts/probe_y93_native_tail.py'),
            ref('pipeline/checks/y93_support_' + token + '.json')]))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(candidate=CID, program=ref(program), native_body=ref(native_path), native_executed=False)))


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
