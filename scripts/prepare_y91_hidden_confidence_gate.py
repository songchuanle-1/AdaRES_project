"""Prepare an independent gated connection inside the shared confidence MLP."""
import argparse
import ast
import base64
import hashlib
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick

CID = 'AdaRES-Y91'
PACKAGE = 'adares_y91_hidden_confidence_gate'
PARENT = 'adares_w2_pointmass_competition'
NAMES = ('__init__.py', 'backbone.py', 'dec.py', 'ipdn.py', 'loss.py', 'sample_model.py')
MODULE = '''class ConfidenceHiddenGate(nn.Module):
    """Condition the second confidence hidden state on the first."""

    def __init__(self, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(d_model, d_model))

    def forward(self, first, second):
        gate = 1 + torch.tanh((first @ self.weight) / (first.shape[-1] ** .5))
        return second * gate


'''
OLD_RETURN = '        return self.head(self.evidence_norm(evidence))'
NEW_RETURN = '''        normalized = self.evidence_norm(evidence)
        first = self.head[1](self.head[0](normalized))
        second = self.head[3](self.head[2](first))
        return self.head[4](self.hidden_gate(first, second))'''
EDITS = [
    ('class QueryCompetitionIndicator(nn.Module):', MODULE + 'class QueryCompetitionIndicator(nn.Module):'),
    ('        self.evidence_norm = nn.LayerNorm(3 * d_model + 1)',
     '        self.evidence_norm = nn.LayerNorm(3 * d_model + 1)\n        self.hidden_gate = ConfidenceHiddenGate(d_model)'),
    (OLD_RETURN, NEW_RETURN),
]


def files():
    return [ROOT / 'AdaRES/ipdn/model_paper5' / PACKAGE / name for name in NAMES] + [
        ROOT / 'AdaRES/configs_song/model_paper5' / (PACKAGE + '_res.yaml')]


def bodies():
    cpu, anchor, _ = (ROOT / 'scripts/probe_y83_cpu.py').read_text(encoding='utf-8').partition('parameters=sum(')
    assert anchor
    core = (ROOT / 'scripts/y91_observer_core.py').read_text(encoding='utf-8')
    cpu += (ROOT / 'scripts/probe_y91_cpu_controls.py').read_text(encoding='utf-8')
    cpu += '\n' + core + '\n' + (ROOT / 'scripts/probe_y91_cpu_tail.py').read_text(encoding='utf-8')
    native, anchor, _ = (ROOT / 'scripts/probe_y83_native_smoke.py').read_text(encoding='utf-8').partition('losses=[];')
    assert anchor and native.count("'AdaRES-Y83'") == 1 and native.count('164632197') == 1
    native = native.replace("'AdaRES-Y83'", "'AdaRES-Y91'").replace('164632197', '164695173')
    native += '\nlosses,gradient_checks,batches,modes,step_rows,path_rows,state=[],[],[],[],[],[],{}\n'
    native += 'head=model.dec.indi_embedding;gate=head.hidden_gate;original_prediction=model.dec.prediction_head\n'
    native += core + '\n' + (ROOT / 'scripts/probe_y91_native_tail.py').read_text(encoding='utf-8')
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
            raw = b'# AdaRES-Y91: gated hidden connection in the original shared confidence MLP.\nfrom .ipdn import IPDN\n'
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
    program = ROOT / ('pipeline/checks/y91_cpu_' + token + '_program.py')
    native_path = ROOT / ('pipeline/checks/y91_native_' + token + '_body.py')
    history = ROOT / ('pipeline/checks/y91_targeted_history_' + token + '.json')
    assert not any(path.exists() for path in (program, native_path, history))
    program.write_text('PAYLOAD=' + repr(payload) + '\n' + cpu, encoding='utf-8', newline='\n')
    native_path.write_text(native, encoding='utf-8', newline='\n')
    matches = [
        dict(id=candidate['id'], stage=candidate['stage'], hypothesis=candidate.get('hypothesis', ''),
             matched_control=candidate.get('matched_control', ''))
        for candidate in work['candidates']
        if candidate['id'] in ('AdaRES-Y26', 'AdaRES-Y60', 'AdaRES-Y63', 'AdaRES-Y76',
                              'AdaRES-Y86', 'AdaRES-Y89', 'AdaRES-Y90')]
    tick.save(history, dict(complete=True, scope='Targeted connection/consumer comparison, not novelty proof.',
                            matches=matches, parent_source=ref(parent / 'dec.py')))
    work['candidates'].append(dict(
        id=CID, profile='experiment_first', stage='implementation',
        hypothesis='The original shared confidence MLP maps normalized evidence through two GELU hidden states. Conditioning the second state channel-wise on the first may ease fitting nonlinear evidence interactions. The original MLP can represent such interactions implicitly; no defect, missing information, novelty or trained benefit is presumed.',
        matched_control='Independent W2. h1=GELU(L1(evidence_norm(original_evidence))), h2=GELU(L2(h1)); final logits=L3(h2*(1+tanh(h1@W/sqrt256))). Shared zeroW256x256/no RNG/+65536. Original five MLP layers and both norms retain initial weights and seven-stage sharing; all original query/mask/mass/overlap/context/evidence, masks/scores/Boolean gates, full inputs/existing2D/releasedbackbone/sampling/SCG/targets-loss-selector/seed1999/fixed30/native YAML unchanged. Exact zero output/RNG and close input-gradient anchors; confidence-only intervention/bypass and original hidden consumers/finite new gradients/two updates required. No numerical/feature/validation scan.',
        historical_scope='Y63 adds focal-peer bilinear logits outside the MLP; Y91 gates an internal learned hidden representation with the first hidden representation. Y89 changes evidence-norm sharing and Y90 changes peer aggregation; neither is combined. No new mask statistic, visual-readback route, paused Y57 mechanism, Y86 scalar shortcut or rejected qla normalization is reopened.',
        comparison_baseline='AdaRES-W2', implementation_package=PACKAGE, config=payload['variant']['config'],
        prepared_native_body=ref(native_path), observer_core=ref('scripts/y91_observer_core.py'),
        next_action='Run original CPU YAML/full model/released backbone/TRAIN collation, scalar/bounded/no-RNG/permutation controls, all-stage W2/input-gradient anchors, confidence-only gate intervention/bypass and exact original hidden/evidence consumers with two synthetic updates. Retain tested version and authorized GPU0 native plans only after actual CPU PASS. Earlier Y89/Y90 and original Y87/Y88 remain preserved; CPU is not GPU PASS, admission or efficacy.',
        evidence=[ref(path) for path in paths] + [
            ref(program), ref(native_path), ref(history), ref(__file__), ref('scripts/y91_observer_core.py'),
            ref('scripts/probe_y91_cpu_controls.py'), ref('scripts/probe_y91_cpu_tail.py'),
            ref('scripts/probe_y91_native_tail.py')]))
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
