"""Independent product residual at the existing self/language sum before its original norm."""
import argparse
import ast
import base64
import hashlib
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick

CID = 'AdaRES-Y87'
PACKAGE = 'adares_y87_self_language_product'
PARENT = 'adares_w2_pointmass_competition'
NAMES = ('__init__.py', 'backbone.py', 'dec.py', 'ipdn.py', 'loss.py', 'sample_model.py')
MODULE = '''class SelfLanguageProductFusion(nn.Module):
    """Retain the original sum; add a learned product interaction before norm_qqa."""

    def __init__(self, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(d_model, d_model))

    def forward(self, self_query, language_query):
        return self_query + language_query + (self_query * language_query) @ self.weight


'''


def files():
    return [ROOT / 'AdaRES/ipdn/model_paper5' / PACKAGE / name for name in NAMES] + [
        ROOT / 'AdaRES/configs_song/model_paper5' / (PACKAGE + '_res.yaml')]


def bodies():
    old = (ROOT / 'scripts/probe_y82_cpu.py').read_text(encoding='utf-8')
    before, marker, _ = old.partition('"""Callbacks shared literally by CPU decoder and future native TRAIN checks."""')
    assert marker
    before = before.replace('ScaleProductFusion', 'SelfLanguageProductFusion').replace('scale_product', 'self_language_product')
    before = before.replace('(c+f)/2', '(c+f)').replace('(a+a)/2', '(a+a)').replace('(b+z)/2', '(b+z)')
    before = before.replace('(a[0]+a[1])/2', '(a[0]+a[1])')
    before = before.replace('(c[b,i,d].double()+f[b,i,d].double())/2+', '(c[b,i,d].double()+f[b,i,d].double())+')
    before = before.replace('.5*torch.ones_like(c)', 'torch.ones_like(c)').replace('.5*torch.ones_like(f)', 'torch.ones_like(f)')
    before = before.replace('zero_average_and_input_gradients_exact', 'zero_sum_and_input_gradients_exact')
    before = before.replace('same_average_different_product_distinguished', 'same_sum_different_product_distinguished')
    assert not any(isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
                   and isinstance(node.right, ast.Constant) and node.right.value == 2
                   for node in ast.walk(ast.parse(before))), 'The original self/language baseline is a sum, never a half-average'
    _, marker, tail = old.partition('hooks=install_hooks();model.dec.prediction_head=checked_prediction')
    assert marker
    tail = marker + tail
    tail = tail.replace('original_branch_identities_and_qla_consumption_verified',
                        'original_self_language_and_norm_ffn_consumption_verified')
    core = (ROOT / 'scripts/y87_observer_core.py').read_text(encoding='utf-8')
    cpu = before + '\n' + core + '\n' + tail
    native = (ROOT / 'scripts/probe_y82_native_smoke.py').read_text(encoding='utf-8')
    prefix, marker, _ = native.partition('def check_path_calls(state):')
    assert marker
    _, marker, suffix = native.partition('class FirstTwoBatches:')
    assert marker
    native = prefix + core + '\n' + marker + suffix
    native = native.replace("'AdaRES-Y82'", "'AdaRES-Y87'").replace('scale_product', 'self_language_product')
    native = native.replace('fusion_calls=12,qla_consumption_calls=12,',
                            'fusion_calls=12,original_norm_consumption_calls=12,original_ffn_consumption_calls=12,')
    for source in (cpu, native):
        ast.parse(source, feature_version=(3, 8))
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
         '        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)\n        self.self_language_product = SelfLanguageProductFusion(d_model)'),
        ('            query = self.norm_qqa(query_self + query_qla)',
         '            query = self.norm_qqa(self.self_language_product(query_self, query_qla))')]
    for old, new in edits:
        assert source.count(old) == 1
        source = source.replace(old, new)
    assert 'query_qia = (coarse_query + fine_query)/2' in source
    paths[0].parent.mkdir()
    for path in paths[:-1]:
        raw = source.encode() if path.name == 'dec.py' else (parent / path.name).read_bytes()
        if path.name == '__init__.py':
            raw = b'# AdaRES-Y87: self/language product before unchanged original norm_qqa.\nfrom .ipdn import IPDN\n'
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
    program = ROOT / ('pipeline/checks/y87_cpu_' + token + '_program.py')
    native_body = ROOT / ('pipeline/checks/y87_native_' + token + '_body.py')
    history = ROOT / ('pipeline/checks/y87_targeted_history_' + token + '.json')
    assert not any(path.exists() for path in (program, native_body, history))
    cpu, native = bodies()
    program.write_text('PAYLOAD=' + repr(payload) + '\n' + cpu, encoding='utf-8', newline='\n')
    native_body.write_text(native, encoding='utf-8', newline='\n')
    ast.parse(program.read_text(encoding='utf-8'), feature_version=(3, 8))
    matches = [dict(id=candidate['id'], stage=candidate['stage'], hypothesis=candidate.get('hypothesis', ''),
        matched_control=candidate.get('matched_control', '')) for candidate in work['candidates']
        if candidate['id'] in ('AdaRES-Y26', 'AdaRES-Y63', 'AdaRES-Y70', 'AdaRES-Y71',
                              'AdaRES-Y74', 'AdaRES-Y75', 'AdaRES-Y82', 'AdaRES-Y83', 'AdaRES-Y86')]
    tick.save(history, dict(complete=True, scope='Targeted consumer/location comparison, not novelty proof.',
                           matches=matches, parent_source=ref(parent / 'dec.py')))
    work['candidates'].append(dict(id=CID, profile='experiment_first', stage='implementation',
        hypothesis='Original recurrent query refinement adds its self-attention and language-attention states before a shared active LayerNorm and stage FFN. A shared zero-initialized projection of their product at that exact sum may ease fitting conditional interactions. Both states already carry language and spatial context; no missing information, defect, novelty or trained benefit is presumed.',
        matched_control='Independent W2 plus self+language+(self*language)@W before the SAME existing shared norm_qqa and original stage FFN, six calls. One zero W256x256/no RNG/+65536. Retain original coarse/fine half-average, all original tensors/queries/channels/full inputs/2D/backbone/coordinates/sampling/SCG/k/gamma/static masks/gate algorithm, seven shared confidence heads/targets/loss/selector/seed1999/fixed30/native YAML. Initial output/input-gradient anchors exact; nonzero intervention/bypass must affect intended refined masks then restore.',
        historical_scope='Y82 multiplies coarse/fine visual updates before later language/self attention; this combines the already computed self/language outputs at a distinct recurrent fusion consumer. Y74 gates the attended-language update, Y75 modifies initial query/token interaction, Y83 unties norm affines. None is combined. Rejected Y26 discarded-norm reconnection remains untouched: original active norm_qqa and qla path are retained, not bypassed.',
        comparison_baseline='AdaRES-W2', implementation_package=PACKAGE, config=payload['variant']['config'],
        prepared_native_body=ref(native_body),
        next_action='Run original CPU model/backbone/TRAIN load, independent float32/64 sum-product/no-RNG/scalar/permutation/swap/same-sum controls, all-stage W2 and input-gradient anchors, original self/language tensor identities and exact norm/FFN consumers, shared head/gates and two synthetic CPU optimizer steps. Retain exact seven tested files and GPU0-only native plan. Y86 precedes Y87; no CPU-based GPU admission or efficacy.',
        evidence=[ref(path) for path in paths] + [ref(program), ref(native_body), ref(history),
            ref('scripts/y87_observer_core.py'), ref(__file__)]))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(candidate=CID, program=ref(program), native_body=ref(native_body), native_GPU_executed=False)))


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
