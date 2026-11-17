"""Prepare a direct language-conditioned mask-feature path, preserving live jobs."""
import argparse
import ast
import base64
from datetime import datetime, timedelta
import hashlib
import json
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick

CID = 'AdaRES-Y68'
PACKAGE = 'adares_y68_language_mask_modulation'
PARENT = 'adares_w2_pointmass_competition'
NAMES = ('__init__.py', 'backbone.py', 'dec.py', 'ipdn.py', 'loss.py', 'sample_model.py')

MODULATOR = '''class LanguageMaskModulation(nn.Module):
    """Zero-initialized channel modulation from the mean of valid language tokens."""

    def __init__(self, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(d_model, d_model))

    def forward(self, mask_feats, language, language_padding):
        valid = (~language_padding).to(language.dtype)
        clean = language.masked_fill(language_padding.unsqueeze(-1), 0.)
        sentence = clean.sum(1) / valid.sum(1, keepdim=True).clamp_min(1.)
        scale = torch.tanh(sentence @ self.weight)
        return mask_feats * (1. + scale.unsqueeze(1))


'''


def files():
    return [ROOT / 'AdaRES/ipdn/model_paper5' / PACKAGE / n for n in NAMES] + [
        ROOT / 'AdaRES/configs_song/model_paper5' / (PACKAGE + '_res.yaml')]


def prepare(token):
    view = tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')
    assert view['research']['creation_budget'] >= 1
    work = tick.load(ROOT / 'pipeline/research_work.json')
    assert not any(c['id'] == CID for c in work['candidates'])
    target = files(); assert not any(p.exists() for p in target) and not target[0].parent.exists()
    parent = target[0].parent.parent / PARENT
    source = (parent / 'dec.py').read_text(encoding='utf-8')
    assert source.count('class DEC(nn.Module):') == 1
    source = source.replace('class DEC(nn.Module):', MODULATOR + 'class DEC(nn.Module):')
    marker = '        self.indi_norm = nn.LayerNorm(d_model)'
    assert source.count(marker) == 1
    source = source.replace(marker, marker + '\n        self.language_mask_modulation = LanguageMaskModulation(d_model)')
    marker = '        mask_feats, _ = self.get_batches(mask_feats, batch_offsets)'
    assert source.count(marker) == 1
    source = source.replace(marker, marker + '\n        mask_feats = self.language_mask_modulation(mask_feats, lang_feats, lang_masks)')
    target[0].parent.mkdir()
    for path in target[:-1]:
        raw = source.encode('utf-8') if path.name == 'dec.py' else (parent / path.name).read_bytes()
        if path.name == '__init__.py':
            raw = b'# AdaRES-Y68: valid-language-conditioned mask-feature modulation.\nfrom .ipdn import IPDN\n'
        path.write_bytes(raw)
        ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 8))
    target[-1].write_bytes((ROOT / 'AdaRES/configs_song/model_paper5' / (PARENT + '_res.yaml')).read_bytes())
    assert len(list((ROOT / 'AdaRES/configs_song').rglob(target[-1].name))) == 1
    def encoded(p): return dict(base64=base64.b64encode(p.read_bytes()).decode(), sha256=ref(p)['sha256'])
    payload = dict(variant=dict(id=CID, package=PACKAGE,
        config=target[-1].relative_to(ROOT / 'AdaRES').as_posix(),
        python_files={p.name: encoded(p) for p in target[:-1]}, config_data=encoded(target[-1])),
        parent_config='configs_song/model_paper5/' + PARENT + '_res.yaml',
        goal_sha256=ref('pipeline/goal.json')['sha256'],
        parent_dec_lf_sha256=hashlib.sha256((parent / 'dec.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        train_source_lf_sha256=hashlib.sha256((ROOT / 'AdaRES/tools/train.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest())
    body = ROOT / 'scripts/probe_y68_cpu.py'
    program = ROOT / ('pipeline/checks/y68_cpu_' + token + '_program.py'); assert not program.exists()
    program.write_text('PAYLOAD=' + repr(payload) + '\n' + body.read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
    ast.parse(program.read_text(encoding='utf-8'), feature_version=(3, 8))
    work['candidates'].append(dict(id=CID, profile='experiment_first', stage='implementation',
        hypothesis='Native mask features x_mask(x) are expression-independent before their dot product '
            'with language-conditioned queries. A direct valid-language-conditioned channel modulation '
            'of the shared mask features may make conditional mask fitting easier. Existing query '
            'cross-attention can represent language interactions; this is an architectural comparison, '
            'not a missing-information, novelty or efficacy claim.',
        matched_control='Unchanged W2 plus one mask-feature multiplication after native get_batches: '
            'f_next=f*(1+tanh(mean_valid(projected_language)@W)); W[256,256] exact zero/no RNG, '
            '+65536 parameters. Same modulated tensor feeds all seven native prediction heads. '
            'No change to shared confidence head, targets, losses, Boolean attention gate, selector, '
            'all-channel input, released SPFormer, seed1999, Q128 or fixed30 native YAML.',
        historical_scope='Targeted ledger scan found no FiLM/modulation or language-to-mask-feature '
            'path. Y28/Y33/Y40 rejected final-confidence readbacks/alignment; this changes the mask '
            'feature formation before all prediction stages and adds no confidence-only readback. '
            'Y53 rejected a padding-leak premise; no such premise is reused, and the new mean explicitly '
            'excludes padding. Y25 soft gate and Y26 discarded norm reconnect are unchanged. '
            'Y65/Y66 query feedback and Y67 stage normalization are absent. No rejected ID is reopened.',
        comparison_baseline='AdaRES-W2', implementation_package=PACKAGE, config=payload['variant']['config'],
        next_action='Execute CPU native YAML/full model/released backbone/real TRAIN collation plus '
            'independent valid-token formula, zero output/input-gradient/RNG anchors, token padding/'
            'permutation/duplication and wrong-language controls, seven-stage decoder execution and '
            'finite nonzero modulation gradients. Actual sparse TRAIN two-step loss/backward/optimizer '
            'on an exclusive authorized GPU is mandatory before admission.',
        evidence=[ref(p) for p in target] + [ref(program), ref(body), ref(__file__)]))
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(id=CID, program=ref(program), native_executed=False, admitted=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--token', required=True); args = parser.parse_args()
    lease(args.token, 180)
    lock = monitor.acquire_lock(str(ROOT)); assert lock is not None
    try: prepare(args.token)
    finally: lock.close()
