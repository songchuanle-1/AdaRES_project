"""CPU controls on actual frozen independent Y46 production code, not admission."""
import ast
import base64
import gc
import hashlib
import importlib
import json
from pathlib import Path
import time
import zlib
import gorilla
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import torch
from torch import nn
assert P['physical_gpu'] is None and (not torch.cuda.is_available())
torch.set_num_threads(4)
root = Path(P['root'])
parent = root / 'ipdn/model_paper5/adares_w2_pointmass_competition'
package = root / 'ipdn/model_paper5/adares_y46_component_coherence'
config = root / 'configs_song/model_paper5/adares_y46_component_coherence_res.yaml'
assert (package / 'dec.py').is_file()
hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(package.glob('*.py'))}
assert len(hashes) == 6
assert hashes['ipdn/model_paper5/adares_y46_component_coherence/dec.py'] == '363506506bf0e289d67dfe72c1a0041e66a67fd1962247a726c34d47b838c225'
for name in ('__init__.py', 'backbone.py', 'ipdn.py', 'loss.py', 'sample_model.py'):
    assert (package / name).read_bytes() == (parent / name).read_bytes()
assert config.read_bytes() == (root / 'configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml').read_bytes()
tree = ast.parse((package / 'dec.py').read_text())
tree.body = tree.body[3:]
tree.body = [node for node in tree.body if not (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in ('component_inputs', 'AllComponentReadout', 'FinalIndicatorAttachment'))]
decoder_ast = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'DEC'))
constructor = next((node for node in decoder_ast.body if isinstance(node, ast.FunctionDef) and node.name == '__init__'))
constructor.body = constructor.body[:-3]
assert ast.dump(tree) == ast.dump(ast.parse((parent / 'dec.py').read_text()))
for source in package.glob('*.py'):
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.ImportFrom):
            assert 'model_paper5' not in (node.module or '')
        if isinstance(node, ast.Import):
            assert not any(('model_paper5' in name.name for name in node.names))
module = importlib.import_module('ipdn.model_paper5.adares_y46_component_coherence.dec')
reference = importlib.import_module('ipdn.model_paper5.adares_w2_pointmass_competition.dec')
cfg = gorilla.Config.fromfile(str(config))
torch.manual_seed(1999)
physical_parent = reference.DEC(**cfg.model.dec, sampling_module=cfg.model.sampling_module)
parent_rng = torch.random.get_rng_state().clone()
parent_state = physical_parent.state_dict()
torch.manual_seed(1999)
physical_candidate = module.DEC(**cfg.model.dec, sampling_module=cfg.model.sampling_module)
candidate_state = physical_candidate.state_dict()
assert torch.equal(parent_rng, torch.random.get_rng_state())
new_keys = set(candidate_state) - set(parent_state)
assert len(new_keys) == 8 and all((key.startswith('mask_component_readout.') for key in new_keys))
assert all((torch.equal(value, candidate_state[key]) for key, value in parent_state.items()))
new_parameters = sum((value.numel() for key, value in candidate_state.items() if key in new_keys))
assert new_parameters == 328961
assert physical_candidate._component_attachment.enabled
assert physical_candidate._component_attachment.prototype is physical_candidate.mask_component_readout
physical_candidate._component_attachment.close()
del physical_parent, physical_candidate, parent_state, candidate_state
gc.collect()

class HookHarness(nn.Module):

    def __init__(self):
        super().__init__()
        self.num_layer = 6
        self.ila_ffn_layers = nn.ModuleList([nn.Identity()])
        self.indi_embedding = module.QueryCompetitionIndicator(256).double()
        self.query = torch.randn(2, 128, 256, dtype=torch.double)
        self.features = torch.randn(2, 16, 256, dtype=torch.double)
        self.masks = torch.randn(2, 128, 16, dtype=torch.double)
        self.padding = torch.zeros(2, 16, dtype=torch.bool)
        self.mass = torch.randint(1, 16, (2, 16)).double()

    def forward(self, x, fps, offsets, language, language_masks, points):
        self.ila_ffn_layers[-1](self.features)
        return [self.indi_embedding(self.query, self.masks, self.padding, self.mass) for _ in range(self.num_layer + 1)]
hook_cases = []
for seed in range(32):
    torch.manual_seed(1999 + seed)
    decoder = HookHarness()
    readout = module.AllComponentReadout(256).double()
    attachment = module.FinalIndicatorAttachment(decoder, readout)
    points = torch.tensor([[i % 4, i // 4, 0] for i in range(16)], dtype=torch.double).repeat(2, 1)
    inputs = (None, None, torch.tensor([0, 16, 32]), None, None, points)
    baseline = decoder(*inputs)
    attachment.enabled = True
    changed = decoder(*inputs)
    assert all((torch.equal(a, b) for a, b in zip(baseline[:-1], changed[:-1])))
    residual = torch.stack(attachment.residuals)
    assert torch.equal(changed[-1], baseline[-1] + torch.stack((torch.zeros_like(residual), residual), -1))
    targets = torch.arange(128).remainder(2).repeat(2, 1)
    nn.functional.cross_entropy(changed[-1].reshape(-1, 2), targets.flatten()).backward()
    gradients = {name: float(value.grad.norm()) for name, value in readout.named_parameters()}
    assert len(gradients) == 8 and all((np.isfinite(value) and value > 0 for value in gradients.values()))
    with torch.no_grad():
        repeated = decoder(*inputs)
        assert all((torch.equal(a, b) for a, b in zip(changed, repeated)))
        attachment.wrong_correspondence = True
        wrong = decoder(*inputs)
        attachment.enabled = False
        disabled = decoder(*inputs)
    assert all((torch.equal(a, b) for a, b in zip(baseline, disabled)))
    wrong_gap = float((wrong[-1] - changed[-1]).abs().max())
    bound = 8 * torch.finfo(torch.double).eps * 128 * max(1.0, float(changed[-1].abs().max()), float(wrong[-1].abs().max()))
    assert wrong_gap > bound
    hook_cases.append(dict(seed=seed, only_final_foreground_changed=True, disabled_parent_exact=True, repeat_exact=True, wrong_gap=wrong_gap, gradients=gradients))
    attachment.close()
    del decoder, readout, baseline, changed, repeated, wrong, disabled
    gc.collect()
capture_path = Path('/tmp/adares_y46_native_capture_d179_d179d824a64943e0b06bfa90de0caf52.json')
assert hashlib.sha256(capture_path.read_bytes()).hexdigest() == '5716585e9624d06ff5ecd90ddddec1faf95db9333199daf1b5eaa86907219d89'
capture = json.loads(capture_path.read_text())

def unpack(item):
    raw = zlib.decompress(base64.b64decode(item['zlib_base64']))
    assert hashlib.sha256(raw).hexdigest() == item['sha256']
    return np.frombuffer(raw, dtype=np.dtype(item['dtype'])).reshape(item['shape']).copy()

def uncached(graph, masks, masses, tokens):
    rows, weights, groups = ([], [], [])
    for query, mask in enumerate(masks):
        for side, selected in enumerate((mask, ~mask)):
            indices = np.flatnonzero(selected)
            if not len(indices):
                continue
            count, labels = connected_components(graph[indices][:, indices], directed=False)
            component_mass = np.bincount(labels, weights=masses[indices], minlength=count)
            features = np.zeros((count, tokens.shape[1]), dtype=np.float64)
            np.add.at(features, labels, tokens[indices] * masses[indices, None])
            fraction = component_mass / masses[indices].sum()
            rows.append(np.concatenate((features / component_mass[:, None], fraction[:, None]), 1))
            weights.append(fraction)
            groups.append(np.full(count, 2 * query + side, dtype=np.int64))
    return tuple((torch.from_numpy(np.concatenate(values)) for values in (rows, weights, groups)))
torch.manual_seed(1999)
readout = module.AllComponentReadout(256).double()
archive = Path('/tmp/adares_y46_revision1_dec_f56b.py')
assert hashlib.sha256(archive.read_bytes()).hexdigest() == '7dd952f18632392f5e5a050b4aabcfd10de5bcfdad9c81f494de353dfa56cf8f'
old_class = next((n for n in ast.parse(archive.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'AllComponentReadout'))
old_ns = dict(torch=torch, nn=nn)
exec(compile(ast.Module(body=[old_class], type_ignores=[]), str(archive), 'exec'), old_ns)
old_readout = old_ns['AllComponentReadout'](256).double()
old_readout.load_state_dict(readout.state_dict())
old_new_errors = []
rows = []
for scene in capture['rows']:
    for expression in scene['expressions']:
        started = time.monotonic()
        arrays = {key: unpack(value) for key, value in expression['packed_native_inputs'].items()}
        tokens, masses = (arrays['tokens'].astype(np.float64), arrays['masses'].astype(np.float64))
        masks, edges = (arrays['masks'] > 0, arrays['graph_edges'])
        graph = csr_matrix((np.ones(len(edges), bool), (edges[:, 0], edges[:, 1])), shape=(len(masses), len(masses)))
        query = torch.from_numpy(arrays['query'].astype(np.float64))
        actual = module.component_inputs(graph, masks, masses, tokens)
        expected = uncached(graph, masks, masses, tokens)
        assert all((torch.equal(a, b) for a, b in zip(actual, expected)))
        readout.zero_grad(set_to_none=True)
        score = readout(query, actual)
        with torch.no_grad():
            old_score = old_readout(query, actual)
            error = float((score - old_score).abs().max())
            envelope = 8 * torch.finfo(torch.double).eps * max(1, len(masses)) * max(1.0, float(score.abs().max()), float(old_score.abs().max()))
            assert error <= envelope
            old_new_errors.append(error)
        score.square().mean().backward()
        gradients = {name: float(value.grad.norm()) for name, value in readout.named_parameters()}
        assert all((np.isfinite(value) and value > 0 for value in gradients.values()))
        with torch.no_grad():
            assert torch.equal(score, readout(query, expected))
            assert torch.equal(score, readout(query, actual))
        rows.append(dict(scan_id=scene['scan_id'], ann_id=expression['ann_id'], rows_weights_groups_exact=True, scores_exact=True, CPU_square_gradients=gradients, elapsed_seconds=time.monotonic() - started))
        del actual, expected, score, arrays
        gc.collect()
assert len(rows) == 32
result = dict(complete=True, id='AdaRES-Y46', implementation_revision=2, scientific_status='production_CPU_controls_passed_native_production_pending', production_source_hashes=hashes, config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(), stripped_parent_AST_exact=True, physical_imports_independent=True, physical_DEC_parent_initialization_exact=True, RNG_preserved=True, added_parameters=new_parameters, new_tensors=8, CPU_hook_cases=32, CPU_captured_expressions=len(rows), native_capture_scenes=8, exact_production_vs_uncached_all_expressions=True, minimum_hook_CE_gradient=min((value for row in hook_cases for value in row['gradients'].values())), minimum_capture_square_gradient=min((value for row in rows for value in row['CPU_square_gradients'].values())), minimum_hook_wrong_correspondence_gap=min((row['wrong_gap'] for row in hook_cases)), hook_cases=hook_cases, rows=rows, limitations=['No GPU allocation, native full-model loss/backward, released-init optimizer step or resource admission.', 'HookHarness CE is a declared surrogate; captured native tensors test production readout, not the native training graph.', 'Production DEC constructor/import/state registration tested physically; full CPU DEC forward uses inherited CUDA-only paths and was not run.', 'Revision1 memoization retained; revision2 only stable grouped reduction. Original failures/cost risks retained.', 'No validation, threshold, component, feature, loss-weight or epoch selection; no efficacy/novelty/Pareto claim.'])
result['revision2_old_new_CPU64_max_error'] = max(old_new_errors)
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
