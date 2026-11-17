import ast,hashlib,importlib,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import gorilla
assert P['physical_gpu'] is None
torch.set_num_threads(4)
root=Path(P['root'])
package=root/'ipdn/model_paper5/adares_y49_resolvent_coherence'
parent=root/'ipdn/model_paper5/adares_w2_pointmass_competition'
tree=ast.parse((package/'dec.py').read_text())
tree.body=tree.body[4:]
tree.body=[n for n in tree.body if not ((isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('reference_inverse','inverse_diagonal','descriptors','GlobalResolventReadout','FinalResolventAttachment')) or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='forest_counts' for t in n.targets)))]
decoder=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DEC')
constructor=next(n for n in decoder.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
constructor.body=constructor.body[:-3]
assert ast.dump(tree)==ast.dump(ast.parse((parent/'dec.py').read_text()))
module=importlib.import_module('ipdn.model_paper5.adares_y49_resolvent_coherence.dec')
reference=importlib.import_module('ipdn.model_paper5.adares_w2_pointmass_competition.dec')
config=root/'configs_song/model_paper5/adares_y49_resolvent_coherence_res.yaml'
assert config.read_bytes()==(root/'configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml').read_bytes()
for name in ('__init__.py','backbone.py','ipdn.py','loss.py','sample_model.py'):
    assert (package/name).read_bytes()==(parent/name).read_bytes()
cfg=gorilla.Config.fromfile(str(config))
torch.manual_seed(1999)
physical_parent=reference.DEC(**cfg.model.dec,sampling_module=cfg.model.sampling_module)
parent_rng=torch.random.get_rng_state().clone()
parent_state=physical_parent.state_dict()
torch.manual_seed(1999)
physical_candidate=module.DEC(**cfg.model.dec,sampling_module=cfg.model.sampling_module)
state=physical_candidate.state_dict()
new=set(state)-set(parent_state)
assert len(new)==4 and all(k.startswith('mask_resolvent_readout.') for k in new)
assert all(torch.equal(v,state[k]) for k,v in parent_state.items())
assert torch.equal(parent_rng,torch.random.get_rng_state())
assert sum(state[k].numel() for k in new)==66561
physical_candidate._resolvent_attachment.close()
GlobalResolventReadout=module.GlobalResolventReadout
FinalResolventAttachment=module.FinalResolventAttachment
def cpu_mode():
    import ast
    source = Path(P['root']) / 'ipdn/model_paper5/adares_w2_pointmass_competition/dec.py'
    tree = ast.parse(source.read_text())
    cls = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'QueryCompetitionIndicator'))
    namespace = dict(torch=torch, nn=nn)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), namespace)

    class HookHarness(nn.Module):

        def __init__(self):
            super().__init__()
            self.num_layer = 6
            self.indi_embedding = namespace['QueryCompetitionIndicator'](256).double()
            self.query = torch.randn(2, 128, 256, dtype=torch.double)
            self.masks = torch.randn(2, 128, 16, dtype=torch.double)
            self.padding = torch.zeros(2, 16, dtype=torch.bool)
            self.mass = torch.randint(1, 16, (2, 16)).double()

        def forward(self, x, fps, offsets, language, language_masks, points):
            return [self.indi_embedding(self.query, self.masks, self.padding, self.mass) for _ in range(self.num_layer + 1)]
    rows = []
    for seed in range(32):
        torch.manual_seed(1999 + seed)
        decoder, readout = (HookHarness(), GlobalResolventReadout(256).double())
        attachment = FinalResolventAttachment(decoder, readout)
        points = torch.tensor([[i % 4, i // 4, 0] for i in range(16)], dtype=torch.double).repeat(2, 1)
        inputs = (None, None, torch.tensor([0, 16, 32]), None, None, points)
        parent = decoder(*inputs)
        attachment.enabled = True
        candidate = decoder(*inputs)
        assert all((torch.equal(left, right) for left, right in zip(parent[:-1], candidate[:-1])))
        assert torch.equal(parent[-1][..., 0], candidate[-1][..., 0])
        assert attachment.capture['calls'] == 7 and len(attachment.cached) == 2
        targets = torch.arange(128).remainder(2).repeat(2, 1)
        nn.functional.cross_entropy(candidate[-1].reshape(-1, 2), targets.flatten()).backward()
        gradients = {name: float(value.grad.norm()) for name, value in readout.named_parameters()}
        assert len(gradients) == 4 and all((np.isfinite(value) and value > 0 for value in gradients.values()))
        with torch.no_grad():
            repeated = decoder(*inputs)
            assert all((torch.equal(left, right) for left, right in zip(candidate, repeated)))
            attachment.wrong_correspondence = True
            wrong = decoder(*inputs)
            attachment.enabled = False
            disabled = decoder(*inputs)
        assert all((torch.equal(left, right) for left, right in zip(parent, disabled)))
        wrong_gap = float((candidate[-1] - wrong[-1]).abs().max())
        assert wrong_gap > 8 * torch.finfo(torch.double).eps * 128 * max(1.0, float(candidate[-1].abs().max()), float(wrong[-1].abs().max()))
        rows.append(dict(seed=seed, only_final_foreground_changed=True, auxiliary6_exact=True, disabled_parent_exact=True, exact_repeat=True, wrong_gap=wrong_gap, gradients=gradients))
        attachment.close()
    return dict(complete=True, id='AdaRES-Y49', CPU_cases=32, new_tensors=4, prototype_parameters=66561, minimum_CPU_CE_gradient=min((value for row in rows for value in row['gradients'].values())), minimum_wrong_correspondence_gap=min((row['wrong_gap'] for row in rows)), rows=rows, native_backward_executed=False, scientific_status='CPU_final_indicator_instrumentation_only', limitations=['HookHarness CE and exact source functions are not native model/loss execution.', 'No GPU, independent production package, native optimizer/resource/efficacy or admission claim.'])
result=cpu_mode()
result.update(physical_package_independent=True,physical_DEC_parent_state_exact=True,RNG_preserved=True,added_parameters=66561,new_tensors=4,implementation_revision=1,production_decoder_sha256=hashlib.sha256((package/"dec.py").read_bytes()).hexdigest())
Path(P["output"]).write_text(json.dumps(result),encoding="utf-8")
