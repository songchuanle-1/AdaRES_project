"""Independent native package, parent identity, RNG and final-only controls."""
import ast
import hashlib
import importlib
import json
from pathlib import Path
import gorilla
import numpy as np
import torch
from torch import nn

assert P['physical_gpu'] is None
torch.set_num_threads(4)
root = Path(P['root'])
package = root / 'ipdn/model_paper5/adares_y50_joint_incidence'
parent = root / 'ipdn/model_paper5/adares_w2_pointmass_competition'
module = importlib.import_module('ipdn.model_paper5.adares_y50_joint_incidence.dec')
reference = importlib.import_module('ipdn.model_paper5.adares_w2_pointmass_competition.dec')
decoder_sha = hashlib.sha256((package / 'dec.py').read_bytes()).hexdigest()
assert decoder_sha == '1587458066bd8b33396ad927f8268146ed4d8ac4043f44adf534aeef8fd5ef37'
config = root / 'configs_song/model_paper5/adares_y50_joint_incidence_res.yaml'
assert config.read_bytes() == (root / 'configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml').read_bytes()
for name in ('__init__.py','backbone.py','ipdn.py','loss.py','sample_model.py'):
    assert (package / name).read_bytes() == (parent / name).read_bytes()
tree = ast.parse((package / 'dec.py').read_text())
tree.body = [n for n in tree.body if not (isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('joint_incidence_descriptor','JointIncidenceReadout','FinalJointIncidenceAttachment'))
             and not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str) and n.value.value.startswith('Attach the fixed joint-incidence'))]
decoder = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == 'DEC')
constructor = next(n for n in decoder.body if isinstance(n,ast.FunctionDef) and n.name == '__init__')
assert ast.unparse(constructor.body[-1]) == 'self._joint_attachment.enabled = True'
constructor.body = constructor.body[:-3]
assert ast.dump(tree,include_attributes=False) == ast.dump(ast.parse((parent/'dec.py').read_text()),include_attributes=False)
cfg = gorilla.Config.fromfile(str(config))
torch.manual_seed(1999)
physical_parent = reference.DEC(**cfg.model.dec, sampling_module=cfg.model.sampling_module)
parent_rng = torch.random.get_rng_state().clone()
parent_state = physical_parent.state_dict()
torch.manual_seed(1999)
physical_candidate = module.DEC(**cfg.model.dec, sampling_module=cfg.model.sampling_module)
state = physical_candidate.state_dict()
new = set(state) - set(parent_state)
assert len(new) == 4 and all(k.startswith('mask_joint_readout.') for k in new)
assert all(torch.equal(v,state[k]) for k,v in parent_state.items())
assert torch.equal(parent_rng,torch.random.get_rng_state())
assert sum(state[k].numel() for k in new) == 66561
physical_candidate._joint_attachment.close()


class Harness(nn.Module):
    def __init__(self, indicator):
        super().__init__()
        self.indi_embedding = indicator
        self.num_layer = 6

    def forward(self, query, masks, padding, masses):
        return [self.indi_embedding(query,masks,padding,masses) for _ in range(7)]


rows = []
for seed in range(1999,2031):
    torch.manual_seed(seed)
    harness = Harness(reference.QueryCompetitionIndicator(256).double())
    head = module.JointIncidenceReadout(256).double()
    attachment = module.FinalJointIncidenceAttachment(harness,head)
    query = torch.randn((2,128,256),dtype=torch.double)
    masks = torch.randn((2,128,19),dtype=torch.double)
    padding = torch.zeros((2,19),dtype=torch.bool)
    padding[1,-3:] = True
    masses = torch.arange(1,20,dtype=torch.double).repeat(2,1)
    parent_values = harness(query,masks,padding,masses)
    attachment.enabled = True
    values = harness(query,masks,padding,masses)
    assert all(torch.equal(a,b) for a,b in zip(parent_values[:6],values[:6]))
    assert torch.equal(parent_values[-1][...,0],values[-1][...,0])
    assert not torch.equal(parent_values[-1][...,1],values[-1][...,1])
    repeated = harness(query,masks,padding,masses)
    assert all(torch.equal(a,b) for a,b in zip(values,repeated))
    attachment.wrong_correspondence = True
    wrong = harness(query,masks,padding,masses)
    gap = float((values[-1]-wrong[-1]).abs().max())
    assert gap > 0
    head.zero_grad(set_to_none=True)
    nn.functional.cross_entropy(values[-1].reshape(-1,2),torch.arange(256).remainder(2)).backward()
    gradients = {n:float(v.grad.norm()) for n,v in head.named_parameters()}
    assert len(gradients) == 4 and all(np.isfinite(v) and v>0 for v in gradients.values())
    attachment.enabled = False
    disabled = harness(query,masks,padding,masses)
    assert all(torch.equal(a,b) for a,b in zip(parent_values,disabled))
    attachment.close()
    rows.append(dict(seed=seed,auxiliary6_exact=True,background_exact=True,disabled_parent_exact=True,
                     repeated_exact=True,wrong_gap=gap,gradients=gradients))
result = dict(complete=True,id='AdaRES-Y50',physical_package_independent=True,parent_stripped_AST_exact=True,
              nondecoder_parent_bytes_exact=True,config_parent_bytes_exact=True,parent_init_state_exact=True,RNG_preserved=True,
              added_parameters=66561,new_tensors=4,production_decoder_sha256=decoder_sha,rows=rows,
              limitations=['Physical CPU/import/hooks/proxy gradients, not native model/loss or utility.',
                           'No validation selection, no source/optimizer/queue modification.'])
Path(P['output']).write_text(json.dumps(result),encoding='utf-8')
