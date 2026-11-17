"""Build the same revised Y49 mechanism as an independent additive package."""
import argparse,ast,copy,hashlib,json,os,sys,tarfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import foreground_tick as tick
parser=argparse.ArgumentParser()
parser.add_argument('--token',required=True)
parser.add_argument('--native-parent',action='store_true')
args=parser.parse_args()
owner=tick.load(ROOT/'pipeline/headless/.foreground.lockdir/owner.json')
assert owner['token']==args.token and owner['thread_id']==os.environ['CODEX_THREAD_ID']
assert datetime.now(timezone.utc)<datetime.fromisoformat(owner['deadline_at'])
work=tick.load(ROOT/'pipeline/research_work.json')
candidate=next(c for c in work['candidates'] if c['id']=='AdaRES-Y49')
assert candidate['revision_history'][0]['result']=='passed'
parent=ROOT/'AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition'
package=ROOT/'AdaRES/ipdn/model_paper5/adares_y49_resolvent_coherence'
generation='nativeparent_' if args.native_parent else ''
if package.exists():
    assert args.native_parent
    previous=tick.load(ROOT/('pipeline/analyses/y49_production_bundle_f56b_'+args.token+'.json'))
    assert all(hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in previous['files'])
    assert (ROOT/('pipeline/analyses/y49_production_bundle_f56b_'+args.token+'.tar')).is_file()
else:
    assert not args.native_parent
native=ROOT/('pipeline/analyses/y49_revision1_native_program_f56b_'+args.token+'.py')
assert hashlib.sha256(native.read_bytes()).hexdigest()=='8ae82051f11ab3956d2fb355fe3ffd12a4640fd750f94df41fa0aeb251cd665f'
original=ast.parse(native.read_text())
names=('reference_inverse','inverse_diagonal','descriptors','GlobalResolventReadout','FinalResolventAttachment')
helpers=[copy.deepcopy(n) for n in original.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
assert len(helpers)==5
counter=next(copy.deepcopy(n) for n in original.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='forest_counts' for t in n.targets))
if args.native_parent:
    parent=ROOT/'AdaRES/ipdn/model_paper5/adares_y46_component_coherence'
    tree=ast.parse((parent/'dec.py').read_text(encoding='utf-8'))
    tree.body=tree.body[3:]
    tree.body=[n for n in tree.body if not (isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('component_inputs','AllComponentReadout','FinalIndicatorAttachment'))]
    base=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DEC')
    next(n for n in base.body if isinstance(n,ast.FunctionDef) and n.name=='__init__').body=next(n for n in base.body if isinstance(n,ast.FunctionDef) and n.name=='__init__').body[:-3]
else:
    tree=ast.parse((parent/'dec.py').read_text(encoding='utf-8'))
imports=ast.parse('import numpy as np\nfrom scipy.sparse import csr_matrix,diags,eye\nfrom scipy.sparse.csgraph import connected_components\nfrom scipy.sparse.linalg import splu\n').body
insert=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.ClassDef))
tree.body[insert:insert]=[counter,*helpers]
tree.body=imports+tree.body
decoder=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DEC')
constructor=next(n for n in decoder.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
constructor.body.extend(ast.parse('with torch.random.fork_rng(devices=[]):\n    self.mask_resolvent_readout=GlobalResolventReadout(d_model)\nself._resolvent_attachment=FinalResolventAttachment(self,self.mask_resolvent_readout)\nself._resolvent_attachment.enabled=True\n').body)
text=ast.unparse(ast.fix_missing_locations(tree))+'\n'
compile(text,'<independent-y49>','exec')
package.mkdir(exist_ok=args.native_parent)
records=[]
for name in ('__init__.py','backbone.py','dec.py','ipdn.py','loss.py','sample_model.py'):
    data=text.encode('utf-8') if name=='dec.py' else (parent/name).read_bytes()
    path=package/name
    with path.open('wb' if args.native_parent else 'xb') as stream:stream.write(data)
    records.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(data).hexdigest()))
config=ROOT/'AdaRES/configs_song/model_paper5/adares_y49_resolvent_coherence_res.yaml'
config_parent='adares_y46_component_coherence_res.yaml' if args.native_parent else 'adares_w2_pointmass_competition_res.yaml'
with config.open('wb' if args.native_parent else 'xb') as stream:stream.write((ROOT/'AdaRES/configs_song/model_paper5'/config_parent).read_bytes())
records.append(dict(path=config.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(config.read_bytes()).hexdigest()))
bundle=ROOT/('pipeline/analyses/y49_production_bundle_f56b_'+generation+args.token+'.json')
tick.save(bundle,dict(complete=True,id='AdaRES-Y49',implementation_revision=1,created_at=datetime.now().astimezone().isoformat(),files=records,helper_AST_identical=True,no_shared_entry_change=True,no_admission_or_training=True,limitations=['Native task diagnosis remains incomplete; zero observed native equal-component-partition witnesses retained.','Original/revised runtime risks retained; local package preparation is not native production/resources PASS.']))
archive=ROOT/('pipeline/analyses/y49_production_bundle_f56b_'+generation+args.token+'.tar')
with tarfile.open(archive,'x') as stream:
    for row in records:stream.add(ROOT/row['path'],arcname=row['path'])
cpu=ROOT/('pipeline/analyses/y49_production_cpu_f56b_'+generation+args.token+'.py')
cpu_text='''import ast,hashlib,importlib,json\nfrom pathlib import Path\nimport numpy as np\nimport torch\nfrom torch import nn\nimport gorilla\nassert P['physical_gpu'] is None\ntorch.set_num_threads(4)\nroot=Path(P['root'])\npackage=root/'ipdn/model_paper5/adares_y49_resolvent_coherence'\nparent=root/'ipdn/model_paper5/adares_w2_pointmass_competition'\nmodule=importlib.import_module('ipdn.model_paper5.adares_y49_resolvent_coherence.dec')\nreference=importlib.import_module('ipdn.model_paper5.adares_w2_pointmass_competition.dec')\nconfig=root/'configs_song/model_paper5/adares_y49_resolvent_coherence_res.yaml'\nassert config.read_bytes()==(root/'configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml').read_bytes()\nfor name in ('__init__.py','backbone.py','ipdn.py','loss.py','sample_model.py'):\n    assert (package/name).read_bytes()==(parent/name).read_bytes()\ncfg=gorilla.Config.fromfile(str(config))\ntorch.manual_seed(1999)\nphysical_parent=reference.DEC(**cfg.model.dec,sampling_module=cfg.model.sampling_module)\nparent_rng=torch.random.get_rng_state().clone()\nparent_state=physical_parent.state_dict()\ntorch.manual_seed(1999)\nphysical_candidate=module.DEC(**cfg.model.dec,sampling_module=cfg.model.sampling_module)\nstate=physical_candidate.state_dict()\nnew=set(state)-set(parent_state)\nassert len(new)==4 and all(k.startswith('mask_resolvent_readout.') for k in new)\nassert all(torch.equal(v,state[k]) for k,v in parent_state.items())\nassert torch.equal(parent_rng,torch.random.get_rng_state())\nassert sum(state[k].numel() for k in new)==66561\nphysical_candidate._resolvent_attachment.close()\nGlobalResolventReadout=module.GlobalResolventReadout\nFinalResolventAttachment=module.FinalResolventAttachment\n'''
cpu_text+=ast.unparse(next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='cpu_mode'))
cpu_text+='\nresult=cpu_mode()\nresult.update(physical_package_independent=True,physical_DEC_parent_state_exact=True,RNG_preserved=True,added_parameters=66561,new_tensors=4,implementation_revision=1,production_decoder_sha256=hashlib.sha256((package/"dec.py").read_bytes()).hexdigest())\nPath(P["output"]).write_text(json.dumps(result),encoding="utf-8")\n'
compile(cpu_text,'<y49-production-cpu>','exec')
with cpu.open('x',encoding='utf-8') as stream:stream.write(cpu_text)
sha=records[2]['sha256']
template=ROOT/'scripts/probe_y46_production_native_131e.py'
replacements={'AdaRES-Y46':'AdaRES-Y49','configs_song/model_paper5/adares_y46_component_coherence_res.yaml':'configs_song/model_paper5/adares_y49_resolvent_coherence_res.yaml','ipdn/model_paper5/adares_y46_component_coherence/dec.py':'ipdn/model_paper5/adares_y49_resolvent_coherence/dec.py','adares_y46_component_coherence':'adares_y49_resolvent_coherence','7dd952f18632392f5e5a050b4aabcfd10de5bcfdad9c81f494de353dfa56cf8f':sha}
class Bind(ast.NodeTransformer):
    def visit_Constant(self,node):
        if isinstance(node.value,str) and node.value in replacements:return ast.copy_location(ast.Constant(value=replacements[node.value]),node)
        if isinstance(node.value,int) and not isinstance(node.value,bool) and node.value in (328961,8):return ast.copy_location(ast.Constant(value={328961:66561,8:4}[node.value]),node)
        return node
    def visit_Attribute(self,node):
        self.generic_visit(node)
        node.attr={'mask_component_readout':'mask_resolvent_readout','_component_attachment':'_resolvent_attachment','prototype':'readout'}.get(node.attr,node.attr)
        return node
bound=Bind().visit(ast.parse(template.read_text()))
destination=ROOT/('pipeline/analyses/y49_production_native_f56b_'+generation+args.token+'.py')
native_text=ast.unparse(ast.fix_missing_locations(bound))+'\n'
compile(native_text,'<y49-production-native>','exec')
with destination.open('x',encoding='utf-8') as stream:stream.write(native_text)
print(json.dumps(dict(complete=True,files=records,CPU_program=str(cpu),native_program=str(destination))))
