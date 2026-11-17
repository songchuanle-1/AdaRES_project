"""Bind the existing fixed TRAIN8/large-pair cost recipe to physical Y49."""
import argparse,ast,hashlib,json,os,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import foreground_tick as tick
parser=argparse.ArgumentParser()
parser.add_argument('--token',required=True)
args=parser.parse_args()
owner=tick.load(ROOT/'pipeline/headless/.foreground.lockdir/owner.json')
assert owner['token']==args.token and owner['thread_id']==os.environ['CODEX_THREAD_ID']
assert datetime.now(timezone.utc)<datetime.fromisoformat(owner['deadline_at'])
decoder=ROOT/'AdaRES/ipdn/model_paper5/adares_y49_resolvent_coherence/dec.py'
new_sha=hashlib.sha256(decoder.read_bytes()).hexdigest()
mapping={'AdaRES-Y46':'AdaRES-Y49','configs_song/model_paper5/adares_y46_component_coherence_res.yaml':'configs_song/model_paper5/adares_y49_resolvent_coherence_res.yaml','ipdn/model_paper5/adares_y46_component_coherence/dec.py':'ipdn/model_paper5/adares_y49_resolvent_coherence/dec.py','adares_y46_component_coherence':'adares_y49_resolvent_coherence','7dd952f18632392f5e5a050b4aabcfd10de5bcfdad9c81f494de353dfa56cf8f':new_sha,'363506506bf0e289d67dfe72c1a0041e66a67fd1962247a726c34d47b838c225':new_sha,'dec.mask_component_readout.':'dec.mask_resolvent_readout.'}
class Bind(ast.NodeTransformer):
    def visit_Constant(self,node):
        if isinstance(node.value,str) and node.value in mapping:return ast.copy_location(ast.Constant(value=mapping[node.value]),node)
        if isinstance(node.value,int) and not isinstance(node.value,bool) and node.value==328961:return ast.copy_location(ast.Constant(value=66561),node)
        return node
    def visit_Attribute(self,node):
        self.generic_visit(node)
        node.attr={'mask_component_readout':'mask_resolvent_readout','_component_attachment':'_resolvent_attachment','cached_inputs':'cached','residuals':'cached'}.get(node.attr,node.attr)
        return node
    def visit_keyword(self,node):
        self.generic_visit(node)
        if node.arg=='implementation_revision':node.value=ast.Constant(value=1)
        return node
records=[]
for kind,name in (('inference','probe_y46_production_controls_f56b.py'),('stress','probe_y46_revision2_stress_f56b.py')):
    source=ROOT/'scripts'/name
    tree=Bind().visit(ast.parse(source.read_text()))
    if kind=='inference':
        body=[]
        skip=False
        for node in tree.body:
            targets=[t.id for t in node.targets if isinstance(t,ast.Name)] if isinstance(node,ast.Assign) else []
            if 'queries' in targets:skip=True
            if 'data_cfg' in targets:skip=False
            if not skip:body.append(node)
        tree.body=body
        tree.body.insert(-1,ast.parse("result['cost_only']=True; result['limitations'].append('This cost-only harness omits the native cached-control loop; original/revised native diagnostic and production tests are separately recorded. No efficacy or admission.')").body[0])
    text=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    compile(text,'<y49-fixed-cost>','exec')
    path=ROOT/('pipeline/analyses/y49_production_'+kind+'_program_f56b_'+args.token+'.py')
    with path.open('x',encoding='utf-8') as stream:stream.write(text)
    records.append(dict(kind=kind,path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),template_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
tick.save(ROOT/('pipeline/analyses/y49_production_cost_prereg_f56b_'+args.token+'.json'),dict(complete=True,id='AdaRES-Y49',implementation_revision=1,preregistered_at=datetime.now().astimezone().isoformat(),programs=records,recipe='Same fixed TRAIN8 indices0,44,60,101,115,126,137,171 plus728/6780, serial predict2warmups/3readings and native batch2 stress2warmups/3readings; no numeric model/protocol changes.',no_admission=True))
print(json.dumps(records))
