"""Mechanically bind existing production tests to the preregistered revision2."""
import argparse, ast, copy, hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import foreground_tick as tick
parser=argparse.ArgumentParser()
parser.add_argument('--token',required=True)
parser.add_argument('--suffix',choices=('', 'compat'),default='')
args=parser.parse_args()
owner=tick.load(ROOT/'pipeline/headless/.foreground.lockdir/owner.json')
assert owner['token']==args.token and owner['thread_id']==os.environ['CODEX_THREAD_ID']
assert datetime.now(timezone.utc)<datetime.fromisoformat(owner['deadline_at'])
old_sha='7dd952f18632392f5e5a050b4aabcfd10de5bcfdad9c81f494de353dfa56cf8f'
new_path=ROOT/'AdaRES/ipdn/model_paper5/adares_y46_component_coherence/dec.py'
new_sha=hashlib.sha256(new_path.read_bytes()).hexdigest()
old_path=ROOT/('pipeline/analyses/y46_revision1_dec_f56b_'+args.token+'.py')
assert hashlib.sha256(old_path.read_bytes()).hexdigest()==old_sha
old_tree,new_tree=(ast.parse(p.read_text()) for p in (old_path,new_path))
def forward(tree):
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AllComponentReadout')
    return next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
forward(old_tree).body=copy.deepcopy(forward(new_tree).body)
assert ast.dump(old_tree)==ast.dump(new_tree),'Revision changed more than preregistered pooling'
class Bind(ast.NodeTransformer):
    def visit_Constant(self,node):
        if node.value==old_sha:return ast.copy_location(ast.Constant(value=new_sha),node)
        if isinstance(node.value,str) and node.value.startswith('Revision1 component memoization and original source/cost risks retained;'):
            return ast.copy_location(ast.Constant(value='Revision1 memoization retained; revision2 only stable grouped reduction. Original failures/cost risks retained.'),node)
        return node
    def visit_keyword(self,node):
        self.generic_visit(node)
        if node.arg=='implementation_revision':node.value=ast.Constant(value=2)
        return node
records=[]
for mode,name in (('cpu','probe_y46_production_cpu_131e.py'),('native','probe_y46_production_native_131e.py'),('controls','probe_y46_production_controls_f56b.py')):
    source=ROOT/'scripts'/name
    tree=Bind().visit(ast.parse(source.read_text()))
    if mode=='cpu':
        revised=[]
        for node in tree.body:
            revised.append(node)
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='readout' for t in node.targets):
                revised.extend(ast.parse("""
archive=Path('/tmp/adares_y46_revision1_dec_f56b.py')
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='7dd952f18632392f5e5a050b4aabcfd10de5bcfdad9c81f494de353dfa56cf8f'
old_class=next(n for n in ast.parse(archive.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='AllComponentReadout')
old_ns=dict(torch=torch,nn=nn)
exec(compile(ast.Module(body=[old_class],type_ignores=[]),str(archive),'exec'),old_ns)
old_readout=old_ns['AllComponentReadout'](256).double()
old_readout.load_state_dict(readout.state_dict())
old_new_errors=[]
""").body)
        tree.body=revised
        class Compare(ast.NodeTransformer):
            def visit_Assign(self,node):
                if any(isinstance(t,ast.Name) and t.id=='score' for t in node.targets):
                    return [node,*ast.parse("""
with torch.no_grad():
    old_score=old_readout(query,actual)
    error=float((score-old_score).abs().max())
    envelope=8*torch.finfo(torch.double).eps*max(1,len(masses))*max(1.0,float(score.abs().max()),float(old_score.abs().max()))
    assert error<=envelope
    old_new_errors.append(error)
""").body]
                return node
        tree=Compare().visit(tree)
        tree.body.insert(-1,ast.parse("result['revision2_old_new_CPU64_max_error']=max(old_new_errors)").body[0])
    text=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    compile(text,'<bound-y46-revision2>','exec')
    relative='pipeline/analyses/y46_revision2_'+mode+'_program_f56b_'+args.suffix+args.token+'.py'
    path=ROOT/relative
    with path.open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
    records.append(dict(kind=mode,path=relative,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),template=dict(path=source.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(source.read_bytes()).hexdigest())))
tick.save(ROOT/('pipeline/analyses/y46_revision2_test_bindings_f56b_'+args.suffix+args.token+'.json'),dict(complete=True,id='AdaRES-Y46',implementation_revision=2,created_at=datetime.now().astimezone().isoformat(),new_decoder_sha256=new_sha,only_preregistered_pooling_AST_changed=True,programs=records,exact_native_repeat_assertion_unchanged=True))
print(json.dumps(records))
