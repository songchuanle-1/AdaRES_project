"""Bind the unchanged original Y102 scientific files to actual-native Y101/current owners."""
import argparse,ast,json,re
from prepare_y60_stage_heads import ROOT,lease,ref
from y101_deployment import Y100,Y101,git
OLD='e87aca94bc89ab107b2db887bc37c9d4fb4a3f73'
OWNERS={'server-1':'AdaRES-Y101','server-2':'AdaRES-Y100'}
REPLACEMENTS={
    'afc93861cf66b0da4835391597ee2d5ce90f6584':OLD,
    'refs/heads/adares-y95-query-set-mean-on-y94':'refs/heads/adares-y102-geometry-mask-skip-on-y101',
    'refs/heads/adares-y95-query-set-mean':'refs/heads/adares-y102-geometry-mask-skip',
    'run_y95_bound_native_y94':'run_y102_bound_native_y101','run_y95_native':'run_y102_native',
    'retain_y95_cpu_native':'retain_y102_cpu_native','test_y95_runtime_plan':'test_y102_runtime_plan',
    'rebind_y95_on_y94':'rebind_y102_on_y101','prepare_y95_query_set_mean':'prepare_y102_geometry_mask_skip',
    'y95_contract':'y102_contract','advance_y94':'advance_y101','y94_deployment':'y101_deployment',
    'y95_y94_additive_binding_':'y102_y101_additive_binding_','original_pre_y94_binding':'original_pre_y101_binding',
    'AdaRES-Y95':'AdaRES-Y102','AdaRES-Y94':'AdaRES-Y101','AdaRES-Y93':'AdaRES-Y100',
    'HEADY94':'HEADY101','Y94/server1':'Y101/server1','Y93/server2':'Y100/server2',
    'Y95 bytes':'Y102 bytes','Y95 additions':'Y102 additions','native-tested Y94':'native-tested Y101',
    'CPU-tested Y95':'CPU-tested Y102','exact Y94 parent':'exact Y101 parent'}
PATTERN=re.compile('|'.join(re.escape(k) for k in sorted(REPLACEMENTS,key=len,reverse=True)))
RUNNER='''
class Runner(ast.NodeTransformer):
    def visit_Constant(self,node):
        if node.value=="294de23f52e4ff93ebbf370c17c13d9a7d02e10b":node.value=Y101
        elif isinstance(node.value,str):node.value=node.value.replace("run_y102_native.py","run_y102_bound_native_y101.py")
        return node
    def visit_Assign(self,node):
        node=self.generic_visit(node)
        if any(isinstance(t,ast.Name) and t.id=="OWNERS" for t in node.targets):node.value=ast.parse(repr(OWNERS),mode="eval").body
        return node
    def visit_FunctionDef(self,node):
        if node.name=="validate_binding":return ast.parse(BINDING).body[0]
        return self.generic_visit(node)
'''
CURRENT='''
def test_actual_parent_native_and_preserved_current_run_sources(self):
    self.assertEqual(parent_evidence(),self.proposal["parent_native"])
    self.assertEqual(list(contract(Y100,Y101,"AdaRES-Y100")),[Y100,Y101])
    queue=monitor.load_queue(str(ROOT))
    for name,cid,pid,version in (("server-1","AdaRES-Y101",2263581,Y101),("server-2","AdaRES-Y100",19305,Y100)):
        task=next(r for r in queue["gpu_queues"][name] if r["id"]==cid)
        self.assertEqual((task["status"],task["pid"],task["code_commit"]),("running",pid,version))
        self.assertTrue(task["launch_verified"]);closure.admission(ROOT,task)
'''
class Specialize(ast.NodeTransformer):
    def text(self,v):return PATTERN.sub(lambda m:REPLACEMENTS[m.group()],v)
    def visit_Name(self,n):n.id={'Y93':'Y100','Y94':'Y101'}.get(n.id,n.id);return n
    def visit_ImportFrom(self,n):
        n.module=self.text(n.module)
        for a in n.names:a.name={'Y93':'Y100','Y94':'Y101'}.get(a.name,a.name)
        return n
    def visit_Constant(self,n):
        if isinstance(n.value,str):n.value=self.text(n.value)
        return n
    def visit_Assign(self,n):
        if any(isinstance(t,ast.Name) and t.id=='OWNERS' for t in n.targets):n.value=ast.parse(repr(OWNERS),mode='eval').body;return n
        return self.generic_visit(n)
    def visit_ClassDef(self,n):
        if n.name=='Runner':return ast.parse(RUNNER).body[0]
        return self.generic_visit(n)
    def visit_FunctionDef(self,n):
        if n.name=='test_actual_parent_native_and_preserved_current_run_sources':return ast.parse(CURRENT).body[0]
        return self.generic_visit(n)
def rebind(token):
    lease(token,300);assert git('rev-parse','HEAD').decode().strip()==Y101
    source=ROOT/'scripts/rebind_y95_on_y94.py';original=ref(source)
    tree=Specialize().visit(ast.parse(source.read_text(encoding='utf-8')));body=ast.unparse(ast.fix_missing_locations(tree))+'\n';ast.parse(body,feature_version=(3,8))
    out=ROOT/('pipeline/checks/y102_y101_binding_wrapper_'+token+'.py');assert not out.exists();out.write_text(body,encoding='utf-8',newline='\n')
    namespace={'__name__':'y102_y101_binding_generation','__file__':__file__};exec(compile(body,str(out),'exec'),namespace)
    assert namespace['CID']=='AdaRES-Y102' and namespace['OLD']==OLD and namespace['Y100']==Y100 and namespace['Y101']==Y101 and namespace['OWNERS']==OWNERS
    assert namespace['BRANCH']=='refs/heads/adares-y102-geometry-mask-skip-on-y101' and namespace['RUNNER']=='scripts/run_y102_bound_native_y101.py'
    namespace['rebind'](token);assert ref(source)==original
    print(json.dumps(dict(wrapper=ref(out),original_helper=original,scientific_CPU_replayed=False,native_GPU_executed=False,production_HEAD_preserved=Y101)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);rebind(p.parse_args().token)
