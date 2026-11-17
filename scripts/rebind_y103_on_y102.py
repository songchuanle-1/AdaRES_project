"""Bind unchanged Y103 bytes to actual-native Y102 and exact current run owners."""
import argparse,ast,json,re
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from y102_deployment import Y101,Y102,git
OLD='960babebd812098efde2eb1b188e612028ae571e'
OWNERS={'server-1':'AdaRES-Y101','server-2':'AdaRES-Y102'}
REPLACEMENTS={
    'afc93861cf66b0da4835391597ee2d5ce90f6584':OLD,
    'refs/heads/adares-y95-query-set-mean-on-y94':'refs/heads/adares-y103-spatial-input-skip-on-y102',
    'refs/heads/adares-y95-query-set-mean':'refs/heads/adares-y103-spatial-input-skip',
    'run_y95_bound_native_y94':'run_y103_bound_native_y102','run_y95_native':'run_y103_native',
    'retain_y95_cpu_native':'retain_y103_cpu_native','test_y95_runtime_plan':'test_y103_runtime_plan',
    'rebind_y95_on_y94':'rebind_y103_on_y102','prepare_y95_query_set_mean':'prepare_y103_spatial_input_skip',
    'y95_contract':'y103_contract','advance_y94':'advance_y102','y94_deployment':'y102_deployment',
    'y95_y94_additive_binding_':'y103_y102_additive_binding_','original_pre_y94_binding':'original_pre_y102_binding',
    'AdaRES-Y95':'AdaRES-Y103','AdaRES-Y94':'AdaRES-Y102','AdaRES-Y93':'AdaRES-Y101',
    'HEADY94':'HEADY102','Y94/server1':'Y102/server2','Y93/server2':'Y101/server1',
    'Y95 bytes':'Y103 bytes','Y95 additions':'Y103 additions','native-tested Y94':'native-tested Y102',
    'CPU-tested Y95':'CPU-tested Y103','exact Y94 parent':'exact Y102 parent'}
PATTERN=re.compile('|'.join(re.escape(k) for k in sorted(REPLACEMENTS,key=len,reverse=True)))
RUNNER='''
class Runner(ast.NodeTransformer):
    def visit_Constant(self,node):
        if node.value=="6fb26984ae652efbba177ab20c39575c44a69517":node.value=Y102
        elif isinstance(node.value,str):node.value=node.value.replace("run_y103_native.py","run_y103_bound_native_y102.py")
        return node
    def visit_Assign(self,node):
        node=self.generic_visit(node)
        if any(isinstance(t,ast.Name) and t.id=="OWNERS" for t in node.targets):node.value=ast.parse(repr(OWNERS),mode="eval").body
        return node
    def visit_FunctionDef(self,node):
        if node.name=="validate_binding":return ast.parse(BINDING).body[0]
        return self.generic_visit(node)
'''

class Specialize(ast.NodeTransformer):
    def __init__(self,pid):self.pid=pid
    def text(self,v):return PATTERN.sub(lambda m:REPLACEMENTS[m.group()],v)
    def visit_Name(self,n):n.id={'Y93':'Y101','Y94':'Y102'}.get(n.id,n.id);return n
    def visit_Attribute(self,n):n.attr={'Y93':'Y101','Y94':'Y102'}.get(n.attr,n.attr);return self.generic_visit(n)
    def visit_ImportFrom(self,n):
        n.module=self.text(n.module)
        for a in n.names:a.name={'Y93':'Y101','Y94':'Y102'}.get(a.name,a.name)
        return n
    def visit_keyword(self,n):
        if n.arg:n.arg=self.text(n.arg)
        return self.generic_visit(n)
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
        if n.name=='test_actual_parent_native_and_preserved_current_run_sources':
            return ast.parse('''
def test_actual_parent_native_and_preserved_current_run_sources(self):
    self.assertEqual(parent_evidence(),self.proposal["parent_native"])
    self.assertEqual(list(contract(Y101,Y102,"AdaRES-Y101")),[Y101,Y102])
    queue=monitor.load_queue(str(ROOT))
    for name,cid,pid,version in (("server-1","AdaRES-Y101",2263581,Y101),("server-2","AdaRES-Y102",'''+str(self.pid)+''',Y102)):
        task=next(r for r in queue["gpu_queues"][name] if r["id"]==cid)
        self.assertEqual((task["status"],task["pid"],task["code_commit"]),("running",pid,version))
        self.assertTrue(task["launch_verified"]);closure.admission(ROOT,task)
''').body[0]
        return self.generic_visit(n)

def rebind(token):
    lease(token,300);assert git('rev-parse','HEAD').decode().strip()==Y102
    queue=monitor.load_queue(str(ROOT));task=next(r for r in queue['gpu_queues']['server-2'] if r['id']=='AdaRES-Y102')
    assert task['status']=='running' and task['launch_verified'] is True and task['code_commit']==Y102 and type(task['pid']) is int and task['pid']>0
    dispatch=tick.load(ROOT/('pipeline/checks/adares-y102_dispatch_'+token+'.json'));assert dispatch['task']==task
    source=ROOT/'scripts/rebind_y95_on_y94.py';original=ref(source)
    tree=Specialize(task['pid']).visit(ast.parse(source.read_text(encoding='utf-8')));body=ast.unparse(ast.fix_missing_locations(tree))+'\n';ast.parse(body,feature_version=(3,8))
    out=ROOT/('pipeline/checks/y103_y102_binding_wrapper_'+token+'.py');assert not out.exists();out.write_text(body,encoding='utf-8',newline='\n')
    namespace={'__name__':'y103_y102_binding_generation','__file__':__file__};exec(compile(body,str(out),'exec'),namespace)
    assert namespace['CID']=='AdaRES-Y103' and namespace['OLD']==OLD and namespace['Y101']==Y101 and namespace['Y102']==Y102 and namespace['OWNERS']==OWNERS
    assert namespace['BRANCH']=='refs/heads/adares-y103-spatial-input-skip-on-y102' and namespace['RUNNER']=='scripts/run_y103_bound_native_y102.py'
    namespace['rebind'](token);assert ref(source)==original
    print(json.dumps(dict(wrapper=ref(out),original_helper=original,scientific_CPU_replayed=False,native_GPU_executed=False,production_HEAD_preserved=Y102)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);rebind(p.parse_args().token)
