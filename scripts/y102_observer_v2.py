"""Correct only the observer's physical server mapping; preserve launch artifacts."""
import y102_deployment as original
from y102_deployment import *
OWNERS={'AdaRES-Y101':('server-1',Y101,350.0),'AdaRES-Y102':('server-2',Y102,280.0)}

def corrected_observer():
    tree=ast.parse(inspect.getsource(original.observe));assignments=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Tuple) and [getattr(v,'id',None) for v in t.elts]==['name','commit','power'] for t in node.targets):assignments.append(node)
    assert len(assignments)==1
    assignments[0].value=ast.parse(repr(OWNERS),mode='eval').body
    assignments[0].value=ast.Subscript(value=assignments[0].value,slice=ast.Name(id='cid',ctx=ast.Load()),ctx=ast.Load())
    text=ast.unparse(ast.fix_missing_locations(tree))+'\n';ast.parse(text,feature_version=(3,8))
    namespace=dict(vars(original));namespace['__file__']=__file__
    exec(compile(text,__file__,'exec'),namespace)
    return namespace['observe']

def observe(token,cid,label):
    assert cid in OWNERS
    queue=monitor.load_queue(str(ROOT));name,version,power=OWNERS[cid]
    task=next(r for r in queue['gpu_queues'][name] if r['id']==cid)
    assert task['status']=='running' and task['launch_verified'] is True
    assert task['server']==name and task['code_commit']==version and str(task['gpu'])=='0'
    corrected_observer()(token,cid,label)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=('observe',));p.add_argument('--id',required=True,choices=tuple(OWNERS));p.add_argument('--token',required=True);p.add_argument('--label',default='progress');a=p.parse_args()
    lease(a.token,240);writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:observe(a.token,a.id,a.label)
    finally:writer.close()
