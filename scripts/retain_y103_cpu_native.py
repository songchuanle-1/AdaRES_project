"""Retain the first CPU-tested Y103 and prepare exact unexecuted native GPU0 tasks."""
import argparse,ast,json
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from retain_y100_cpu_native import Rewrite
from y101_deployment import Y101
RUNNER='''
def create_runner():
    source=ROOT/'scripts/run_y101_native.py';target=ROOT/'scripts/run_y103_native.py'
    old_test=ROOT/'scripts/test_y101_runtime_plan.py';new_test=ROOT/'scripts/test_y103_runtime_plan.py'
    assert not target.exists() and not new_test.exists()
    changes={'run_y101_native':'run_y103_native','original_Y99_Y100_preserved':'original_Y101_Y100_preserved','Y101':'Y103','y101':'y103'}
    tree=Rewrite(changes).visit(ast.parse(source.read_text(encoding='utf-8')))
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PARENT' for t in n.targets):n.value=ast.Constant(value=HEAD)
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='OWNERS' for t in n.targets):n.value=ast.parse(repr(OWNERS),mode='eval').body
    text=ast.unparse(ast.fix_missing_locations(tree))+'\\n'
    text=text.replace('all six complete original FFNs/current-query bypasses/static masks/shared heads/hard gates/gradients','complete original spatial/appearance projections, whole spatial-input skip, sum_norm/visual memory/static7/shared heads/hard gates/gradients')
    ast.parse(text,feature_version=(3,8));target.write_text(text,encoding='utf-8',newline='\\n')
    tree=Rewrite(changes).visit(ast.parse(old_test.read_text(encoding='utf-8')))
    for n in ast.walk(tree):
        if isinstance(n,ast.FunctionDef) and n.name=='test_matching_current_owner_full_completion_required':
            loop=next(x for x in n.body if isinstance(x,ast.For));loop.iter=ast.parse(repr(tuple(OWNERS.items())),mode='eval').body
    text=ast.unparse(ast.fix_missing_locations(tree))+'\\n';ast.parse(text,feature_version=(3,8));new_test.write_text(text,encoding='utf-8',newline='\\n')
    return [ref(source),ref(target),ref(old_test),ref(new_test)]
'''
def operate(token,operation):
    lease(token,400);source=ROOT/'scripts/retain_y101_cpu_native.py';original=ref(source)
    changes={
        'prepare_y101_query_ffn_skip':'prepare_y103_spatial_input_skip',
        'refs/heads/adares-y101-query-ffn-skip':'refs/heads/adares-y103-spatial-input-skip',
        'y100_deployment':'y101_deployment','advance_y100':'advance_y101',
        'original_Y99_Y100_preserved':'original_Y101_Y100_preserved',
        'shared current-query FFN bypass':'whole spatial-input visual-memory bypass',
        'all six complete original FFNs/current-query bypasses/static masks/shared heads/hard gates/gradients':'whole original spatial-input/projection/fusion/visual-memory/static7/shared heads/hard gates/gradients',
        'all six complete original query FFNs/current pre-FFN identity/twelve stage consumers/original static masks7/shared heads/hard gates/full gradients/native modes/budget':'complete original spatial and appearance projections/full32 raw spatial bypass/sum_norm/visual memory/static7/shared heads/hard gates/full gradients/native modes/budget',
        'scalarfloat32/64/no-RNG/permutations/slot-locality/seven-stage W2/input-gradient/exact initial readout/causal refined masks/bypass':'scalarfloat32/64/no-RNG/permutations/sign/row-locality/seven-stage W2/input-gradient/causal initial-final masks/exact bypass',
        'all six original FFNs and shared skip gradients verified':'whole raw spatial input/original projections/fusion/static mask and shared skip gradients verified',
        'Y101':'Y103','y101':'y103'}
    tree=Rewrite(changes).visit(ast.parse(source.read_text(encoding='utf-8')))
    for i,n in enumerate(tree.body):
        if isinstance(n,ast.ImportFrom) and n.module=='y101_deployment':n.names=[ast.alias(name='Y100'),ast.alias(name='Y101'),ast.alias(name='contract')]
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='HEAD' for t in n.targets):n.value=ast.Name(id='Y101',ctx=ast.Load())
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='OWNERS' for t in n.targets):n.value=ast.parse(repr({'server-1':'AdaRES-Y101','server-2':'AdaRES-Y100'}),mode='eval').body
        if isinstance(n,ast.FunctionDef) and n.name=='create_runner':tree.body[i]=ast.parse(RUNNER).body[0]
    for n in ast.walk(tree):
        if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and [getattr(x,'id',None) for x in n.target.elts]==['name','cid','pid','commit']:
            n.iter=ast.parse("(('server-1','AdaRES-Y101',2263581,Y101),('server-2','AdaRES-Y100',19305,Y100))",mode='eval').body
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='update' and {'stage','next_action','unblock_condition','blocker'}.issubset({k.arg for k in n.keywords}):
            for k in n.keywords:
                if k.arg=='blocker':k.value=ast.Constant('Original Y101/server1GPU0 and Y100/server2GPU0 remain reserved; first CPU feasibility passed, actual native sparse TRAIN still required.')
                if k.arg=='unblock_condition':k.value=ast.Constant('At exact HEADY101 after matching original Y101/server1 or Y100/server2 naturally fixed30/full finite exact checkpoint/canonical archive, then fresh exclusive authorized idle GPU0.')
                if k.arg=='next_action':k.value=ast.Constant('Y102 precedes Y103. Do not replay passed CPU/version/plan tests. At HEADY101 ONLY after matching original Y101/server1 or Y100/server2 naturally fixed30/full finite original checkpoint/canonical archive and fresh authorized exclusive idle GPU0 use run_y103_native.py; changed HEAD/owner requires NEW tested same-byte binding. Actual native PASS must continue through complete original selected-device freeze/admission/source selection/dispatch/increasing counters same tick. GPU1/Y57 pauses remain; efficacy unknown.')
    body=ast.unparse(ast.fix_missing_locations(tree))+'\n';ast.parse(body,feature_version=(3,8))
    wrapper=ROOT/('pipeline/checks/y103_retention_'+token+'_program.py')
    if operation=='retain':assert not wrapper.exists();wrapper.write_text(body,encoding='utf-8',newline='\n')
    else:assert wrapper.read_text(encoding='utf-8')==body
    ns={'__name__':'y103_retention_operations','__file__':__file__};exec(compile(body,str(wrapper),'exec'),ns)
    assert ns['CID']=='AdaRES-Y103' and ns['HEAD']==Y101 and ns['OWNERS']=={'server-1':'AdaRES-Y101','server-2':'AdaRES-Y100'}
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        ns[operation](token);assert ref(source)==original
        w=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in w['candidates'] if r['id']=='AdaRES-Y103');c['evidence'].extend([original,ref(wrapper),ref(__file__)]);tick.save(ROOT/'pipeline/research_work.json',w)
    finally:writer.close()
    print(json.dumps(dict(retention_program=ref(wrapper),immutable_operations=original,operation=operation)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=('retain','plan'));p.add_argument('--token',required=True);a=p.parse_args();operate(a.token,a.operation)
