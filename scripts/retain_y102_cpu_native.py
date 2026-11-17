"""Specialize preserved retention/runtime operations for tested Y102 geometry masks."""
import argparse,ast,json
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from retain_y100_cpu_native import Rewrite

def operate(token,operation):
    lease(token,400)
    source=ROOT/'scripts/retain_y101_cpu_native.py';original=ref(source)
    changes={
        'prepare_y101_query_ffn_skip':'prepare_y102_geometry_mask_skip',
        'refs/heads/adares-y101-query-ffn-skip':'refs/heads/adares-y102-geometry-mask-skip',
        'shared current-query FFN bypass':'all-scene centered-coordinate static-mask connection',
        'all six complete original FFNs/current-query bypasses/static masks/shared heads/hard gates/gradients':'full-scene geometry/mass/original masks and batching/static seven consumers/shared heads/hard gates/gradients',
        'all six complete original query FFNs/current pre-FFN identity/twelve stage consumers/original static masks7/shared heads/hard gates/full gradients/native modes/budget':'original full3 scene coordinates/point masses/centroid/padding/original mask batching/static seven consumers/shared heads/hard gates/full gradients/native modes/budget',
        'scalarfloat32/64/no-RNG/permutations/slot-locality/seven-stage W2/input-gradient/exact initial readout/causal refined masks/bypass':'scalarfloat32/64/no-RNG/permutations/translation/point-mass scale/padded NaN/scene isolation/seven-stage W2/input-gradient/exact initial scores/causal masks/bypass',
        'all six original FFNs and shared skip gradients verified':'original full-scene geometry/point-mass/static-mask/projection and shared skip gradients verified',
        'Y101':'Y102','y101':'y102',
    }
    tree=Rewrite(changes).visit(ast.parse(source.read_text(encoding='utf-8')))
    # Keep both original-owner plans, choosing server2 as the default for this independent candidate.
    for n in ast.walk(tree):
        if isinstance(n,ast.FunctionDef) and n.name=='plan':
            for call in ast.walk(n):
                if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='update':
                    keys={kw.arg for kw in call.keywords}
                    if 'native_smoke_plan' in keys and 'required_server' in keys:
                        for kw in call.keywords:
                            if kw.arg=='required_server':kw.value=ast.Constant(value='server-2')
                            if kw.arg=='native_smoke_plan':kw.value=ast.parse("plans['server-2']",mode='eval').body
    body=ast.unparse(ast.fix_missing_locations(tree))+'\n';ast.parse(body,feature_version=(3,8))
    wrapper=ROOT/('pipeline/checks/y102_retention_'+token+'_program.py')
    if operation=='retain':
        assert not wrapper.exists();wrapper.write_text(body,encoding='utf-8',newline='\n')
    else:
        assert wrapper.read_text(encoding='utf-8')==body
    ns={'__name__':'y102_retention_operations','__file__':__file__}
    exec(compile(body,str(wrapper),'exec'),ns)
    assert ns['CID']=='AdaRES-Y102' and ns['OWNERS']=={'server-1':'AdaRES-Y99','server-2':'AdaRES-Y100'}
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        ns[operation](token)
        assert ref(source)==original
        work=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in work['candidates'] if r['id']=='AdaRES-Y102')
        c['evidence'].extend([original,ref(wrapper),ref(__file__)])
        if operation=='plan':
            c['next_action']='Y101 precedes Y102. '+c['next_action']
        tick.save(ROOT/'pipeline/research_work.json',work)
    finally:writer.close()
    print(json.dumps(dict(retention_program=ref(wrapper),immutable_operations=original,operation=operation)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=('retain','plan'));p.add_argument('--token',required=True);a=p.parse_args();operate(a.token,a.operation)
