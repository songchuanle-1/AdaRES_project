"""Create one independent W2 whole-spatial-input visual-memory connection."""
import argparse,ast,json,re
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from y101_deployment import Y101,git
from create_y100_launch_helpers import Rewrite

def sub(text,changes):
    pattern=re.compile('|'.join(re.escape(k) for k in sorted(changes,key=len,reverse=True)))
    return pattern.sub(lambda m:changes[m.group()],text)
def create(token):
    lease(token,600);assert git('rev-parse','HEAD').decode().strip()==Y101
    before={p:ref(p) for p in ('pipeline/queue.json','pipeline/running.json','pipeline/research_work.json','pipeline/goal.json','SERVERS.yaml')};index=git('ls-files','--stage','-z')
    w=tick.load(ROOT/'pipeline/research_work.json');assert not any(r['id']=='AdaRES-Y103' for r in w['candidates'])
    sources=['scripts/prepare_y100_appearance_input_skip.py','scripts/y100_observer_core.py','scripts/y100_contract.py','scripts/test_y100_preparation.py']
    originals=[ref(p) for p in sources]
    oldcore=(ROOT/sources[1]).read_text(encoding='utf-8')
    # Exchange only the two visual-memory sources; static masks still consume original spatial x.
    core=sub(oldcore,{'input_proj_2d':'input_proj','input_proj':'input_proj_2d','appearance':'spatial','spatial':'appearance','1024':'32','32':'1024'})
    original_tree=ast.parse(oldcore);tree=ast.parse(core)
    mask=next(n for n in original_tree.body if isinstance(n,ast.FunctionDef) and n.name=='capture_base_mask')
    for i,n in enumerate(tree.body):
        if isinstance(n,ast.FunctionDef) and n.name=='capture_base_mask':tree.body[i]=mask
    core=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    core=core.replace("state['appearance_projected'] + state['spatial_adapted']","state['spatial_adapted'] + state['appearance_projected']")
    core=core.replace("if raw.requires_grad:\n        assert raw.grad is not None and torch.isfinite(raw.grad).all()","assert raw.requires_grad and raw.grad is not None and torch.isfinite(raw.grad).all()")
    changes={'prepare_y100_appearance_input_skip':'prepare_y103_spatial_input_skip','y100_observer_core':'y103_observer_core','y100_contract':'y103_contract','Y100':'Y103','y100':'y103','AppearanceInputSkip':'SpatialInputSkip','appearance_input_skip':'spatial_input_skip','appearance_input':'spatial_input','164891781':'164637829','262144':'8192','original_appearance_projection_and_fusion_consumers_verified':'original_spatial_projection_and_fusion_consumers_verified'}
    tree=Rewrite(changes).visit(ast.parse((ROOT/sources[0]).read_text(encoding='utf-8')))
    module='''class SpatialInputSkip(nn.Module):
    """Keep complete native spatial MLP; all sparse channels bypass it in parallel."""
    def __init__(self,input_dim,d_model):
        super().__init__()
        self.weight=nn.Parameter(torch.zeros(input_dim,d_model))
    def forward(self,projected,spatial):
        return projected+spatial@self.weight


'''
    edits=(('class DEC(nn.Module):',module+'class DEC(nn.Module):'),('        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)','        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)\n        self.spatial_input_skip = SpatialInputSkip(self.input_proj[0].in_features, d_model)'),('        inst_feats = self.input_proj(x)','        inst_feats = self.spatial_input_skip(self.input_proj(x), x)'))
    bodies='''
def bodies():
    from prepare_y100_appearance_input_skip import bodies as original_bodies
    from create_y103_preparation import sub
    cpu,native=original_bodies()
    original_core=(ROOT/'scripts/y100_observer_core.py').read_text(encoding='utf-8')
    new_core=(ROOT/'scripts/y103_observer_core.py').read_text(encoding='utf-8')
    changes={'AppearanceInputSkip':'SpatialInputSkip','appearance_input_skip':'spatial_input_skip','appearance_input':'spatial_input','164891781':'164637829','262144':'8192',"'AdaRES-Y100'":repr(CID),'original_appearance_projection_and_fusion_consumers_verified':'original_spatial_projection_and_fusion_consumers_verified'}
    result=[]
    for body in (cpu,native):
        assert body.count(original_core)==1
        body=body.replace(original_core,'# SHARED_NEW_SPATIAL_CORE')
        body=sub(body,changes).replace('# SHARED_NEW_SPATIAL_CORE',new_core)
        ast.parse(body,feature_version=(3,8));assert body.count(new_core)==1
        result.append(body)
    return tuple(result)
'''
    for i,n in enumerate(tree.body):
        if isinstance(n,ast.ImportFrom) and n.module=='y99_deployment_v2':n.module='y101_deployment';n.names=[ast.alias(name='Y101'),ast.alias(name='git')]
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MODULE' for t in n.targets):n.value=ast.Constant(module)
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='EDITS' for t in n.targets):n.value=ast.parse(repr(edits),mode='eval').body
        if isinstance(n,ast.FunctionDef) and n.name=='bodies':tree.body[i]=ast.parse(bodies).body[0]
    for n in ast.walk(tree):
        if isinstance(n,ast.Name) and n.id=='Y99':n.id='Y101'
        if isinstance(n,ast.FunctionDef) and n.name=='prepare':
            for loop in ast.walk(n):
                if isinstance(loop,ast.For) and isinstance(loop.target,ast.Tuple) and [getattr(x,'id',None) for x in loop.target.elts]==['name','cid','pid']:
                    loop.iter=ast.parse(repr((('server-1','AdaRES-Y101',2263581),('server-2','AdaRES-Y100',19305))),mode='eval').body
        if isinstance(n,ast.Call) and any(k.arg=='hypothesis' for k in n.keywords):
            for k in n.keywords:
                if k.arg=='hypothesis':k.value=ast.Constant('Whole original32D sparse features enter visual memory through the complete original input_proj MLP. An independent zero-initialized full32 linear bypass beside that same MLP may ease spatial visual-memory fitting before the same original sum_norm. Original MLP and all modalities remain; no missing information, bug, novelty or benefit is presumed.')
                elif k.arg=='matched_control':k.value=ast.Constant('Independent W2: spatial=input_proj(all32 sparse)+all32 sparse@zeroW32x256 BEFORE SAME sum_norm(spatial+original full1024 appearance), +8192/total164637829/noRNG. Preserve complete original spatial/appearance/mask projections, all raw channels/points, static7/query generator/refinements/sampling/SCG/k/gamma/shared heads/hard gates/targets/loss/selector/backbone/data/seed1999fixed30/native YAML. No feature subset, threshold, temperature, loss, epoch or validation scan.')
                elif k.arg=='historical_scope':k.value=ast.Constant('Y103 connects whole original spatial features to visual memory before sum_norm. Y99 connects spatial features directly to static masks; Y100 bypasses appearance projection; Y102 connects centered scene coordinates to masks; Y71 multiplies projected modalities. Independent W2, no combination, tuning, reopened rejected route or relaxed numerical control.')
        if isinstance(n,ast.Constant) and isinstance(n.value,str):
            if n.value.startswith('# AdaRES-Y103:'):n.value='# AdaRES-Y103: complete spatial projection plus whole-input visual-memory bypass.\nfrom .ipdn import IPDN\n'
            if n.value.startswith('Y103 adds full raw1024'):n.value='Y103 routes all32 original sparse features beside complete input_proj BEFORE original sum_norm/visual memory. Y99 spatial-to-static-mask, Y100 appearance-to-memory, Y102 centered geometry-to-mask and Y71 projected product have different sources or destinations. No features removed, no combination, tuning or reopening.'
    contract=sub((ROOT/sources[2]).read_text(encoding='utf-8'),changes)
    contract=sub(contract,{'raw_appearance_requires_grad':'raw_spatial_requires_grad','appearance_feature_delta_max':'spatial_feature_delta_max','original_appearance_projection_identity':'original_spatial_projection_identity','float64_full_appearance_projection_reference':'float64_full_spatial_projection_reference'})
    contract=contract.replace('assert type(r["raw_spatial_requires_grad"]) is bool','assert r["raw_spatial_requires_grad"] is True')
    test_tree=Rewrite(changes).visit(ast.parse((ROOT/sources[3]).read_text(encoding='utf-8')))
    test=ast.unparse(ast.fix_missing_locations(test_tree))+'\n'
    test=sub(test,{'appearance_feature_delta_max':'spatial_feature_delta_max','original_appearance_projection_identity':'original_spatial_projection_identity','float64_full_appearance_projection_reference':'float64_full_spatial_projection_reference'})
    test=test.replace('c[\'AdaRES-Y99\'][\'stage\']','c[\'AdaRES-Y101\'][\'stage\']').replace('c[\'AdaRES-Y98\'][\'stage\']','c[\'AdaRES-Y100\'][\'stage\']')
    outputs={'scripts/prepare_y103_spatial_input_skip.py':ast.unparse(ast.fix_missing_locations(tree))+'\n','scripts/y103_observer_core.py':core,'scripts/y103_contract.py':contract,'scripts/test_y103_preparation.py':test}
    for p,body in outputs.items():assert not (ROOT/p).exists();ast.parse(body,feature_version=(3,8))
    for p,body in outputs.items():(ROOT/p).write_text(body,encoding='utf-8',newline='\n')
    assert all(ref(p)==r for p,r in before.items()) and git('ls-files','--stage','-z')==index
    out=ROOT/('pipeline/checks/y103_preparation_helpers_'+token+'.json');assert not out.exists();tick.save(out,dict(complete=True,token=token,original_inputs=originals,generated=[ref(p) for p in outputs],science_created=False,CPU_executed=False,native_executed=False))
    print(json.dumps(dict(manifest=ref(out),generated=len(outputs))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);create(p.parse_args().token)
