"""Specialize preserved server1 terminal tools for the exact original Y101 run."""
import argparse,ast,json,re
from prepare_y60_stage_heads import ROOT,lease,ref
from y101_deployment import Y101,git
CHANGES={
    'y99_deployment_v2':'y101_deployment','advance_y99_v2':'advance_y101',
    'adares_y99_spatial_mask_skip':'adares_y101_query_ffn_skip',
    '53f5b028bdf24e8b41dd934931d1fedc99843d8d':Y101,
    '4d87ad8b3a6efe78be8dc9e8214bba9226160a22b649f64f8e1dd26d73988ede':'379a5b050624141f20345f614877f062f1a61309b55531fe977e023538fecd0c',
    'fd07c0536891d905a7cd89e2b4211b32ed31a532b7b70ac9d71a8d7a66e97dbc':'e624a80505b35aa2fa09fa55bde64ed8f8e1915dbe385da3073db5b44bc81db1',
    'Y99':'Y101','y99':'y101'}
PATTERN=re.compile('|'.join(re.escape(k) for k in sorted(CHANGES,key=len,reverse=True)))
class Specialize(ast.NodeTransformer):
    def text(self,v):return PATTERN.sub(lambda m:CHANGES[m.group()],v)
    def visit_Name(self,n):n.id=self.text(n.id);return n
    def visit_Attribute(self,n):n.attr=self.text(n.attr);return self.generic_visit(n)
    def visit_ImportFrom(self,n):n.module=self.text(n.module);return self.generic_visit(n)
    def visit_alias(self,n):
        n.name=self.text(n.name)
        if n.asname:n.asname=self.text(n.asname)
        return n
    def visit_Constant(self,n):
        if isinstance(n.value,str):n.value=self.text(n.value)
        elif type(n.value) is int and n.value==2980836:n.value=2263581
        return n
def create(token):
    lease(token,300);assert git('rev-parse','HEAD').decode().strip()==Y101
    source=ROOT/'scripts/create_y99_terminal_helpers.py';original=ref(source)
    tree=Specialize().visit(ast.parse(source.read_text(encoding='utf-8')));body=ast.unparse(ast.fix_missing_locations(tree))+'\n';ast.parse(body,feature_version=(3,8))
    out=ROOT/('pipeline/checks/y101_terminal_wrapper_'+token+'.py');assert not out.exists();out.write_text(body,encoding='utf-8',newline='\n')
    ns={'__name__':'y101_terminal_generation','__file__':__file__};exec(compile(body,str(out),'exec'),ns)
    assert ns['CID']=='AdaRES-Y101' and ns['ADMISSION']==CHANGES['4d87ad8b3a6efe78be8dc9e8214bba9226160a22b649f64f8e1dd26d73988ede']
    ns['create'](token);assert ref(source)==original
    print(json.dumps(dict(wrapper=ref(out),original_generator=original,checkpoint_loaded=False,archived=False)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);create(p.parse_args().token)
