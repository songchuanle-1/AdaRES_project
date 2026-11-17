"""Specialize preserved server2 lifecycle generation for original Y102, without GPU use."""
import argparse,ast,json,re
from prepare_y60_stage_heads import ROOT,lease,ref
from y102_deployment import Y102,git
CHANGES={
    'y100_deployment':'y102_deployment','advance_y100':'advance_y102',
    'adares_y100_appearance_input_skip':'adares_y102_geometry_mask_skip',
    '7091f206c21429c02349c73e68d5e2b06ca3adc9f585465450870ac2a0e019f2':'c37ae30c544492ea9e00e7c35ad4290b1062521e8182c13b885a3afb4e96968a',
    'cb9b9a80fb6f9f9b116f9e9a35a515b3c4fb542eb6f4128cbee0ed5d389eff01':'f24acc3ac40dcca556edecc4106e202cb1387aa49f564efdfe91435b99e898c4',
    'Y100':'Y102','y100':'y102','19305':'4747'}
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
    def visit_keyword(self,n):
        if n.arg:n.arg=self.text(n.arg)
        return self.generic_visit(n)
    def visit_Constant(self,n):
        if isinstance(n.value,str):n.value=self.text(n.value)
        elif type(n.value) is int and n.value==19305:n.value=4747
        return n
def create(token):
    lease(token,300);assert git('rev-parse','HEAD').decode().strip()==Y102
    source=ROOT/'scripts/create_y100_terminal_helpers.py';original=ref(source)
    tree=Specialize().visit(ast.parse(source.read_text(encoding='utf-8')));body=ast.unparse(ast.fix_missing_locations(tree))+'\n';ast.parse(body,feature_version=(3,8))
    out=ROOT/('pipeline/checks/y102_terminal_generation_'+token+'.py');assert not out.exists();out.write_text(body,encoding='utf-8',newline='\n')
    namespace={'__name__':'y102_terminal_generation','__file__':__file__};exec(compile(body,str(out),'exec'),namespace)
    assert namespace['ADMISSION']==CHANGES['7091f206c21429c02349c73e68d5e2b06ca3adc9f585465450870ac2a0e019f2']
    assert namespace['FREEZE']==CHANGES['cb9b9a80fb6f9f9b116f9e9a35a515b3c4fb542eb6f4128cbee0ed5d389eff01']
    namespace['create'](token);assert ref(source)==original
    print(json.dumps(dict(wrapper=ref(out),original_generator=original,checkpoint_loaded=False,archived=False,training_restart=False)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);create(p.parse_args().token)
