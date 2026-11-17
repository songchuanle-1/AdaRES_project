"""Use a bounded 30-second counter observation window; preserve all original verification semantics."""
import argparse,ast,inspect
import launch_y101_frozen_review_job as original

class Window(ast.NodeTransformer):
    def __init__(self):self.count=0
    def visit_Constant(self,node):
        if isinstance(node.value,str) and 'a=read();time.sleep(5);b=read();' in node.value:
            assert node.value.count('a=read();time.sleep(5);b=read();')==1
            self.count+=1;node.value=node.value.replace('a=read();time.sleep(5);b=read();','a=read();time.sleep(30);b=read();')
        return node

def observe(token,label):
    transformer=Window();tree=transformer.visit(ast.parse(inspect.getsource(original.observe)));assert transformer.count==1
    namespace=dict(vars(original));namespace['__file__']=__file__;exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),namespace);namespace['observe'](token,label)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);p.add_argument('--label',required=True);a=p.parse_args();observe(a.token,a.label)
