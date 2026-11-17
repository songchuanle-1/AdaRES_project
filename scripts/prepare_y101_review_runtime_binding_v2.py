"""Bind the import guard to the verified original wrapper plus private binary, retaining the failed assumption."""
import argparse,ast,inspect
from pathlib import PurePosixPath
import prepare_y101_review_runtime_binding as original
from prepare_y60_stage_heads import ROOT,lease,ref,tick
import closure

class Labels(ast.NodeTransformer):
    def __init__(self):self.counts={k:0 for k in ('pipeline/checks/y101_review_import_','pipeline/checks/y101_review_runtime_')}
    def visit_Constant(self,node):
        if isinstance(node.value,str) and node.value in self.counts:
            self.counts[node.value]+=1;node.value=node.value.replace('review_import_','review_import_v2_').replace('review_runtime_','review_runtime_v2_')
        return node

def run(token):
    lease(token,600)
    work=tick.load(ROOT/'pipeline/research_work.json');candidate=next(r for r in work['candidates'] if r['id']=='AdaRES-Y101');assert candidate['stage']=='training' and not candidate.get('frozen_review_runtime_binding')
    closure.artifact(ROOT,candidate['frozen_review_runtime_initial_failure'])
    origin=tick.load(closure.artifact(ROOT,candidate['frozen_review_pointnet_origin']));assert origin['complete'] and not origin['CUDA_used']
    wrappers={r['resolved']:r for r in origin['wrappers'] if r['exists']};assert len(wrappers)==1
    wrapper=next(iter(wrappers.values()));assert wrapper['origin']=='/data1/songcl/Project/AdaRES_project/AdaRES/pointnet2/pointnet2_utils.py'
    binaries=origin['original_live_PID_pointnet_binaries'];assert len(binaries)==1
    binary=binaries[0];assert binary['path'].startswith(origin['pointnet_egg']+'/pointnet2/_ext.') and binary['bytes']>0
    guard="assert P['pointnet_egg']+'/' in pointnet2_utils.__file__"
    assert original.BODY.count(guard)==1
    replacement="assert str(pathlib.Path(pointnet2_utils.__file__).resolve())=="+repr(wrapper['resolved'])+"\nassert hashlib.sha256(pathlib.Path(pointnet2_utils.__file__).read_bytes()).hexdigest()=="+repr(wrapper['sha256'])+"\nfrom pointnet2 import _ext as original_pointnet_ext\nassert str(pathlib.Path(original_pointnet_ext.__file__).resolve())=="+repr(binary['path'])+"\nassert hashlib.sha256(pathlib.Path(original_pointnet_ext.__file__).read_bytes()).hexdigest()=="+repr(binary['sha256'])
    body=original.BODY.replace(guard,replacement);assert body.replace(replacement,guard)==original.BODY
    source=inspect.getsource(original.prepare);transformer=Labels();tree=transformer.visit(ast.parse(source));assert transformer.counts=={'pipeline/checks/y101_review_import_':1,'pipeline/checks/y101_review_runtime_':1}
    namespace=dict(vars(original));namespace.update(__file__=__file__,BODY=body);exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),namespace);namespace['prepare'](token)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--token',required=True);run(parser.parse_args().token)
