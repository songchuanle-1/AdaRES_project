"""Correct only evaluator provenance lookup; the frozen evaluator and tested child/bootstrap remain unchanged."""
import argparse,ast,inspect,json
import launch_y101_frozen_review_job as original
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
import closure

class ArtifactSource(ast.NodeTransformer):
    def __init__(self):self.count=0
    def visit_For(self,node):
        if isinstance(node.target,ast.Name) and node.target.id=='path' and isinstance(node.iter,ast.Tuple) and [getattr(n,'value',None) for n in node.iter.elts]==['tools/train.py','tools/evaluate_adares_frozen_full.py']:
            self.count+=1
            return ast.parse("for path in ('tools/train.py',):\n raw=git('show',Y101+':AdaRES/'+path)\n sources.append(dict(path=path,sha256=hashlib.sha256(raw.replace(b'\\r\\n',b'\\n')).hexdigest(),LF=True))\nevaluator=closure.artifact(ROOT,adapter['original_evaluator'])\nassert ref(evaluator)['sha256']=='7824e7807b44c6344b158570626d7ad90fb5fc44eb6852dbd475d34440b1dc13'\nsources.append(dict(path='tools/evaluate_adares_frozen_full.py',sha256=ref(evaluator)['sha256'],LF=False))").body
        return self.generic_visit(node)

def launch(token):
    lease(token,600);work=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in work['candidates'] if r['id']=='AdaRES-Y101');assert c['stage']=='validation' and not c.get('frozen_review_job') and not c.get('frozen_review_launch_intent')
    failure=ROOT/('pipeline/checks/y101_review_source_binding_failure_'+token+'.json');assert not failure.exists()
    adapter=tick.load(closure.artifact(ROOT,c['frozen_review_resource_adapter']));closure.artifact(ROOT,adapter['original_evaluator'])
    tick.save(failure,dict(complete=True,token=token,id='AdaRES-Y101',failed_command='python -B scripts/launch_y101_frozen_review_job.py launch --token '+token,observed_failure='git show Y101:AdaRES/tools/evaluate_adares_frozen_full.py returned128; frozen evaluator exists as an artifact outside the training commit',remote_job_created=False,launch_intent_created=False,checkpoint_loaded=False,GPU_used=False,correction='Bind original evaluator through its already frozen exact artifact SHA instead of a nonexistent Git path; preserve native train Git/source identity and all evaluation/resource guards.',evidence=[ref(original.__file__),adapter['original_evaluator'],c['frozen_review_resource_adapter'],c['frozen_review_bootstrap_test'],ref(__file__)]))
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        work=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in work['candidates'] if r['id']=='AdaRES-Y101');c['frozen_review_source_binding_failure']=ref(failure);c['evidence'].extend([ref(failure),ref(__file__)]);tick.save(ROOT/'pipeline/research_work.json',work)
    finally:writer.close()
    transformation=ArtifactSource();tree=transformation.visit(ast.parse(inspect.getsource(original.launch)));assert transformation.count==1
    namespace=dict(vars(original));namespace['__file__']=__file__;exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),namespace);namespace['launch'](token)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--token',required=True);launch(parser.parse_args().token)
