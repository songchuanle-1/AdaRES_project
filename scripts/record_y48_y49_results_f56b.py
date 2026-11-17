"""Persist frozen Y48 harm and the completed Y49 implementation tests."""
import argparse,hashlib,json,os,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import foreground_tick as tick
import pipeline_monitor as monitor
parser=argparse.ArgumentParser()
parser.add_argument('--token',required=True)
args=parser.parse_args()
owner=tick.load(ROOT/'pipeline/headless/.foreground.lockdir/owner.json')
assert owner['token']==args.token and owner['thread_id']==os.environ['CODEX_THREAD_ID']
assert datetime.now(timezone.utc)<datetime.fromisoformat(owner['deadline_at'])
def ref(path):return dict(path=path,sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
work_path=ROOT/'pipeline/research_work.json'
work=tick.load(work_path)
c48=next(c for c in work['candidates'] if c['id']=='AdaRES-Y48')
cluster='pipeline/analyses/y48_clustered_12be_'+args.token+'.json'
stats=tick.load(ROOT/cluster)
assert stats['frozen_harm_triggered_groups']['w2']==['small']
cost='pipeline/analyses/y48_trained_cost_12be_'+args.token+'.json'
assert tick.load(ROOT/cost)['complete']
c48['evidence'].extend([ref('pipeline/analyses/y48_frozen_12be_'+args.token+'.json'),ref(cluster),ref(cost),ref('pipeline/analyses/AdaRES-Y48_frozen_review_f56b.md')])
c48.update(stage='rejected',failure_scope='implementation',scientific_status='trained_realization_rejected_frozen_small_subgroup_harm',next_action='Do not promote, retune, restart or rename this realized OR-supervision method. Retain frozen full/repeat/scene/subgroup/trained cost evidence; broader mechanism not universally falsified.',rejection_reason='Frozen small subgroup Y48-W2 delta-.0233349321,sceneCI95[-.0392752156,-.0079237523],upper<0 on2816expressions/124scenes; original harm rule triggered. Overall intervals cross0.')
c49=next(c for c in work['candidates'] if c['id']=='AdaRES-Y49')
refs=[ref('pipeline/analyses/y49_revision1_cpu_f56b_'+args.token+'.json'),ref('pipeline/analyses/y49_revision1_native_f56b_'+args.token+'.json')]
cpu,native=(tick.load(ROOT/r['path']) for r in refs)
assert cpu['complete'] and native['complete']
assert cpu['maximum_descriptor_error']<=8*2.220446049250313e-16
assert native['native_scenes']==8 and native['minimum_native_gradient']>0
c49['revision_history'][0].update(result='passed',evidence=refs)
c49['evidence'].extend(refs)
c49.update(stage='validation',scientific_status='revision1_CPU_equivalence_and_native_backward_passed_not_production',next_action='Continue same revision1: resolve task-relevant equal-component/different-global information diagnosis, then independent production/released-init tools.train/restore, serial/native-stress resources and six-gate review. Original costs and zero native collision witnesses remain; no admission from prototype.')
c49['diagnostic_gates']['native_backward']='Revision1 actual TRAIN8 all4 gradients finite/nonzero; temporary W2-inherited prototype only.'
c49['diagnostic_gates']['resources']='CPU descriptors19.979->6.692s; warmed native candidate.369-2.912s vs parent.213-.301s. Descriptive, not resource admission.'
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
try:tick.save(work_path,work)
finally:lock.close()
print(json.dumps(dict(Y48=c48['scientific_status'],Y49=c49['scientific_status'])))
