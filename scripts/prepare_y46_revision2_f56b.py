"""Freeze the second and final bounded implementation repair before changing code."""
import argparse, hashlib, os, sys
from datetime import datetime, timezone
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
def ref(relative):return dict(path=relative,sha256=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest())
work_path=ROOT/'pipeline/research_work.json'
work=tick.load(work_path)
c=next(c for c in work['candidates'] if c['id']=='AdaRES-Y46')
assert len(c['revision_history'])==1 and c['revision_history'][0]['result']=='passed'
source='AdaRES/ipdn/model_paper5/adares_y46_component_coherence/dec.py'
assert ref(source)['sha256']=='7dd952f18632392f5e5a050b4aabcfd10de5bcfdad9c81f494de353dfa56cf8f'
assert not any(e['path']==source for e in c['evidence']),'Historical direct source evidence needs explicit archive binding'
failure='pipeline/analyses/y46_repeat_forensic_f56b_'+args.token+'.json'
raw=tick.load(ROOT/failure)
assert raw['native_repeat_gate']=='FAILED_UNDER_ORIGINAL_EXACT_RULE'
revision=dict(revision=2,failure_scope='implementation',reason='Unchanged production CUDA index_add pooling yields unequal cached outputs across all8 repeats on each of4 first-scene expressions; max errors2.98e-8 to5.96e-8. Original exact-repeat rule failed, regardless of small magnitude.',change='Only replace atomic index_add pooling with stable group ordering, the same per-row weights, and per-group sum reductions. No component, graph, mask, feature, head, label or optimizer change; cache revision1 remains.',prediction='Preserve component rows/weights/groups exactly and old/new CPU64 output within EXISTING8eps*nodes envelope; original exact cached native repeats must now pass on TRAIN8. All8 native gradients/updates, exact model/AdamW/PolyLR restore, independence and resource tests must be rerun. No widened tolerance for cached repeats.',preregistered_at=datetime.now().astimezone().isoformat(),failure_evidence=[ref(failure),ref('pipeline/analyses/y46_production_controls_f56b_'+args.token+'.log')],result='pending',evidence=[])
path='pipeline/analyses/y46_revision2_prereg_f56b_'+args.token+'.json'
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    tick.save(ROOT/path,dict(complete=True,id=c['id'],revision=revision,original_production_source=ref(source),source_archive='pipeline/analyses/y46_revision1_dec_f56b_'+args.token+'.py',forensic_clarification='Raw inherited cached_controls.repeat_exact=true is NOT a checked result in the forensic-only adapter. Actual cached_repeat_forensics.exact_repeats are all false; gate FAILED. Raw artifact retained unmodified.'))
    c['revision_history'].append(revision)
    c['evidence'].extend([ref(path),ref(failure),ref('pipeline/analyses/y46_production_controls_f56b_'+args.token+'.log')])
    c['stage']='implementation'
    c['failure_scope']='implementation'
    c['scientific_status']='revision2_preregistered_exact_native_repeat_repair_pending'
    c['pending_gates']=['revision2_CPU_controls','production_native_matched_controls','released_init_tools_train_restore','serial_inference','native_stress','resources','independence','final_six_gate_review']
    c['next_action']='Apply only preregistered deterministic pooled sum, preserve revision1 bytes/failures, then test CPU64 equivalence and exact native TRAIN8 repeats before all affected production/update/restore/resource gates. No third revision or admission from old gates.'
    tick.save(work_path,work)
    print(revision['preregistered_at'])
finally:lock.close()
