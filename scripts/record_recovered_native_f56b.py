"""Bind interrupted-tick diagnostic evidence without rerunning or admitting it."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, 'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import foreground_tick as tick
import pipeline_monitor as monitor
parser = argparse.ArgumentParser()
parser.add_argument('--token', required=True)
args = parser.parse_args()
owner = tick.load(ROOT/'pipeline/headless/.foreground.lockdir/owner.json')
assert owner['token']==args.token and owner['thread_id']==os.environ['CODEX_THREAD_ID']
assert datetime.now(timezone.utc)<datetime.fromisoformat(owner['deadline_at'])
def evidence(relative):
    return dict(path=relative,sha256=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest())
paths = dict(y46='pipeline/analyses/y46_production_native_12be_12be943f16874034808ff67e741653ad.json',y49='pipeline/analyses/y49_native_12be_12be943f16874034808ff67e741653ad.json')
y46,y49=(tick.load(ROOT/paths[key]) for key in ('y46','y49'))
assert y46['complete'] and y46['actual_tools_train_steps']==2
assert y46['model_optimizer_scheduler_restore_exact'] and y46['production_hook_registration_after_restore']
assert y49['complete'] and y49['native_scenes']==8 and y49['native_backward_executed']
review=dict(complete=True,observed_at_local=datetime.now().astimezone().isoformat(),source_tick='12be943f16874034808ff67e741653ad',binding_tick=args.token,reexecuted_tests=0,evidence=[evidence(p) for p in paths.values()],y46=dict(actual_tools_train_steps=2,new_tensors=8,minimum_gradient=min(v for row in y46['step_records'] for v in row['gradients'].values()),all8_updated=all(v>0 for v in y46['new_parameter_updates'].values()),exact_model_optimizer_scheduler_restore=True,pending=['production matched TRAIN8 controls','serial inference','native stress728/6780','resources','six gate review']),y49=dict(native_scenes=8,new_tensors=4,minimum_gradient=y49['minimum_native_gradient'],cached_control_count=sum(len(row['cached_controls']) for row in y49['rows']),minimum_wrong_gap=min(c['wrong_gap'] for row in y49['rows'] for c in row['cached_controls']),warmed_parent_seconds=[r['parent']['seconds'] for r in y49['rows'][1:]],warmed_candidate_seconds=[r['candidate']['seconds'] for r in y49['rows'][1:]],native_same_component_partition_witnesses=0,pending=['native task diagnosis','independent production','released initialization/tools.train/restore','resources','six gate review']),limitations=['The source tests ran in the interrupted previous tick, not again in this tick.','All previous failures and cost risks retained.','No efficacy/novelty/necessity/resource admission or training launch established.'])
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None,'Pipeline writer busy'
try:
    relative='pipeline/analyses/recovered_native_f56b_'+args.token+'.json'
    destination=ROOT/relative
    assert not destination.exists()
    tick.save(destination,review)
    work_path=ROOT/'pipeline/research_work.json'
    work=tick.load(work_path)
    for identity,key in (('AdaRES-Y46','y46'),('AdaRES-Y49','y49')):
        candidate=next(c for c in work['candidates'] if c['id']==identity)
        assert candidate['stage']=='validation'
        for item in (evidence(paths[key]),evidence(relative)):
            if item not in candidate['evidence']:candidate['evidence'].append(item)
        candidate.setdefault('history',[]).append(dict(observed_at=review['observed_at_local'],action='recovered_previous_tick_native_results_no_rerun',source_tick=review['source_tick']))
        if key=='y46':
            candidate['scientific_status']='production_tools_train_restore_passed_resources_and_matched_controls_pending'
            candidate['diagnostic_gates']['native_backward']='Actual released-init production tools.train2steps; all8 finite nonzero gradients/updates. Not efficacy.'
            candidate['diagnostic_gates']['production_restore']='Exact model/AdamW/PolyLR numeric state and hook restore passed.'
            candidate['pending_gates']=['production_native_matched_controls','serial_inference','native_stress','resources','final_six_gate_review']
            candidate['next_action']='Execute frozen production TRAIN8 cached controls/serial costs and native stress728/6780 on idle authorized exclusive GPU; retain severe original/cache costs; no admission before actual resource and six-gate review.'
        else:
            candidate['scientific_status']='native_diagnostic_passed_production_diagnosis_and_resources_pending'
            candidate['diagnostic_gates']['native_backward']='Actual W2 TRAIN8 temporary head all4 finite nonzero native gradients. Not released-init production.'
            candidate['diagnostic_gates']['controls']='32 cached correct/wrong/query-only controls separated with exact repeats; no full-loader repeat claim.'
            candidate['diagnostic_gates']['resources']='Warmed parent .213-.304s versus candidate .396-11.854s; descriptive single readings, not lightweight Pareto PASS.'
            candidate['pending_gates']=['native_task_diagnosis','resources','independent_production','released_init_tools_train_restore','six_gate_review']
            candidate['next_action']='Resolve native task-relevant global-vs-component information diagnosis and severe actual runtime risk before production admission; same stable ID, preserve zero native equal-component-partition witnesses and fixed controls. Production/released-init/restore/resources still pending.'
    tick.save(work_path,work)
    print(json.dumps(review,indent=2))
finally:
    lock.close()
