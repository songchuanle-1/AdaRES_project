"""Record real Y57 task evidence and its remaining exclusive GPU dependency."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import authorize_prepared_candidates as scoped
import foreground_tick as tick
import pipeline_monitor as monitor

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--token',required=True)
parser.add_argument('--observed-from',required=True)
parser.add_argument('--observed-until',required=True)
args=parser.parse_args()
owner=scoped.lease(args.token)
start,end=map(datetime.fromisoformat,(args.observed_from,args.observed_until))
now=datetime.now(timezone.utc)
assert start.tzinfo is not None and end.tzinfo is not None
assert datetime.fromisoformat(owner['started_at']) <= start <= end <= now
assert (now-end).total_seconds()<600
writer=monitor.acquire_lock(str(ROOT));assert writer is not None
try:
    base='pipeline/analyses/'
    gt_path=base+'y57_gt_binding_result_'+args.token+'.json'
    pair_path=base+'y57_pair_binding_'+args.token+'_result.json'
    native_plan_path=base+'y57_native_task_'+args.token+'_prereg.json'
    occupation_path=base+'adares_live_posteval_'+args.token+'.json'
    first_occupation_path=base+'adares_live_progress_'+args.token+'.json'
    gt,pair,native,occupation=(tick.load(ROOT/path) for path in (gt_path,pair_path,native_plan_path,occupation_path))
    assert gt['complete'] and gt['all_bindings_passed'] and gt['expressions']==32
    assert pair['complete'] and pair['all_GT_mass_membership_geometry_bindings_exact'] and pair['expressions']==32
    assert native['complete'] and native['id']=='AdaRES-Y57' and native['missing_or_unauthorized_GPU_entry_rejected']
    assert not native['actual_GPU_execution'] and not native['full_native_backward']
    assert scoped.ref(native['program']['path'])==native['program']
    assert occupation['complete'] and occupation['free_authorized_gpus']==[]
    assert all(occupation['native_steps_increased'].values())
    assert {row['index'] for row in occupation['second']['GPUs']}=={0,1}
    assert all(row['pids'] and row['memory_mib']>=1024 for row in occupation['second']['GPUs'])
    assert sorted(row['pid'] for row in occupation['second']['runs'])==[47332,69064]
    review_path=base+'y57_task_review_'+args.token+'.json'
    assert not (ROOT/review_path).exists()
    retry=(end+timedelta(hours=1)).isoformat()
    review=dict(complete=True,id='AdaRES-Y57',token=args.token,
        recorded_at=datetime.now().astimezone().isoformat(),
        GT_reconstruction=scoped.ref(gt_path),CPU_batch2_binding=scoped.ref(pair_path),
        frozen_next_native_capture=scoped.ref(native_plan_path),
        earlier_bounded_progress_window=scoped.ref(first_occupation_path),current_occupation=scoped.ref(occupation_path),
        occupation_observed_from=args.observed_from,occupation_observed_until=args.observed_until,
        actual_associations=gt['associations'],all32_GT_bindings_passed=True,
        single_vs_pair_labels_counts_geometry_exact=True,query_permutation_correlation_error=0,
        scientific_status='Inconclusive task-causal relevance; descriptive association partly persists under rotated-label control',
        interpretation='Positive cache correlation is not independent incremental information or a task-defect proof. No fitted predictor, feature selection, changed selector, native correction or admission.',
        provenance_limit='Current reconstruction is source/GT/count/geometry bound; original capture did not hash raw point-GT/checkpoint at capture time. Future fresh capture cannot retroactively certify old provenance.',
        pending='Fresh native batch2/all7layer observational capture only after an idle authorized exclusive GPU, then review task-relevance before any formal model.',
        non_GPU_alternatives_completed=['Current same32expression GT/geometry/count replay and exact point-IoU references',
            'Pre-registered all-query descriptor/error/quality correlation with joint permutation and fixed rotated-label control',
            'Current four native batch2 CPU collates matched exactly to single-scene labels/membership/counts/raw geometry',
            'Source-bound frozen native program, Python3.8 parse/compile and missing/unauthorized GPU entry rejection'],
        no_model_or_training_changes=True,GPU_tests=0,new_candidate_ids=0,implementation_revisions=0,
        admissions=0,training_launches=0,retry_after=retry)
    tick.save(ROOT/review_path,review)
    work=tick.load(ROOT/'pipeline/research_work.json')
    candidate=next(c for c in work['candidates'] if c['id']=='AdaRES-Y57')
    assert candidate['stage']=='design'
    candidate['stage']='blocked'
    candidate['failure_scope']='inconclusive'
    candidate['scientific_status']=review['scientific_status']
    candidate['blocker_kind']='exclusive_gpu_slot'
    candidate['required_server']='server-2'
    candidate['blocker']='Actual current CPU GT/quality controls, batch2 data binding and source-frozen native-program preparation are complete. Both authorized GPUs remain occupied by original Y54/Y52. The next source/checkpoint/GT-bound sparse native observational capture requires a genuinely idle exclusive GPU; no source/mount/human-authorization block is claimed.'
    candidate['retry_after']=retry
    candidate['unblock_condition']='A live authorized server-2 GPU0/1 is idle and its exclusive diagnostic lease is available; use a new owning token and recheck current source/checkpoint/GT bindings, then run the SAME frozen native-task program without interrupting Y52/Y54.'
    candidate['attempted_alternatives']=review['non_GPU_alternatives_completed']
    candidate['next_action']='Run frozen '+native['program']['path']+' through run_adares_authorized_probe.py with --seconds600 --gpu --gpu-id <fresh-idle-authorized-card> under a fresh owning token; analyze final layer6 as primary, all other layers exploratory. No formal model or admission from association alone.'
    candidate['gate_summary']=dict(diagnosis='Current32TRAIN GT/geometry/count binding and descriptive associations executed; causal/incremental relevance inconclusive',
        controls='Joint query permutation exact; rotated-label error association remains positive, cannot treat correlation as task-defect PASS; CPU single/pair bindings exact',
        native_backward='Full model not executed; native program prepared/entry-tested only',
        resources='20:29 fresh double sampling: both authorized GPUs occupied by original Y54/Y52 with increasing steps; GPU tests0',
        independence='No formal model/config implemented',constraints='No new IDs, queue, admissions, model revisions or training; fixed30/seed1999/Q128 preserved')
    for path in (review_path,occupation_path,first_occupation_path):
        reference=scoped.ref(path)
        if reference not in candidate['evidence']:candidate['evidence'].append(reference)
    tick.save(ROOT/'pipeline/research_work.json',work)
    print(json.dumps(dict(id=candidate['id'],stage=candidate['stage'],review=scoped.ref(review_path),
        retry_after=retry,scientific_status=candidate['scientific_status']),indent=2))
finally:writer.close()
