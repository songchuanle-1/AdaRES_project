"""Freeze the narrowed auxiliary-target mechanism after the source control."""
from datetime import datetime
import json
from research_tick_4567 import ROOT, TOKEN, lease, ref, tick, monitor

lease()
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y52')
    assert candidate['stage'] == 'diagnosis'
    first = 'pipeline/analyses/y52_mass_target_4567_' + TOKEN + '.json'
    result = tick.load(ROOT / first)
    assert result['complete'] and result['auxiliary_quality_mass_mismatch_confirmed']
    path = 'pipeline/analyses/y52_train_controls_prereg_4567_' + TOKEN + '.json'
    assert not (ROOT / path).exists()
    plan = dict(complete=True, id=candidate['id'], token=TOKEN,
        preregistered_at=datetime.now().astimezone().isoformat(), previous_control=ref(first),
        clarification='Direct indicator quality-regression premise is false: it is matched CE. Preserve that negative result. The remaining narrower hypothesis concerns ONLY the existing auxiliary score MSE and its gradient through shared queries.',
        causal_hypothesis='Equal-superpoint auxiliary targets can disagree with actual point-level mask IoU and select different existing >0.5 supervision supports. Correcting point intersection/union counts may supply metric-consistent shared-query supervision; trained benefit and influence on final selection remain unknown.',
        intervention='Keep every mask, indicator, loss weight, native >0.5 filter and schedule unchanged. Replace ONLY the auxiliary quality target by detached exact point IoU using per-superpoint total and positive-point counts, including mixed-label superpoints. No feature subset or numerical scan.',
        CPU_test='Replay existing frozen eight TRAIN scenes/32 expressions with seed1999 at each dataset access; compare ann IDs, object IDs, coordinates, masses and GT component masses to capture before using masks. Evaluate ALL128 frozen masks per expression, not selected good cases. This does NOT provide native matched-query frequencies.',
        task_diagnosis='Compare exact point-IoU, native hard-superpoint-IoU and mass-weighted hard-superpoint-IoU. Record continuous differences and existing-half-filter disagreements; no fixed effect-size gate and no efficacy claims.',
        controls='Implement the O(Q*M) count kernel. For uniform, highly nonuniform, empty, mixed-label and fixed32 random fixtures, compare exactly to native get_iou on expanded points; compare loss and score gradients against native MSE at identical targets. Confirm finite nonzero gradients of all four unchanged native score-head tensors on an explicit component fixture, not full native backward.',
        repeatability='Replay each CPU dataset input twice at the same seed; require point labels/membership/annotation identity and packed masses/coordinates to agree before pairing. Mismatch is inconclusive; never silently substitute another augmentation.',
        next_GPU_test='After CPU preparation, a fresh idle authorized exclusive GPU is required to run actual native TRAIN matching, corrected-loss paired forward/backward and shared-query gradient audit plus resource/restore tests. No GPU currently reserved and no admission granted.',
        limits=['The auxiliary score is not directly consumed by RES selection.', 'All-query frozen-mask frequency is not matched-query supervision frequency.', 'No post-training accuracy, novelty, resource PASS or native-backward PASS from these CPU controls.', 'Y50 commit authorization is not authorization to commit this new candidate.'])
    tick.save(ROOT / path, plan)
    candidate['evidence'].extend([ref('scripts/probe_y52_mass_target_4567.py'), ref(first), ref(path)])
    candidate.update(stage='diagnosis', next_action=plan['CPU_test'],
        mechanism_clarification=plan['clarification'],
        remaining_hypothesis=plan['causal_hypothesis'])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(id=candidate['id'], stage=candidate['stage'], plan=ref(path)), indent=2))
finally:
    lock.close()
