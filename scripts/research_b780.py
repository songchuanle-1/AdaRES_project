"""Utilities and first durable diagnosis for this reserve-only tick."""

from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TOKEN = 'b7801e9894f742b09038e89d033a8945'
sys.path.insert(0, 'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import foreground_tick as tick
import pipeline_monitor as monitor


def lease(reserve=90):
    base = ROOT / 'pipeline/headless'
    owner = tick.load(base / '.foreground.lockdir/owner.json')
    assert owner['token'] == TOKEN and owner['thread_id'] == os.environ['CODEX_THREAD_ID']
    assert (datetime.fromisoformat(owner['deadline_at'])-datetime.now(timezone.utc)).total_seconds() > reserve
    for name, key in (('.tick_context.json','context_sha256'),('.tick_context_state.next.json','pending_sha256')):
        assert tick.digest(base/name) == owner[key]
    return owner


def ref(path):
    path = Path(path)
    path = path if path.is_absolute() else ROOT/path
    assert path.resolve().is_relative_to(ROOT.resolve()) and path.is_file()
    return dict(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def save(path, value):
    assert not (ROOT/path).exists(), 'Never overwrite archived evidence'
    tick.save(ROOT/path,value)
    return ref(path)


def bind(candidate,*rows):
    for row in rows:
        if row not in candidate['evidence']:
            candidate['evidence'].append(row)


def bootstrap():
    lease()
    lock=monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        work=tick.load(ROOT/'pipeline/research_work.json')
        assert not any(c['id']=='AdaRES-Y58' for c in work['candidates'])
        context=tick.load(ROOT/'pipeline/headless/.tick_model_view.json')
        assert context['research']['reserve_only'] and context['research']['reserve_budget']==2
        plan=save('pipeline/analyses/y58_prereg_'+TOKEN+'.json',dict(
            complete=True,id='AdaRES-Y58',preregistered_at=datetime.now().astimezone().isoformat(),
            hypothesis_id='cross_scene_normalization_mixture_confidence_drift',
            hypothesis='Pooling normalization moments over unrelated TRAIN scenes may alter the fixed focal '
                       'scene representation/confidence when only its batch mate changes. This is a '
                       'task-relevance diagnosis, not an assumption that BatchNorm is harmful.',
            first_test='Actual source-derived BatchNorm configuration/learned affine, float32/64 exact '
                       'focal feature rows with matched-count independent mates; pooled vs separate-scene '
                       'training moments. Repeat/within-scene row permutation/eval-stat negatives. '
                       'Actual trained W2 confidence head must remain batch-separable for fixed inputs.',
            required_native='Fixed native focal TRAIN item with two prespecified mates, dropout disabled '
                            'in every arm, BN train vs fixed-stat-only control, same frozen weights/RNG/GT '
                            'and restored buffers. Preserve upstream sampling and point correspondences; '
                            'confidence/IoU direction and repeat error must be reported, not inferred from CPU.',
            sampling='Scene cluster is independent unit; component rows are not task samples.',
            limits='No validation tuning, new model package, queue admission, GPU use or normalization '
                   'replacement until native mechanism evidence. Do not reopen an existing rejected route.',
            goal=ref('pipeline/goal.json'),source=[
                ref('AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition/backbone.py'),
                ref('AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition/ipdn.py'),
                ref('AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition/dec.py')],
            result='pending'))
        candidate=dict(id='AdaRES-Y58',stage='diagnosis',created_at=datetime.now().astimezone().isoformat(),
            hypothesis_id='cross_scene_normalization_mixture_confidence_drift',
            hypothesis='Cross-scene training normalization moments can introduce unrelated-mate dependence '
                       'into focal confidence; diagnose actual native task relevance before any correction.',
            matched_control='Fixed focal features/data/weights; only unrelated mate changes, train BN versus '
                            'fixed-stat and per-scene component moments. Exact repeat/permutation and '
                            'fixed-input native-indicator separability negatives, no validation selection.',
            next_action='Read the exact BN source/config and targeted related ledger entries; execute '
                        'frozen component/actual-trained-head CPU controls before preparing a native causal test.',
            evidence=[plan,ref(__file__)],failure_scope='inconclusive',
            gate_summary={k:'pending' for k in ('diagnosis','controls','native_backward','resources','independence','constraints')})
        work['candidates'].append(candidate)
        work.pop('design_cycle',None)
        work.pop('tick_handoff',None)
        tick.save(ROOT/'pipeline/research_work.json',work)
        print(json.dumps(dict(id=candidate['id'],stage='diagnosis',preregistration=plan)))
    finally:
        lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['bootstrap'])
    parser.add_argument('--token',required=True)
    args=parser.parse_args()
    assert args.token==TOKEN
    bootstrap()
