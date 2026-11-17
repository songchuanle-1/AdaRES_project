"""Bind exact numeric control program and positive-control definition before test."""
from datetime import datetime,timezone
import hashlib,os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import foreground_tick as tick
import pipeline_monitor as monitor
TOKEN='fbd16ee677ad4d3aa2bbc403acdc2a67'
owner=tick.load(ROOT/'pipeline/headless/.foreground.lockdir/owner.json')
assert owner['token']==TOKEN and owner['thread_id']==os.environ['CODEX_THREAD_ID']
assert datetime.now(timezone.utc)<datetime.fromisoformat(owner['deadline_at'])
def ref(p):return dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest())
program='scripts/probe_y51_gauge_controls_fbd1.py'
compile((ROOT/program).read_text(),program,'exec')
relative='pipeline/analyses/y51_controls_freeze_fbd1_'+TOKEN+'.json'
record=dict(complete=True,id='AdaRES-Y51',preregistered_at=datetime.now().astimezone().isoformat(),program=ref(program),
    source=ref('pipeline/analyses/y51_source_fbd1_'+TOKEN+'.json'),
    actual_functions='Unmodified native IPDN.predict_by_feat and permutation helper AST; res task checked from native configuration. Literal Criterion.loss_indi stays untouched and is bitwise tested in native float32 on the exact half-integer fixture. For float64 fixtures ONLY, the single hardcoded float32 one-hot-buffer dtype leaf is explicitly adapted to logits.dtype, with all0/1values/formula/labels/weights unchanged.',
    supplementary_controls='Exact half-integer fixture: q0=[0,1],q1=[1.5,1.5],others=[0,-4], same fixed alternating +/-1 common shifts. Raw foreground top must flip1to0 while native softmax top stays0. Run32frozen native TRAIN expressions using same trained W2 indicator checkpoint.',
    uncertainty='Report top gaps and original8eps*nodes*Q bound; an order change near a numerical tie is uncertainty, not causal gauge dependence. Never relax the bound or silently discard a case.',
    dtype_instrumentation_preregistered=True,no_model_or_decision_rule_change=True)
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    assert not (ROOT/relative).exists()
    tick.save(ROOT/relative,record)
    work=tick.load(ROOT/'pipeline/research_work.json')
    c=next(c for c in work['candidates'] if c['id']=='AdaRES-Y51')
    assert c['stage']=='diagnosis'
    c['evidence'].extend([ref(relative),ref(program),record['source'],ref('scripts/probe_y51_source_fbd1.py')])
    c['next_action']='Execute frozen exact native prediction/loss32fixture+positive+TRAIN8/32 gauge test, preserve every numeric/order result; no model changes or admission.'
    tick.save(ROOT/'pipeline/research_work.json',work)
    print(relative)
finally:lock.close()
