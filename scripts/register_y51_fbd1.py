"""Register a falsifiable deployment-confidence audit before source review."""
from datetime import datetime,timezone
import hashlib
import os
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
relative='pipeline/analyses/y51_gauge_prereg_fbd1_'+TOKEN+'.json'
record=dict(complete=True,id='AdaRES-Y51',preregistered_at=datetime.now().astimezone().isoformat(),
    hypothesis='The deployed native confidence ordering might depend on a per-query common logit shift that is invisible to its native two-class confidence CE. Test the exact native inference/loss extraction before considering any change; this is a deployment-consistency diagnosis, not a proposed calibration/ranking variant.',
    matched_control='Same query masks, labels, logits and native extractor, with fixed common shift k_q=(-1)^q added to BOTH class logits per query. Compare actual native CE, confidence values, chosen single-argmax and selected mask. Mask contents, selector, source and checkpoint are never changed.',
    protocol=['Read only native W2 ipdn.py/loss.py/dec.py, hash exact source and extract the actual confidence-to-selection AST.',
              'Fixed32 seeds1999..2030/full128 queries/float64 fixtures, same original8eps*nodes*Q numerical envelope; exact integer margin fixtures separately verify order without floating uncertainty.',
              'If the actual native score is gauge-invariant and all matched selections stay unchanged, reject this hypothesized mismatch, not all confidence mechanisms.',
              'If a mismatch is observed, retain native evidence and investigate task relevance and goal constraints; no automatic selector change or admission.',
              'Preserve historical ranking/readout closures, Y50 unchanged human-authorization block, validation freezes and all GPU permissions.'],
    no_model_variant=True,no_validation_selection=True,no_GPU_required=True,no_admission=True)
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work=tick.load(ROOT/'pipeline/research_work.json')
    assert not any(c['id']=='AdaRES-Y51' for c in work['candidates'])
    assert not (ROOT/relative).exists()
    tick.save(ROOT/relative,record)
    evidence=dict(path=relative,sha256=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest())
    work['candidates'].append(dict(id='AdaRES-Y51',stage='diagnosis',hypothesis=record['hypothesis'],
        matched_control=record['matched_control'],next_action='Execute exact native confidence/loss source audit, then fixed32 common-mode shift controls; do not invent an alternative selector.',
        evidence=[evidence],failure_scope='inconclusive'))
    work.pop('design_cycle',None)
    tick.save(ROOT/'pipeline/research_work.json',work)
    print(relative)
finally:lock.close()
