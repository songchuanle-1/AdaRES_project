"""Preserve failed utility import; extract the exact get_iou from native loss."""
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
source='scripts/probe_y51_gauge_controls_fbd1.py'
text=(ROOT/source).read_text()
changes=[('from ipdn.utils import get_iou','from typing import Union'),
         ("exec(compile(ast.Module(body=[predict]+loss_nodes,type_ignores=[])","iou=next(copy.deepcopy(n) for n in trees['loss.py'].body if isinstance(n,ast.FunctionDef) and n.name=='get_iou')\nexec(compile(ast.Module(body=[predict,iou]+loss_nodes,type_ignores=[])")]
for old,new in changes:
    assert text.count(old)==1
    text=text.replace(old,new)
compile(text,'<y51-utility-import-v2>','exec')
destination='pipeline/analyses/y51_gauge_controls_v2_program_fbd1_'+TOKEN+'.py'
relative='pipeline/analyses/y51_gauge_v2_freeze_fbd1_'+TOKEN+'.json'
def ref(p):return dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest())
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    with (ROOT/destination).open('x',encoding='utf-8') as stream:stream.write(text)
    tick.save(ROOT/relative,dict(complete=True,id='AdaRES-Y51',preregistered_at=datetime.now().astimezone().isoformat(),
        original=ref(source),failure=ref('pipeline/analyses/y51_gauge_controls_fbd1_'+TOKEN+'.log'),program=ref(destination),
        correction='Native IPDN imports get_iou from its loss.py, not ipdn.utils. Extract that exact native get_iou AST and import typing.Union for its unchanged annotations. Original failure precedes all numeric tests; preserve it. No scientific criterion, model, fixture, dtype adaptation or tolerance change.'))
    work=tick.load(ROOT/'pipeline/research_work.json')
    candidate=next(c for c in work['candidates'] if c['id']=='AdaRES-Y51')
    candidate['evidence'].extend([ref(relative),ref(destination)])
    tick.save(ROOT/'pipeline/research_work.json',work)
    print(destination)
finally:lock.close()
