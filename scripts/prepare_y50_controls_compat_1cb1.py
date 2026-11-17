"""Unpack the documented cached (query, descriptor) pair; preserve failed harness."""
from datetime import datetime,timezone
import hashlib
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import foreground_tick as tick
TOKEN='1cb15474bfd346dcbc34837f3e9b79d3'
owner=tick.load(ROOT/'pipeline/headless/.foreground.lockdir/owner.json')
assert owner['token']==TOKEN and owner['thread_id']==os.environ['CODEX_THREAD_ID']
assert datetime.now(timezone.utc)<datetime.fromisoformat(owner['deadline_at'])
source=ROOT/('pipeline/analyses/y50_production_controls_program_1cb1_'+TOKEN+'.py')
old='for query, values in zip(queries, attachment.cached):\n            correct = head(query, values)'
new='for query, (cached_query, values) in zip(queries, attachment.cached):\n            assert torch.equal(query, cached_query)\n            correct = head(query, values)'
text=source.read_text()
assert text.count(old)==1
text=text.replace(old,new)
compile(text,'<y50-cached-control-compat>','exec')
destination=ROOT/('pipeline/analyses/y50_production_controls_v2_program_1cb1_'+TOKEN+'.py')
with destination.open('x',encoding='utf-8') as stream:stream.write(text)
def ref(path):return dict(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
path=ROOT/('pipeline/analyses/y50_controls_compat_prereg_1cb1_'+TOKEN+'.json')
tick.save(path,dict(complete=True,id='AdaRES-Y50',preregistered_at=datetime.now().astimezone().isoformat(),
                    failure=ref(ROOT/('pipeline/analyses/y50_production_controls_1cb1_'+TOKEN+'.log')),
                    original=ref(source),program=ref(destination),
                    change='Only correctly unpack cached(query,descriptor) and require independently captured query exact equality. Native gradient checks preceded failure, but no complete native PASS is inferred. No production model, source fixture, sampling or tolerance change.',
                    no_implementation_revision=True))
print(destination)
