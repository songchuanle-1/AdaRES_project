"""Preserve failed CPU harness; only replace unavailable ast.unparse assertion."""
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
source=ROOT/'scripts/probe_y50_production_cpu_1cb1.py'
old="assert ast.unparse(constructor.body[-1]) == 'self._joint_attachment.enabled = True'"
new="assert ast.dump(constructor.body[-1],include_attributes=False) == ast.dump(ast.parse('self._joint_attachment.enabled = True').body[0],include_attributes=False)"
text=source.read_text()
assert text.count(old)==1
text=text.replace(old,new)
compile(text,'<y50-cpu-compat>','exec')
destination=ROOT/'scripts/probe_y50_production_cpu_1cb1_v2.py'
with destination.open('x',encoding='utf-8') as stream:stream.write(text)
def ref(path):return dict(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
path=ROOT/('pipeline/analyses/y50_cpu_compat_prereg_1cb1_'+TOKEN+'.json')
tick.save(path,dict(complete=True,id='AdaRES-Y50',preregistered_at=datetime.now().astimezone().isoformat(),
                    failure=ref(ROOT/('pipeline/analyses/y50_production_cpu_1cb1_'+TOKEN+'.log')),
                    original=ref(source),program=ref(destination),
                    change='Only ast.unparse statement assertion becomes exact ast.dump equality; remotePython3.8 lacks ast.unparse. Original preserved. No model or tolerance change.',
                    no_implementation_revision=True))
print(destination)
