"""Bind existing guarded candidate-only deployment to Y50, no policy change."""
import ast
from datetime import datetime,timezone
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
text=(ROOT/'scripts/deploy_y49_production_f56b.py').read_text()
for old,new in [('AdaRES-Y49','AdaRES-Y50'),('adares_y49_resolvent_coherence','adares_y50_joint_incidence'),
                ('/tmp/adares_y49_nativeparent_bundle_f56b.json','/tmp/adares_y50_bundle_1cb1.json'),
                ('/tmp/adares_y49_nativeparent_bundle_f56b.tar','/tmp/adares_y50_bundle_1cb1.tar')]:
    assert old in text
    text=text.replace(old,new)
compile(text,'<guarded-y50-deploy>','exec')
path=ROOT/('pipeline/analyses/y50_deploy_program_1cb1_'+TOKEN+'.py')
with path.open('x',encoding='utf-8') as stream:stream.write(text)
print(path)
