"""Isolate control replay from the intervention hook before first execution."""

import ast
from datetime import datetime
from research_y56_dd8e import ROOT,TOKEN,lease,ref,save,bind,tick,monitor
lease(300)
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work=tick.load(ROOT/'pipeline/research_work.json')
    candidate=next(c for c in work['candidates'] if c['id']=='AdaRES-Y57')
    original=ROOT/('pipeline/analyses/y57_anchor_'+TOKEN+'_program.py')
    payload=ast.literal_eval(ast.parse(original.read_text(encoding='utf-8')).body[0].value)
    body=ROOT/'scripts/probe_y57_anchor_v2_dd8e.py'
    setup=ROOT/'scripts/setup_y57_prototype_5103.py'
    program='PAYLOAD='+repr(payload)+'\n'+setup.read_text(encoding='utf-8')+'\n'+body.read_text(encoding='utf-8')
    ast.parse(program,feature_version=(3,8))
    path='pipeline/analyses/y57_anchor_v2_'+TOKEN+'_program.py'
    assert not (ROOT/path).exists()
    (ROOT/path).write_text(program,encoding='utf-8',newline='\n')
    plan=save(path.replace('_program.py','_prereg.json'),dict(complete=True,id='AdaRES-Y57',revision=1,
        preregistered_at=datetime.now().astimezone().isoformat(),program=ref(path),body=ref(body),
        driver_only='Remove intervention hook before isolated shifted-feature replay; otherwise hook would undo the shift. '
                    'Corrected by static inspection before ANY diagnostic execution; v1unrun kept.',
        scientific_control_or_production_change=False,result='pending'))
    bind(candidate,ref(path),plan,ref(body),ref(__file__))
    tick.save(ROOT/'pipeline/research_work.json',work)
    print('Frozen isolated control replay before first execution; no original gate altered.')
finally:
    lock.close()
