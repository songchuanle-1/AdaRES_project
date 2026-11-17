"""Keep the remote diagnostic compatible with its Python3.8 runtime."""

import ast
from datetime import datetime
from research_b780 import ROOT,TOKEN,lease,ref,save,bind,tick,monitor
lease()
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work=tick.load(ROOT/'pipeline/research_work.json')
    c=next(c for c in work['candidates'] if c['id']=='AdaRES-Y58')
    old=ROOT/('pipeline/analyses/y58_cpu_'+TOKEN+'_program.py')
    payload=ast.literal_eval(ast.parse(old.read_text(encoding='utf-8')).body[0].value)
    body=ROOT/'scripts/probe_y58_cpu_v2_b780.py'
    program='PAYLOAD='+repr(payload)+'\n'+body.read_text(encoding='utf-8')
    ast.parse(program,feature_version=(3,8))
    path='pipeline/analyses/y58_cpu_v2_'+TOKEN+'_program.py'
    assert not (ROOT/path).exists()
    (ROOT/path).write_text(program,encoding='utf-8',newline='\n')
    plan=save(path.replace('_program.py','_prereg.json'),dict(complete=True,id='AdaRES-Y58',
        preregistered_at=datetime.now().astimezone().isoformat(),program=ref(path),body=ref(body),
        static_driver_correction='Replace Python3.9-only removeprefix with prefix-length slicing. '
                                 'Original version unrun; no outcomes, fixture or scientific gate changed.',result='pending'))
    bind(c,ref(path),plan,ref(body),ref(__file__))
    tick.save(ROOT/'pipeline/research_work.json',work)
    print('Frozen Python3.8-compatible diagnostic before first execution.')
finally:
    lock.close()
