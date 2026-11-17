"""Freeze exact physical source after the preregistered connection diagnostic."""

import ast,base64
from datetime import datetime
from research_y56_dd8e import ROOT,TOKEN,lease,ref,save,bind,tick,monitor
lease(300)
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    work=tick.load(ROOT/'pipeline/research_work.json')
    candidate=next(c for c in work['candidates'] if c['id']=='AdaRES-Y57')
    diagnosis=ref('pipeline/analyses/y57_anchor_gpu1_'+TOKEN+'.json')
    result=tick.load(ROOT/diagnosis['path'])
    assert all(r['original_control_passed'] for r in result['rows'] if r['role']!='original')
    assert all(not r['original_control_passed'] for r in result['rows'] if r['role']=='original')
    parent=ROOT/'AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition'
    package=parent.with_name('adares_y57_peer_second_moment')
    config=ROOT/'AdaRES/configs_song/model_paper5/adares_y57_peer_second_moment_res.yaml'
    files=[ref(p) for p in sorted(package.glob('*.py'))]+[ref(config)]
    assert len(files)==7
    for p in package.glob('*.py'):
        ast.parse(p.read_text(encoding='utf-8'),feature_version=(3,8))
        if p.name!='dec.py':assert p.read_bytes()==(parent/p.name).read_bytes()
    prior=ROOT/('pipeline/analyses/y57_anchor_v2_'+TOKEN+'_program.py')
    payload=ast.literal_eval(ast.parse(prior.read_text(encoding='utf-8')).body[0].value)
    bundle=save('pipeline/analyses/y57_anchor_bundle_'+TOKEN+'.json',dict(complete=True,id='AdaRES-Y57',revision=1,
        files=files,previous_bundle=candidate['tested_bundle'],diagnostic=diagnosis,
        reason='Preregistered initial-moment connection, all769parent inputs/full256moments retained.',
        created_at=datetime.now().astimezone().isoformat(),formal_uncommitted=True))
    payload['bundle']=bundle
    payload['files']={(ROOT/r['path']).name:dict(sha256=r['sha256'],
        base64=base64.b64encode((ROOT/r['path']).read_bytes()).decode('ascii')) for r in files}
    setup=(ROOT/'scripts/setup_y57_prototype_5103.py').read_text(encoding='utf-8')
    setup=setup.replace('/ P["token"] / PAYLOAD["package"]','/ P["token"] / "anchor_revision1" / PAYLOAD["package"]')
    for name,body in (('cpu','scripts/probe_y57_cpu_5103.py'),('physical','scripts/probe_y57_anchor_physical_dd8e.py')):
        program='PAYLOAD='+repr(payload)+'\n'+setup+'\n'+(ROOT/body).read_text(encoding='utf-8')
        ast.parse(program,feature_version=(3,8))
        path='pipeline/analyses/y57_anchor_'+name+'_'+TOKEN+'_program.py'
        assert not (ROOT/path).exists()
        (ROOT/path).write_text(program,encoding='utf-8',newline='\n')
        plan=save(path.replace('_program.py','_prereg.json'),dict(complete=True,id='AdaRES-Y57',revision=1,
            preregistered_at=datetime.now().astimezone().isoformat(),program=ref(path),body=ref(body),bundle=bundle,
            protocol='Original CPU controls; actual4TRAIN batches/all7native anchor-gradient sites, same shifted-vector bound; '
                     '2warm3stress on largest input,8scene paired costs, unchanged seed30/config.',result='pending'))
        bind(candidate,ref(path),plan,ref(body))
    bind(candidate,bundle,diagnosis,ref('pipeline/analyses/y57_anchor_gpu1_'+TOKEN+'.log'),ref(__file__))
    candidate['tested_bundle']=bundle
    candidate['revision_history'][-1]['evidence'].extend([diagnosis,bundle])
    candidate['revision_history'][-1]['result_evidence']=[diagnosis]
    candidate.update(stage='validation',next_action='Execute physical anchored Y57 CPU/native/gradient/resource tests now; '
        'original failure retained, no local commit consent for this new seven-file bundle.')
    tick.save(ROOT/'pipeline/research_work.json',work)
    print('Frozen independent physical Y57 connection revision and remaining affected tests.')
finally:
    lock.close()
