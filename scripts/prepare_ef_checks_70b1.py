"""Freeze exact independent bytes and update existing IDs for the new workflow."""

import ast
import base64
from datetime import datetime
import hashlib
from experiment_first_70b1 import ROOT,TOKEN,PACKAGES,FILES,lease,ref,save,bind,tick,monitor

lease()
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    rows=[]
    parent=ROOT/'AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition'
    for number,name in PACKAGES.items():
        directory=parent.parent/name
        files={}
        for filename in FILES:
            path=directory/filename
            ast.parse(path.read_text(encoding='utf-8'),feature_version=(3,8))
            files[filename]=dict(base64=base64.b64encode(path.read_bytes()).decode(),sha256=ref(path)['sha256'])
            if filename not in ('__init__.py','ipdn.py'):
                assert path.read_bytes()==(parent/filename).read_bytes()
        config=ROOT/('AdaRES/configs_song/model_paper5/'+name+'_res.yaml')
        matches=list((ROOT/'AdaRES/configs_song').rglob(config.name))
        assert matches==[config]
        rows.append(dict(id='AdaRES-Y'+str(number),package=name,python_files=files,
            config=config.relative_to(ROOT/'AdaRES').as_posix(),
            config_data=dict(base64=base64.b64encode(config.read_bytes()).decode(),sha256=ref(config)['sha256'])))
    train=ROOT/'AdaRES/tools/train.py'
    payload=dict(variants=rows,parent_config='configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml',
        train_source_lf_sha256=hashlib.sha256(train.read_bytes().replace(b'\r\n',b'\n')).hexdigest())
    body=ROOT/'scripts/probe_ef_cpu_70b1.py'
    program='PAYLOAD='+repr(payload)+'\n'+body.read_text(encoding='utf-8')
    ast.parse(program,feature_version=(3,8))
    program_path='pipeline/analyses/ef_cpu_'+TOKEN+'_program.py'
    assert not (ROOT/program_path).exists()
    (ROOT/program_path).write_text(program,encoding='utf-8',newline='\n')
    policy=tick.load(ROOT/'pipeline/closure_policy.json')['experiment_first']
    plan=save('pipeline/analyses/ef_implementation_'+TOKEN+'_plan.json',dict(complete=True,
        created_at=datetime.now().astimezone().isoformat(),ids=[r['id'] for r in rows],
        authorization=policy['authorization'],program=ref(program_path),body=ref(body),
        implementation=dict(Y58='Keep only sparse-backbone BN running statistics fixed; affine stays trainable; decoder BN and frozen-text baseline mode unchanged.',
            Y59='Keep the fully frozen text encoder in eval during outer train; leave a trainable encoder and all backbone/decoder normalization unchanged.'),
        parent='AdaRES-W2',protocol='Native YAML values identical to W2: seed1999/fixed30/releasedSPFormer/Q128/singleGPU, no validation tuning.',
        preserve='All original diagnosis plans/results retained. This approved prospective workflow no longer requires task-benefit diagnosis before implementation.',
        native_smoke='Pending actual two-step TRAIN loss/backward via native YAML/class/dataset/train function on a genuinely free authorized exclusive GPU; no admission before this.',
        no_training_or_queue_change=True))
    work=tick.load(ROOT/'pipeline/research_work.json')
    for row in rows:
        c=next(c for c in work['candidates'] if c['id']==row['id'])
        assert c['stage']=='blocked' and c['blocker_kind']=='exclusive_gpu_slot'
        c.setdefault('workflow_history',[]).append(dict(recorded_at=datetime.now().astimezone().isoformat(),
            previous_next_action=c['next_action'],new_profile='experiment_first',authorization=policy['authorization'],
            evidence=plan,reason='Prospective approved policy allows implementation and real native smoke without prior efficacy proof. Original negative evidence remains unchanged.'))
        c.update(implementation_package=row['package'],config=row['config'],comparison_baseline='AdaRES-W2',
            blocked_from_stage='implementation',profile='experiment_first',
            next_action='Execute exact-package CPU constructor/mode and real native-YAML TRAIN data checks now; '
                        'then perform the two-step native GPU smoke on the first actually free authorized slot. '
                        'Save tested bytes and admit/dispatch in the same tick only after real smoke passes.')
        bind(c,plan,ref(program_path),ref(body),ref(__file__),policy['authorization'])
        for filename in FILES:bind(c,ref(parent.parent/row['package']/filename))
        bind(c,ref(ROOT/'AdaRES'/row['config']))
    tick.save(ROOT/'pipeline/research_work.json',work)
    print('Existing Y58/Y59 exact implementations registered; old evidence preserved; CPU test prepared, native GPU smoke pending.')
finally:
    lock.close()
