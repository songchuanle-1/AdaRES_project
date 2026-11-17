"""Freeze future two-step native checks without touching occupied GPUs."""

import ast
from experiment_first_70b1 import ROOT,TOKEN,lease,ref,save,bind,tick,monitor

lease()
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    original=ROOT/('pipeline/analyses/ef_cpu_'+TOKEN+'_program.py')
    payload=ast.literal_eval(ast.parse(original.read_text(encoding='utf-8')).body[0].value)
    version=tick.load(ROOT/('pipeline/analyses/ef_version_'+TOKEN+'.json'))
    body=ROOT/'scripts/probe_ef_native_smoke_70b1.py'
    work=tick.load(ROOT/'pipeline/research_work.json')
    for row in payload['variants']:
        frozen=dict(variant=row,train_source_lf_sha256=payload['train_source_lf_sha256'],
            code_commit=version['code_commit'],goal_sha256=ref('pipeline/goal.json')['sha256'])
        program='FROZEN='+repr(frozen)+'\n'+body.read_text(encoding='utf-8')
        ast.parse(program,feature_version=(3,8))
        code=compile(program,'<native-smoke>','exec')
        refused=False
        try:exec(code,dict(P=dict(physical_gpu=None)))
        except AssertionError:refused=True
        assert refused
        path='pipeline/analyses/'+row['id'].lower().replace('-','_')+'_smoke_'+TOKEN+'_program.py'
        assert not (ROOT/path).exists()
        (ROOT/path).write_text(program,encoding='utf-8',newline='\n')
        plan=save(path.replace('_program.py','_plan.json'),dict(complete=True,id=row['id'],
            profile='experiment_first',program=ref(path),code_commit=version['code_commit'],
            config=row['config'],goal_sha256=frozen['goal_sha256'],
            planned_command='python -B scripts/run_experiment_first_probe.py '+row['id'].lower().replace('-','_')+'_smoke '
                +path+' --token <NEW_CURRENT_TOKEN> --seconds 600 --gpu --gpu-id <IDLE_AUTHORIZED_GPU>',
            required_resource='One genuinely idle authorized server-2 GPU0/1, exclusive lease and no compute process.',
            checks='Native YAML resolver, released backbone, actual native TRAIN dataloader and tools/train.py::train for exactly two optimizer steps; finite loss/gradients, exercised mode path and GPU peak.',
            Python38_compile=True,missing_GPU_entry_refused=True,native_executed=False,
            note='No smoke PASS or admission yet. Use a new current lease and verify all frozen bytes and goal before execution.'))
        c=next(c for c in work['candidates'] if c['id']==row['id'])
        c['native_smoke_plan']=plan
        bind(c,plan,ref(path),ref(body),ref(__file__),ref('scripts/run_experiment_first_probe.py'))
    tick.save(ROOT/'pipeline/research_work.json',work)
    print('Two native YAML/train-entry two-step smoke programs frozen; missing-GPU guard passed; CUDA not executed.')
finally:
    lock.close()
