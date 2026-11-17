"""Save only the fourteen CPU-tested variant files; do not push or admit."""

import ast
from datetime import datetime
import hashlib
import json
import subprocess
from experiment_first_70b1 import ROOT,TOKEN,lease,ref,save,bind,tick,monitor

lease()
lock=monitor.acquire_lock(str(ROOT))
assert lock is not None
def git(*args,binary=False):
    result=subprocess.check_output(['git','-c','core.autocrlf=false',*args],cwd=ROOT)
    return result if binary else result.decode('utf-8').strip()
try:
    program=ROOT/('pipeline/analyses/ef_cpu_'+TOKEN+'_program.py')
    packet=ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
    test_path='pipeline/analyses/ef_cpu_'+TOKEN+'.json'
    result=tick.load(ROOT/test_path)
    assert result['complete'] and result['implementation_test_passed'] and not result['CUDA_initialized']
    assert result['native_smoke_passed'] is False
    paths=[]
    for row in packet['variants']:
        assert any(r['id']==row['id'] and r['full_model_CPU_constructed'] and
                   r['configured_TRAIN_data_collated'] for r in result['rows'])
        for filename,value in row['python_files'].items():
            path='AdaRES/ipdn/model_paper5/'+row['package']+'/'+filename
            assert ref(path)['sha256']==value['sha256']
            paths.append(path)
        path='AdaRES/'+row['config']
        assert ref(path)['sha256']==row['config_data']['sha256']
        paths.append(path)
    assert len(paths)==14 and len(set(paths))==14
    assert not git('diff','--cached','--name-only'),'Preserve unrelated staged work; do not mix commits'
    parent=git('rev-parse','HEAD')
    assert parent=='05dc2c0424e33f460f96a4ab4938b0d05e7dc6ea'
    git('add','--',*paths)
    assert set(git('diff','--cached','--name-only').splitlines())==set(paths)
    for path in paths:
        assert hashlib.sha256(git('show',':'+path,binary=True)).hexdigest()==ref(path)['sha256']
    message=git('commit','--only','-m','Add independent Y58 and Y59 mode-contract experiments','--',*paths)
    commit=git('rev-parse','HEAD')
    assert set(git('diff-tree','--no-commit-id','--name-only','-r',commit).splitlines())==set(paths)
    for path in paths:
        assert hashlib.sha256(git('show',commit+':'+path,binary=True)).hexdigest()==ref(path)['sha256']
    binding=save('pipeline/analyses/ef_version_'+TOKEN+'.json',dict(complete=True,
        saved_at=datetime.now().astimezone().isoformat(),parent_commit=parent,code_commit=commit,
        exact_files=[ref(p) for p in paths],actual_test=ref(test_path),raw_log=ref(test_path.replace('.json','.log')),
        test_scope='CPU native constructor, released pretrain, YAML resolution, actual TRAIN collate and mode contracts.',
        native_two_step_smoke_passed=False,admission=False,pushed=False,synchronized=False,
        authorization=tick.load(ROOT/'pipeline/closure_policy.json')['experiment_first']['authorization']))
    work=tick.load(ROOT/'pipeline/research_work.json')
    for row in packet['variants']:
        c=next(c for c in work['candidates'] if c['id']==row['id'])
        c['code_commit']=commit
        bind(c,binding,ref(test_path),ref(test_path.replace('.json','.log')),ref(__file__),
             ref('scripts/run_ef_cpu_70b1.py'))
    tick.save(ROOT/'pipeline/research_work.json',work)
    print(json.dumps(dict(commit=commit,files=len(paths),binding=binding,git_output=message,
                          pushed=False,admitted=False,training_started=False),indent=2))
finally:
    lock.close()
