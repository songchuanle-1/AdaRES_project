import ast,hashlib,json,math,os,pathlib,re,subprocess,sys,time,traceback
IDS=['AdaRES-Y101','AdaRES-Y36','AdaRES-W2','AdaRES-B1']
def validate(p):
    assert p['id']=='AdaRES-Y101' and p['server']=='server-1' and p['gpu']==0 and p['power']==350.0
    assert p['code']=='/data1/songcl/Project/AdaRES_project/AdaRES' and p['old_pid']==2263581
    assert [r['id'] for r in p['methods']]==IDS and p['freeze']['evaluation']['methods']==IDS
    assert p['freeze']['evaluation']['repeats']==2 and p['freeze']['evaluation']['primary']=='FIRST'
    assert p['freeze']['evaluation']['scene_cluster_bootstrap']==10000 and p['freeze']['evaluation']['bootstrap_seed']==1999 and p['freeze']['evaluation']['TRAIN_size_cuts']==[1487,3325]
    assert p['freeze']['resource_condition']==dict(server='server-1',gpu=0,power_limit_watts=350.0,note=p['freeze']['resource_condition']['note'])
    t=p['terminal'];assert t['complete'] and t['id']=='AdaRES-Y101' and t['process_absent'] and t['CUDA_initialized'] is False
    assert t['completed_epochs']=={k:list(range(1,31)) for k in ('Train','Val')}
    assert len(t['checkpoints'])==2 and t['checkpoints'][0]['epoch']==30
    base=pathlib.PurePosixPath('/data1/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_y101_query_ffn_skip_res/20261112_200238')
    for index,row in enumerate(t['checkpoints']):
        assert pathlib.PurePosixPath(row['path'])==base/('last.pth' if index==0 else 'best.pth')
        assert row['model_finite'] and row['optimizer_finite'] and row['optimizer_state_count']>0
        assert row['scheduler_last_epoch']==row['epoch'] and 1<=row['epoch']<=30 and math.isfinite(row['best_miou'])
        assert re.fullmatch('[a-f0-9]{64}',row['sha256'])
    assert p['methods'][0]['checkpoint']==t['checkpoints'][1]['path'] and p['methods'][0]['checkpoint_sha256']==t['checkpoints'][1]['sha256']
    for row in p['methods']:
        assert row['config'].startswith('/data1/songcl/Project/AdaRES_project/') and row['checkpoint'].startswith('/data1/songcl/Project/AdaRES_project/AdaRES_checkpoints/')
        assert re.fullmatch('[a-f0-9]{64}',row['checkpoint_sha256']) and re.fullmatch('[a-f0-9]{64}',row['config_sha256'])
    return True

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for raw in iter(lambda:f.read(8*1024*1024),b''):h.update(raw)
    return h.hexdigest()

def stripped(source):
    tree=ast.parse(source);main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');removed=[];body=[]
    for node in main.body:
        assignment=isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='lease' and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='open' and isinstance(node.value.args[0],ast.Constant) and node.value.args[0].value=='/data1/songcl/Project/AdaRES_project/.foreground_gpu0_diagnostic.lock'
        flock=isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='fcntl' and node.value.func.attr=='flock' and len(node.value.args)==2 and isinstance(node.value.args[0],ast.Name) and node.value.args[0].id=='lease'
        if assignment or flock:removed.append(('assignment' if assignment else 'flock',node))
        else:body.append(node)
    assert [r[0] for r in removed]==['assignment','flock'];main.body=body
    return ast.fix_missing_locations(tree)

def main():
    import fcntl
    validate(P);assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
    os.environ['PYTHONDONTWRITEBYTECODE']='1';os.chdir(P['code']);sys.path[:0]=P['python_paths']
    folder=pathlib.Path(P['folder']);assert folder.is_dir() and str(folder).startswith('/data1/songcl/Project/AdaRES_project/pipeline/review_jobs/AdaRES-Y101/')
    lock=open('/data1/songcl/Project/AdaRES_project/.foreground_gpu0_diagnostic.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def state(value):
        value.update(id=P['id'],token=P['token'],worker_pid=os.getpid(),observed_at=time.time())
        temporary=folder/'state.next.json';temporary.write_text(json.dumps(value));os.replace(str(temporary),str(folder/'state.json'))
    def preserved():
        assert not pathlib.Path('/proc/2263581').exists()
        root=pathlib.Path(P['code']);assert (root/'.codex_source_commit').read_text().strip()==P['code_commit']
        for row in P['source_files']:
            raw=(root/row['path']).read_bytes();raw=raw.replace(b'\r\n',b'\n') if row['LF'] else raw
            assert hashlib.sha256(raw).hexdigest()==row['sha256'],row['path']
        for command in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
            try:argv=command.read_bytes().split(b'\0')
            except OSError:continue
            assert not (b'tools/train.py' in argv and P['candidate_config'].encode() in argv)
    def idle():
        power=float(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=power.limit','--format=csv,noheader,nounits']).decode().strip());assert power==350.0
        assert not subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip()
        used,util=[float(v.strip()) for v in subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(',')];assert used<1024 and util<10
    captures={};state(dict(status='guarding',completed_methods=[]))
    try:
        preserved();idle()
        assert hashlib.sha256(P['adapter_source'].encode()).hexdigest()==P['adapter_sha256']
        compile(stripped(P['adapter_source']),'resource_adapter_outer_owned','exec')
        bootstrap="import ast,sys;P0="+repr(dict(paths=P['python_paths'],source=P['adapter_source']))+";sys.path[:0]=P0['paths'];"+P['strip_bootstrap']
        for row in P['methods']:
            preserved();idle();assert sha(row['config'])==row['config_sha256'] and sha(row['checkpoint'])==row['checkpoint_sha256']
            output=folder/(row['id'].lower()+'.json');log=folder/(row['id'].lower()+'.log');assert not output.exists() and not log.exists()
            argv=[sys.executable,'-B','-S','-c',bootstrap,'--config',row['config'],'--checkpoint',row['checkpoint'],'--output',str(output),'--gpu-id','0']
            environment=os.environ.copy();environment['CUDA_VISIBLE_DEVICES']='0';environment['PYTHONPATH']=os.pathsep.join(P['python_paths'])
            with log.open('xb') as stream:
                child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,env=environment)
                state(dict(status='evaluating',method=row['id'],child_pid=child.pid,completed_methods=list(captures),log=str(log),output=str(output)))
                code=child.wait();assert code==0,'Preserve evaluation failure, no automatic retry'
            capture=json.loads(output.read_text());assert capture['complete'] and capture['completed_passes']==2 and capture['seed']==1999
            assert capture['config_sha256']==row['config_sha256'] and capture['checkpoint_sha256']==row['checkpoint_sha256']
            assert capture['active_queries']==[128] and capture['selected_queries']==1
            assert all(r['summary']['expressions']==9508 and r['summary']['scenes']==141 for r in capture['passes'])
            captures[row['id']]=capture;preserved();state(dict(status='method_complete',method=row['id'],completed_methods=list(captures)))
            time.sleep(2)
        ns={'__name__':'paired_analysis_functions_only'};assert hashlib.sha256(P['analysis_source'].encode()).hexdigest()==P['analysis_sha256'];exec(compile(P['analysis_source'],'original_paired_scene_analysis','exec'),ns)
        comparisons={cid:ns['comparison'](captures['AdaRES-Y101'],captures[cid],seed=1999,draws=10000) for cid in IDS[1:]}
        for result in comparisons.values():
            for group in result.values():
                if 'scene_clustered_ci95' in group:group['prespecified_CI_upper_below_zero']=group['scene_clustered_ci95'][1]<0
        result=dict(complete=True,id=P['id'],token=P['token'],methods=IDS,full_passes=8,primary='FIRST',summaries={k:v['passes'][0]['summary'] for k,v in captures.items()},comparisons=comparisons,raw_files={cid:dict(path=str(folder/(cid.lower()+'.json')),sha256=sha(folder/(cid.lower()+'.json'))) for cid in IDS},original_resource=dict(server='server-1',gpu=0,power_limit_watts=350.0),training_restarted=False,automatic_promotion=False,formal_complete_cost_review=False,remaining='Reconcile all raw captures and full prespecified subgroup/paired outcomes, complete original trained-cost protocol and scientific review; no automatic target or promotion declaration.')
        (folder/'paired_result.json').write_text(json.dumps(result));state(dict(status='complete',completed_methods=IDS,result=str(folder/'paired_result.json')))
    except BaseException as error:
        state(dict(status='failed',completed_methods=list(captures),error=repr(error)));traceback.print_exc();raise
if __name__=='__main__':main()
