"""Close actual admitted launch while honestly preserving a later loader stall."""
import argparse,ast,json,subprocess,sys
from datetime import datetime
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
from finish_y92_binding_tick import command,SKILL
import closure

def finish(token):
    lease(token,300)
    command(['scripts/y102_observer_v2.py','observe','--id','AdaRES-Y101','--token',token,'--label','final_diagnostic'])
    attempted=['scripts/y102_observer_v2.py','observe','--id','AdaRES-Y102','--token',token,'--label','final_diagnostic']
    response=subprocess.run([sys.executable,'-B']+attempted,cwd=ROOT,capture_output=True,timeout=60)
    error=ROOT/('pipeline/checks/y102_final_progress_'+token+'_stderr.log');assert not error.exists();error.write_bytes(response.stdout+response.stderr)
    rawpath=ROOT/('pipeline/checks/production_adares-y102_'+token+'_final_diagnostic.log')
    raw=json.loads(rawpath.read_text(encoding='utf-8'))
    assert response.returncode!=0 and raw['complete'] is True and raw['process_identity_verified'] is True and raw['native_progress_confirmed'] is False
    assert raw['pid']==4747 and raw['source_marker']=='0c6c844719c8466ddd8888d054fe3b7ee658e643' and raw['before']['latest']==raw['after']['latest']
    assert raw['after']['latest']==dict(phase='Train',epoch=1,total=30,step=7,count=4686) and raw['no_bounded_tail_error'] is True
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        evidence=[ref(rawpath),ref(error),ref('pipeline/checks/production_adares-y102_'+token+'_release.log'),ref('pipeline/checks/production_adares-y102_'+token+'_after_diagnostic.log')]
        evidence += [ref('pipeline/checks/'+prefix+token+'.json') for prefix in ('y102_progress_diagnostic_','y102_worker_stack_','y102_loader_source_','y102_observer_mapping_failure_')]
        stack=tick.load(closure.artifact(ROOT,evidence[-3]));assert 'Operation not permitted' in stack['stack']
        diagnostic=ROOT/('pipeline/checks/y102_stalled_progress_'+token+'.json');assert not diagnostic.exists()
        value=dict(complete=True,token=token,id='AdaRES-Y102',observed_local=datetime.now().astimezone().isoformat(),pid=4747,observation=raw,current_native_progress_confirmed=False,status='loader_stalled_after_verified_initial_launch',observed_loader_pid=5241,initial_launch_verified=True,no_restart=True,no_source_data_seed_protocol_change=True,administrator_stack_help_required=True,evidence=evidence+[ref(__file__)])
        tick.save(diagnostic,value)
        work=tick.load(ROOT/'pipeline/research_work.json');candidate=next(r for r in work['candidates'] if r['id']=='AdaRES-Y102')
        assert candidate['stage']=='training';candidate['latest_training_diagnostic']=ref(diagnostic)
        candidate['next_action']='Initial launch genuinely advanced to Train1 step7; current counters stalled while loader5241 uses CPU. Preserve original PID4747/source/data/seed/fixed30/receipts and all logs; no restart or checkpoint claim. Read-only follow-up first. Deeper worker stack inspection needs administrator assistance because ptrace denied; no sudo/root grant inferred.'
        candidate['evidence'].extend([ref(diagnostic)]+evidence)
        tick.save(ROOT/'pipeline/research_work.json',work)
    finally:writer.close()
    for name in ('server-1','server-2'):command([SKILL+'/gpu_occupation.py','AdaRES_project','--token',token,'--server',name,'--label','release'])
    source=(ROOT/'scripts/finish_y102_launch_tick_v2.py').read_text(encoding='utf-8')
    original="""entry=c[cid]['latest_training_observation'];v=tick.load(closure.artifact(ROOT,entry));fresh(v,'observed_local')
            assert v['token']==token and v['native_progress_confirmed'] is True and v['pid']==pid and v['source_marker'] in (commit,Y102)
            observations[cid]=dict(evidence=entry,phase=v['after']['latest'],pid=pid)"""
    replacement="""if cid=='AdaRES-Y101':
                entry=c[cid]['latest_training_observation'];v=tick.load(closure.artifact(ROOT,entry));fresh(v,'observed_local')
                assert v['token']==token and v['native_progress_confirmed'] is True and v['pid']==pid and v['source_marker'] in (commit,Y102)
                observations[cid]=dict(evidence=entry,phase=v['after']['latest'],pid=pid,native_progress_confirmed=True)
            else:
                entry=c[cid]['latest_training_diagnostic'];v=tick.load(closure.artifact(ROOT,entry));fresh(v,'observed_local')
                assert v['token']==token and v['current_native_progress_confirmed'] is False and v['pid']==pid and v['observation']['process_identity_verified'] is True
                assert v['observation']['before']['latest']==v['observation']['after']['latest'] and v['status']=='loader_stalled_after_verified_initial_launch'
                observations[cid]=dict(evidence=entry,phase=v['observation']['after']['latest'],pid=pid,native_progress_confirmed=False,status=v['status'])"""
    assert source.count(original)==1;source=source.replace(original,replacement)
    source=source.replace('formal_FIRST_or_cost_executed=False,evidence=',"formal_FIRST_or_cost_executed=False,current_Y102_progress_confirmed=False,unresolved_loader_stall=c['AdaRES-Y102']['latest_training_diagnostic'],evidence=")
    source=source.replace("guards,failedobserver,ref(","guards,failedobserver,c['AdaRES-Y102']['latest_training_diagnostic'],ref(")
    source=source.replace("'; exact source/YAML/private runtime/power/increasing native counters '+evidence(observations['AdaRES-Y102']['evidence'])", "'; exact source/YAML/private runtime/power verified, but current native counters STALLED atTrain1step7; initial dispatch increase only, not current progress '+evidence(observations['AdaRES-Y102']['evidence'])")
    source=source.replace("'- No formal FIRST/cost/promotion. Trainingbest/native feasibility/increasing counters do not prove efficacy.","'- Y102 current progress STALLED at Train1step7. PID4747 alive, loader5241 CPU busy; main waits on futex, GPU0util0/memory16186MB, no bounded-tail error. Ordinary-account gdb denied ptrace and lacks py-bt; no stack obtained, no root/sudo or restart attempted. Need administrator help to obtain original loader stack before any disposition. Exact diagnostic '+evidence(c['AdaRES-Y102']['latest_training_diagnostic'])+'.',\n            '- No formal FIRST/cost/promotion. Trainingbest/native feasibility/increasing counters do not prove efficacy.")
    source=source.replace('精确源码/数据/私有环境/功耗和原生步数递增均已验证。','精确源码/数据/私有环境/功耗已验证。Y101原生步数继续递增；Y102启动时曾真实递增至第7步，当前停止推进，不能报告为持续运行正常。')
    source=source.replace("'Y100保留原产物、收据和完整原设备后评协议；", "'需处理：Y102数据加载子进程5241持续CPU忙，主进程4747等待，原生计数停在Train1step7；普通帐号调用栈读取被系统ptrace权限拒绝且无py-bt。需要管理员协助获取该原子进程调用栈；不推断sudo/root权限，不停止/重启训练、不改源码/数据协议。Y100保留原产物、收据和完整原设备后评协议；")
    source=source.replace('canonical dispatch/increasing native counters. Y101 continues unchanged.','canonical dispatch/initial native counter increase; later Y102 loader stall persisted with all original processes and evidence. Y101 continues unchanged.')
    source=source.replace('Goal/old receipts/data/runtime/mappings/seed1999fixed30/350W280W/bans/GPU1+Y57 pauses/foreign jobs/scheduler intact. Closeout','CurrentY102progressSTALLED atTrain1step7/administrator worker-stack help required. Goal/old receipts/data/runtime/mappings/seed1999fixed30/350W280W/bans/GPU1+Y57 pauses/foreign jobs/scheduler intact. Closeout')
    tree=ast.parse(source);tree.body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)) or isinstance(n,ast.Assign) or isinstance(n,ast.FunctionDef) and n.name in ('record','evidence')]
    body=ast.unparse(ast.fix_missing_locations(tree))+'\n';ast.parse(body,feature_version=(3,8))
    generated=ROOT/('pipeline/checks/y102_stalled_closeout_'+token+'_program.py');assert not generated.exists();generated.write_text(body,encoding='utf-8',newline='\n')
    namespace={'__file__':__file__,'__name__':'y102_stalled_closeout'};exec(compile(body,str(generated),'exec'),namespace)
    out,notes,lines=namespace['record'](token)
    command([SKILL+'/foreground_tick.py','AdaRES_project','report','--token',token,'--notes',str(notes)])
    command([SKILL+'/foreground_tick.py','AdaRES_project','finish','--token',token])
    receipt=tick.load(ROOT/('pipeline/tick_receipts/'+token+'.json'));assert receipt['status']=='FINALIZED' and receipt['postflight']=='PASS' and receipt['context_committed'] is True and receipt['lease_release']=='released'
    assert not (ROOT/'pipeline/headless/.foreground.lockdir').exists() and not (ROOT/'pipeline/headless/.tick_context_state.next.json').exists()
    pending=ROOT/'pipeline/archive_pending';assert not pending.exists() or not any(p.is_file() for p in pending.rglob('*'))
    print(json.dumps(dict(closeout=ref(out),status=receipt['status'],postflight=receipt['postflight'],context_committed=True,lease_release=receipt['lease_release'],memory_lines=lines,Y102_current_progress_confirmed=False,Y102_state='loader_stalled_after_verified_initial_launch',administrator_help_required=True,observations={cid:r['phase'] for cid,r in tick.load(out)['observations'].items()},native_GPU_tasks=1,admissions=1,launches=1,archives=1)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);finish(p.parse_args().token)
