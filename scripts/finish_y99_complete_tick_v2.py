"""Preserve the authored note-field comparison failure and reuse its existing exact diagnostic."""
import argparse,ast,inspect,json
import finish_y99_complete_tick as original
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
import closure

def finish(token):
    lease(token,300);failure=ROOT/('pipeline/checks/y99_closeout_note_field_failure_'+token+'.json');assert not failure.exists()
    diagnostic=ROOT/('pipeline/checks/y99_complete_'+token+'_y102_diagnostic.json');d=tick.load(diagnostic);assert d['complete'] and not d['current_native_progress_confirmed'];raw=d['observation'];assert raw['pid']==4747 and raw['process_identity_verified'] and raw['source_files_verified']==6 and raw['native_YAML_semantics_verified'] and raw['no_bounded_tail_error']
    tick.save(failure,dict(complete=True,token=token,observed_failure='Authored report audit compared complete resource_condition to only server/gpu/power dict; unchanged legacy freezes also contain scientific limitation note. Failed before closeout/memory/report/finish. Original observer diagnostic already completed and retained.',original_helper=ref(original.__file__),original_diagnostic=ref(diagnostic),correction='Require exact candidate/server-1/GPU0/350W identity fields, retain the additional original note unchanged. No old freeze/record/receipt/canonical checker/scientific protocol changes. Reuse exact existing diagnostic rather than repeat SSH.',evidence=[ref(original.__file__),ref(diagnostic),ref(__file__)]))
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        w=tick.load(ROOT/'pipeline/research_work.json');c=next(x for x in w['candidates'] if x['id']=='AdaRES-Y99');c['evidence'].extend([ref(failure),ref(__file__)]);tick.save(ROOT/'pipeline/research_work.json',w)
    finally:writer.close()
    source=inspect.getsource(original.finish)
    needle="f['resource_condition']==dict(server='server-1',gpu=0,power_limit_watts=350.)";assert source.count(needle)==1
    source=source.replace(needle,"f['resource_condition']['server']=='server-1' and f['resource_condition']['gpu']==0 and f['resource_condition']['power_limit_watts']==350.")
    start=source.index('    response=subprocess.run(');end=source.index('    occupations=',start)
    source=source[:start]+"    diagnostic=ROOT/('pipeline/checks/y99_complete_'+token+'_y102_diagnostic.json');existing=tick.load(closure.artifact(ROOT,ref(diagnostic)));raw=existing['observation'];confirmed=existing['current_native_progress_confirmed'];rawpath=closure.artifact(ROOT,existing['evidence'][0]);log=closure.artifact(ROOT,existing['evidence'][1])\n    assert existing['complete'] and raw['complete'] and raw['process_identity_verified'] and raw['pid']==4747 and raw['source_marker']==Y102 and raw['native_YAML_semantics_verified'] and raw['source_files_verified']==6 and raw['no_bounded_tail_error'] and not confirmed\n    assert raw['before']['latest']==raw['after']['latest']==dict(phase='Train',epoch=1,total=30,step=7,count=4686)\n"+source[end:]
    namespace=dict(vars(original));namespace['__file__']=__file__;command=original.command
    def audited_command(args):
        if args[:3]==[original.SKILL+'/foreground_tick.py','AdaRES_project','report']:
            notes=ROOT/('pipeline/reports/notes_'+token+'.json');v=tick.load(notes);v['actions'].append('收尾操作修复：原冻结resource_condition含note说明，初次本地整对象比较误报失败；原脚本/诊断/失败记录均保留，改为逐项核验候选/服务器/GPU0/350W，说明原样保留。复用已完成停滞诊断，没有重跑科学测试或更改官方校验器。');v['evidence'].extend([ref(failure),ref(__file__)]);tick.save(notes,v)
            memory=ROOT/'pipeline/codex_memory.md';lines=memory.read_text(encoding='utf-8').splitlines();lines.append('- Authored closeout note-field equality failure preserved '+ref(failure)['path']+' SHA'+ref(failure)['sha256']+'. Only exact server/GPU0/350W field checks corrected; original extra note, freezes/receipts/science/canonical validators unchanged. Reused actual earlier diagnostic, no extra SSH probe.');assert len(lines)<120;memory.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
        return command(args)
    namespace['command']=audited_command;exec(compile(ast.parse(source),__file__,'exec'),namespace);namespace['finish'](token)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--token',required=True);finish(parser.parse_args().token)
