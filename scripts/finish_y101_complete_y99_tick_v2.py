"""Resolve only legacy omitted GPU fields from immutable original freezes; keep canonical validators untouched."""
import argparse,ast,copy,inspect
import finish_y101_complete_y99_tick as original
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
import closure

LABELS={'release':'retry','_release':'_retry','_release_':'_retry_','_y102_command.log':'_y102_command_retry.log','_y102_diagnostic.json':'_y102_diagnostic_retry.json'}
class Labels(ast.NodeTransformer):
    def __init__(self,mapping):self.mapping=mapping;self.count=0
    def visit_Constant(self,node):
        if isinstance(node.value,str) and node.value in self.mapping:node.value=self.mapping[node.value];self.count+=1
        return node

def required_gpu(candidate):
    if 'required_gpu' in candidate:return candidate['required_gpu']
    assert candidate['id'] in ('AdaRES-Y62','AdaRES-Y63','AdaRES-Y64') and candidate['stage']=='blocked' and candidate['blocker_kind']=='exclusive_gpu_slot' and candidate['resource_work_kind']=='frozen_evaluation' and candidate['required_server']=='server-1'
    freeze=tick.load(closure.artifact(ROOT,candidate['post_training_freeze']));assert freeze['complete'] and freeze['id']==candidate['id']
    condition=freeze['resource_condition'];assert condition['server']=='server-1' and condition['gpu']==0 and condition['power_limit_watts']==350.
    server=monitor.load_servers(str(ROOT))['server-1'];assert server['allowed_gpus']==[0] and server['max_tasks_per_gpu']==1
    return condition['gpu']

def finish(token):
    lease(token,300)
    assert not (ROOT/('pipeline/checks/y101_complete_y99_closeout_'+token+'.json')).exists() and not (ROOT/('pipeline/snapshots/codex_memory_before_'+token+'.md')).exists()
    failure=ROOT/('pipeline/checks/y101_complete_y99_legacy_gpu_failure_'+token+'.json');assert not failure.exists()
    w=tick.load(ROOT/'pipeline/research_work.json');c={r['id']:r for r in w['candidates']};legacy=[c[i] for i in ('AdaRES-Y62','AdaRES-Y63','AdaRES-Y64')];assert all('required_gpu' not in r and required_gpu(r)==0 for r in legacy)
    tick.save(failure,dict(complete=True,token=token,observed_failure="KeyError: required_gpu in the authored due-record report audit after fresh physical occupation; official report/finish and memory write not reached",original_helper=ref(original.__file__),legacy_ids=[r['id'] for r in legacy],strict_resolution='Read GPU0/350W from each unchanged original hashed freeze, require same candidate/server identity and current whitelist [0]. No record, receipt, resource authorization, scientific gate or canonical validator edits.',freezes=[r['post_training_freeze'] for r in legacy],original_observations=[ref('pipeline/checks/gpu_occupation_'+name+'_release_'+token+'.json') for name in ('server-1','server-2')],new_observations_use_new_immutable_labels=True,evidence=[ref(__file__),ref(original.__file__)]+[r['post_training_freeze'] for r in legacy]))
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        w=tick.load(ROOT/'pipeline/research_work.json');x=next(r for r in w['candidates'] if r['id']=='AdaRES-Y101');x['evidence'].extend([ref(failure),ref(__file__)]);tick.save(ROOT/'pipeline/research_work.json',w)
    finally:writer.close()
    namespace=dict(vars(original));namespace.update(__file__=__file__,required_gpu=required_gpu)
    due=inspect.getsource(original.due_records);needle="x['required_gpu']==0";assert due.count(needle)==2;changed_due=due.replace(needle,'required_gpu(x)==0');assert changed_due.replace('required_gpu(x)==0',needle)==due;exec(compile(ast.parse(changed_due),__file__,'exec'),namespace)
    source=inspect.getsource(original.finish);before=ast.parse(source);transform=Labels(LABELS);tree=transform.visit(copy.deepcopy(before));assert transform.count>=5
    restored=Labels({v:k for k,v in LABELS.items()}).visit(copy.deepcopy(tree));assert ast.dump(restored,include_attributes=False)==ast.dump(before,include_attributes=False)
    original_command=namespace['command'];authored={'done':False}
    def command(args):
        if args[:3]==[namespace['SKILL']+'/foreground_tick.py','AdaRES_project','report']:
            assert not authored['done'];lease(token,120);notes=ROOT/('pipeline/reports/notes_'+token+'.json');value=tick.load(notes);value['actions'].append('收尾兼容：Y62/Y63/Y64旧资源挂起记录没有required_gpu字段，初次审计因此KeyError。原失败和原脚本保留；仅从各自原签名冻结资源中读取GPU0/350W，并再次要求当前白名单[0]和相同候选/服务器身份。未改旧记录、收据、资源权限或官方检查器，刷新使用新不可变标签。');value.setdefault('evidence',[]).extend([ref(failure),ref(__file__)]);tick.save(notes,value)
            memory=ROOT/'pipeline/codex_memory.md';lines=memory.read_text(encoding='utf-8').splitlines();lines.append('- Closeout legacy missing-GPU metadata compatibility only: original Y62/Y63/Y64 freezes explicitly bind server1GPU0/350W and current whitelist[0]; original records/receipts/validators unchanged. Failed first audit retained '+ref(failure)['path']+' SHA'+ref(failure)['sha256']+'.');assert len(lines)<120 and sum(s.startswith('## ') for s in lines)==7;memory.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n');authored['done']=True
        return original_command(args)
    namespace['command']=command;exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),namespace);namespace['finish'](token);assert authored['done']

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--token',required=True);finish(parser.parse_args().token)
