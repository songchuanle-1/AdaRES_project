"""Close this resource-adapter preparation event using unchanged canonical validation."""
import argparse,json,os
from datetime import datetime
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
import closure
import finish_adares_zero_budget_observation_v3 as base

BASE_SHA='6296d158f24de33c2f403424c97cde885fa0ed1410c1d44bc097024e8be19c3b'

def finish(token):
    lease(token,300)
    assert os.environ['CODEX_THREAD_ID']=='01a07e32-78bf-7670-925b-370911ea48db'
    assert ref(base.__file__)['sha256']==BASE_SHA
    work=tick.load(ROOT/'pipeline/research_work.json');candidate=next(r for r in work['candidates'] if r['id']=='AdaRES-Y101')
    assert candidate['stage']=='training'
    adapter_ref=candidate['frozen_review_resource_adapter'];adapter=tick.load(closure.artifact(ROOT,adapter_ref))
    assert adapter['complete'] and adapter['token']==token and adapter['id']=='AdaRES-Y101' and adapter['tests_passed']==14
    assert adapter['scientific_AST_identical_after_reversing_only_resource_edits'] and adapter['original_idle_thresholds_unchanged']
    assert not any(adapter[k] for k in ('checkpoint_loaded','CUDA_used','real_GPU_queries','actual_resource_lock_acquired','adapter_deployed','evaluation_executed','evaluation_ready'))
    for entry in adapter['evidence']:closure.artifact(ROOT,entry)
    tests=tick.load(closure.artifact(ROOT,adapter['tests']));assert tests['passed'] and tests['tests_run']==14 and tests['errors']==tests['failures']==0
    out=ROOT/('pipeline/checks/y101_review_adapter_closeout_'+token+'.json');assert not out.exists()
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        tick.save(out,dict(complete=True,token=token,id='AdaRES-Y101',observed_at=datetime.now().astimezone().isoformat(),resource_only_tests_passed=14,checkpoint_loaded=False,CUDA_used=False,adapter_deployed=False,formal_FIRST_or_cost_executed=False,training_restarted=False,new_scientific_candidates=0,admissions=0,launches=0,archives=0,remaining=adapter['remaining'],report_composition='Unmodified v3 observer/occupation/validator/report/finish with this genuine operational preparation added to authored notes and seven-section memory under the owned lease before canonical report.',evidence=[ref(__file__),ref(base.__file__),adapter_ref,adapter['tests'],candidate['frozen_review_reference_stage'],candidate['post_training_freeze']]))
        work=tick.load(ROOT/'pipeline/research_work.json');candidate=next(r for r in work['candidates'] if r['id']=='AdaRES-Y101');assert candidate['stage']=='training' and candidate['frozen_review_resource_adapter']==adapter_ref
        candidate['frozen_review_adapter_closeout']=ref(out);candidate['evidence'].extend([ref(out),ref(__file__)]);tick.save(ROOT/'pipeline/research_work.json',work)
    finally:writer.close()
    original_command=base.command;authored={'done':False}
    def command(args):
        if args[:3]==[base.SKILL+'/foreground_tick.py','AdaRES_project','report']:
            assert not authored['done'] and len(args)==7 and args[3:6]==['--token',token,'--notes'];lease(token,120)
            notes=ROOT/('pipeline/reports/notes_'+token+'.json');assert os.path.normcase(os.path.abspath(args[6]))==os.path.normcase(str(notes.resolve()))
            writer=monitor.acquire_lock(str(ROOT));assert writer is not None
            try:
                value=tick.load(notes);value['actions'].append('新准备：Y101原定server-1 GPU0/350W后评的独立资源适配副本通过14项标准库CPU边界检查，拒绝GPU1、错服务器/功耗、已有计算PID、忙显存/利用率、查询失败及锁竞争。完整评测AST除明确资源编辑外保持一致；测试只触及独立临时锁，未查询或占用GPU、未加载检查点、未部署或执行正式评测。')
                value['decisions'].append('沿用原完整后评协议、旧权重SHA与既有路径映射；原训练自然30轮及精确有限last/best/模型/优化器/调度器核验、原始归档、独占原server-1 GPU0/350W和私有运行环境/导入绑定齐备后才能后评。资源边界测试通过不等于运行环境、完整评测或方法效果通过。保持零创建/准入预算及原GPU1/Y57暂停。')
                value.setdefault('evidence',[]).extend([ref(out),adapter_ref,adapter['tests'],ref(__file__)]);tick.save(notes,value)
                memory=ROOT/'pipeline/codex_memory.md';lines=memory.read_text(encoding='utf-8').splitlines();assert any(s.startswith('- Current token '+token+';') for s in lines)
                target=[i for i,s in enumerate(lines) if s.startswith('- Y101 review still requires natural original fixed30/')];assert len(target)==1
                lines[target[0]]='- Y101 resource-only evaluator adapter14 CPU boundary checks PASS; scientific AST unchanged after reversing only resource edits; GPU0-only CLI/exactserver1/350W/shared diagnostic lock/no compute PID guards. '+base.evidence(adapter_ref)+'. Not deployed, no real GPU query/lock/checkpoint load/evaluation. Still require natural original fixed30/full finite checkpoint/canonical archive, fresh exclusive originalserver1GPU0/350W, private-runtime/code import launch binding and full paired/bootstrap/subgroup/cost. No formal FIRST/efficacy/goal attainment.'
                lines.append('- This tick newly prepared and CPU-tested Y101 resource-only review adapter14 guards; operational preparation, no scientific replay or new training. '+base.evidence(ref(out))+'.')
                assert len(lines)<120 and sum(s.startswith('## ') for s in lines)==7
                memory.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n');authored['done']=True
            finally:writer.close()
        return original_command(args)
    base.command=command
    try:base.finish(token)
    finally:base.command=original_command
    assert authored['done']
    print(json.dumps(dict(adapter_closeout=ref(out),resource_only_tests_passed=14,adapter_deployed=False,formal_evaluation_executed=False,memory_lines=len((ROOT/'pipeline/codex_memory.md').read_text(encoding='utf-8').splitlines()))))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--token',required=True);finish(parser.parse_args().token)
