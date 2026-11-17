"""Record the genuine CPU import binding and retain its original path-assumption failure."""
import argparse,json,os
from datetime import datetime
from prepare_y60_stage_heads import ROOT,lease,monitor,ref,tick
import closure
import finish_adares_zero_budget_observation_v3 as base

def finish(token):
    lease(token,300);assert ref(base.__file__)['sha256']=='6296d158f24de33c2f403424c97cde885fa0ed1410c1d44bc097024e8be19c3b'
    assert os.environ['CODEX_THREAD_ID']=='01a07e32-78bf-7670-925b-370911ea48db'
    work=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in work['candidates'] if r['id']=='AdaRES-Y101');assert c['stage']=='training'
    binding_ref=c['frozen_review_runtime_binding'];binding=tick.load(closure.artifact(ROOT,binding_ref))
    assert binding['complete'] and binding['token']==token and binding['id']=='AdaRES-Y101' and binding['isolated_CPU_imports_passed']==4
    assert binding['private_python_and_site_packages_verified'] and binding['private_pointnet_egg_and_sparseconv_imported'] and binding['all_four_native_YAML_semantics_verified']
    assert not any(binding[k] for k in ('model_created','dataset_created','checkpoint_loaded','CUDA_initialized','real_GPU_queries','adapter_deployed','evaluation_executed','evaluation_ready'))
    for entry in binding['evidence']:closure.artifact(ROOT,entry)
    failure=c['frozen_review_runtime_initial_failure'];origin_ref=c['frozen_review_pointnet_origin'];closure.artifact(ROOT,failure);origin=tick.load(closure.artifact(ROOT,origin_ref))
    assert origin['complete'] and not origin['CUDA_used'] and len(origin['original_live_PID_pointnet_binaries'])==1
    out=ROOT/('pipeline/checks/y101_review_runtime_closeout_'+token+'.json');assert not out.exists()
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        tick.save(out,dict(complete=True,token=token,id='AdaRES-Y101',recorded_at=datetime.now().astimezone().isoformat(),isolated_CPU_import_bindings=4,original_pointnet_path_assumption_failed_and_preserved=True,correction='Project PointNet wrapper exact SHA plus original live PID private egg binary exact path/SHA; no environment or scientific-control change.',model_created=False,dataset_created=False,checkpoint_loaded=False,CUDA_initialized=False,evaluation_executed=False,adapter_deployed=False,admissions=0,launches=0,archives=0,remaining=binding['remaining'],report_composition='Unchanged v3 observation, physical occupation and canonical validation/report/finish; add actual operational preparation and failure/correction to authored notes and seven-section memory before official report under owned lease.',evidence=[ref(__file__),ref(base.__file__),binding_ref,failure,origin_ref,c['frozen_review_resource_adapter'],c['frozen_review_reference_stage']]))
        work=tick.load(ROOT/'pipeline/research_work.json');c=next(r for r in work['candidates'] if r['id']=='AdaRES-Y101');assert c['stage']=='training' and c['frozen_review_runtime_binding']==binding_ref
        c['frozen_review_runtime_closeout']=ref(out);c['evidence'].extend([ref(out),ref(__file__)]);tick.save(ROOT/'pipeline/research_work.json',work)
    finally:writer.close()
    original_command=base.command;authored={'done':False}
    def command(args):
        if args[:3]==[base.SKILL+'/foreground_tick.py','AdaRES_project','report']:
            assert not authored['done'] and len(args)==7 and args[3:6]==['--token',token,'--notes'];lease(token,120)
            notes=ROOT/('pipeline/reports/notes_'+token+'.json');assert os.path.normcase(os.path.abspath(args[6]))==os.path.normcase(str(notes.resolve()))
            writer=monitor.acquire_lock(str(ROOT));assert writer is not None
            try:
                value=tick.load(notes)
                value['actions'].extend(['新准备：Y101/Y36/W2/B1四个原冻结后评配置分别在server-1原私有Python中通过CPU导入、原生train模型解析、私有依赖与原YAML语义核验；资源适配副本仅导入定义，未调用main。全程CUDA屏蔽，权重加载与CUDA初始化禁用，未创建模型/数据集、未执行科学测试或正式后评。','初次导入预检的PointNet包装文件必须来自egg断言失败已保留；只读原PID映射确认实际为项目包装代码加私有egg扩展，修正为两者精确路径和SHA绑定后四配置通过。原失败脚本/程序/日志与来源诊断均保留，无环境或科学控制修改。一次本地内联诊断漏设scripts导入路径，失败于任何远端操作前；修正本地入口后执行只读诊断。'])
                value['decisions'].append('保留零创建/准入预算、两条原运行与全部旧暂停。CPU导入和14项资源边界检查不能替代原训练自然30轮、精确有限检查点/优化器/调度器核验及归档、独占原GPU0/350W、最终受保护可执行启动和全量配对FIRST/置信区间/子组/成本；不宣布达标，不迁移原GPU1后评。')
                value.setdefault('evidence',[]).extend([ref(out),binding_ref,failure,origin_ref,ref(__file__)]);tick.save(notes,value)
                memory=ROOT/'pipeline/codex_memory.md';lines=memory.read_text(encoding='utf-8').splitlines();assert any(s.startswith('- Current token '+token+';') for s in lines)
                positions=[i for i,s in enumerate(lines) if s.startswith('- Y101 resource-only evaluator adapter14 ')];assert len(positions)==1
                index=positions[0];lines[index]=lines[index].replace('private-runtime/code import launch binding and full','final guarded executable launch and full')
                lines[index+1:index+1]=['- Y101/Y36/W2/B1 four isolated original-private-runtime CPU import/config/class bindings PASS '+base.evidence(binding_ref)+'. Project PointNet wrapper plus original livePID private egg binary exact SHA verified; initial wrong wrapper-from-egg assumption failure '+base.evidence(failure)+' preserved, descriptive origin '+base.evidence(origin_ref)+'. No model/dataset/checkpoint/CUDA/evaluation. Adapter not deployed; original natural terminal/archive/exclusive resource/final guarded launch/full review pending.']
                lines.append('- This tick new4 operational CPU import bindings, original wrong PointNet path assumption failure and exact-source correction preserved; no scientific replay/new training. '+base.evidence(ref(out))+'.')
                assert len(lines)<120 and sum(s.startswith('## ') for s in lines)==7;memory.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n');authored['done']=True
            finally:writer.close()
        return original_command(args)
    base.command=command
    try:base.finish(token)
    finally:base.command=original_command
    assert authored['done'];print(json.dumps(dict(runtime_closeout=ref(out),isolated_import_bindings=4,initial_failure_preserved=True,checkpoint_loaded=False,GPU_used=False,evaluation_executed=False,memory_lines=len((ROOT/'pipeline/codex_memory.md').read_text(encoding='utf-8').splitlines()))))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--token',required=True);finish(parser.parse_args().token)
