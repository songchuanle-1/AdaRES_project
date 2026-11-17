"""Compose unchanged observation closeout with this tick's audited asset preparation."""
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
    view=tick.load(ROOT/'pipeline/headless/.tick_model_view.json')
    assert view['queue']['candidate_budget']==view['research']['creation_budget']==view['research']['reserve_budget']==0
    work=tick.load(ROOT/'pipeline/research_work.json')
    candidate=next(r for r in work['candidates'] if r['id']=='AdaRES-Y101')
    assert candidate['stage']=='training'
    stage_ref=candidate['frozen_review_reference_stage']
    stage=tick.load(closure.artifact(ROOT,stage_ref))
    clarification_ref=candidate['frozen_review_asset_clarification']
    clarification=tick.load(closure.artifact(ROOT,clarification_ref))
    inventory_ref=candidate['frozen_review_asset_inventory']
    inventory=tick.load(closure.artifact(ROOT,inventory_ref))
    for value in (stage,clarification,inventory):
        assert value['complete'] and value['token']==token and value['id']=='AdaRES-Y101'
    assert stage['all_three_reference_bytes_match_frozen_SHA']
    assert len(stage['published'])==len(stage['adapted_native_YAML'])==3
    assert stage['checkpoint_loaded'] is False and stage['CUDA_used'] is False
    assert stage['formal_FIRST_or_cost_executed'] is False and stage['evaluation_ready'] is False
    assert stage['active_source_or_native_configs_modified'] is False and stage['queue_or_running_modified'] is False
    for entry in stage['evidence']:
        closure.artifact(ROOT,entry)
    assert clarification['initial_byte_guard_not_passed'] and clarification['all_reference_sources_match_original_frozen_lf']
    assert len(clarification['source_lf_checks'])==18 and all(r['matches_frozen_lf'] for r in clarification['source_lf_checks'])
    closure.artifact(ROOT,clarification['original_failed_byte_guard'])
    rows=[tick.load(closure.artifact(ROOT,r)) for r in stage['published']]
    assert {r['id'] for r in rows}=={'AdaRES-Y36','AdaRES-W2','AdaRES-B1'}
    assert all(r['complete'] and r['published_without_overwrite'] and r['temporary_removed'] and not r['checkpoint_loaded'] and not r['CUDA_used'] for r in rows)
    supplement=ROOT/('pipeline/checks/y101_review_asset_closeout_'+token+'.json')
    assert not supplement.exists()
    supplement_value=dict(complete=True,token=token,id='AdaRES-Y101',recorded_at=datetime.now().astimezone().isoformat(),
        kind='operational_reference_preparation',original_frozen_resource=dict(server='server-1',gpu=0,power_limit_watts=350),
        reference_files=3,reference_bytes=sum(r['bytes'] for r in rows),adapted_native_YAML_files=3,
        source_files_matching_original_frozen_LF=18,initial_raw_byte_guard_failed=True,raw_failure_preserved=True,
        all_reference_checkpoint_bytes_match_original_frozen_SHA=True,original_checkpoint_and_native_configs_preserved=True,
        checkpoint_loaded=False,CUDA_used=False,new_scientific_candidates=0,scientific_CPU_replays=0,native_GPU_tasks=0,
        admissions=0,launches=0,archives=0,formal_FIRST_or_cost_executed=False,target_achievement_claimed=False,
        remaining=stage['remaining'],
        closeout_composition='Unmodified v3 observation/occupation/report/finish; add authored notes and seven-section memory only immediately before canonical report while lease owned. No validator, snapshot, metric, phase gate or deadline changes.',
        evidence=[ref(__file__),ref(base.__file__),inventory_ref,clarification_ref,stage_ref,clarification['original_failed_byte_guard']]+stage['published']+stage['adapted_native_YAML'])
    writer=monitor.acquire_lock(str(ROOT));assert writer is not None
    try:
        tick.save(supplement,supplement_value)
        work=tick.load(ROOT/'pipeline/research_work.json');candidate=next(r for r in work['candidates'] if r['id']=='AdaRES-Y101')
        assert candidate['stage']=='training' and candidate['frozen_review_reference_stage']==stage_ref
        candidate['frozen_review_asset_closeout']=ref(supplement)
        candidate['evidence'].extend([ref(supplement),ref(__file__)])
        tick.save(ROOT/'pipeline/research_work.json',work)
    finally:writer.close()
    original_command=base.command
    authored={'done':False}
    def command(args):
        if args[:3]==[base.SKILL+'/foreground_tick.py','AdaRES_project','report']:
            assert not authored['done'] and len(args)==7 and args[3:6]==['--token',token,'--notes']
            lease(token,120)
            notes=ROOT/('pipeline/reports/notes_'+token+'.json')
            assert os.path.normcase(os.path.abspath(args[6]))==os.path.normcase(str(notes.resolve()))
            writer=monitor.acquire_lock(str(ROOT));assert writer is not None
            try:
                value=tick.load(notes)
                value['actions'].extend([
                    '新准备：Y101原定server-1 GPU0/350W完整后评的Y36、W2、B1三份对照权重已预置；复制前后完整SHA均与原冻结值一致，未覆盖原文件。三份独立原生YAML只使用既有两项路径映射，不改活动训练代码或配置。',
                    '首次参考源码原始字节检查失败已保留；按原冻结LF清单复核18文件全部一致，W2/B1仅CRLF差异。未重跑科学测试，未加载检查点，未使用GPU或执行正式FIRST/成本。'])
                value['decisions'].append('Y101后评仍须等原训练自然30轮、精确原有限last/best/模型/优化器/调度器与完整Train-Val核验及归档、原server-1 GPU0独占、经测试的资源锁适配和完整配对后评。本次预置不迁移任何原GPU1挂起后评，不改变零准入预算或宣布达标。')
                value.setdefault('evidence',[]).extend([ref(supplement),stage_ref,clarification_ref,ref(__file__)])
                tick.save(notes,value)
                memory=ROOT/'pipeline/codex_memory.md';lines=memory.read_text(encoding='utf-8').splitlines()
                assert any(line.startswith('- Current token '+token+';') for line in lines)
                position=lines.index('## Closed Or Failed Directions')
                lines[position:position]=[
                    '- Y101 ORIGINAL server1GPU0/350W full-review references Y36/W2/B1 staged3 exact frozen-SHA checkpoints and3 independent native YAML copies using only existing two path mappings; active source/native configs untouched. '+base.evidence(stage_ref)+'. No checkpoint load/GPU/evaluation or old GPU1 review migration.',
                    '- Y101 initial raw-byte source guard FAILED and preserved; original LF-manifest18/18 match (12 W2/B1 CRLF-only). Clarification '+base.evidence(clarification_ref)+'; preparation audit '+base.evidence(ref(supplement))+'. Original negative evidence/control/receipt intact.',
                    '- Y101 review still requires natural original fixed30/full finite exact checkpoint proof/canonical archive, exclusive originalserver1GPU0/350W, tested resource-only evaluator lock adapter and full paired/bootstrap/subgroup/cost execution. No formal FIRST/efficacy/goal attainment.']
                lines.append('- This tick also prepared3 frozen reference weight files and3 path-adapted independent configs; operational preparation only, no new training. '+base.evidence(ref(supplement))+'.')
                assert len(lines)<120 and sum(s.startswith('## ') for s in lines)==7
                memory.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
                authored['done']=True
            finally:writer.close()
        return original_command(args)
    base.command=command
    try:base.finish(token)
    finally:base.command=original_command
    assert authored['done']
    print(json.dumps(dict(asset_closeout=ref(supplement),reference_files=3,configs=3,checkpoint_loaded=False,GPU_used=False,evaluation_executed=False,memory_lines=len((ROOT/'pipeline/codex_memory.md').read_text(encoding='utf-8').splitlines()))))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--token',required=True);finish(parser.parse_args().token)
