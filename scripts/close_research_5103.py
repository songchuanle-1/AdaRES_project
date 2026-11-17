"""Persist this turn's scientific limits and honest continuation disposition."""

import argparse
from datetime import datetime, timedelta
import importlib.util
import io
import json
import statistics
import unittest

from research_5103 import ROOT, TOKEN, lease, ref, save, bind, tick, monitor, closure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    parser.add_argument('--deadline', action='store_true')
    args = parser.parse_args()
    assert args.token == TOKEN
    owner, remaining = lease(45)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        work = tick.load(ROOT / 'pipeline/research_work.json')
        review_path = 'pipeline/analyses/continuation_review_' + TOKEN + '.json'
        if args.deadline:
            assert remaining <= 180, 'Actual deadline not yet near'
            work['design_cycle'] = dict(stop_reason='tick_deadline_near',
                attempted_ids=['AdaRES-Y52','AdaRES-Y54','AdaRES-Y56','AdaRES-Y57'],
                evidence=[ref(review_path)], actual_deadline_at=owner['deadline_at'],
                resume_condition='Continue SAME Y56 native loss-derived correction controls/independent '
                                 'implementation as evidence permits; Y57 original numerical-control hold '
                                 'requires new explanatory evidence or a preregistered bounded code revision, '
                                 'never lowering frozen error bound. No automatic new training.')
            tick.save(ROOT/'pipeline/research_work.json',work)
            print('Actual near-deadline disposition saved.')
            return
        paths = {name: 'pipeline/analyses/' + name + '_' + TOKEN + '.json' for name in (
            'final_terminal_comparison','y57_cpu','y57_activity_audit','y57_resources_v2_gpu1','y56_native_loss')}
        paths['y56_native_loss'] = 'pipeline/analyses/y56_native_loss_gpu1_' + TOKEN + '.json'
        values = {name: tick.load(ROOT / path) for name,path in paths.items()}
        assert all(value['complete'] for value in values.values())
        final = values['final_terminal_comparison']
        audit = values['y57_activity_audit']
        resources = values['y57_resources_v2_gpu1']
        loss56 = values['y56_native_loss']
        released = [r for r in audit['rows'] if r['state']=='released_initialization']
        trained = [r for r in audit['rows'] if r['state']=='trained_W2_diagnostic_ONLY']
        assert len(released)==len(trained)==4
        assert all(r['every_indicator_tensor_nonzero'] and r['added_columns_gradient']>0 for r in audit['rows'])
        assert all(not r['controls'][6]['original_control_passed'] for r in released)
        assert all(r['controls'][6]['original_control_passed'] for r in trained)
        assert resources['parameter_delta']==66048 and len(resources['stress'])==5
        assert all(r['finite_backward'] and r['peak_reserved_bytes'] < 24*2**30 for r in resources['stress'])
        assert len(resources['paired_native_predict_profile'])==16
        observations56 = []
        for pair in (0,8):
            rows = {r['mode']:r for r in loss56['rows'] if r['pair']==pair}
            assert set(rows)=={'original','repeat','inverse_multiplicity'}
            observations56.append(dict(pair=pair,
                original_loss=rows['original']['native_loss'], repeat_loss=rows['repeat']['native_loss'],
                corrected_loss=rows['inverse_multiplicity']['native_loss'],
                repeat_loss_abs=abs(rows['original']['native_loss']-rows['repeat']['native_loss']),
                corrected_minus_original=rows['inverse_multiplicity']['native_loss']-rows['original']['native_loss'],
                maximum_logit_change=max(r['logit_change'] for r in rows['inverse_multiplicity']['layer_activity']),
                copies=sum(sum(r['duplicate_excess']) for r in rows['inverse_multiplicity']['layer_activity'])))
        test_path = ROOT/'scripts/test_y57_prototype_5103.py'
        spec = importlib.util.spec_from_file_location('y57_structural_tests',str(test_path))
        tests = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tests)
        output=io.StringIO()
        result=unittest.TextTestRunner(stream=output,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
        assert result.wasSuccessful() and result.testsRun==5
        now=datetime.now().astimezone()
        review=dict(complete=True,token=TOKEN,reviewed_at=now.isoformat(),
            evidence={name:ref(path) for name,path in paths.items()},
            final_comparison='Y52/Y54 fixed30+all frozen references complete. Y54 large subgroup '
                             'versus Y36 triggers original harm rule; no automatic promotion/rerun.',
            Y56=observations56,
            Y57=dict(released_final_control_passed=0,released_batches=4,
                     diagnostic_trained_final_control_passed=4,
                     every_new_or_indicator_tensor_native_finite_nonzero=True,
                     formal_initialization='released_SPFormer_only',
                     diagnosis='Reachable gradients, but original released fixed final-layer control '
                               'below numerical bound. Trained diagnostic separation does not erase failure.',
                     released_final_signals=[r['controls'][6]['same_width_shift_signal'] for r in released],
                     released_final_bounds=[r['controls'][6]['arithmetic_bound'] for r in released],
                     trained_final_signals=[r['controls'][6]['same_width_shift_signal'] for r in trained],
                     resources=dict(parameter_delta=66048,stress_peak_reserved_bytes=max(
                         r['peak_reserved_bytes'] for r in resources['stress']),
                         paired_median_seconds={role:statistics.median(r['median_seconds'] for r in
                             resources['paired_native_predict_profile'] if r['role']==role)
                                                for role in ('parent','candidate')})),
            structural_tests=dict(tests=5,passed=True,output=output.getvalue(),source=ref(test_path)),
            interruptions='First native driver train() return mismatch retained. native_v2 interrupted '
                          'with no output/live process; unchanged retry exposed numerical criterion failure. '
                          'Resource GPU0 precheck transiently occupied, no GPU work launched; later GPU1 '
                          'retry with live precheck completed. No foreign job stopped.',
            optimizer_steps=0,weights_saved=False,new_training=0,Git_commits=0,
            limits='No trained Y56/Y57 benefit, final-selector harm, population causality, seed robustness '
                   'or novelty proof. No inherited numerical-control failure converted to PASS.')
        review_ref=save(review_path,review)
        candidates={c['id']:c for c in work['candidates']}
        c56,c57=candidates['AdaRES-Y56'],candidates['AdaRES-Y57']
        bind(c56,ref(paths['y56_native_loss']),ref(paths['y56_native_loss'].replace('.json','.log')),review_ref)
        c56.update(stage='design',scientific_status='actual_copy_native_task_loss_test_completed',
            next_action='Review actual copy/no-copy native task-loss and per-layer gradient evidence; '
                        'then preregister SAME Y56 independent inverse-multiplicity realization and '
                        'matched controls/native backward/resource tests if justified. Do not infer '
                        'trained gain or population harm from conditioned pair8.')
        c56['gate_summary']['diagnosis']='Actual trained-W2 native TRAIN pair8+no-copy pair0 task-loss/gradient witness completed; review signed deltas and repeat, no efficacy.'
        bind(c57,review_ref,ref(test_path),ref(paths['y57_cpu']),ref(paths['y57_activity_audit']),
             ref(paths['y57_resources_v2_gpu1']),ref(paths['y57_resources_v2_gpu1'].replace('.json','.log')),
             ref('pipeline/analyses/y57_resources_v2_'+TOKEN+'.log'))
        for revision in c57.get('diagnostic_revision_history',[]):
            if revision['scope']=='driver_train_return_and_output_scalar_compatibility':
                revision['result']='Driver compatibility resolved; scientific numerical-control failure preserved.'
                revision['result_evidence']=[ref(paths['y57_activity_audit']),ref(paths['y57_resources_v2_gpu1'])]
        c57.update(stage='blocked',failure_scope='implementation',blocker_kind='numerical_control_failure',
            blocker='All four released-initialization final-layer same-width shifted signals '
                    'remain below the unchanged frozen arithmetic bound, despite finite nonzero '
                    'native gradients. CPU and trained-W2-diagnostic controls and resources completed; '
                    'the original control is NOT passed. Not missing GPU/user authorization.',
            retry_after=(now+timedelta(hours=4)).isoformat(),
            unblock_condition='New evidence explaining original numerical-control failure, or a '
                              'same-ID evidence-backed preregistered bounded code revision, then retest '
                              'original frozen controls/native/resource gates. Do not lower bound, '
                              'deploy diagnostic weights, reinitialize from trained W2 or rename candidate.',
            next_action='Preserve exact seven-file Y57 bundle and original failed controls. Recheck '
                        'only changed explanatory evidence or a bounded same-ID revision; no commit, '
                        'admission or training until affected gates actually pass.',
            attempted_alternatives=['Physical independent CPU float32/64 controls.',
                'All4actual batch2 released native backward and all7evidence gradients.',
                'Separate strict shape-mapped trainedW2 diagnostic controls, never production init.',
                'Same bundle5stress iterations and16paired native predict profiles.'],
            wake_paths=[review_path,c57['tested_bundle']['path']])
        c57['gate_summary'].update(native_backward='Actual released four TRAIN batch2/32expressions/all7layers: every indicator tensor and new columns finite nonzero gradients.',
            controls='FAIL original released final layer shift separation4/4below frozen bound; CPU and trained diagnostic separation retained but not substitute PASS.',
            resources='Measured parameter delta66048; five finite native stress runs under24GiB;16paired native single-scene profiles, bounded scope.',
            independence='Six physically independent files/unique unchanged-config YAML; unchanged nonindicator AST and five structural regressions pass.',
            constraints='No optimizer/formal training, commit, admission, validation tuning or diagnostic-weight deployment.')
        work.pop('design_cycle',None)
        for candidate in (c56,c57):
            for row in candidate['evidence']:
                closure.artifact(ROOT,row)
        tick.save(ROOT/'pipeline/research_work.json',work)
        notes=dict(actions=[
            '本轮21:27规范快照design，队列0/容量2；补完Y36新9508表达/141场景双次FIRST评估，未重复上轮四方法8次评估。',
            'Y36新FIRST mIoU .510725438294；Y52/Y54对其总体区间跨0。Y54 large组差-.0134863/CI[-.0267899,-.00070265]触发预注册伤害审查，Y52无子组触发。',
            '实现Y57独立六文件包及唯一YAML；CPU双精度、实际四组batch2全7层反向、训练父模型隔离诊断及资源已执行，原所有失败保留。',
            'Y57正式初始化全部新增参数有有限非零梯度，但4/4最终层错位信号仍低于冻结数值界；按实现级numerical_control_failure暂挂，不改门槛、不称已通过。',
            'Y57增加66048参数，5次原生后向压力及16组配对推理资源记录完成；5项结构回归通过。',
            'Y56真实复制pair8/无复制pair0，16表达全部7损失层original/repeat/inverse-count原生损失及梯度已完成；有符号差值和零对照保留。'],
            decisions=['Y52/Y54固定训练和五方法冻结比较现已闭环，但不是性能成功；不自动晋升、调参或重跑。',
                'Y57数值控制失败是实现级研究暂挂，不是缺授权或假设证伪；已完成可执行替代诊断，不提交未通过候选。',
                'Y56保持原ID继续由实际损失证据推进；不新增占位候选、队列、Git提交或训练。完整报告0-3保留preflight时间，finish前不预称PASS。'])
        tick.save(ROOT/('pipeline/reports/notes_'+TOKEN+'.json'),notes)
        memory=f'''# AdaRES_project Research Memory
## Goal And Phase
- 当前token {TOKEN}；上海快照2026-10-20 21:27:35.736825/deadline22:27:35.736825不修改；原manual继续中接到同项目heartbeat，续用同一租约而非重进begin。
- 仅AdaRES_project/PROJECT_ROOT D:/project/python -B；server-2 GPU0/1各单实验，owner真实CODEX_THREAD_ID已核实，不改调度不跨项目。queue0/capacity2/budget2/reserve_budget0。
## Champion And Baseline
- 目标.5150，历史CSV冠军Y36 .5110/B1 .5020/差.0040；本轮新冻结Y36 FIRST .510725438294、repeat .510722322121，9508表达/141scene各双次，不混用历史rounded。
- 上轮四方法FIRST：Y52 .509708428366/Y54 .506342952986/W2 .510107995189/B1 .502221764356，本轮未重跑。
- 五方法比较pipeline/analyses/final_terminal_comparison_{TOKEN}.json，Y52-Y36 -.00101701/CI[-.00899731,.00675352]，Y54-Y36 -.00438249/CI[-.01261417,.00369061]，总体均跨0。
- Y54 large组3388表达/135scene，对Y36差-.01348630762/CI[-.02678985305,-.000702650758]触发原CI上界<0伤害审查；Y52无触发。Y52/Y54 complete表示fixed30和冻结比较闭环，不是模型成功，不自动晋升/重跑/全假设证伪。
## Active Hypotheses
- Y56保持原ID design；历史40TRAIN场景160表达1120相关层记录中一个场景初始层3精确复制，最终层0。hardmask相等不等于精确query+softmask复制，条件case不估计总体。
- 本轮真实复制pair8+无复制pair0，original/repeat/inverse-count各实际native任务loss/backward/all7层，16表达。结果{paths['y56_native_loss']}；有符号损失/重复/逐层梯度见本轮review，不把局部扰动当训练收益。
- Y56下一项同ID审查损失证据后冻结独立逆复制实现和控制/后向/资源；未创建正式Y56包、不称准入。
- Y57独立包adares_y57_peer_second_moment，六Python+唯一res YAML，保留W2全部769输入再拼全256逐通道方差、1025宽/原7挂接层/原loss/seed1999/fixed30/Q128/单argmax。
- 包清单pipeline/analyses/y57_prototype_bundle_{TOKEN}.json，七文件字节冻结，非indicator AST与其余五文件完全保持，5结构回归通过。实际新增66048参数，未提交。
- CPU float32/64重复/排列/错位/相同peer/空支持/padding及组件梯度通过；不当原生PASS。
- 实际released SPFormer四TRAIN batch2/32表达/全部7层：每个indicator tensor、新输入列和variance证据梯度均有限非零；无optimizer/存权重。
- 但原固定最终层错位控制4/4信号4.70e-6至6.17e-6低于原1.5258789e-5界，失败保留。分开trainedW2诊断状态相同控制4/4通过(signal1.0886-1.9394)，不是正式初始化或撤销失败。
- Y57 blocked/failure_scope implementation/blocker_kind numerical_control_failure；非缺GPU/人工授权、非假设证伪。retry {c57['retry_after']}，新说明证据/同ID预注册有界修订可唤醒；不降低界、改名或用trainedW2正式初始化。
## Closed Or Failed Directions
- 原Y53/Y55/Y50/Y51/Y49rev1/Y46/Y48等关闭/拒绝保持。Y57原native train()链式返回None诊断脚本失败已兼容修正，不是生产代码修订。
- native_v2执行中断无结果/现场无进程确认后同程序唯一名重试；重试明确原数值控制失败，全部日志保留。资源GPU0一度占用precheck拒绝，后GPU1新实时检查执行，无外来作业被停止。
- 上轮标量Y57全输入gain负、向量对全输入CI跨0、两个下降场景均保留；不从当前梯度/资源证明训练效果或新颖性。
## Running And Queue
- 本轮未新增训练、队列、准入、Git提交或push；已存Y52/Y54 finished checkpoints不改。HEAD8943c8c460501b200a8dea958964579ffea65f76预期不变，无关dirty保留。
- Y57资源5次最大固定输入native backward/2warm3测、16单场景parent/candidate配对profile完成，峰值{review['Y57']['resources']['stress_peak_reserved_bytes']}bytes；不是全量最坏/Pareto保证。
- 原Y52/Y54精确提交授权不覆盖新Y57包；但当前先缺科学控制，不制造新授权问题或重复索要。原始GPU授权足够，新的GPU执行每次仍实时占用/租约核查。
## Open Questions And Risks
- Y56原生损失活动是否支持正式逆复制实现需审查，不等于最终伤害/总体效益。Y57需解释未过原数值控制或预注册有界局部修订后重测全部受影响gate。
- 本轮review {review_path} SHA{review_ref['sha256']}；冻结原生审计/资源/损失和失败证据已逐项绑定。
- 单训练seed、scene CI非seed不确定性；时延只固定TRAIN输入，official unique/multiple未知；未确认文献新颖性。
## Last Tick
- 实际新增Y36双次全量/完整五方法比较、Y57独立实现与CPU+native原生审计/资源、Y56原生损失检验及5结构回归；未把上轮四方法评估算成本轮执行。
- 当前先持久化事实与下一项，最终以canonical finish/FINALIZED/PASS/context提交及锁/pending消失核实，不预称成功；若checker不能接受诚实disposition则abort说明，释放后不补写。
'''
        assert len(memory.splitlines())<120
        (ROOT/'pipeline/codex_memory.md').write_text(memory,encoding='utf-8',newline='\n')
        monitor.log_decision(str(ROOT),'next_batch','继续AdaRES原候选与冻结评估，不能以梯度代替科学控制。',
            'Y52/Y54完整比较闭环，Y54大型目标触发伤害；Y57数值控制实现级暂挂，Y56保留原生损失证据续做。',
            '原始门槛未改；独立包/CPU/native/资源与替代诊断真实完成，原负结果保留。',review_path)
        print(json.dumps(dict(review=review_ref,Y56=observations56,Y57=review['Y57'],remaining=lease()[1]),indent=2))
    finally:
        lock.close()


if __name__=='__main__':
    main()
