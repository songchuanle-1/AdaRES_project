"""Bind actual CPU execution of the unchanged native Y81 evidence callbacks."""
import argparse
import ast
import json
from prepare_y81_query_mask_refinement import ROOT, lease, monitor, ref, tick
import closure


def save(token):
    work=tick.load(ROOT/'pipeline/research_work.json');candidate=next(c for c in work['candidates'] if c['id']=='AdaRES-Y81')
    body=ROOT/'scripts/probe_y81_native_smoke.py';program=ROOT/('pipeline/checks/y81_native_observer_CPU_'+token+'_program.py')
    result=ROOT/('pipeline/analyses/y81_native_observer_cpu_'+token+'.json');value=tick.load(result)
    def functions(path):
        return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
    expected,actual=functions(body),functions(program)
    assert len(expected)==13 and all(actual[key]==content for key,content in expected.items())
    payload=ast.literal_eval(ast.parse(program.read_text(encoding='utf-8')).body[0].value)
    assert value['id']==candidate['id']=='AdaRES-Y81' and value['code_commit']==candidate['code_commit']==payload['code_commit']
    assert value['native_body_sha256']==payload['native_body_sha256']==ref(body)['sha256']
    assert all(value[key] is True for key in ('complete','CPU_native_callbacks_executed','configured_TRAIN_data_collated','released_backbone_loaded','full_model_CPU_constructed'))
    assert all(value[key] is False for key in ('CUDA_initialized','native_sparse_task','native_smoke_passed','admission'))
    assert value['train_points']==162738 and value['train_expressions']==8 and value['synthetic_CPU_decoder_optimizer_steps']==2
    assert len(value['gradient_checks'])==len(value['synthetic_CPU_decoder_losses'])==2
    from test_y81_native_smoke_plan import N
    N['check_response_rows'](value['refinement_paths'])
    for row in value['gradient_checks']:
        assert row['all_finite'] and row['both_input_gradients_finite'] and row['tensors']==387 and row['indicator_tensors']==10
        assert row['indicator_calls']==row['query_norm_consumption_calls']==7
        assert row['mask_refinement_calls']==row['prediction_consumption_calls']==row['original_coarse_gate_consumption_calls']==6
        assert row['refinement_weight_gradient_max']>0 and row['parameter_update_max']>0
    preparation=tick.load(closure.artifact(ROOT,candidate['native_preparation']))
    assert preparation['native_body']==ref(body) and preparation['guard_tests_passed']==11 and not preparation['native_executed']
    out=ROOT/('pipeline/checks/y81_native_observer_verification_'+token+'.json');assert not out.exists()
    evidence=[ref(result),ref(result.with_suffix('.log')),ref(program),ref(body),ref(__file__),ref('scripts/prepare_y81_native_observer_cpu.py')]
    tick.save(out,dict(complete=True,id=candidate['id'],token=token,code_commit=candidate['code_commit'],CPU_native_callbacks_actually_executed=True,
        native_body=ref(body),CPU_program=ref(program),CPU_result=ref(result),exact_callback_functions_and_classes=13,
        original_full_model_pretrain_TRAIN_loaded=True,two_synthetic_CPU_decoder_optimizer_steps=True,exact_static_mask_query_padding_and_prediction_consumption=True,
        original_hard_gate_coarse_consumers_and_shared_heads=True,finite_both_input_and_new_weight_gradients_and_updates=True,
        zero_then_nonzero_response=True,evidence=evidence,CUDA_initialized=False,native_sparse_task=False,native_smoke_passed=False,admission=False,efficacy_unknown=True))
    candidate['CPU_native_observer_verification']=ref(out);candidate['evidence'].extend([ref(out)]+evidence)
    candidate['next_action']='Do not replay completed CPU/full model/TRAIN/version/native-plan/callback preparation. Preserve active HEAD e0c and original Y77/Y78/Y79 plus paused Y57. Y80 precedes Y81. Only canonical matching owner full fixed30 archive plus fresh exclusive authorized server2GPU0/280W or GPU1/390W idle/no compute/no reservation permits the exact frozen native two-step YAML/TRAIN task. Actual finite task loss/backward/optimizer and all six original static mask base/current refined query/padding/prediction consumers, original hard-gate/coarse consumers and seven shared heads must PASS before original complete evaluation/cost freeze, new receipt/admission, exact retained-version source contract, safe deployment and increasing steps. Server1 requires a separate private-runtime/path/350W plan. No GPU PASS/admission/efficacy from CPU checks.'
    work['tick_attempt']=dict(tick_started_at=tick.load(ROOT/'pipeline/headless/.tick_model_view.json')['generated_at'],candidate_id=candidate['id'],kind='implementation_test',result='passed',
        command='python -B scripts/run_experiment_first_probe.py y81_native_observer_cpu '+ref(program)['path']+' --token '+token+' --seconds 480',
        observation='Thirteen exact frozen native callbacks/classes executed on two synthetic CPU decoder optimizer steps after original full model/pretrain/TRAIN loading; original static mask/current query/padding and six prediction consumers, original hard-gate/coarse and seven shared heads, zero/next-step response and finite original/new/input gradients and real updates passed. No GPU/native sparse task/admission/efficacy.',evidence=[ref(out),ref(result)])
    tick.save(ROOT/'pipeline/research_work.json',work);print(json.dumps(dict(verification=ref(out),functions=13,CPU_only=True)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--token',required=True);args=parser.parse_args();lease(args.token,180)
    lock=monitor.acquire_lock(str(ROOT));assert lock is not None
    try:save(args.token)
    finally:lock.close()
