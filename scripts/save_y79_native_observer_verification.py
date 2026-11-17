"""Bind actual CPU execution of the exact native callbacks, without GPU admission."""
import argparse
import ast
import json
from prepare_y79_query_language_feedback import ROOT, CID, lease, monitor, ref, tick
import closure


def save(token):
    body = ROOT / 'scripts/probe_y79_native_smoke.py'
    program = ROOT / ('pipeline/checks/y79_native_observer_CPU_' + token + '_program.py')
    result = ROOT / ('pipeline/analyses/y79_native_observer_cpu_' + token + '.json')
    native = ast.parse(body.read_text(encoding='utf-8'))
    wrapper = ast.parse(program.read_text(encoding='utf-8'))
    functions = {n.name: ast.dump(n, include_attributes=False) for n in native.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    actual = {n.name: ast.dump(n, include_attributes=False) for n in wrapper.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert len(functions) == 16 and all(actual[k] == v for k, v in functions.items())
    payload = ast.literal_eval(wrapper.body[0].value)
    value = tick.load(result)
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == CID)
    assert candidate['stage'] == 'blocked' and candidate['blocker_kind'] == 'exclusive_gpu_slot'
    assert value['id'] == CID and value['code_commit'] == candidate['code_commit'] == payload['code_commit']
    assert value['native_body_sha256'] == payload['native_body_sha256'] == ref(body)['sha256']
    assert all(value[k] is True for k in ('complete', 'CPU_native_callbacks_executed', 'configured_TRAIN_data_collated', 'released_backbone_loaded', 'full_model_CPU_constructed'))
    assert all(value[k] is False for k in ('CUDA_initialized', 'native_sparse_task', 'native_smoke_passed', 'admission'))
    assert value['train_points'] == 162738 and value['train_expressions'] == 8 and value['synthetic_CPU_decoder_optimizer_steps'] == 2
    assert len(value['gradient_checks']) == 2 and len(value['synthetic_CPU_decoder_losses']) == 2
    controls = {k: v for k, v in globals().items() if False}
    for name in ('check_path_calls', 'check_response_rows'):
        node = next(n for n in native.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(body), 'exec'), controls)
    controls['check_response_rows'](value['feedback_paths'])
    for record in value['gradient_checks']:
        assert record['all_finite'] and record['both_input_gradients_finite']
        assert record['tensors'] == 387 and record['indicator_tensors'] == 10
        assert record['indicator_calls'] == record['query_norm_consumption_calls'] == 7
        assert record['feedback_calls'] == record['qla_consumption_calls'] == 6 and record['next_lla_consumption_calls'] == 5
        assert record['feedback_weight_gradient_max'] > 0 and record['parameter_update_max'] > 0
    preparation = tick.load(closure.artifact(ROOT, candidate['native_preparation']))
    assert preparation['native_body'] == ref(body) and preparation['guard_tests_passed'] == 11 and not preparation['native_executed']
    out = ROOT / ('pipeline/checks/y79_native_observer_verification_' + token + '.json')
    assert not out.exists()
    evidence = [ref(result), ref(result.with_suffix('.log')), ref(program), ref(body), ref(__file__), ref('scripts/prepare_y79_native_observer_cpu.py')]
    tick.save(out, dict(complete=True, id=CID, token=token, code_commit=candidate['code_commit'],
        CPU_native_callbacks_actually_executed=True, native_body=ref(body), CPU_program=ref(program), CPU_result=ref(result),
        exact_callback_functions_and_classes=16, original_full_model_pretrain_TRAIN_loaded=True,
        two_synthetic_CPU_decoder_optimizer_steps=True, full_seven_shared_confidence_query_norm_calls=True,
        six_exact_current_query_language_ffn_padding_identities=True, six_qla_and_five_next_lla_consumers=True,
        finite_both_input_and_new_weight_gradients_and_updates=True, zero_then_nonzero_response=True,
        evidence=evidence, CUDA_initialized=False, native_sparse_task=False, native_smoke_passed=False, admission=False, efficacy_unknown=True))
    candidate['CPU_native_observer_verification'] = ref(out)
    candidate['evidence'].extend([ref(out)] + evidence)
    candidate['next_action'] = 'Do not replay completed CPU/model/TRAIN/version/native-plan/callback preparation. Preserve HEAD498e and original Y75/Y76/Y77 plus paused Y57. Y78 precedes Y79. Only canonical slot-owner fixed30 archive plus fresh matching server2GPU0/280W or GPU1/390W exclusive idle/no compute/no reservation permits the exact frozen native YAML/tools.train TRAIN two-step task. Actual finite task loss/backward/optimizer and all original query/language/padding/qla/next-lla/shared-head paths must PASS before original complete evaluation/cost freeze, new receipt/admission, exact retained-version source contract, safe production selection/sync, canonical dispatch and increasing native steps. Server1 requires its own private runtime/path/350W plan. No GPU PASS or efficacy from CPU checks.'
    work['tick_attempt'] = dict(tick_started_at=tick.load(ROOT / 'pipeline/headless/.tick_model_view.json')['generated_at'], candidate_id=CID,
        kind='implementation_test', result='passed',
        command='python -B scripts/run_experiment_first_probe.py y79_native_observer_cpu ' + ref(program)['path'] + ' --token ' + token + ' --seconds 480',
        observation='Exact sixteen frozen native callback functions/classes actually executed on two synthetic CPU decoder optimizer steps after full original YAML/model/pretrain/TRAIN loading; finite gradients and updates, original shared consumers and zero/next-step response passed. No CUDA/native sparse TRAIN task/GPU PASS/admission/efficacy.', evidence=[ref(out), ref(result)])
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(verification=ref(out), functions=16, CPU_only=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 180)
    lock = monitor.acquire_lock(str(ROOT))
    assert lock is not None
    try:
        save(args.token)
    finally:
        lock.close()
