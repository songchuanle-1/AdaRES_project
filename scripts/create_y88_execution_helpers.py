"""Reuse the guarded server1 admission/dispatch code with Y88-specific validation."""
import ast
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def create():
    source = (SCRIPTS / 'advance_y84.py').read_text(encoding='utf-8')
    source = source[:source.index('\ndef activate(token):')]
    source = source.replace('Y84', 'Y88').replace('y84', 'y88')
    begin = source.index('def validate_result(')
    end = source.index('\ndef verify(token):', begin)
    source = source[:begin] + 'from y88_contract import validate_result\n\n' + source[end:]
    source = source.replace('python -B scripts/y88_server1_alternative.py run --token ',
                            'python -B scripts/run_y88_server1_native.py --token ')
    source = source.replace('independent visual query initialization', 'independent mask-only quadratic query readout')
    source = source.replace('Preserve Y83', 'Preserve Y87')
    source += '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('verify', 'admit', 'dispatch'))
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    lease(args.token, 300)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        dict(verify=verify, admit=admit, dispatch=dispatch)[args.operation](args.token)
    finally:
        writer.close()
'''
    ast.parse(source, feature_version=(3, 8))
    output = SCRIPTS / 'advance_y88.py'
    assert not output.exists()
    output.write_text(source, encoding='utf-8', newline='\n')
    tests = (SCRIPTS / 'test_y87_admission.py').read_text(encoding='utf-8')
    for old, new in {
        'Y87': 'Y88', 'y87': 'y88',
        'fusion_calls': 'projection_calls', 'original_norm_consumption_calls': 'original_mask_consumers',
        'original_ffn_consumption_calls': 'original_score_consumers', 'fusion_paths': 'projection_paths',
        'fusion_delta_max': 'delta_max', 'original_self_identity': 'original_normalized_query_identity',
        'original_language_identity': 'original_mask_consumer', 'both_inputs_undetached': 'input_undetached',
        'float64_sum_product_projection_reference': 'float64_quadratic_projection_reference',
        'both_input_gradients_finite': 'input_gradients_finite',
        "'original_initial_prediction_preserved'": "'static_mask_features_preserved'",
    }.items():
        tests = tests.replace(old, new)
    tests = tests.replace("advance.idle(dict(gpu='0,1,280', compute=''), 280.)",
                          "advance.idle(dict(gpu='0,1,350', compute=''), 350.)")
    tests = tests.replace("[('nan,0,280', ''), ('0,0,350', ''), ('0,2000,280', ''), ('0,0,280', '99')]",
                          "[('nan,0,350', ''), ('0,0,390', ''), ('0,2000,350', ''), ('0,0,350', '99')]")
    tests = tests.replace('advance.idle(dict(gpu=gpu, compute=compute), 280.)',
                          'advance.idle(dict(gpu=gpu, compute=compute), 350.)')
    ast.parse(tests, feature_version=(3, 8))
    output = SCRIPTS / 'test_y88_admission.py'
    assert not output.exists()
    output.write_text(tests, encoding='utf-8', newline='\n')
    print('Created Y88-specific admission/dispatch helper and fifteen actual-evidence guards; originals unchanged.')


if __name__ == '__main__':
    create()
