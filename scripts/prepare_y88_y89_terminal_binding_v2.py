"""Retain the failed wrapper and repair only its test expectation and log labels."""
import argparse
import ast
from datetime import datetime
from prepare_y60_stage_heads import ROOT, lease, ref, tick


def validate_existing_generated():
    finalizer = (ROOT / 'scripts/finalize_y88_fixed30.py').read_text(encoding='utf-8')
    finalizer = finalizer.replace('y88_terminal_contract', 'y88_y89_terminal_contract')
    finalizer = finalizer.replace('test_y88_terminal_handoff.py', 'test_y88_terminal_handoff_y89.py')
    finalizer = finalizer.replace('scripts/finalize_y88_fixed30.py', 'scripts/finalize_y88_fixed30_y89.py')
    tests = (ROOT / 'scripts/test_y88_terminal_handoff.py').read_text(encoding='utf-8')
    tests = tests.replace('import finalize_y88_fixed30 as archive', 'import finalize_y88_fixed30_y89 as archive')
    tests = tests.replace('from y88_terminal_contract import', 'from y88_y89_terminal_contract import')
    paths = (ROOT / 'scripts/finalize_y88_fixed30_y89.py',
             ROOT / 'scripts/test_y88_terminal_handoff_y89.py')
    for path, expected in zip(paths, (finalizer, tests)):
        assert path.read_bytes() == expected.encode()
        ast.parse(expected, feature_version=(3, 8))
    return paths


def prepare(token):
    lease(token, 300)
    original = ROOT / 'scripts/prepare_y88_y89_terminal_binding.py'
    lifecycle = ROOT / ('pipeline/checks/y88_y89_terminal_binding_' + token + '_lifecycle.log')
    failed = ROOT / ('pipeline/checks/y88_y89_terminal_binding_' + token + '_additive.log')
    assert b'Ran 25 tests' in lifecycle.read_bytes() and b'\nOK' in lifecycle.read_bytes()
    assert b'FAILED (errors=1)' in failed.read_bytes() and b"key='allowed_gpus'" in failed.read_bytes()
    assert b'ValueError: AdaRES server-1 requires its exact scoped resource grant' in failed.read_bytes()
    repair = ROOT / ('pipeline/checks/y88_y89_binding_test_repair_' + token + '.json')
    assert not repair.exists()
    tick.save(repair, dict(complete=True, token=token, observed_at=datetime.now().astimezone().isoformat(),
        scope='test_expectation_only', actual_guard_rejected_unauthorized_resource=True,
        observation='Original test expected AssertionError but canonical resource guard raised ValueError. '
            'Only the exact allowed_gpus case now requires ValueError; all other test expectations and controls unchanged.',
        model_changed=False, native_plan_changed=False, resource_guard_changed=False,
        original_finalizers_changed=False, scientific_revision=False,
        evidence=[ref(original), ref(lifecycle), ref(failed), ref('scripts/test_y88_y89_binding.py'),
            ref('scripts/test_y88_y89_binding_v2.py'), ref(__file__)]))
    source = original.read_text(encoding='utf-8')
    replacements = {
        'generated = generate()': 'generated = validate_existing_generated()',
        "('test_y88_y89_binding.py', 6, 'additive')": "('test_y88_y89_binding_v2.py', 6, 'additive')",
        "'pipeline/checks/y88_y89_terminal_binding_'": "'pipeline/checks/y88_y89_terminal_binding_v2_'",
        "ref('scripts/test_y88_y89_binding.py')": "ref('scripts/test_y88_y89_binding_v2.py')",
        "dependencies = old['dependencies'] +": "dependencies = old['dependencies'] + [ref(repair_path)] +",
    }
    for before, after in replacements.items():
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    namespace = dict(__file__=__file__, __name__='y88_y89_binding_v2',
                     validate_existing_generated=validate_existing_generated, repair_path=repair)
    exec(compile(source, str(original), 'exec'), namespace)
    namespace['prepare'](token)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    prepare(parser.parse_args().token)
