"""Specialize preserved observers for a different, explicitly checked connection."""
import argparse
import ast
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick

STRINGS = {
    'hidden_gate_calls': 'evidence_skip_calls',
    'active_hidden_gate_changes_confidence_only': 'active_evidence_skip_changes_confidence_only',
    'gated_hidden_consumed': 'residual_hidden_consumed',
    'first_and_second_hidden_identity': 'evidence_and_second_hidden_identity',
    'float64_gate_reference': 'float64_residual_reference',
    'final_linear_consumes_gated_hidden': 'final_linear_consumes_residual_hidden',
    'gate_delta_max': 'projection_delta_max',
}

RESPONSE = '''
def check_response_rows(rows):
    import math
    assert len(rows) == 14
    for index, row in enumerate(rows):
        assert type(row['step']) is int and row['step'] == index // 7
        assert type(row['stage']) is int and row['stage'] == index % 7
        assert row['weight_nonzero'] is bool(index // 7)
        for key in ('original_query_identity', 'original_mask_identity', 'original_mass_identity',
                    'original_evidence_exact', 'evidence_and_second_hidden_identity',
                    'both_inputs_undetached', 'float64_residual_reference',
                    'final_linear_consumes_residual_hidden'):
            assert row[key] is True, key
        for key in ('projection_delta_max', 'hidden_delta_max'):
            assert math.isfinite(row[key])
            assert row[key] > 0 if index // 7 else row[key] == 0
'''

CAPTURE = '''
def capture_residual(module, args, value):
    index = len(state['gated'])
    assert module is residual and len(args) == 2
    evidence, second = args
    assert evidence is state['evidence_outputs'][index] and second is state['second_hidden'][index]
    assert evidence.shape[-1] == 769 and second.shape[-1] == 256
    assert evidence.requires_grad and second.requires_grad and value.requires_grad
    reference = second.double() + evidence.double() @ module.weight.double()
    torch.testing.assert_allclose(value.double(), reference, rtol=1e-5, atol=1e-6)
    projected = evidence @ module.weight
    assert torch.equal(value, second + projected) and torch.isfinite(value).all()
    nonzero = bool(torch.count_nonzero(module.weight))
    assert nonzero == bool(len(step_rows))
    projection_delta = float(projected.abs().max())
    hidden_delta = float((value - second).abs().max())
    assert projection_delta > 0 and hidden_delta > 0 if nonzero else projection_delta == hidden_delta == 0
    value.retain_grad()
    state['gated'].append(value)
    path_rows.append(dict(step=len(step_rows), stage=index, weight_nonzero=nonzero,
        projection_delta_max=projection_delta, hidden_delta_max=hidden_delta,
        original_query_identity=True, original_mask_identity=True, original_mass_identity=True,
        original_evidence_exact=True, evidence_and_second_hidden_identity=True,
        both_inputs_undetached=True, float64_residual_reference=True,
        final_linear_consumes_residual_hidden=False))
'''


class Specialize(ast.NodeTransformer):
    def visit_Name(self, node):
        node.id = {'gate': 'residual', 'capture_gate': 'capture_residual'}.get(node.id, node.id)
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = STRINGS.get(node.value, node.value)
            node.value = node.value.replace('AdaRES-Y91', 'AdaRES-Y93').replace('y91_', 'y93_')
            node.value = node.value.replace('gate.weight', 'residual.weight')
            node.value = node.value.replace('hidden_gate_calls=14', 'evidence_skip_calls=14')
        elif type(node.value) is int:
            node.value = {164695173: 164826501, 65536: 196864}.get(node.value, node.value)
        return node

    def visit_keyword(self, node):
        node.arg = STRINGS.get(node.arg, node.arg)
        return self.generic_visit(node)

    def visit_ImportFrom(self, node):
        node.module = node.module.replace('y91_observer_core', 'y93_observer_core')
        node.module = node.module.replace('y91_contract', 'y93_contract')
        node.module = node.module.replace('prepare_y91_hidden_confidence_gate', 'prepare_y93_evidence_hidden_skip')
        node.module = node.module.replace('run_y91_native', 'run_y93_native')
        return node

    def visit_FunctionDef(self, node):
        if node.name == 'check_response_rows':
            return ast.parse(RESPONSE).body[0]
        if node.name == 'capture_gate':
            return ast.parse(CAPTURE).body[0]
        node = self.generic_visit(node)
        if node.name == 'test_live_or_incomplete_owner_never_unblocks':
            for item in ast.walk(node):
                if isinstance(item, ast.Constant) and item.value == 'AdaRES-Y88':
                    item.value = 'AdaRES-Y90'
        return node


def create(token):
    lease(token, 180)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        names = ('y91_observer_core.py', 'probe_y91_cpu_tail.py', 'probe_y91_native_tail.py',
                 'y91_contract.py', 'test_y91_preparation.py', 'test_y91_runtime_plan.py')
        original = [ref('scripts/' + name) for name in names]
        generated = []
        for name in names:
            destination = ROOT / 'scripts' / name.replace('y91', 'y93')
            assert not destination.exists()
            parsed = Specialize().visit(ast.parse((ROOT / 'scripts' / name).read_text(encoding='utf-8')))
            text = ast.unparse(ast.fix_missing_locations(parsed)) + '\n'
            ast.parse(text, feature_version=(3, 8))
            destination.write_text(text, encoding='utf-8', newline='\n')
            generated.append(ref(destination))
        assert [ref(item['path']) for item in original] == original
        out = ROOT / ('pipeline/checks/y93_support_' + token + '.json')
        assert not out.exists()
        tick.save(out, dict(complete=True, token=token, originals=original, generated=generated,
            connection='full normalized evidence plus original second hidden, not an internal multiplicative gate',
            original_helpers_preserved=True, CPU_executed=False, native_executed=False, helper=ref(__file__)))
        print(__import__('json').dumps(dict(support=ref(out), generated=generated)))
    finally:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    create(parser.parse_args().token)
