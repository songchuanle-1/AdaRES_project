"""One-time specialization of retention mechanics; historical signed helpers stay intact."""
import ast
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

STEP_CHECK = '''    for index, row in enumerate(cpu['CPU_decoder_steps']):
        assert row['step'] == index and row['all_finite'] and row['indicator_tensors'] == 10
        assert row['indicator_calls'] == row['query_norm_consumption_calls'] == 7
        assert row['fusion_calls'] == row['original_norm_consumption_calls'] == row['original_ffn_consumption_calls'] == 6
        assert row['both_input_gradients_finite'] is True
        assert row['new_gradient_max'] > 0 and row['parameter_update_max'] > 0
'''


def create():
    source = (SCRIPTS / 'retain_y86_cpu_native.py').read_text(encoding='utf-8')
    source = source.replace('Y86', 'Y87').replace('y86', 'y87')
    source = source.replace('prepare_y87_overlap_skip', 'prepare_y87_self_language_product')
    source = source.replace('adares-y87-overlap-readout-skip', 'adares-y87-self-language-product')
    source = source.replace("BRANCH = 'refs/heads/adares-y87-self-language-product'",
        "BRANCH = 'refs/heads/adares-y87-self-language-product'\nPARENT = '2d2a0b98865977a6e2f37d316979344ffc56817c'")
    source = source.replace("'nonzero_confidence_only_intervention_restored'",
                            "'nonzero_fusion_changes_refined_and_final_masks', 'bypassed_fusion_negative_control_exact'")
    source = source.replace("'original_static_masks_and_hard_gates_verified'",
                            "'static_mask_and_original_hard_gate_preserved', 'original_self_language_and_norm_ffn_consumption_verified'")
    source = source.replace("(164629639, 2)", "(164695173, 65536)")
    source = source.replace("len(cpu['CPU_shortcut_paths']) == 14", "len(cpu['CPU_fusion_paths']) == 12")
    start = source.index("    for index, row in enumerate(cpu['CPU_decoder_steps']):")
    end = source.index("    program = ROOT /", start)
    source = source[:start] + STEP_CHECK + source[end:]
    source = source.replace("    git('read-tree', Y85, env=env)",
        "    assert git('rev-parse', PARENT + '^').decode().strip() == Y85\n"
        "    previous = next(item for item in work['candidates'] if item['id'] == 'AdaRES-Y86')\n"
        "    assert previous['code_commit'] == PARENT and previous['implementation_test_passed']\n"
        "    closure.artifact(ROOT, previous['source_manifest'])\n"
        "    prior_delta = [row.split('\\t') for row in git('diff', '--no-renames', '--name-status', Y85, PARENT).decode().splitlines()]\n"
        "    assert len(prior_delta) == 7 and all(status == 'A' and 'adares_y86_overlap_readout_skip' in path for status, path in prior_delta)\n"
        "    assert all((ROOT / path).read_bytes() == git('show', PARENT + ':' + path) for _, path in prior_delta)\n"
        "    git('read-tree', PARENT, env=env)")
    source = source.replace("'--name-status', Y85, tree", "'--name-status', PARENT, tree")
    source = source.replace("git('commit-tree', tree, '-p', Y85, '-m', 'Add Y87 direct readout of existing maximum overlap')",
                            "git('commit-tree', tree, '-p', PARENT, '-m', 'Add Y87 self-language query product before original norm')")
    source = source.replace('parent_commit=Y85', 'parent_commit=PARENT')
    start = source.index("    prefix, separator, _ = ")
    end = source.index("    expected = ", start)
    source = source[:start] + (
        "    core = ROOT / 'scripts/y87_observer_core.py'\n"
        "    body = closure.artifact(ROOT, candidate['prepared_native_body']).read_text(encoding='utf-8')\n"
    ) + source[end:]
    source = source.replace('assert len(expected) == 7', 'assert len(expected) == 14')
    source = source.replace('exact_callback_functions_and_classes=7', 'exact_callback_functions_and_classes=14')
    source = source.replace("ref('scripts/probe_y87_native_tail.py')", "candidate['prepared_native_body']")
    source = source.replace('fourteen exact all-stage original confidence/max-overlap consumers and scalar projection references',
                            'twelve exact original self/language sum-product paths and original norm/FFN consumers')
    source = source.replace('seven-stage W2/input-gradient anchors, original consumers and two synthetic updates',
                            'seven-stage W2/input-gradient anchors, original self/language/norm/FFN consumers and two synthetic updates')
    source = source.replace('seven identical actually executed callbacks/nine guards', 'fourteen identical actually executed callbacks/nine guards')
    source = source.replace('GPU0-only native plan with seven identical', 'GPU0-only native plan with fourteen identical')
    source = source.replace('Only fresh exclusive authorized GPU0 permits this exact native YAML/TRAIN two-step task.',
                            'Y86 precedes Y87. Only fresh exclusive authorized GPU0 permits this exact native YAML/TRAIN two-step task.')
    ast.parse(source, feature_version=(3, 8))
    output = SCRIPTS / 'retain_y87_cpu_native.py'
    assert not output.exists()
    output.write_text(source, encoding='utf-8', newline='\n')
    print('Created Y87 retention/native-plan specialization; old helper unchanged.')


if __name__ == '__main__':
    create()
