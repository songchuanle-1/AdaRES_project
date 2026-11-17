"""One-time mechanical specialization; all signed historical lifecycle files stay intact."""
import ast
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def specialize():
    outputs = {}
    for filename in ('finalize_y80_fixed30.py', 'prepare_y80_terminal_handoff.py',
                     'test_y80_terminal_handoff.py'):
        source = (SCRIPTS / filename).read_text(encoding='utf-8')
        text = source.replace('Y80', 'Y83').replace('y80', 'y83')
        replacements = {
            'adares_y83_query_visual_feedback': 'adares_y83_stage_query_norm',
            '42226': '20849',
            '20261102_175917': '20261103_175050',
            '4e68cd84d56d890222adc2bec8ab59cb0c8715b6': '2818322e6049b2541f1590a4fa9ad6854b3c598d',
            '9ff6b809a01d04aa59be9d3d3ec300fffac6169195d50c085057532a764e7d3c': 'e9f1a93e18201ff6902f7a478f7f2c78bec6661c06bf6a4bfe7a5ef821a6705a',
            'scripts/y82_deployment.py': 'scripts/y84_deployment.py',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        if filename.startswith('test_'):
            text = text.replace('import Y83,Y81,Y82,validate_marker', 'import Y83,Y84,validate_marker')
            text = text.replace('for value in (Y83,Y81,Y82)', 'for value in (Y83,Y84)')
            text = text.replace("for cid in ('AdaRES-Y81','AdaRES-Y82'):", "for cid in ('AdaRES-Y84',):")
        if filename.startswith('finalize_'):
            anchor = "    closure.artifact(ROOT, candidate['post_training_freeze'])"
            assert text.count(anchor) == 1
            text = text.replace(anchor,
                "    assert candidate['post_training_freeze']['sha256'] == '9d4da2437c977c4772b8105b7bdd83b421eaad349cdf37a7df273f05e94e9997'\n" + anchor)
        for stale in ('query_visual_feedback', '42226', '20261102_175917', 'AdaRES-Y80', 'y80_terminal'):
            assert stale not in text, stale
        ast.parse(text, feature_version=(3, 8))
        outputs[SCRIPTS / filename.replace('y80', 'y83')] = text
    assert all(not path.exists() for path in outputs)
    for path, text in outputs.items():
        path.write_text(text, encoding='utf-8', newline='\n')
    print('Created exact Y83 lifecycle specializations; original signed helpers unchanged.')


if __name__ == '__main__':
    specialize()
