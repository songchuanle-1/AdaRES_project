"""Mechanical specialization of the signed Y79 lifecycle helpers; retain originals."""
import ast
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def specialize():
    outputs = {}
    for original in ('finalize_y79_fixed30.py', 'prepare_y79_terminal_handoff.py', 'test_y79_terminal_handoff.py'):
        source = (SCRIPTS / original).read_text(encoding='utf-8')
        text = source.replace('Y79', 'Y82').replace('y79', 'y82')
        replacements = {
            'adares_y82_query_language_feedback': 'adares_y82_scale_product',
            '2218231': '1121817',
            '20261102_230148': '20261104_010148',
            'e0c7dba199fcf1a009941450d04aeb799418a7e6': '5773768ffb87d7fea7432e85fd0da561556aa2fc',
            '202eea9e539eb32bec278717496f68d223766663fa2ba936e7a16ef253185382': 'dd2f49300cf194b9dbe83a80145de6082d9f4a5d2e6bbf4d52d7061fe4445b5f',
            'scripts/y81_deployment.py': 'scripts/y83_deployment.py',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        if original.startswith('test_'):
            text = text.replace('import Y82,Y80,Y81,validate_marker', 'import Y82,Y83,validate_marker')
            text = text.replace('for value in (Y82,Y80,Y81)', 'for value in (Y82,Y83)')
            text = text.replace("for cid in ('AdaRES-Y80','AdaRES-Y81'):", "for cid in ('AdaRES-Y83',):")
        if original.startswith('finalize_'):
            anchor = "    closure.artifact(ROOT, candidate['post_training_freeze'])"
            assert text.count(anchor) == 1
            text = text.replace(anchor,
                "    assert candidate['post_training_freeze']['sha256'] == 'c294d2b9ef7014ee3c977a98bb1a8df7c0320e6d09ca5130fc98b181f34c2554'\n" + anchor)
        for unwanted in ('query_language_feedback', '2218231', '20261102_230148', 'AdaRES-Y79', 'y79_terminal'):
            assert unwanted not in text, unwanted
        ast.parse(text, feature_version=(3, 8))
        outputs[SCRIPTS / original.replace('y79', 'y82')] = text
    assert all(not path.exists() for path in outputs)
    for path, text in outputs.items():
        path.write_text(text, encoding='utf-8', newline='\n')
    print('Created exact Y82 finalizer, preparation and 25 lifecycle rejection controls.')


if __name__ == '__main__':
    specialize()
