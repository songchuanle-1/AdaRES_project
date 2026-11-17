"""Specialize the proven server1 lifecycle without modifying historical signed helpers."""
import ast
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def create():
    outputs = {}
    replacements = {
        'Y82': 'Y84', 'y82': 'y84', 'adares_y84_scale_product': 'adares_y84_visual_query_init',
        '1121817': '243952', '20261104_010148': '20261105_032303',
        '20261104_010138': '20261105_032256',
        '5773768ffb87d7fea7432e85fd0da561556aa2fc': '7bef2cd313b870583c970c56580be5a78097234d',
        'dd2f49300cf194b9dbe83a80145de6082d9f4a5d2e6bbf4d52d7061fe4445b5f': 'c8ba93aa472ad4c54bd4af1904b5205a7426c7650445cae391cb603cca75f8aa',
        'c294d2b9ef7014ee3c977a98bb1a8df7c0320e6d09ca5130fc98b181f34c2554': 'd3e38757764ede03fb84028fee837a8ed392d477fc32e0eb391d7676cb3b650c',
        'scripts/y83_deployment.py': 'scripts/y87_deployment.py',
    }
    for name in ('finalize_y82_fixed30.py', 'prepare_y82_terminal_handoff.py', 'test_y82_terminal_handoff.py'):
        text = (SCRIPTS / name).read_text(encoding='utf-8')
        for old, new in replacements.items():
            text = text.replace(old, new)
        if name.startswith('test_'):
            text = text.replace('import Y84,Y83,validate_marker', 'import Y84,Y85,Y87,validate_marker')
            text = text.replace('for value in (Y84,Y83)', 'for value in (Y84,Y85,Y87)')
            text = text.replace("for cid in ('AdaRES-Y83',):", "for cid in ('AdaRES-Y85','AdaRES-Y87'):")
        ast.parse(text, feature_version=(3, 8))
        outputs[SCRIPTS / name.replace('y82', 'y84')] = text
    text = (SCRIPTS / 'y82_terminal_contract.py').read_text(encoding='utf-8')
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace('from y83_deployment import Y84, Y83, contract, git, validate_delta, paths',
        'from y87_deployment import Y84, Y85, Y87, VERSIONS, contract, git, paths')
    text = text.replace('marker in (Y84, Y83)', 'marker in (Y84, Y85, Y87)')
    text = text.replace('    versions = (Y84, Y83)', '    versions = (Y84, Y85, Y87)')
    text += '''

def validate_delta(rows, cid):
    assert cid in ('AdaRES-Y85', 'AdaRES-Y87')
    assert len(rows) == 7 and all(len(row) == 2 and row[0] == 'A' for row in rows)
    assert {row[1] for row in rows} == set(paths(cid))
'''
    ast.parse(text, feature_version=(3, 8))
    outputs[SCRIPTS / 'y84_terminal_contract.py'] = text
    assert all(not path.exists() for path in outputs)
    for path, text in outputs.items():
        path.write_text(text, encoding='utf-8', newline='\n')
    print('Created four exact Y84 server1 terminal helpers; original signed files unchanged.')


if __name__ == '__main__':
    create()
