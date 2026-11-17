"""Specialize established GPU0 lifecycle checks; never alter signed old helpers."""
import argparse
import ast
from pathlib import Path
from prepare_y60_stage_heads import lease, monitor, ROOT

SCRIPTS = Path(__file__).resolve().parent


def create(token):
    lease(token, 300)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        outputs = {}
        substitutions = {
            'adares_y85_stage_query_norm': 'adares_y85_mask_feature_norm',
            '20849': '22090',
            '20261103_175050': '20261104_160627',
            '2818322e6049b2541f1590a4fa9ad6854b3c598d': 'aa9d8db3ea05432181ada752eaeeca27952aae27',
            'e9f1a93e18201ff6902f7a478f7f2c78bec6661c06bf6a4bfe7a5ef821a6705a': 'c191a90dda4fdf0c2a1992ab5fca9d6164150c20025f6211cd2c3d9708f71c7a',
            '9d4da2437c977c4772b8105b7bdd83b421eaad349cdf37a7df273f05e94e9997': '4380dc32199f13af68e88833726f19399f3556d57446c2b8d62532298b4117bf',
            'scripts/y84_deployment.py': 'scripts/y85_deployment.py',
        }
        for filename in ('finalize_y83_fixed30.py', 'prepare_y83_terminal_handoff.py',
                         'test_y83_terminal_handoff.py'):
            source = (SCRIPTS / filename).read_text(encoding='utf-8')
            text = source.replace('Y83', 'Y85').replace('y83', 'y85')
            for old, new in substitutions.items():
                text = text.replace(old, new)
            if filename.startswith('test_'):
                text = text.replace('import Y85,Y84,validate_marker', 'import Y85,validate_marker')
                text = text.replace('for value in (Y85,Y84)', 'for value in (Y85,)')
                text = text.replace("for cid in ('AdaRES-Y84',):", "for cid in ('AdaRES-Y85',):")
            for stale in ('stage_query_norm', '20849', '20261103_175050', 'AdaRES-Y83',
                          'y83_terminal', '2818322e6049b2541f1590a4fa9ad6854b3c598d'):
                assert stale not in text, stale
            ast.parse(text, feature_version=(3, 8))
            outputs[SCRIPTS / filename.replace('y83', 'y85')] = text
        assert all(not path.exists() for path in outputs)
        for path, source in outputs.items():
            path.write_text(source, encoding='utf-8', newline='\n')
        print('Created three exact Y85 lifecycle helpers; historical files unchanged.')
    finally:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    create(parser.parse_args().token)
