"""Specialize the proven read-only guarded runner without changing signed Y89 files."""
import ast
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def create():
    outputs = {}
    for original in ('run_y89_native.py','test_y89_runtime_plan.py'):
        source = (SCRIPTS/original).read_text(encoding='utf-8')
        text = source.replace('Y89','Y90').replace('y89','y90')
        ast.parse(text,feature_version=(3,8))
        outputs[SCRIPTS/original.replace('y89','y90')] = text
    assert all(not path.exists() for path in outputs)
    for path,text in outputs.items():
        path.write_text(text,encoding='utf-8',newline='\n')
    print('Created exact Y90 native runner and six matching-runtime rejection tests; original Y89 helpers unchanged.')


if __name__=='__main__':
    create()
