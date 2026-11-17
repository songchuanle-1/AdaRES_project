"""Specialize preserved server-1 execution helpers for the tested Y92 version."""
import argparse
import ast
from datetime import datetime
import re
from prepare_y60_stage_heads import ROOT, lease, monitor, ref, tick
from y91_deployment import Y90, Y91, git
import closure

Y92 = '27d6f67e7121a5168bbfeca9db745377e0bd49e5'
SYMBOLS = {'Y88': 'Y90', 'Y89': 'Y91', 'Y90': 'Y92',
           'y88': 'y90', 'y89': 'y91', 'y90': 'y92'}
PATTERN = re.compile('|'.join(SYMBOLS))
PARENT_FREEZE_SHA = 'ad60ab5a94db03b12405a043be738aae978786d560d3b707357150994ec95395'


def symbols(value):
    return PATTERN.sub(lambda match: SYMBOLS[match.group()], value)


class Specialize(ast.NodeTransformer):
    def __init__(self, name):
        self.name = name
        self.function = None

    def visit_FunctionDef(self, node):
        previous = self.function
        self.function = node.name
        node = self.generic_visit(node)
        self.function = previous
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            value = symbols(node.value).replace('query_peer_affinity', 'mask_feature_quadratic')
            if self.name == 'y92_deployment.py' and value == Y90:
                value = Y92
            if self.name == 'advance_y92.py':
                value = value.replace(
                    '93bf51b65a580cbdd5a7ade69fae370d8d0e6e6db5ab665645892c775d84aa81',
                    PARENT_FREEZE_SHA)
                value = value.replace('independent bounded query-pair weighting of original confidence peers',
                                      'independent visual-side quadratic static mask readout')
            if self.name == 'test_y92_execution_guards.py':
                if self.function == 'test_reversed_unknown_or_divergent_branches_refused' and value == 'AdaRES-Y91':
                    value = 'AdaRES-Y93'
                if self.function == 'test_modified_or_unrelated_source_refused' and value == 'AdaRES-Y91':
                    value = 'AdaRES-Y93'
            node.value = value
        return node

    def visit_Name(self, node):
        node.id = symbols(node.id)
        return node

    def visit_Attribute(self, node):
        node.attr = symbols(node.attr)
        return self.generic_visit(node)

    def visit_alias(self, node):
        node.name = symbols(node.name)
        return node

    def visit_ImportFrom(self, node):
        node.module = symbols(node.module).replace('query_peer_affinity', 'mask_feature_quadratic')
        return self.generic_visit(node)

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if self.name == 'y92_deployment.py' and self.function == 'observe':
            target = node.targets[0]
            if isinstance(target, ast.Tuple) and [item.id for item in target.elts] == ['name', 'commit', 'power']:
                node.value = ast.parse(
                    "{'AdaRES-Y91': ('server-2', Y91, 280.), "
                    "'AdaRES-Y92': ('server-1', Y92, 350.)}[cid]", mode='eval').body
        return node


def create(token):
    lease(token, 180)
    writer = monitor.acquire_lock(str(ROOT))
    assert writer is not None
    try:
        assert git('rev-parse', 'HEAD').decode().strip() == Y91
        assert git('rev-parse', Y92 + '^').decode().strip() == Y91
        work = tick.load(ROOT / 'pipeline/research_work.json')
        candidate = next(row for row in work['candidates'] if row['id'] == 'AdaRES-Y92')
        assert candidate['code_commit'] == Y92 and candidate['stage'] == 'blocked'
        assert candidate['blocker_kind'] == 'exclusive_gpu_slot'
        assert not candidate.get('native_smoke_result') and not candidate.get('admission_receipt')
        binding = tick.load(closure.artifact(ROOT, candidate['additive_version_binding']))
        assert binding['code_commit'] == Y92 and binding['parent_commit'] == Y91
        for dependency in binding['evidence']:
            closure.artifact(ROOT, dependency)
        before = {path: ref(path) for path in (
            'pipeline/queue.json', 'pipeline/running.json', 'pipeline/goal.json', 'SERVERS.yaml')}
        index = git('ls-files', '--stage', '-z')
        names = ('advance_y92.py', 'y92_deployment.py', 'test_y92_execution_guards.py')
        originals = [ref('scripts/' + name.replace('y92', 'y90')) for name in names]
        outputs = []
        for name in names:
            destination = ROOT / 'scripts' / name
            assert not destination.exists()
            source = (ROOT / 'scripts' / name.replace('y92', 'y90')).read_text(encoding='utf-8')
            parsed = Specialize(name).visit(ast.parse(source))
            text = ast.unparse(ast.fix_missing_locations(parsed)) + '\n'
            ast.parse(text, feature_version=(3, 8))
            destination.write_text(text, encoding='utf-8', newline='\n')
            outputs.append(ref(destination))
        assert [ref(item['path']) for item in originals] == originals
        assert all(ref(path) == value for path, value in before.items())
        assert git('ls-files', '--stage', '-z') == index
        output = ROOT / ('pipeline/checks/y92_execution_helpers_' + token + '.json')
        assert not output.exists()
        tick.save(output, dict(complete=True, token=token, id=candidate['id'], code_commit=Y92,
            prepared_at=datetime.now().astimezone().isoformat(),
            server='server-1', physical_gpu=0, power_limit_watts=350.,
            originals=originals, evidence=[ref(__file__)] + outputs,
            original_helpers_unchanged=True, native_executed=False, admitted=False,
            production_HEAD_preserved=Y91, index_queue_running_preserved=True,
            scope='Server1 execution preparation only, not a runtime activation or resource substitution. '
                'Server2 original native alternative remains immutable; this helper does not consume it.'))
        candidate['evidence'].append(ref(output))
        tick.save(ROOT / 'pipeline/research_work.json', work)
        print(__import__('json').dumps(dict(helpers=ref(output), native_executed=False, admitted=False)))
    finally:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    create(parser.parse_args().token)
