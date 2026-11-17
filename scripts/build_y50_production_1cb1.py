"""Mechanically derive an independent Y50 package from bound native W2 bytes."""
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, 'C:/Users/chuanle song/.codex/skills/songcl_research_pipeline')
import foreground_tick as tick
TOKEN = '1cb15474bfd346dcbc34837f3e9b79d3'
owner = tick.load(ROOT / 'pipeline/headless/.foreground.lockdir/owner.json')
assert owner['token'] == TOKEN and owner['thread_id'] == os.environ['CODEX_THREAD_ID']
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner['deadline_at'])
diagnosis = tick.load(ROOT / ('pipeline/analyses/y50_diagnosis_1cb1_' + TOKEN + '.json'))
assert diagnosis['complete'] and diagnosis['maximum_actual_parent_indicator_error'] == 0
source = ROOT / 'scripts/y50_joint_incidence_1cb1.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == diagnosis['extension_sha256']
parent = ROOT / 'AdaRES/ipdn/model_paper5/adares_y49_resolvent_coherence'
assert hashlib.sha256((parent / 'dec.py').read_bytes()).hexdigest() == '50cfd1e821c188ddcb44bb929a6ffb84a186f0af89e1deb002aae10ca7e3bf31'
tree = ast.parse((parent / 'dec.py').read_text())
tree.body = tree.body[4:]
removed = {'reference_inverse', 'inverse_diagonal', 'descriptors', 'GlobalResolventReadout', 'FinalResolventAttachment'}
tree.body = [n for n in tree.body if not (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in removed)
             and not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'forest_counts' for t in n.targets))]
decoder = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DEC')
constructor = next(n for n in decoder.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
assert ast.unparse(constructor.body[-1]) == 'self._resolvent_attachment.enabled = True'
constructor.body = constructor.body[:-3]
helper_nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
attachment_source = ROOT / 'scripts/y50_attachment_1cb1.py'
helper_nodes += ast.parse(attachment_source.read_text()).body
insert = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.ClassDef))
tree.body[insert:insert] = helper_nodes
constructor.body.extend(ast.parse('with torch.random.fork_rng(devices=[]):\n    self.mask_joint_readout=JointIncidenceReadout(d_model)\nself._joint_attachment=FinalJointIncidenceAttachment(self,self.mask_joint_readout)\nself._joint_attachment.enabled=True\n').body)
text = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
compile(text, '<independent-y50>', 'exec')
package = ROOT / 'AdaRES/ipdn/model_paper5/adares_y50_joint_incidence'
assert not package.exists()
package.mkdir()
records = []
for name in ('__init__.py', 'backbone.py', 'dec.py', 'ipdn.py', 'loss.py', 'sample_model.py'):
    data = text.encode('utf-8') if name == 'dec.py' else (parent / name).read_bytes()
    path = package / name
    with path.open('xb') as stream:
        stream.write(data)
    records.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(data).hexdigest()))
config = ROOT / 'AdaRES/configs_song/model_paper5/adares_y50_joint_incidence_res.yaml'
data = (ROOT / 'AdaRES/configs_song/model_paper5/adares_y49_resolvent_coherence_res.yaml').read_bytes()
with config.open('xb') as stream:
    stream.write(data)
records.append(dict(path=config.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(data).hexdigest()))
manifest = ROOT / ('pipeline/analyses/y50_production_bundle_1cb1_' + TOKEN + '.json')
tick.save(manifest, dict(complete=True, id='AdaRES-Y50', created_at=datetime.now().astimezone().isoformat(), files=records,
                         source_sha256=diagnosis['extension_sha256'], helper_AST_identical=True,
                         native_parent_binding='SAME independently tested Y49 nativeparent, remove all Y49 helper AST and three constructor statements.',
                         no_shared_entry_change=True, no_admission=True))
archive = manifest.with_suffix('.tar')
with tarfile.open(archive, 'x') as stream:
    for row in records:
        stream.add(ROOT / row['path'], arcname=row['path'])
sha = records[2]['sha256']
mapping = {'AdaRES-Y46': 'AdaRES-Y50',
           'configs_song/model_paper5/adares_y46_component_coherence_res.yaml': 'configs_song/model_paper5/adares_y50_joint_incidence_res.yaml',
           'ipdn/model_paper5/adares_y46_component_coherence/dec.py': 'ipdn/model_paper5/adares_y50_joint_incidence/dec.py',
           'adares_y46_component_coherence': 'adares_y50_joint_incidence',
           '7dd952f18632392f5e5a050b4aabcfd10de5bcfdad9c81f494de353dfa56cf8f': sha,
           '363506506bf0e289d67dfe72c1a0041e66a67fd1962247a726c34d47b838c225': sha,
           'dec.mask_component_readout.': 'dec.mask_joint_readout.'}


class Bind(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value in mapping:
            return ast.copy_location(ast.Constant(value=mapping[node.value]), node)
        if isinstance(node.value, int) and not isinstance(node.value, bool) and node.value == 328961:
            return ast.copy_location(ast.Constant(value=66561), node)
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        node.attr = {'mask_component_readout': 'mask_joint_readout', '_component_attachment': '_joint_attachment',
                     'prototype': 'readout', 'cached_inputs': 'cached', 'residuals': 'cached'}.get(node.attr, node.attr)
        return node


programs = []
for kind, template in (('native', 'probe_y46_production_native_131e.py'), ('controls', 'probe_y46_production_controls_f56b.py'), ('stress', 'probe_y46_revision2_stress_f56b.py')):
    tree = Bind().visit(ast.parse((ROOT / 'scripts' / template).read_text()))
    if kind == 'native':
        # Only four new tensors; preserve unrelated four-scene and two-batch values.
        class Tensors(ast.NodeTransformer):
            def visit_Compare(self, node):
                self.generic_visit(node)
                if len(node.comparators) == 1 and isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value == 8:
                    node.comparators[0] = ast.Constant(value=4)
                return node
        tree = Tensors().visit(tree)
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'result' for t in node.targets):
                for keyword in node.value.keywords:
                    if keyword.arg == 'new_parameter_tensors':
                        keyword.value = ast.Constant(value=4)
    if kind == 'controls':
        class Cached(ast.NodeTransformer):
            def visit_Compare(self, node):
                self.generic_visit(node)
                if ast.unparse(node.left) == 'len(gradients)' and len(node.comparators) >= 1 and isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value == 8:
                    node.comparators[0] = ast.Constant(value=4)
                return node
            def visit_Assign(self, node):
                if isinstance(node.targets[0], ast.Tuple) and ast.unparse(node.targets[0]) == '(rows, weights, groups)':
                    return ast.Pass()
                if isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'wrong':
                    return ast.parse('wrong=head(query,values.roll(1,dims=0))').body[0]
                return self.generic_visit(node)
        tree = Cached().visit(tree)
    text = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    compile(text, '<y50-' + kind + '>', 'exec')
    path = ROOT / ('pipeline/analyses/y50_production_' + kind + '_program_1cb1_' + TOKEN + '.py')
    with path.open('x', encoding='utf-8') as stream:
        stream.write(text)
    programs.append(dict(kind=kind, path=path.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         template_sha256=hashlib.sha256((ROOT / 'scripts' / template).read_bytes()).hexdigest()))
prereg = ROOT / ('pipeline/analyses/y50_production_tests_prereg_1cb1_' + TOKEN + '.json')
tick.save(prereg, dict(complete=True, id='AdaRES-Y50', preregistered_at=datetime.now().astimezone().isoformat(), programs=programs,
                       recipe='SAME fixed TRAIN8/serial2warmup3readings/largepair728,6780 stress plus released-init actual tools.train2steps and full exact restore.',
                       controls='Actual parent AST/config/initialization/RNG; only final foreground score changes; cached-repeat exact and wrong descriptor correspondence; all4 native gradients/updates.',
                       no_validation_selection=True, no_admission=True))
print(json.dumps(dict(bundle=str(manifest), decoder_sha256=sha, programs=programs), indent=2))
