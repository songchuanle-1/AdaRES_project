"""Mechanically isolate W2 and change only auxiliary point-count targets."""
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
from research_tick_4567 import ROOT, TOKEN, lease, ref, tick, monitor
import yaml

lease()
SOURCE = ROOT / 'AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition'
DEST = ROOT / 'AdaRES/ipdn/model_paper5/adares_y52_point_target'
CONFIG = ROOT / 'AdaRES/configs_song/model_paper5/adares_y52_point_target_res.yaml'
assert not DEST.exists() and not CONFIG.exists()
loss_tree = ast.parse((SOURCE / 'loss.py').read_text(encoding='utf-8'))
model_tree = ast.parse((SOURCE / 'ipdn.py').read_text(encoding='utf-8'))
count_source = '''
def expression_point_targets(superpoints, batch_offsets, scenes_len, gt_pmasks):
    targets = []
    expression = 0
    for scene, count in enumerate(scenes_len):
        start, end = int(batch_offsets[scene]), int(batch_offsets[scene + 1])
        selected = (superpoints >= start) & (superpoints < end)
        membership = superpoints[selected].long() - start
        total = torch.bincount(membership, minlength=end - start).float()
        for _ in range(int(count)):
            target = gt_pmasks[expression].reshape(-1)
            if target.numel() != membership.numel():
                raise ValueError('Point target and scene membership disagree')
            positive = torch.bincount(membership, weights=(target > .5).float(), minlength=end - start)
            targets.append((total, positive))
            expression += 1
    if expression != len(gt_pmasks):
        raise ValueError('Expression point-target count mismatch')
    return targets
'''
quality_source = '''
def get_point_iou(inputs, total_counts, positive_counts):
    if inputs.shape[-1] != total_counts.numel() or total_counts.shape != positive_counts.shape:
        raise ValueError('Point counts and superpoint masks disagree')
    binary = (inputs.sigmoid() >= .5).to(inputs.dtype)
    intersection = (binary * positive_counts).sum(-1)
    union = (binary * total_counts).sum(-1) + positive_counts.sum() - intersection
    return intersection / (union + 1e-6)
'''
model_tree.body.append(ast.parse(count_source).body[0])
loss_tree.body.append(ast.parse(quality_source).body[0])
ipdn = next(n for n in model_tree.body if isinstance(n, ast.ClassDef) and n.name == 'IPDN')
loss = next(n for n in ipdn.body if isinstance(n, ast.FunctionDef) and n.name == 'loss')
position = next(i for i, n in enumerate(loss.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'point_mass' for t in n.targets))
loss.body[position:position] = ast.parse('point_targets = expression_point_targets(superpoints, batch_offsets, scenes_len, gt_pmasks)').body
criterion_calls = [n for n in ast.walk(loss) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == 'self' and n.func.attr == 'criterion']
assert len(criterion_calls) == 1
criterion_calls[0].keywords.append(ast.keyword(arg='point_targets', value=ast.Name(id='point_targets', ctx=ast.Load())))
criterion = next(n for n in loss_tree.body if isinstance(n, ast.ClassDef) and n.name == 'Criterion')
masks = next(n for n in criterion.body if isinstance(n, ast.FunctionDef) and n.name == 'loss_masks')
targets = [n for n in ast.walk(masks) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'get_iou']
assert len(targets) == 1
targets[0].func.id = 'get_point_iou'
targets[0].args = [ast.Name(id='pred_mask', ctx=ast.Load()), ast.Starred(value=ast.Subscript(value=ast.Name(id='tgt', ctx=ast.Load()), slice=ast.Constant(value='point_counts'), ctx=ast.Load()), ctx=ast.Load())]
forward = next(n for n in criterion.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
forward.args.args.append(ast.arg(arg='point_targets'))
forward.args.defaults.append(ast.Constant(value=None))
position = next(i for i, n in enumerate(forward.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'num_insts' for t in n.targets))
forward.body[position:position] = ast.parse('''
if point_targets is None or len(point_targets) != len(target):
    raise ValueError('Exact RES point targets are required')
if any(len(ids) > 1 for ids in object_idss):
    raise ValueError('Point-target candidate is restricted to single-reference RES')
for item, counts in zip(target, point_targets):
    item['point_counts'] = counts
''').body
texts = {'loss.py': ast.unparse(ast.fix_missing_locations(loss_tree)) + '\n',
         'ipdn.py': ast.unparse(ast.fix_missing_locations(model_tree)) + '\n'}
for name, text in texts.items():
    compile(text, str(DEST / name), 'exec')
cfg_path = ROOT / 'AdaRES/configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml'
cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
assert cfg['model']['task_type'] == 'res'
lock = monitor.acquire_lock(str(ROOT))
assert lock is not None
try:
    lease()
    DEST.mkdir()
    files = []
    for source in sorted(SOURCE.glob('*.py')):
        path = DEST / source.name
        if source.name in texts:
            path.write_text(texts[source.name], encoding='utf-8', newline='\n')
        else:
            path.write_bytes(source.read_bytes())
        files.append(ref(path.relative_to(ROOT).as_posix()))
    CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')
    files.append(ref(CONFIG.relative_to(ROOT).as_posix()))
    assert len(files) == 7
    for filename, allowed in (('ipdn.py', {'loss'}), ('loss.py', {'forward', 'loss_masks'})):
        old_tree = ast.parse((SOURCE / filename).read_text(encoding='utf-8'))
        new_tree = ast.parse((DEST / filename).read_text(encoding='utf-8'))
        for old_cls in [n for n in old_tree.body if isinstance(n, ast.ClassDef)]:
            new_cls = next(n for n in new_tree.body if isinstance(n, ast.ClassDef) and n.name == old_cls.name)
            for old_method in [n for n in old_cls.body if isinstance(n, ast.FunctionDef) and n.name not in allowed]:
                new_method = next(n for n in new_cls.body if isinstance(n, ast.FunctionDef) and n.name == old_method.name)
                assert ast.dump(old_method, include_attributes=False) == ast.dump(new_method, include_attributes=False)
    record = 'pipeline/analyses/y52_prototype_bundle_4567_' + TOKEN + '.json'
    assert not (ROOT / record).exists()
    tick.save(ROOT / record, dict(complete=True, id='AdaRES-Y52', created_at_local=datetime.now().astimezone().isoformat(),
        token=TOKEN, files=files, source_files=[ref(p.relative_to(ROOT).as_posix()) for p in sorted(SOURCE.glob('*.py'))],
        code_status='local independent uncommitted prototype; not deployed, admitted or trained',
        method_scope='ONLY IPDN.loss, Criterion.forward, Criterion.loss_masks and two pure count helpers; other class methods exact AST matches',
        new_trainable_parameters=0, inference_code_and_decoder_byte_identical=True,
        CPU_physical_tests='pending; manifest is not a gate PASS'))
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y52')
    candidate['evidence'].extend([ref('scripts/probe_y52_train_cpu_4567.py'), ref('pipeline/analyses/y52_train_cpu_4567_' + TOKEN + '.json'), ref(record)])
    candidate.update(stage='implementation', next_action='Test exact isolated candidate loss and point-target plumbing on CPU against expanded-point reference and unchanged native BCE/Dice/indicator paths; then native matched TRAIN backward requires a genuinely idle exclusive GPU.')
    tick.save(ROOT / 'pipeline/research_work.json', work)
    print(json.dumps(dict(id=candidate['id'], stage=candidate['stage'], files=len(files), manifest=ref(record), new_trainable_parameters=0), indent=2))
finally:
    lock.close()
