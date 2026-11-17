"""Import and construct exact independent model bytes on CPU, without forward."""
import base64
import gc
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import gorilla
import torch

assert P['physical_gpu'] is None
assert torch.cuda.device_count() == 0, 'This construction probe must remain CPU-only'
torch.set_num_threads(2)
directory = Path(tempfile.mkdtemp(prefix='adares_y52_construct_' + P['token'] + '_'))
package_directory = directory / 'adares_y52_point_target'
package_directory.mkdir()
for name, row in PAYLOAD['python_files'].items():
    data = base64.b64decode(row['base64'])
    assert hashlib.sha256(data).hexdigest() == row['sha256']
    (package_directory / name).write_bytes(data)
config_data = base64.b64decode(PAYLOAD['config']['base64'])
assert hashlib.sha256(config_data).hexdigest() == PAYLOAD['config']['sha256']
config_path = directory / 'adares_y52_point_target_res.yaml'
config_path.write_bytes(config_data)
cfg = gorilla.Config.fromfile(str(config_path))
assert cfg.train.seed == 1999 and cfg.train.epochs == 30 and cfg.model.task_type == 'res'
importlib.import_module('ipdn.model_paper5')
namespace = 'ipdn.model_paper5.adares_y52_point_target'
assert namespace not in sys.modules
spec = importlib.util.spec_from_file_location(namespace, str(package_directory / '__init__.py'),
    submodule_search_locations=[str(package_directory)])
module = importlib.util.module_from_spec(spec)
sys.modules[namespace] = module
spec.loader.exec_module(module)
Candidate = importlib.import_module(namespace + '.ipdn').IPDN
Parent = importlib.import_module('ipdn.model_paper5.adares_w2_pointmass_competition.ipdn').IPDN


def construct(model_class):
    gorilla.set_random_seed(1999)
    model = model_class(**cfg.model)
    assert all(parameter.device.type == 'cpu' for parameter in model.parameters())
    state = model.state_dict()
    digest = hashlib.sha256()
    shape = []
    for name, value in sorted(state.items()):
        tensor = value.detach().contiguous()
        shape.append([name, list(tensor.shape), str(tensor.dtype)])
        digest.update(name.encode('utf-8'))
        digest.update(tensor.numpy().tobytes())
    result = dict(parameters=sum(parameter.numel() for parameter in model.parameters()),
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        parameter_tensors=sum(1 for _ in model.parameters()), state_tensors=len(state),
        state_shapes_sha256=hashlib.sha256(json.dumps(shape, sort_keys=True).encode()).hexdigest(),
        exact_initial_state_sha256=digest.hexdigest(), class_module=model_class.__module__,
        all_parameters_CPU=True)
    del state, model
    gc.collect()
    return result


candidate = construct(Candidate)
parent = construct(Parent)
assert candidate['class_module'].startswith(namespace + '.')
for key in ('parameters', 'trainable_parameters', 'parameter_tensors', 'state_tensors',
            'state_shapes_sha256', 'exact_initial_state_sha256'):
    assert candidate[key] == parent[key], 'Constructor/state changed: ' + key
result = dict(complete=True, id='AdaRES-Y52', no_GPU=True, candidate=candidate, parent=parent,
    additional_parameters=candidate['parameters']-parent['parameters'],
    physical_namespace_independent=True, fixed_seed_initial_state_exact=True,
    actual_full_model_import_and_CPU_construction=True,
    native_forward_executed=False, native_backward_executed=False,
    released_backbone_training_executed=False, model_or_checkpoint_weights_saved=False,
    scientific_status='CPU_import_constructor_and_initialization_control_only',
    limitation='Full CPU construction does not certify sparse native TRAIN matching, backward, resource/restore gates, admission, trained efficacy or accuracy-resource Pareto superiority.')
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
