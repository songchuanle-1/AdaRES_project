"""Freeze exact native observation callbacks into a clearly synthetic CPU check."""
import argparse
import ast
import hashlib
from prepare_y79_query_language_feedback import ROOT, lease, ref, tick


def main(token):
    lease(token, 600)
    source = ROOT / ('pipeline/checks/y79_cpu_' + token + '_program.py')
    payload = ast.literal_eval(ast.parse(source.read_text(encoding='utf-8')).body[0].value)
    work = tick.load(ROOT / 'pipeline/research_work.json')
    candidate = next(c for c in work['candidates'] if c['id'] == 'AdaRES-Y79')
    body = ROOT / 'scripts/probe_y79_native_smoke.py'
    payload.update(code_commit=candidate['code_commit'], native_body_sha256=ref(body)['sha256'])
    header = (ROOT / 'scripts/probe_y79_cpu.py').read_text(encoding='utf-8').split('\ndec_module=')[0]
    callbacks = body.read_text(encoding='utf-8').split('losses=[]', 1)[1].split('\nhooks=', 1)[0]
    dependencies = '''
import itertools
cfg.dist=False
optimizer=gorilla.build_optimizer(model,cfg.optimizer)
scheduler=gorilla.build_lr_scheduler(optimizer,cfg.lr_scheduler)
loader=native.build_dataloader(dataset,dist=False,**cfg.dataloader.train)
'''
    tail = '''
hooks=[model.dec.register_forward_pre_hook(before_decoder),
    model.dec.sampling_module.register_forward_pre_hook(capture_sampling),feedback.register_forward_hook(capture_feedback),
    head.register_forward_pre_hook(before_indicator),head.query_norm.register_forward_hook(capture_query_norm)]
hooks.extend(m.register_forward_hook(capture_language_ffn) for m in model.dec.lia_ffn_layers)
hooks.extend(m.register_forward_pre_hook(before_qla) for m in model.dec.qla_layers)
hooks.extend(m.register_forward_pre_hook(before_lla) for m in model.dec.lla_layers)
model.dec.prediction_head=checked_prediction;optimizer.step=checked_step
model.train(True);torch.manual_seed(1999);lengths=(512,513);total=sum(lengths)
args=(torch.randn(total,cfg.model.media),torch.arange(cfg.model.fps_num).repeat(2,1),torch.tensor([0,lengths[0],total],dtype=torch.int32),
    torch.randn(2,9,model.dec.lang_proj.in_features),torch.ones(2,9,dtype=torch.long),torch.randn(total,3),
    torch.randn(total,model.dec.input_proj_2d[0].in_features),torch.arange(total).remainder(13).float()+1)
args[4][0,-2:]=0;args[3][0,-2:]=0
try:
    for index in range(2):
        optimizer.zero_grad();before_model(model,(batch,));torch.manual_seed(1999+index);value=model.dec(*args)
        parts=list(value['aux_outputs'])+[value];labels=torch.arange(256).reshape(2,128).remainder(2).long()
        loss=sum(torch.nn.functional.cross_entropy(p['indis'].reshape(-1,2),labels.reshape(-1))+p['masks'].square().mean() for p in parts)
        after_model(model,(batch,),(loss,));loss.backward();optimizer.step()
    check_response_rows(path_rows)
    assert len(losses)==len(gradient_checks)==len(modes)==len(step_rows)==2 and len(path_rows)==12 and not torch.cuda.is_initialized()
finally:
    optimizer.step=step;model.dec.prediction_head=original_prediction
    for hook in hooks:hook.remove()
    if getattr(loader,'_iterator',None) is not None:loader._iterator._shutdown_workers()
Path(P['output']).write_text(json.dumps(dict(complete=True,id=row['id'],code_commit=PAYLOAD['code_commit'],CPU_native_callbacks_executed=True,
    native_body_sha256=PAYLOAD['native_body_sha256'],configured_TRAIN_data_collated=True,released_backbone_loaded=True,full_model_CPU_constructed=True,
    train_points=int(batch['feats'].shape[0]),train_expressions=len(batch['ann_ids']),synthetic_CPU_decoder_optimizer_steps=2,
    synthetic_CPU_decoder_losses=losses,gradient_checks=gradient_checks,feedback_paths=path_rows,
    CUDA_initialized=False,native_sparse_task=False,native_smoke_passed=False,admission=False,
    limitation='Exact frozen native callback functions executed on two synthetic CPU decoder optimizer steps after actual full native model/pretrain/TRAIN loading; not GPU or native sparse TRAIN task execution.')),encoding='utf8')
'''
    program = 'PAYLOAD=' + repr(payload) + '\n' + header + dependencies + 'losses=[]' + callbacks + tail
    path = ROOT / ('pipeline/checks/y79_native_observer_CPU_' + token + '_program.py')
    assert not path.exists()
    tree = ast.parse(program, feature_version=(3, 8))
    native = ast.parse(body.read_text(encoding='utf-8'))
    functions = {n.name: ast.dump(n, include_attributes=False) for n in native.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    actual = {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert len(functions) == 16 and all(actual[k] == v for k, v in functions.items())
    path.write_text(program, encoding='utf-8', newline='\n')
    print(str(path.relative_to(ROOT)), hashlib.sha256(path.read_bytes()).hexdigest(), len(functions))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True)
    main(parser.parse_args().token)
