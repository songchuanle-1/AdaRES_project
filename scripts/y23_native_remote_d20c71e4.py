import os
import sys

ROOT = "/bak_home/songcl/Project/AdaRES_project/AdaRES"
sys.path[:0] = [ROOT, ROOT+"/tools", ROOT+"/ipdn/lib",
    "/home/songcl/.conda/envs/adares_project/lib/python3.8/site-packages",
    "/home/songcl/.local/lib/python3.8/site-packages/pointnet2-0.0.0-py3.8-linux-x86_64.egg"]
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.chdir(ROOT)

import copy
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import time
import types
import numpy as np
import torch
import torch.nn.functional as F
import gorilla
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import resolve_model_class, load_shape_compatible_pretrain, train
from probe_adares_y15_resume import rng, restore, differences, cpu_state
from ipdn.model_paper5.adares_y23_valid_token_support.loss import Criterion as NewCriterion
from ipdn.model_paper5.adares_w2_pointmass_competition.loss import Criterion as ParentCriterion

MODE="__MODE__"
assert subprocess.check_output(["hostname"]).decode().strip()=="dl01-MS-7C60"
lease=open("/bak_home/songcl/Project/AdaRES_project/.y15_gpu0_diagnostic.lock","a")
fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
used,util=[int(v.strip()) for v in subprocess.check_output(
    ["nvidia-smi","-i","0","--query-gpu=memory.used,utilization.gpu",
     "--format=csv,noheader,nounits"]).decode().strip().split(",")]
assert used<1024 and util<10
torch.set_num_threads(4)
gorilla.set_random_seed(1999)
CONFIG="configs_song/model_paper5/adares_y23_valid_token_support_res.yaml"
PARENT="configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml"
CHECKPOINT="/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth"
cfg=gorilla.Config.fromfile(CONFIG)
cls=resolve_model_class(CONFIG)
logger=get_root_logger()
result=dict(mode=MODE,complete=False,gpu_precheck=dict(memory_mib=used,util=util),
    parameters_added=0,limitations=["TRAIN diagnostics and bounded execution tests, not trained efficacy or full-dataloader replay."])

def unique_indices(dataset, count):
    rows=[];seen=set()
    for i,chunk in enumerate(dataset.scene_inputs):
        scan=chunk[0]["scene_id"]
        if scan not in seen:
            seen.add(scan);rows.append(i)
            if len(rows)==count:break
    assert len(rows)==count
    return rows

def load_model(config=CONFIG, checkpoint=True):
    model=resolve_model_class(config)(**cfg.model).cuda()
    if checkpoint:
        ck=torch.load(CHECKPOINT,map_location="cpu")
        state=ck["model"]
        loaded=model.load_state_dict(state,strict=False)
        assert not loaded.missing_keys and not set(loaded.unexpected_keys)-{
            "text_encoder.embeddings.position_ids","text_encoder.embeddings.token_type_ids"}
        del ck
    else:
        load_shape_compatible_pretrain(model,cfg.train.pretrain,None)
    return model

def contracts():
    new=NewCriterion(**cfg.model.criterion).cuda()
    old=ParentCriterion(**cfg.model.criterion).cuda()
    rows=[]
    keys=("positive_map","modify_positive_map","pron_positive_map","other_entity_map","rel_positive_map")
    for seed in range(32):
        torch.manual_seed(seed)
        q=F.normalize(torch.randn(2,5,64,device="cuda"),dim=-1).requires_grad_(True)
        tokens=F.normalize(torch.randn(2,7,64,device="cuda"),dim=-1).requires_grad_(True)
        lang=torch.ones(2,7,device="cuda",dtype=torch.long)
        targets=[]
        for b in range(2):
            item={k:torch.zeros(1,7,device="cuda") for k in keys}
            item["positive_map"][0,1]=1
            item["modify_positive_map"][0,2]=1
            item["rel_positive_map"][0,3]=1
            item["other_entity_map"][0,4]=1
            targets.append(item)
        indices=[(torch.tensor([1],device="cuda"),torch.tensor([0],device="cuda")) for _ in range(2)]
        norm=torch.tensor([2.],device="cuda")
        a=old.loss_sem_align(tokens,q,lang,targets,indices,norm)
        b=new.loss_sem_align(tokens,q,lang,targets,indices,norm)
        ag=torch.autograd.grad(a,(tokens,q),retain_graph=True)
        bg=torch.autograd.grad(b,(tokens,q),retain_graph=True)
        padded=torch.cat([tokens,torch.randn(2,3,64,device="cuda")],1)
        padded_targets=[{k:F.pad(v,(0,3)) for k,v in t.items()} for t in targets]
        padded_lang=F.pad(lang,(0,3))
        c=new.loss_sem_align(padded,q,padded_lang,padded_targets,indices,norm)
        cg=torch.autograd.grad(c,(tokens,q))
        rows.append(dict(all_valid_loss_difference=float((a-b).abs()),
            all_valid_gradient_max=max(float((x-y).abs().max()) for x,y in zip(ag,bg)),
            padding_extension_loss_difference=float((b-c).abs()),
            padding_extension_gradient_max=max(float((x-y).abs().max()) for x,y in zip(bg,cg))))
    failures=[]
    for name,lang in (("empty",torch.zeros(2,7,device="cuda",dtype=torch.long)),
                     ("nonprefix",torch.tensor([[1,1,0,1,0,0,0]]*2,device="cuda"))):
        try:
            new.loss_sem_align(tokens,q,lang,targets,indices,norm)
        except ValueError:
            failures.append(name)
        else:
            raise AssertionError(name+" did not fail closed")
    return dict(cases=len(rows),rows=rows,fail_closed=failures,
        all_valid_loss_max=max(r["all_valid_loss_difference"] for r in rows),
        all_valid_gradient_max=max(r["all_valid_gradient_max"] for r in rows),
        padding_extension_loss_max=max(r["padding_extension_loss_difference"] for r in rows),
        padding_extension_gradient_max=max(r["padding_extension_gradient_max"] for r in rows))

if MODE=="contracts":
    result.update(contracts())
else:
    dc=cfg.data.train.copy()
    if MODE in ("inference", "repeat_audit", "scene_repeat", "inference_profile"):
        dc.training=False;dc.aug=False;dc.with_elastic=False
    dataset=build_dataset(dc,logger)
    if MODE=="population":
        model=load_model()
        model.train()
        parent_sem=types.MethodType(ParentCriterion.loss_sem_align,model.criterion)
        native_sem=model.criterion.loss_sem_align
        rows=[]
        def capture_sem(self,tokens,queries,lang,targets,indices,norm):
            result=native_sem(tokens,queries,lang,targets,indices,norm)
            if not cache:
                cache["args"]=(tokens.detach(),queries.detach(),lang,targets,indices,norm)
            return result
        model.criterion.loss_sem_align=types.MethodType(capture_sem,model.criterion)
        for i in unique_indices(dataset,64):
            cache={}
            batch=dataset.collate_fn([dataset[i]])
            with torch.no_grad():model(batch,mode="loss")
            tokens,query,lang,targets,indices,norm=cache["args"]
            tokens=tokens.clone().requires_grad_(True)
            query=query.clone().requires_grad_(True)
            valid=lang.bool();pad=~valid
            replacement=torch.zeros_like(tokens);replacement[...,0]=1
            cf=torch.where(valid.unsqueeze(-1),tokens.detach(),replacement)
            a=parent_sem(tokens,query,lang,targets,indices,norm)
            b=native_sem(tokens,query,lang,targets,indices,norm)
            ac=parent_sem(cf,query,lang,targets,indices,norm)
            bc=native_sem(cf,query,lang,targets,indices,norm)
            ag=torch.autograd.grad(a,tokens,retain_graph=True)[0]
            bg,qg=torch.autograd.grad(b,(tokens,query))
            rows.append(dict(scan_id=batch["scan_ids"][0],expressions=len(batch["ann_ids"]),
                padding_tokens=int(pad.sum()),native_padding_gradient=float(ag[pad].norm()),
                candidate_padding_gradient=float(bg[pad].norm()),
                native_padding_cf=float((a-ac).abs()),candidate_padding_cf=float((b-bc).abs()),
                valid_gradient=float(bg[valid].norm()),query_gradient=float(qg.norm()),
                semantic_loss_change=float(b-a),finite=bool(torch.isfinite(bg).all() and torch.isfinite(qg).all())))
            del batch,tokens,query,a,b,ac,bc,ag,bg,qg,cache
            if len(rows)%8==0:print(json.dumps(dict(scans=len(rows))),flush=True)
        padded=[r for r in rows if r["padding_tokens"]]
        result.update(rows=rows,scans=64,expressions=sum(r["expressions"] for r in rows),
            padded_scans=len(padded),native_padding_gradient_nonzero=sum(r["native_padding_gradient"]>0 for r in padded),
            candidate_padding_gradient_max=max(r["candidate_padding_gradient"] for r in rows),
            candidate_padding_cf_max=max(r["candidate_padding_cf"] for r in rows),
            native_padding_cf_mean=statistics.mean(r["native_padding_cf"] for r in padded),
            all_valid_query_gradients_nonzero=all(r["valid_gradient"]>0 and r["query_gradient"]>0 for r in rows),
            all_finite=all(r["finite"] for r in rows))
    elif MODE=="inference_profile":
        indices=unique_indices(dataset,8)+[728]
        batches=[dataset.collate_fn([dataset[i]]) for i in indices]
        result["rows"]={}
        for name,config in (("candidate",CONFIG),("parent",PARENT)):
            model=load_model(config);model.eval();rows=[]
            with torch.no_grad():
                for batch in batches:
                    for _ in range(2):
                        gorilla.set_random_seed(1999);model(batch,mode="predict")
                    readings=[]
                    for _ in range(3):
                        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        gorilla.set_random_seed(1999);start=time.monotonic()
                        model(batch,mode="predict");torch.cuda.synchronize()
                        readings.append(dict(seconds=time.monotonic()-start,
                            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
                    rows.append(dict(scan_id=batch["scan_ids"][0],points=len(batch["superpoints"]),
                        expressions=len(batch["ann_ids"]),readings=readings))
            result["rows"][name]=rows
            del model;gc.collect();torch.cuda.empty_cache()
        result.update(scans=9,parameters=164629637,
            limitation="Same frozen checkpoint serial single-model inference timings, not fully trained accuracy-cost Pareto proof; 3 readings are not a stable p95 estimate.")
    elif MODE=="scene_repeat":
        index=next(i for i,chunk in enumerate(dataset.scene_inputs) if chunk[0]["scene_id"]=="scene0002_00")
        batch=dataset.collate_fn([dataset[index]])
        result["rows"]={}
        for name,config in (("candidate",CONFIG),("parent",PARENT)):
            model=load_model(config);model.eval()
            baseline=None;rows=[]
            with torch.no_grad():
                for repeat in range(16):
                    gorilla.set_random_seed(1999)
                    pred=model(batch,mode="predict")
                    if baseline is None:
                        baseline=pred
                    rows.append(dict(repeat=repeat,
                        binary_flips=sum(int(((a>=.5)!=(b>=.5)).sum()) for a,b in zip(pred["pred_pmask"],baseline["pred_pmask"])),
                        probability_max=max(float((a-b).abs().max()) for a,b in zip(pred["pred_pmask"],baseline["pred_pmask"])),
                        selection_flips=sum(int(a.argmax()!=b.argmax()) for a,b in zip(pred["indi"],baseline["indi"])),
                        seed_mismatches=sum(int((a!=b).sum()) for a,b in zip(pred["seed_ind"],baseline["seed_ind"])),
                        sample_mismatches=sum(int((a!=b).sum()) for a,b in zip(pred["sample_ind"],baseline["sample_ind"])),
                        piou=[float(v) for v in pred["piou"]]))
            result["rows"][name]=rows
            del model,baseline;gc.collect();torch.cuda.empty_cache()
        result.update(scene="scene0002_00",repeats_per_model=16,
            candidate_flipped_repeats=sum(r["binary_flips"]>0 for r in result["rows"]["candidate"]),
            parent_flipped_repeats=sum(r["binary_flips"]>0 for r in result["rows"]["parent"]),
            candidate_max_binary_flips=max(r["binary_flips"] for r in result["rows"]["candidate"]),
            parent_max_binary_flips=max(r["binary_flips"] for r in result["rows"]["parent"]),
            limitation="Fixed anomaly-scene repeat diagnosis, not a population noise bound. Initial failed repeat retained.")
    elif MODE=="repeat_audit":
        indices=unique_indices(dataset,8)
        batches=[dataset.collate_fn([dataset[i]]) for i in indices]
        model=load_model();model.eval()
        cached=[];candidate_rows=[]
        def clone_inputs(values):
            return tuple(v.detach().clone() if torch.is_tensor(v) else v for v in values)
        capture={}
        def prehook(module,args):
            capture["args"]=clone_inputs(args)
        hook=model.dec.register_forward_pre_hook(prehook)
        with torch.no_grad():
            for batch in batches:
                capture.clear();gorilla.set_random_seed(1999)
                native=model(batch,mode="predict")
                args=capture["args"]
                hook.remove()
                decode=model.dec(*args)
                repeat_decode=model.dec(*args)
                cached.append((args, {k:decode[k].detach().cpu().clone() for k in ("masks","indis","sample_inds")}))
                gorilla.set_random_seed(1999);repeated=model(batch,mode="predict")
                candidate_rows.append(dict(scan_id=batch["scan_ids"][0],
                    cached_repeat_mask_max=float((decode["masks"]-repeat_decode["masks"]).abs().max()),
                    cached_repeat_indicator_max=float((decode["indis"]-repeat_decode["indis"]).abs().max()),
                    native_repeat_binary_flips=sum(int(((a>=.5)!=(b>=.5)).sum()) for a,b in zip(native["pred_pmask"],repeated["pred_pmask"])),
                    native_repeat_selection_flips=sum(int(a.argmax()!=b.argmax()) for a,b in zip(native["indi"],repeated["indi"])),
                    native_repeat_seed_mismatches=sum(int((a!=b).sum()) for a,b in zip(native["seed_ind"],repeated["seed_ind"])),
                    native_repeat_sample_mismatches=sum(int((a!=b).sum()) for a,b in zip(native["sample_ind"],repeated["sample_ind"]))))
                hook=model.dec.register_forward_pre_hook(prehook)
        hook.remove();del model;gc.collect();torch.cuda.empty_cache()
        parent=load_model(PARENT);parent.eval()
        parent_rows=[]
        with torch.no_grad():
            for batch,(args,new),candidate in zip(batches,cached,candidate_rows):
                decoded=parent.dec(*args)
                gorilla.set_random_seed(1999);a=parent(batch,mode="predict")
                gorilla.set_random_seed(1999);b=parent(batch,mode="predict")
                parent_rows.append(dict(scan_id=batch["scan_ids"][0],
                    cached_parent_mask_max=float((decoded["masks"].cpu()-new["masks"]).abs().max()),
                    cached_parent_indicator_max=float((decoded["indis"].cpu()-new["indis"]).abs().max()),
                    cached_parent_sample_mismatches=int((decoded["sample_inds"].cpu()!=new["sample_inds"]).sum()),
                    native_repeat_binary_flips=sum(int(((x>=.5)!=(y>=.5)).sum()) for x,y in zip(a["pred_pmask"],b["pred_pmask"])),
                    native_repeat_selection_flips=sum(int(x.argmax()!=y.argmax()) for x,y in zip(a["indi"],b["indi"])),
                    native_repeat_seed_mismatches=sum(int((x!=y).sum()) for x,y in zip(a["seed_ind"],b["seed_ind"])),
                    native_repeat_sample_mismatches=sum(int((x!=y).sum()) for x,y in zip(a["sample_ind"],b["sample_ind"]))))
        result.update(rows=dict(candidate=candidate_rows,parent=parent_rows),scans=8,
            cached_parent_mask_max=max(r["cached_parent_mask_max"] for r in parent_rows),
            cached_parent_indicator_max=max(r["cached_parent_indicator_max"] for r in parent_rows),
            cached_parent_sample_mismatches=sum(r["cached_parent_sample_mismatches"] for r in parent_rows),
            candidate_native_repeat_binary_flips=sum(r["native_repeat_binary_flips"] for r in candidate_rows),
            parent_native_repeat_binary_flips=sum(r["native_repeat_binary_flips"] for r in parent_rows),
            limitation="Identical cached decoder input audit separates local inference source identity from inherited full-model repeat instability; initial inference failure remains preserved.")
    elif MODE=="inference":
        indices=unique_indices(dataset,8)
        batches=[dataset.collate_fn([dataset[i]]) for i in indices]
        model=load_model();model.eval()
        captures=[];repeat=[]
        with torch.no_grad():
            for batch in batches:
                gorilla.set_random_seed(1999);captures.append(model(batch,mode="predict"))
                gorilla.set_random_seed(1999);repeat.append(model(batch,mode="predict"))
        del model;gc.collect();torch.cuda.empty_cache()
        parent=load_model(PARENT);parent.eval()
        rows=[]
        with torch.no_grad():
            for batch,new,rep in zip(batches,captures,repeat):
                gorilla.set_random_seed(1999);old=parent(batch,mode="predict")
                rows.append(dict(scan_id=batch["scan_ids"][0],
                    parent_binary_flips=sum(int(((a>=.5)!=(b>=.5)).sum()) for a,b in zip(new["pred_pmask"],old["pred_pmask"])),
                    repeat_binary_flips=sum(int(((a>=.5)!=(b>=.5)).sum()) for a,b in zip(new["pred_pmask"],rep["pred_pmask"])),
                    parent_probability_max=max(float((a-b).abs().max()) for a,b in zip(new["pred_pmask"],old["pred_pmask"])),
                    repeat_probability_max=max(float((a-b).abs().max()) for a,b in zip(new["pred_pmask"],rep["pred_pmask"])),
                    selection_flips=sum(int(a.argmax()!=b.argmax()) for a,b in zip(new["indi"],old["indi"])),
                    sample_index_mismatches=sum(int((a!=b).sum()) for a,b in zip(new["sample_ind"],old["sample_ind"]))))
        result.update(rows=rows,scans=8)
    else:
        if MODE in ("stress", "parent_stress"):
            # Frozen prediction-blind 562-scan manifest's largest examples.
            indices=[728,6780,6271]
        else:
            indices=unique_indices(dataset,4)
        batches=[dataset.collate_fn([dataset[i] for i in indices[j:j+2]]) for j in range(0,len(indices),2)]
        model=load_model(PARENT if MODE=="parent_stress" else CONFIG, checkpoint=False)
        opt=gorilla.build_optimizer(model,copy.deepcopy(cfg.optimizer))
        sched=gorilla.build_lr_scheduler(opt,copy.deepcopy(cfg.lr_scheduler))
        def step(batch):
            model.train();opt.zero_grad(set_to_none=True)
            loss,terms=model(batch,mode="loss");loss.backward()
            assert bool(torch.isfinite(loss).all())
            gradients={n:float(p.grad.norm()) for n,p in model.named_parameters() if n.startswith("dec.") and p.grad is not None}
            assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.train.grad_clip);opt.step()
            return dict(loss=float(loss),decoder_gradient_tensors=len(gradients),
                decoder_gradient_nonzero=sum(v>0 for v in gradients.values()))
        result["parameters"]=sum(p.numel() for p in model.parameters())
        if MODE in ("stress", "parent_stress"):
            rows=[]
            for batch in batches:
                for _ in range(2):step(batch)
                readings=[]
                for _ in range(3):
                    opt.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats();start=time.monotonic()
                    value=step(batch);torch.cuda.synchronize()
                    readings.append(dict(seconds=time.monotonic()-start,
                        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,**value))
                rows.append(dict(scans=batch["scan_ids"],points=len(batch["superpoints"]),
                    superpoints=int(batch["batch_offsets"][-1]),readings=readings))
            result["rows"]=rows
        elif MODE=="backward":
            rows=[]
            candidate_sem=model.criterion.loss_sem_align
            parent_sem=types.MethodType(ParentCriterion.loss_sem_align,model.criterion)
            captured={}
            def audit_sem(self,tokens,query,lang,targets,indices,norm):
                value=candidate_sem(tokens,query,lang,targets,indices,norm)
                if not captured:
                    captured.update(tokens=tokens,query=query,lang=lang,
                        candidate=value,parent=parent_sem(tokens,query,lang,targets,indices,norm))
                return value
            model.criterion.loss_sem_align=types.MethodType(audit_sem,model.criterion)
            for batch in batches:
                captured.clear();model.train();opt.zero_grad(set_to_none=True)
                loss,_=model(batch,mode="loss")
                new_grad=torch.autograd.grad(captured["candidate"],captured["tokens"],retain_graph=True)[0]
                old_grad=torch.autograd.grad(captured["parent"],captured["tokens"],retain_graph=True)[0]
                query_grad=torch.autograd.grad(captured["candidate"],captured["query"],retain_graph=True)[0]
                padding=~captured["lang"].bool()
                loss.backward()
                gradients={n:float(p.grad.norm()) for n,p in model.named_parameters()
                    if n.startswith("dec.") and p.grad is not None}
                rows.append(dict(scans=batch["scan_ids"],padding_tokens=int(padding.sum()),
                    candidate_padding_gradient=float(new_grad[padding].norm()),
                    parent_padding_gradient=float(old_grad[padding].norm()),
                    gradient_intervention_norm=float((new_grad-old_grad).norm()),
                    valid_gradient=float(new_grad[~padding].norm()),query_gradient=float(query_grad.norm()),
                    finite=all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None),
                    decoder_gradient_tensors=len(gradients),
                    decoder_nonzero=sum(v>0 for v in gradients.values())))
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.train.grad_clip);opt.step()
            result["rows"]=rows
        else:
            class Batches:
                sampler=None
                def __len__(self):return len(batches)
                def __iter__(self):return iter(batches)
            class Writer:
                def add_scalar(self,*args,**kwargs):pass
            cfg.dist=False
            before=cpu_state(model);calls=[]
            hook=model.register_forward_hook(lambda module,args,out:calls.append(float(out[0])))
            train(1,model,Batches(),opt,sched,cfg,logger,Writer());hook.remove()
            result["native_train"]=dict(losses=calls,model_change=differences(before,cpu_state(model)))
            del before
            checkpoint="/tmp/y23_native_checkpoint_d20c71e4.pth"
            gorilla.save_checkpoint(model,checkpoint,opt,sched,dict(epoch=1,best_miou=0.0))
            saved=torch.load(checkpoint,map_location="cpu");rstate=rng()
            continued=step(batches[0]);final=cpu_state(model);finalopt=copy.deepcopy(opt.state_dict())
            del model,opt,sched;gc.collect();torch.cuda.empty_cache()
            model=load_model(checkpoint=False)
            opt=gorilla.build_optimizer(model,copy.deepcopy(cfg.optimizer))
            sched=gorilla.build_lr_scheduler(opt,copy.deepcopy(cfg.lr_scheduler))
            meta=gorilla.resume(model=model,filename=checkpoint,optimizer=opt,scheduler=sched,strict=False,map_location="cpu")
            allowed={"text_encoder.embeddings.position_ids","text_encoder.embeddings.token_type_ids"}
            result["before_continuation"]=dict(
                model=differences({k:v for k,v in saved["model"].items() if k not in allowed},cpu_state(model)),
                optimizer=differences(saved["optimizer"],opt.state_dict()),
                scheduler=differences(saved["scheduler"],sched.state_dict()),meta_epoch=int(meta["epoch"]))
            assert all(result["before_continuation"][k]["equal"] for k in ("model","optimizer","scheduler"))
            restore(rstate);resumed=step(batches[0])
            result["continuation"]=dict(loss_difference=resumed["loss"]-continued["loss"],
                model=differences(final,cpu_state(model)),optimizer=differences(finalopt,opt.state_dict()),
                gradients=resumed)
result["complete"]=True
Path("/tmp/y23_"+MODE+"_d20c71e4.json").write_text(json.dumps(result))
print(json.dumps({k:v for k,v in result.items() if k!="rows"}),flush=True)
lease.close()
