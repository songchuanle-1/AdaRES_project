import argparse
import gorilla, os
import os.path as osp
import torch
from tqdm import tqdm
import numpy as np
from ipdn.dataset import build_dataloader, build_dataset
from ipdn import IPDN
from ipdn.utils.mask_encoder import rle_decode, rle_encode
from ipdn.utils import get_root_logger_val, save_pred_instances
import json
# torch.cuda.set_device(1)
def get_mask(spmask, superpoint):
    mask = spmask[superpoint]
    return mask

def _print_results_acc(iou_25, iou_50, mious, logger):
    logger.info(f"{'=' * 100}")
    logger.info("{0:<12}{1:<12}{2:<12}{3:<12}{4:<12}{5:<12}{6:<12}"
               .format("IoU", "zt_w_d", "zt_wo_d", "st_w_d", "st_wo_d", "mt", "overall"))
    logger.info(f"{'-' * 100}")
    line_1_str = '{:<12}'.format("0.25")
    for sub_group_type, score in iou_25.items():
        line_1_str += '{:<12.1f}'.format(score * 100)
    logger.info(line_1_str)
    line_2_str = '{:<12}'.format("0.50")
    for sub_group_type, score in iou_50.items():
        line_2_str += '{:<12.1f}'.format(score * 100)
    logger.info(line_2_str)
    line_3_str = '{:<12}'.format("miou")
    for sub_group_type, score in mious.items():
        line_3_str += '{:<12.1f}'.format(score * 100)
    logger.info(line_3_str)
    logger.info(f"{'=' * 100}\n")
    
def decode_stimulus_string(s):
    """
    Split into scene_id, instance_label, # objects, target object id,
    distractors object id.

    :param s: the stimulus string
    """
    if len(s.split('-', maxsplit=4)) == 4:
        scene_id, instance_label, n_objects, target_id = \
            s.split('-', maxsplit=4)
        distractors_ids = ""
    else:
        scene_id, instance_label, n_objects, target_id, distractors_ids = \
            s.split('-', maxsplit=4)

    instance_label = instance_label.replace('_', ' ')
    n_objects = int(n_objects)
    target_id = int(target_id)
    distractors_ids = [int(i) for i in distractors_ids.split('-') if i != '']
    assert len(distractors_ids) == n_objects - 1

    return scene_id, instance_label, n_objects, target_id, distractors_ids


def get_args():
    parser = argparse.ArgumentParser('SoftGroup')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('--checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--filename', type=str, default='best.pth', help='path to checkpoint')
    parser.add_argument('--out', default=None, type=str, help='directory for output results')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--gpu_ids', type=int, default=[0,1], nargs='+', help='ids of gpus to use')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    gorilla.set_cuda_visible_devices(gpu_ids=args.gpu_ids, num_gpu=args.num_gpus)

    cfg = gorilla.Config.fromfile(args.config)
    # args.checkpoint=os.path.join(cfg.work_dir, osp.splitext(osp.basename(args.config))[0], args.filename)
    gorilla.set_random_seed(cfg.test.seed)
    # logger = get_root_logger_val(log_file=args.checkpoint.replace('.pth', '_scanrefer.log'))
    save_file = args.checkpoint.replace('best.pth', 'best0.pth')
    print(save_file)
    # pp
    # lll = 'dec.sem_cls_heads.0.net.0.weight, dec.sem_cls_heads.0.net.1.weight, dec.sem_cls_heads.0.net.1.bias, dec.sem_cls_heads.0.net.1.running_mean, dec.sem_cls_heads.0.net.1.running_var, dec.sem_cls_heads.0.net.1.num_batches_tracked, dec.sem_cls_heads.0.net.4.weight, dec.sem_cls_heads.0.net.5.weight, dec.sem_cls_heads.0.net.5.bias, dec.sem_cls_heads.0.net.5.running_mean, dec.sem_cls_heads.0.net.5.running_var, dec.sem_cls_heads.0.net.5.num_batches_tracked, dec.sem_cls_heads.0.net.8.weight, dec.sem_cls_heads.0.net.8.bias, dec.sem_cls_heads.1.net.0.weight, dec.sem_cls_heads.1.net.1.weight, dec.sem_cls_heads.1.net.1.bias, dec.sem_cls_heads.1.net.1.running_mean, dec.sem_cls_heads.1.net.1.running_var, dec.sem_cls_heads.1.net.1.num_batches_tracked, dec.sem_cls_heads.1.net.4.weight, dec.sem_cls_heads.1.net.5.weight, dec.sem_cls_heads.1.net.5.bias, dec.sem_cls_heads.1.net.5.running_mean, dec.sem_cls_heads.1.net.5.running_var, dec.sem_cls_heads.1.net.5.num_batches_tracked, dec.sem_cls_heads.1.net.8.weight, dec.sem_cls_heads.1.net.8.bias, dec.sem_cls_heads.2.net.0.weight, dec.sem_cls_heads.2.net.1.weight, dec.sem_cls_heads.2.net.1.bias, dec.sem_cls_heads.2.net.1.running_mean, dec.sem_cls_heads.2.net.1.running_var, dec.sem_cls_heads.2.net.1.num_batches_tracked, dec.sem_cls_heads.2.net.4.weight, dec.sem_cls_heads.2.net.5.weight, dec.sem_cls_heads.2.net.5.bias, dec.sem_cls_heads.2.net.5.running_mean, dec.sem_cls_heads.2.net.5.running_var, dec.sem_cls_heads.2.net.5.num_batches_tracked, dec.sem_cls_heads.2.net.8.weight, dec.sem_cls_heads.2.net.8.bias, dec.sem_cls_heads.3.net.0.weight, dec.sem_cls_heads.3.net.1.weight, dec.sem_cls_heads.3.net.1.bias, dec.sem_cls_heads.3.net.1.running_mean, dec.sem_cls_heads.3.net.1.running_var, dec.sem_cls_heads.3.net.1.num_batches_tracked, dec.sem_cls_heads.3.net.4.weight, dec.sem_cls_heads.3.net.5.weight, dec.sem_cls_heads.3.net.5.bias, dec.sem_cls_heads.3.net.5.running_mean, dec.sem_cls_heads.3.net.5.running_var, dec.sem_cls_heads.3.net.5.num_batches_tracked, dec.sem_cls_heads.3.net.8.weight, dec.sem_cls_heads.3.net.8.bias, dec.sem_cls_heads.4.net.0.weight, dec.sem_cls_heads.4.net.1.weight, dec.sem_cls_heads.4.net.1.bias, dec.sem_cls_heads.4.net.1.running_mean, dec.sem_cls_heads.4.net.1.running_var, dec.sem_cls_heads.4.net.1.num_batches_tracked, dec.sem_cls_heads.4.net.4.weight, dec.sem_cls_heads.4.net.5.weight, dec.sem_cls_heads.4.net.5.bias, dec.sem_cls_heads.4.net.5.running_mean, dec.sem_cls_heads.4.net.5.running_var, dec.sem_cls_heads.4.net.5.num_batches_tracked, dec.sem_cls_heads.4.net.8.weight, dec.sem_cls_heads.4.net.8.bias, dec.sem_cls_heads.5.net.0.weight, dec.sem_cls_heads.5.net.1.weight, dec.sem_cls_heads.5.net.1.bias, dec.sem_cls_heads.5.net.1.running_mean, dec.sem_cls_heads.5.net.1.running_var, dec.sem_cls_heads.5.net.1.num_batches_tracked, dec.sem_cls_heads.5.net.4.weight, dec.sem_cls_heads.5.net.5.weight, dec.sem_cls_heads.5.net.5.bias, dec.sem_cls_heads.5.net.5.running_mean, dec.sem_cls_heads.5.net.5.running_var, dec.sem_cls_heads.5.net.5.num_batches_tracked, dec.sem_cls_heads.5.net.8.weight, dec.sem_cls_heads.5.net.8.bias, dec.lqg.0.attn.in_proj_weight, dec.lqg.0.attn.in_proj_bias, dec.lqg.0.attn.out_proj.weight, dec.lqg.0.attn.out_proj.bias, dec.lqg.0.norm.weight, dec.lqg.0.norm.bias, dec.lqg.1.attn.in_proj_weight, dec.lqg.1.attn.in_proj_bias, dec.lqg.1.attn.out_proj.weight, dec.lqg.1.attn.out_proj.bias, dec.lqg.1.norm.weight, dec.lqg.1.norm.bias, dec.lqg.2.attn.in_proj_weight, dec.lqg.2.attn.in_proj_bias, dec.lqg.2.attn.out_proj.weight, dec.lqg.2.attn.out_proj.bias, dec.lqg.2.norm.weight, dec.lqg.2.norm.bias, dec.lqg.3.attn.in_proj_weight, dec.lqg.3.attn.in_proj_bias, dec.lqg.3.attn.out_proj.weight, dec.lqg.3.attn.out_proj.bias, dec.lqg.3.norm.weight, dec.lqg.3.norm.bias, dec.lqg.4.attn.in_proj_weight, dec.lqg.4.attn.in_proj_bias, dec.lqg.4.attn.out_proj.weight, dec.lqg.4.attn.out_proj.bias, dec.lqg.4.norm.weight, dec.lqg.4.norm.bias, dec.lqg.5.attn.in_proj_weight, dec.lqg.5.attn.in_proj_bias, dec.lqg.5.attn.out_proj.weight, dec.lqg.5.attn.out_proj.bias, dec.lqg.5.norm.weight, dec.lqg.5.norm.bias, dec.sem_cls_head.net.0.weight, dec.sem_cls_head.net.1.weight, dec.sem_cls_head.net.1.bias, dec.sem_cls_head.net.1.running_mean, dec.sem_cls_head.net.1.running_var, dec.sem_cls_head.net.1.num_batches_tracked, dec.sem_cls_head.net.4.weight, dec.sem_cls_head.net.5.weight, dec.sem_cls_head.net.5.bias, dec.sem_cls_head.net.5.running_mean, dec.sem_cls_head.net.5.running_var, dec.sem_cls_head.net.5.num_batches_tracked, dec.sem_cls_head.net.8.weight, dec.sem_cls_head.net.8.bias, dec.indi_norm.weight, dec.indi_norm.bias'
    model = IPDN(**cfg.model).cuda()
    optimizer = gorilla.build_optimizer(model, cfg.optimizer)
    lr_scheduler = gorilla.build_lr_scheduler(optimizer, cfg.lr_scheduler)

    # logger.info(f'Load state dict from {args.checkpoint}')
    meta = gorilla.load_checkpoint(model, args.checkpoint, strict=False)
    print(meta['meta'])
    gorilla.save_checkpoint(model, save_file, optimizer,lr_scheduler, meta['meta'])

    # 

if __name__ == '__main__':
    main()
