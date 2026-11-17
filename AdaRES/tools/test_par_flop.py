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
from thop import profile
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
    logger = get_root_logger_val(log_file=args.checkpoint.replace('.pth', '_parameters_gflops.log'))

    model = IPDN(**cfg.model).cuda()
    logger.info(f'Load state dict from {args.checkpoint}')
    meta = gorilla.load_checkpoint(model, args.checkpoint, strict=False)
    res = meta.get('meta')
   
    logger.info(f'Test pth epoch:{res.get("epoch",0)}, best val miou:{res.get("best_miou",0.0)}')
    print(f'Test pth epoch:{res.get("epoch",0)}, best val miou:{res.get("best_miou",0.0)}')
    count_parameters = gorilla.parameter_count(model)['']
    if gorilla.is_main_process():
        logger.info(f'Parameters: {count_parameters / 1e6:.2f}M')


    dataset = build_dataset(cfg.data.val, logger)
    dataloader = build_dataloader(dataset, training=False, **cfg.dataloader.val)

    scan_ids, object_ids, ann_ids, pious, spious, gt_pmasks, pred_pmasks = [], [], [], [], [], [], []
    meta_datas, view_dependents= [], []
    nt_labels = []
    
    seed_ind, sample_ind, indi = [], [], []
    with torch.no_grad():
        model.eval()
        sample_batch = next(iter(dataloader))
        # 将batch移动到模型所在设备
        device = next(model.parameters()).device
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, (sample_batch,'predict')).total()
        logger.info(f"FLOPs: {flops}")
        logger.info(f"FLOPs: {flops / 1e9:.2f} G")


if __name__ == '__main__':
    main()
