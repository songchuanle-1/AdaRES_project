import argparse
import datetime
import gorilla
import os
import os.path as osp
import shutil
import time
import torch
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm

from ipdn.dataset import build_dataloader, build_dataset
from ipdn import IPDN
# from ipdn.model import IPDN
from ipdn.utils import AverageMeter, get_root_logger
import json

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')  # 使用文件系统替代共享内存
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

def _print_results_acc(iou_25, iou_50, logger):
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
    logger.info(f"{'=' * 100}")

    print(f"{'=' * 100}")
    print("{0:<12}{1:<12}{2:<12}{3:<12}{4:<12}{5:<12}{6:<12}"
               .format("IoU", "zt_w_d", "zt_wo_d", "st_w_d", "st_wo_d", "mt", "overall"))
    print(f"{'-' * 100}")
    line_1_str = '{:<12}'.format("0.25")
    for sub_group_type, score in iou_25.items():
        line_1_str += '{:<12.1f}'.format(score * 100)
    print(line_1_str)
    line_2_str = '{:<12}'.format("0.50")
    for sub_group_type, score in iou_50.items():
        line_2_str += '{:<12.1f}'.format(score * 100)
    print(line_2_str)
    print(f"{'=' * 100}")
def get_args():
    parser = argparse.ArgumentParser('SPFormer')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('--resume', type=str, help='path to resume from')
    # parser.add_argument('--work_dir', type=str, help='working directory')

    parser.add_argument('--save_ckpt', type=bool,default=True, help='skip validation')
    parser.add_argument('--skip_validate', action='store_true', help='skip validation')
    parser.add_argument('--skip_training', action='store_true', help='skip training')
    parser.add_argument('--dist', action='store_true', help='if distributed')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--num_machines', type=int, default=1, help='number of machines')
    parser.add_argument('--machine_rank', type=int, default=0, help='rank of machine')
    parser.add_argument('--dist_url', type=str, default="auto", help='distributed training url')
    parser.add_argument('--gpu_ids', type=int, default=[0, 1], nargs='+', help='ids of gpus to use')

    args = parser.parse_args()
    return args

def list_avg(x):
    return sum(x) / len(x)
def dict_val(metric_dict):
    out = {}
    for k, v in metric_dict.items():
        out[k] = v.val
    return out

def train(epoch, model, dataloader, optimizer, lr_scheduler, cfg, logger, writer,save_name='data_train.json'):
    model.train()
    meter_dict = {}
    end = time.time()

    if dataloader.sampler is not None and cfg.dist:
        dataloader.sampler.set_epoch(epoch)
    progress_bar = tqdm(total=len(dataloader))
    true_point_num_list, all_point_num_list, matched_query_num_list, all_sample_num_list = [], [], [], []
    for i, batch in enumerate(dataloader, start=1):
        # if i==2:break
        true_point_num, all_point_num, matched_query_num, all_sample_num = model(batch, mode='loss')
        true_point_num_list+=true_point_num
        all_point_num_list+=all_point_num
        matched_query_num_list+=matched_query_num
        all_sample_num_list+=all_sample_num
        # print(true_point_num_list, all_point_num_list, matched_query_num_list, all_sample_num_list)
        progress_bar.update()
    meter_dict['true_point_num_list']=true_point_num_list
    meter_dict['all_point_num_list']=all_point_num_list
    meter_dict['matched_query_num_list']=matched_query_num_list
    meter_dict['all_sample_num_list']=all_sample_num_list


    save_file = osp.join(cfg.work_dir,save_name)
    # 保存数据到JSON文件
    with open(save_file, "w", encoding="utf-8") as file:
        json.dump(meter_dict, file, ensure_ascii=False, indent=4)























@torch.no_grad()
def eval(epoch, best_iou, model, dataloader, cfg, logger, writer, save_ckp=False):
    if gorilla.is_main_process():
        logger.info('Validation')
        pious, spious, sp_r_ious, p_r_ious, nt_labels, meta_datas, view_dependents, scan_ids = [], [], [], [], [], [], [], []
        progress_bar = tqdm(total=len(dataloader), desc=f'Val[{epoch}/{cfg.train.epochs}]', bar_format="{l_bar}{bar:20}{r_bar}" )

    model.eval()
    for batch in dataloader:
        result = model(batch, mode='predict')
        piou = result['piou']
        spiou = result['spiou']
        nt_label = result['nt_label']
        scan_id = result['scan_id']
        if gorilla.is_main_process():
            pious.extend(piou)
            spious.extend(spiou)
            nt_labels.extend(nt_label)
            scan_ids.extend(scan_id)
            if 'sp_r_iou' in result.keys():
                sp_r_ious.extend(result['sp_r_iou'])
                p_r_ious.extend(result['p_r_iou'])
            if 'meta_datas' in result.keys():
                meta_datas.extend(result['meta_datas'])
                #view_dependents.extend(result['view_dependents'])
            #else:
            #    print('No meta_datas')
            progress_bar.update()
    
    # evaluate
    miou = None
    if gorilla.is_main_process():
        progress_bar.close()
        logger.info('Evaluate referring segmentation: '+str(len(pious)))
        if len(nt_labels) > 0:
            eval_dict = {"zt_w_d": 0, "zt_wo_d": 1, "st_w_d": 2, "st_wo_d": 3, "mt": 4}
            eval_type_mask = np.empty(len(scan_ids))
            for idx, scan_id in enumerate(scan_ids):
                eval_type_mask[idx] = eval_dict[meta_datas[idx]['eval_type']]
                if nt_labels[idx]:
                    pious[idx] = torch.tensor(0.0)
                if meta_datas[idx]['eval_type'] in ("zt_wo_d", "zt_w_d"):
                    if nt_labels[idx]:
                        pious[idx] = torch.tensor(1.0)
                        spious[idx] = torch.tensor(1.0)
                    else:
                        pious[idx] = torch.tensor(0.0)
                        spious[idx] = torch.tensor(0.0)
            pious = torch.stack(pious, dim=0).cpu().numpy()
            acc_half_results = {}
            acc_quarter_results = {}
            for sub_group in ("zt_w_d", "zt_wo_d", "st_w_d", "st_wo_d", "mt"):
                selected_indices = eval_type_mask == eval_dict[sub_group]
                selected = pious[selected_indices]      
                acc_half_results[sub_group] = (selected > 0.5).sum().astype(float) / selected.size
                acc_quarter_results[sub_group] = (selected > 0.25).sum().astype(float) / selected.size
                writer.add_scalar('val/'+ sub_group + '_25', acc_quarter_results[sub_group], epoch)
                writer.add_scalar('val/'+ sub_group + '_50', acc_half_results[sub_group], epoch)
        else:        
            pious = torch.stack(pious, dim=0).cpu().numpy()
        
        precision_half = (pious > 0.5).sum().astype(float) / pious.size
        precision_quarter = (pious > 0.25).sum().astype(float) / pious.size
        miou = pious.mean()

        spious = torch.stack(spious, dim=0).cpu().numpy()
        spprecision_half = (spious > 0.5).sum().astype(float) / spious.size
        spprecision_quarter = (spious > 0.25).sum().astype(float) / spious.size
        spmiou = spious.mean()

        writer.add_scalar('val/mIOU', miou, epoch)
        writer.add_scalar('val/Acc_50', precision_half, epoch)
        writer.add_scalar('val/Acc_25', precision_quarter, epoch)
        writer.add_scalar('val/spmIOU', spmiou, epoch)
        writer.add_scalar('val/spAcc_50', spprecision_half, epoch)
        writer.add_scalar('val/spAcc_25', spprecision_quarter, epoch)
        
        logger.info(f'mIoU : {miou}')
        if len(nt_labels) > 0:
            acc_half_results["overall"] = precision_half
            acc_quarter_results["overall"] = precision_quarter
            _print_results_acc(acc_quarter_results, acc_half_results, logger)
        else:
            logger.info('mIOU: {:.3f}. Acc_50: {:.3f}. Acc_25: {:.3f}'.format(miou, precision_half,
                                                                    precision_quarter))
            print('mIOU: {:.3f}. Acc_50: {:.3f}. Acc_25: {:.3f}'.format(miou, precision_half,
                                                                    precision_quarter))
        logger.info('spmIOU: {:.3f}. spAcc_50: {:.3f}. spAcc_25: {:.3f}'.format(spmiou, spprecision_half,
                                                                      spprecision_quarter))
        print('spmIOU: {:.3f}. spAcc_50: {:.3f}. spAcc_25: {:.3f}'.format(spmiou, spprecision_half,
                                                                      spprecision_quarter))        
        if len(sp_r_ious) > 0:
            sp_r_ious = torch.stack(sp_r_ious, dim=0).cpu().numpy()
            p_r_ious = torch.stack(p_r_ious, dim=0).cpu().numpy()
            sp_rmiou = sp_r_ious.mean()
            p_rmiou = p_r_ious.mean()
            writer.add_scalar('val/sp_r_mIOU', sp_rmiou, epoch)
            writer.add_scalar('val/p_r_mIOU', p_rmiou, epoch)
            logger.info('sp_r_mIOU: {:.3f}'.format(sp_rmiou))
            logger.info('p_r_mIOU: {:.3f}'.format(p_rmiou))
            print('sp_r_mIOU: {:.3f}'.format(sp_rmiou))
            print('p_r_mIOU: {:.3f}'.format(p_rmiou))


    # if gorilla.is_main_process() and save_ckp:
    #     # save
    #     if miou > best_iou:
    #         save_file = osp.join(cfg.work_dir, f'best.pth')
    #         gorilla.save_checkpoint(model, save_file)
    #     elif epoch > cfg.train.save_epoch:
    #         save_file = osp.join(cfg.work_dir, f'epoch_{epoch:04d}.pth')
    #         gorilla.save_checkpoint(model, save_file)

    return miou


def main(args):
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    cfg = gorilla.Config.fromfile(args.config)
    cfg.dist = args.dist

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # if args.work_dir:
    if cfg.work_dir:
        cfg.work_dir = osp.join(cfg.work_dir, osp.splitext(osp.basename(args.config))[0], timestamp)
    else:
        cfg.work_dir = osp.join('./exps', osp.splitext(osp.basename(args.config))[0], timestamp)

    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    if gorilla.is_main_process():
        logger = get_root_logger(log_file=log_file)
        logger.info(f'config: {args.config}')
    else:
        logger = None
    shutil.copy(args.config, osp.join(cfg.work_dir, osp.basename(args.config)))
    writer = SummaryWriter(cfg.work_dir)

    # seed
    gorilla.set_random_seed(cfg.train.seed)

    # model
    model = IPDN(**cfg.model).cuda()
        # optimizer and scheduler
    optimizer = gorilla.build_optimizer(model, cfg.optimizer)
    lr_scheduler = gorilla.build_lr_scheduler(optimizer, cfg.lr_scheduler)

    # multi-gpu
    if args.num_gpus > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gorilla.get_local_rank()], find_unused_parameters=True)
    
    count_parameters = gorilla.parameter_count(model)['']
    if gorilla.is_main_process():
        logger.info(f'Parameters: {count_parameters / 1e6:.2f}M')


    # pretrain or resume
    start_epoch = 1
    best_miou = 0.0
    miou=0.0
    if args.resume:
        # resume_path = os.path.join(cfg.work_dir,args.resume+'.pth')
        resume_path = args.resume
        if gorilla.is_main_process(): logger.info(f'Resume from {resume_path}')
        meta = gorilla.resume(model=model, 
                              filename=resume_path, 
                              optimizer=optimizer, 
                              scheduler=lr_scheduler, 
                              strict=False,
                              map_location='cpu')
        start_epoch = meta.get("epoch", 0) + 1
        best_miou = meta.get("best_miou", 0)
        logger.info(f'resume start_epoch:{start_epoch},best_miou:{best_miou}!')
        print(f'resume start_epoch:{start_epoch},best_miou:{best_miou}!')
        # print(start_epoch)
    elif cfg.train.pretrain:
        if gorilla.is_main_process(): logger.info(f'Load pretrain from {cfg.train.pretrain}')
        gorilla.load_checkpoint(model, cfg.train.pretrain, strict=False, map_location='cpu')

    cfg.dataloader.train.batch_size /= args.num_gpus
    cfg.dataloader.train.batch_size = int(cfg.dataloader.train.batch_size)
    if gorilla.is_main_process():
        logger.info(f'Train batch size per gpu: {cfg.dataloader.train.batch_size}')
    # train and val dataset
    # if not args.skip_training:
    train_dataset = build_dataset(cfg.data.train, logger)
    train_loader = build_dataloader(train_dataset, dist=args.dist, **cfg.dataloader.train)
    # if not args.skip_validate and gorilla.is_main_process():
    val_dataset = build_dataset(cfg.data.val, logger)
    val_loader = build_dataloader(val_dataset, **cfg.dataloader.val)

    # train and val
    if gorilla.is_main_process():
        logger.info('Training')
    # for epoch in range(start_epoch, cfg.train.epochs + 1):
    #     # print(epoch)
    #     if not args.skip_training:
    # train(0, model, train_loader, optimizer, lr_scheduler, cfg, logger, writer)
    train(0, model, val_loader, optimizer, lr_scheduler, cfg, logger, writer,save_name='data_val.json')
       

if __name__ == '__main__':
    args = get_args()
    if args.num_gpus > 1:
        args.dist = True
    gorilla.set_cuda_visible_devices(gpu_ids=args.gpu_ids, num_gpu=args.num_gpus)
    gorilla.launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,) # use tuple to wrap
    )
    
