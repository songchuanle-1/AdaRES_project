import functools
import pointgroup_ops
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_mean
import functools
import numpy as np

from ipdn.utils import cuda_cast
from .backbone import ResidualBlock, UBlock
from .loss import Criterion, get_iou

from .dec import DEC
from transformers import RobertaModel
from pointnet2.pointnet2_utils import FurthestPointSampling

class IPDN(nn.Module):
    def __init__(
        self,
        input_channel: int = 6,
        blocks: int = 5,
        block_reps: int = 2,
        media: int = 32,
        normalize_before=True,
        return_blocks=True,
        pool='mean',
        sampling_module=None,
        dec=None,
        criterion=None,
        test_cfg=None,
        norm_eval=False,
        infer_mode='pos',
        fps_num=512,
        fix_module=[],
        task_type='res',
    ):
        super().__init__()
        self.task_type = task_type

        # backbone and pooling
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_channel,
                media,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1',
            ))
        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        block_list = [media * (i + 1) for i in range(blocks)]
        self.unet = UBlock(
            block_list,
            norm_fn,
            block_reps,
            block,
            indice_key_id=1,
            normalize_before=normalize_before,
            return_blocks=return_blocks,
        )
        self.output_layer = spconv.SparseSequential(norm_fn(media), nn.ReLU(inplace=True))
        self.pool = pool

        self.decoder_param = dec
        self.fps_num = fps_num
        self.text_encoder = RobertaModel.from_pretrained('./backbones/roberta-base')

        self.sampling_module = sampling_module
        self.dec = DEC(**dec, sampling_module=sampling_module, in_channel=media)
        # criterion
        self.criterion = Criterion(**criterion)

        self.test_cfg = test_cfg
        self.norm_eval = norm_eval
        self.infer_mode = infer_mode
        self.init_weights()
        for module in fix_module:
            if '.' in module:
                module, params = module.split('.')
                module = getattr(self, module)
                params = getattr(module, params)
                for param in params.parameters():
                    param.requires_grad = False
            else:
                module = getattr(self, module)
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super(IPDN, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm1d only
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    def forward(self, batch, mode='loss'):
        if mode == 'loss':
            return self.loss(**batch)
        elif mode == 'predict':
            return self.predict(**batch)

    def get_batches(self, x, batch_offsets):
        B = len(batch_offsets) - 1
        max_len = max(batch_offsets[1:] - batch_offsets[:-1])
        new_feats = torch.zeros(B, max_len, x.shape[1]).to(x.device)
        mask = torch.ones(B, max_len, dtype=torch.bool).to(x.device)
        for i in range(B):
            start_idx = batch_offsets[i]
            end_idx = batch_offsets[i + 1]
            cur_len = end_idx - start_idx
            padded_feats = torch.cat([x[start_idx:end_idx], torch.zeros(max_len - cur_len, x.shape[1]).to(x.device)], dim=0)
            new_feats[i] = padded_feats
            mask[i, :cur_len] = False
        mask.detach()
        return new_feats, mask
    def match_sampled_ins(self, sampled_ins_lbl, obj_ids, sampled_pos, obj_pos):
        '''
        params:
            sampled_ins_lbl: [N_query,]
            obj_ids: list [N_gt]
            sampled_pos: [N_query, 3]
            obj_pos: [N_gt, 3]
        return:
            top1_ids: [N_gt,]
            in_obj_ids: list [N_gt,]
            matched_inds: list[(q_ids, tgt_ids)]
            tgt_counts: [N_gt,]
        '''
        matched_inds = []
        # compute the distance matrix
        dist_mat = torch.cdist(sampled_pos, obj_pos, p=2) # [N_query, N_gt]
        # top1 for each gt
        topk_inds = torch.topk(dist_mat, k=1, dim=0, largest=False, sorted=True)[1].squeeze(0) # [N_gt,]

        matched_inds.extend([(q_id[None], torch.tensor([tgt_id], dtype=torch.int64, device=sampled_pos.device)) for tgt_id, q_id in enumerate(topk_inds)])

        in_obj_ids = []
        for tgt_id, obj_id in enumerate(obj_ids):
            in_obj_id = (sampled_ins_lbl==obj_id).nonzero().squeeze(1) # tensor [N_query,]
            in_obj_id = in_obj_id[~np.isin(in_obj_id.cpu(),topk_inds.cpu())]
            in_obj_ids.append(in_obj_id)
            if in_obj_id.numel() == 0: continue
            else:
                matched_inds.extend([(q_id[None], torch.tensor([tgt_id], dtype=torch.int64, device=sampled_pos.device)) for q_id in in_obj_id])
        
        # matched_inds: [(q_id, tgt_id), ...]
        matched_inds_np = np.array([(matched_inds[i][0][0].cpu(),matched_inds[i][1][0].cpu()) for i in range(len(matched_inds))])
        tgt_ids = matched_inds_np[:, 1]
        _, tgt_counts = np.unique(tgt_ids, return_counts=True)

        matched_inds = torch.from_numpy(matched_inds_np).transpose(0,1).to(sampled_pos.device)
        matched_inds = (matched_inds[0], matched_inds[1])
        return topk_inds, in_obj_ids, matched_inds, tgt_counts




    @cuda_cast
    def loss(self, ann_ids, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape, feats, superpoints, batch_offsets, object_idss, gt_pmasks, gt_spmasks, sp_ref_masks, lang_tokenss, lang_masks, coords_float, sp_ins_labels, dense_maps, scenes_len=None, meta_datas=None, view_dependents=None, feats_2d=None):
        batch_size = len(batch_offsets) - 1
        # voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
        # input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

        # sp_feats = self.extract_feat(input, superpoints, p2v_map)
        sp_coords_float = scatter_mean(coords_float, superpoints, dim=0)

        _, sp_coords_float, fps_seed_sp, batch_offsets, sp_ins_labels, feats_2d = self.expand_and_fps(None, sp_coords_float, batch_offsets, sp_ins_labels, scenes_len, feats_2d, mode='tr')

        # lang_feats = self.text_encoder(lang_tokenss, attention_mask=lang_masks)[0]
        # out = self.dec(sp_feats, fps_seed_sp, batch_offsets, lang_feats, lang_masks, sp_coords_float, feats_2d)
        # loss, loss_dict = self.criterion(out, gt_spmasks, sp_ref_masks, object_idss, sp_ins_labels, dense_maps, lang_masks, fps_seed_sp, sp_coords_float, batch_offsets)
        bs = len(object_idss)
       
        # print(sp_coords_float)
        sp_coords_float,batch_mask = self.get_batches(sp_coords_float, batch_offsets)
        # print(bs)
        # print(sp_coords_float.size())
        # print(batch_mask.size())
        fps_seed_sp = fps_seed_sp.long()
        seed_pos = sp_coords_float.gather(1, fps_seed_sp.unsqueeze(-1).repeat(1, 1, 3))
        pad_masks = ~batch_mask

        true_point_num_list, all_point_num_list, matched_query_num_list, all_sample_num_list = [], [], [], []
        for b in range(bs):
            obj_ids = object_idss[b]
            sp_ins_label = sp_ins_labels[b]
            # print(obj_ids)
            if len(obj_ids)==0:
                true_point_num=0
                all_point_num = len(sp_ins_label)
                matched_query_num = 0
            else:

                f = torch.isin(sp_ins_label.detach().cpu(),torch.tensor(obj_ids))

                ## 占用点的数量
                true_point_num = torch.sum(f)
                all_point_num = len(sp_ins_label)
                true_point_num = int(true_point_num.detach().cpu())

                ### seed_pos
                seed_ins_lbl = sp_ins_labels[b][fps_seed_sp[b]]

                # masks
                masks = torch.stack([sp_ins_labels[b]==obj_id for obj_id in obj_ids])
                obj_pos = torch.stack([
                    scatter_mean(sp_coords_float[b].masked_select(pad_masks[b].unsqueeze(1)).view(-1,3), 
                                    mask.long(), 
                                    dim=0)[1]
                    for mask in masks
                ])
                topk_seed, in_obj_seed, matched_seed, tgt_counts_seed= self.match_sampled_ins(seed_ins_lbl, obj_ids, seed_pos[b], obj_pos)
                matched_query_num = len(matched_seed[0])
                

            all_sample_num = len(fps_seed_sp[b])
            true_point_num_list.append(true_point_num)
            all_point_num_list.append(all_point_num)
            matched_query_num_list.append(matched_query_num)
            all_sample_num_list.append(all_sample_num)
            # print(true_point_num,all_point_num,matched_query_num,all_sample_num)
            # print()
            # print()
            # print()
        return true_point_num_list, all_point_num_list, matched_query_num_list, all_sample_num_list
    
    @cuda_cast
    def predict(self, ann_ids, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape, feats, superpoints, batch_offsets, object_idss, gt_pmasks, gt_spmasks, sp_ref_masks, lang_tokenss, lang_masks, coords_float, sp_ins_labels, dense_maps, scenes_len=None, meta_datas=None, view_dependents=None, feats_2d=None):
        batch_size = len(batch_offsets) - 1
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        
        sp_feats = self.extract_feat(input, superpoints, p2v_map)
        sp_coords_float = scatter_mean(coords_float, superpoints, dim=0)

        sp_feats, sp_coords_float, fps_seed_sp, batch_offsets, sp_ins_labels, feats_2d = self.expand_and_fps(sp_feats, sp_coords_float, batch_offsets, sp_ins_labels, scenes_len, feats_2d, mode='pred')
        lang_feats = self.text_encoder(lang_tokenss, attention_mask=lang_masks)[0]
        
        out = self.dec(sp_feats, fps_seed_sp, batch_offsets, lang_feats, lang_masks, sp_coords_float, feats_2d) 
        ret = self.predict_by_feat(scan_ids, object_idss, ann_ids, out, superpoints, gt_pmasks, gt_spmasks, fps_seed_sp)
        if meta_datas[0] is not None: 
            ret['meta_datas'] = meta_datas
            if view_dependents[0] is not None:
                ret['view_dependents'] = view_dependents
        return ret
    
    def predict_by_feat(self, scan_ids, object_idss, ann_ids, out, superpoints, gt_pmasks, gt_spmasks, fps_seed_sp):
        # B is 1 when predecit
        spious, pious, sp_r_ious, p_r_ious, pred_pmasks, scan_idss = [], [], [], [], [], []
        seed_inds, sample_inds, indis = [], [], []
        nt_labels = []
        b = len(object_idss)
        for i in range(b):
            gt_pmask = gt_pmasks[i]
            gt_spmask = gt_spmasks[i]
            pred_spmask = out['masks'][i].squeeze()

            pred_indis = out['indis'][i] # [n_q, 2]
            # take the 1 
            if self.task_type == 'gres':
                indicate = pred_indis.argmax(-1) == 1
                top = indicate.nonzero(as_tuple=False).squeeze(-1)
                # nt_label
                is_nt = False
                if len(top)==0:
                    is_nt = True
                    piou = torch.tensor(0.0, device=pred_spmask.device)
                    spiou = torch.tensor(0.0, device=pred_spmask.device)
                    pred_spmask = torch.zeros_like(pred_spmask[0], device=pred_spmask.device)
                    pred_pmask = pred_spmask[superpoints]

                else:
                    top_mask = pred_spmask[top] # [k, n_sp]
                    pred_spmask = top_mask.max(0)[0] # [n_sp,]
                    spiou = get_iou(pred_spmask, gt_spmask)
                    pred_pmask = pred_spmask[superpoints]
                    piou = get_iou(pred_pmask, gt_pmask)
                    if not is_nt:
                        pred_pmask_binay = pred_pmask.sigmoid() > 0.5
                        is_nt = True if pred_pmask_binay.sum() < 50 else False
                nt_labels.append(is_nt)
            elif self.task_type == 'res':
                softmax_indis = F.softmax(pred_indis, dim=-1)
                indicate = softmax_indis[:,1]
                #print(indicate.max(), (indicate>0.75).sum())
                top = indicate.argmax()

                pred_spmask = pred_spmask[top]
                spiou = get_iou(pred_spmask, gt_spmask)
                pred_pmask = pred_spmask[superpoints]
                piou = get_iou(pred_pmask, gt_pmask)
            else:
                raise NotImplementedError

            spious.append(spiou.cpu())
            pious.append(piou.cpu())
            pred_pmasks.append(pred_pmask.sigmoid().cpu())
            scan_idss.append(scan_ids[0])
            seed_inds.append(fps_seed_sp[i].cpu())
            sample_inds.append(out['sample_inds'][i].cpu())
            indis.append(indicate.cpu())
        gt_pmasks = [gt_pmask.cpu() for gt_pmask in gt_pmasks]
        return dict(scan_id=scan_idss, object_ids=object_idss, ann_id=ann_ids, piou=pious, spiou=spious, gt_pmask=gt_pmasks, pred_pmask=pred_pmasks,
                    sp_r_iou=sp_r_ious, p_r_iou=p_r_ious, nt_label = nt_labels, seed_ind=seed_inds, sample_ind=sample_inds, indi=indis)

    def extract_feat(self, x, superpoints, v2p_map):
        # backbone
        x = self.input_conv(x)
        x, mutil_x = self.unet(x)
        x = self.output_layer(x)
        x = x.features[v2p_map.long()]  # (B*N, media)

        # superpoint pooling
        if self.pool == 'mean':
            x = scatter_mean(x, superpoints, dim=0)  # (B*M, media)
        elif self.pool == 'max':
            x, _ = scatter_max(x, superpoints, dim=0)  # (B*M, media)
        return x
    
    def expand_and_fps(self, sp_feats, sp_coords_float, batch_offsets, sp_ins_labels, scenes_len, feats_2d, mode=None):
        sp_ins_labels_expand = []
        #feats_2d_expand = feats_2d
        
        batch_offsets_expand = batch_offsets[0:1]
        for i in range(len(scenes_len)):
            s = batch_offsets[i]
            e = batch_offsets[i+1]
            if i==0:
                # sp_feats_expand = sp_feats[s:e].repeat(scenes_len[i],1)
                sp_coords_float_expand = sp_coords_float[s:e].repeat(scenes_len[i],1)
                feats_2d_expand = feats_2d[s:e].repeat(scenes_len[i],1)
                fps_seed_sp = FurthestPointSampling.apply(sp_coords_float[s:e].unsqueeze(0), self.fps_num)
                fps_seed_sp_expand = fps_seed_sp.squeeze(0).repeat(scenes_len[i],1)
            else:
                # sp_feats_expand = torch.cat((sp_feats_expand, sp_feats[s:e].repeat(scenes_len[i],1)),dim=0)
                sp_coords_float_expand = torch.cat((sp_coords_float_expand, sp_coords_float[s:e].repeat(scenes_len[i],1)))
                feats_2d_expand = torch.cat((feats_2d_expand, feats_2d[s:e].repeat(scenes_len[i],1)),dim=0)
                fps_seed_sp = FurthestPointSampling.apply(sp_coords_float[s:e].unsqueeze(0), self.fps_num)
                fps_seed_sp_expand = torch.cat((fps_seed_sp_expand, fps_seed_sp.squeeze(0).repeat(scenes_len[i],1)),dim=0)
            for j in range(scenes_len[i]):
                batch_offsets_expand = torch.cat((batch_offsets_expand, batch_offsets_expand[-1:]+batch_offsets[i+1:i+2]-batch_offsets[i:i+1]), dim=0)
                if sp_ins_labels:
                    sp_ins_labels_expand.append(sp_ins_labels[i])
        return None, sp_coords_float_expand, fps_seed_sp_expand, batch_offsets_expand, sp_ins_labels_expand, feats_2d_expand
    
    