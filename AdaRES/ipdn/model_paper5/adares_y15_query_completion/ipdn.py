import functools
import pointgroup_ops
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_mean
import functools
import numpy as np
import copy
import hashlib
import random
from contextlib import contextmanager

from ipdn.utils import cuda_cast
from .backbone import ResidualBlock, UBlock
from .loss import Criterion, get_iou

from .dec import DEC
from transformers import RobertaModel
from pointnet2.pointnet2_utils import FurthestPointSampling


@contextmanager
def isolated_teacher_rng(device):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[device.index]):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)

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
        teacher_checkpoint=None,
        teacher_checkpoint_sha256=None,
    ):
        super().__init__()
        self.task_type = task_type
        self.teacher_checkpoint = teacher_checkpoint
        self.teacher_checkpoint_sha256 = teacher_checkpoint_sha256
        teacher_dec = copy.deepcopy(dec)
        teacher_dec.pop("completion_mode", None)
        teacher_dec.pop("completion_enabled", None)
        teacher_criterion = copy.deepcopy(criterion)
        teacher_criterion.pop("completion_target", None)
        self.teacher_kwargs = dict(
            input_channel=input_channel, blocks=blocks, block_reps=block_reps,
            media=media, normalize_before=normalize_before,
            return_blocks=return_blocks, pool=pool,
            sampling_module=copy.deepcopy(sampling_module), dec=teacher_dec,
            criterion=teacher_criterion, test_cfg=test_cfg, norm_eval=norm_eval,
            infer_mode=infer_mode, fps_num=fps_num, fix_module=fix_module,
            task_type=task_type,
        )
        object.__setattr__(self, "_training_teacher", None)

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
        return self

    def forward(self, batch, mode='loss'):
        if mode == 'loss':
            current = dict(batch)
            if self.criterion.completion_target == "teacher" and (
                    "completion_targets" not in current
                    and "teacher_masks" not in current):
                current["teacher_masks"] = self.training_teacher_masks(batch)
            return self.loss(**current)
        elif mode == 'predict':
            return self.predict(**batch)

    @torch.no_grad()
    def training_teacher_masks(self, batch):
        if self.teacher_checkpoint is None:
            raise ValueError("Teacher completion requires detached training targets")
        device = next(self.parameters()).device
        if device.type != "cuda":
            raise ValueError("Y15 teacher requires the native single-CUDA-device path")
        with isolated_teacher_rng(device):
            return self._training_teacher_masks(batch, device)

    def _training_teacher_masks(self, batch, device):
        teacher = self._training_teacher
        if teacher is None:
            if not self.teacher_checkpoint_sha256:
                raise ValueError("Teacher checkpoint identity must be frozen")
            digest = hashlib.sha256()
            with open(self.teacher_checkpoint, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != self.teacher_checkpoint_sha256:
                raise ValueError("Teacher checkpoint SHA256 mismatch")
            from .teacher_reference.ipdn import IPDN as TrainingReference
            teacher = TrainingReference(**self.teacher_kwargs).to(device)
            state = torch.load(self.teacher_checkpoint, map_location="cpu")
            incompatible = teacher.load_state_dict(state.get("model", state),
                                                   strict=False)
            allowed = {"text_encoder.embeddings.position_ids",
                       "text_encoder.embeddings.token_type_ids"}
            if incompatible.missing_keys or set(incompatible.unexpected_keys) - allowed:
                raise ValueError(str(incompatible))
            teacher.eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            # External frozen reference is intentionally not a student submodule.
            object.__setattr__(self, "_training_teacher", teacher)
        if next(teacher.parameters()).device != device:
            raise ValueError("External teacher device differs from student")
        capture = []
        handle = teacher.dec.register_forward_hook(
            lambda module, inputs, output: capture.append(output))
        try:
            teacher(batch, mode="loss")
            output = capture[-1]
            return [m[:, ~pad].detach() for m, pad in zip(
                output["masks"], output["batch_mask"])]
        finally:
            handle.remove()

    def point_masses(self, superpoints, batch_offsets, scenes_len):
        counts = torch.bincount(
            superpoints.long(), minlength=int(batch_offsets[-1])
        ).to(device=superpoints.device, dtype=torch.float32)
        if counts.numel() != int(batch_offsets[-1]):
            raise ValueError("Superpoint indices and batch offsets disagree")
        return torch.cat([
            counts[int(batch_offsets[i]):int(batch_offsets[i + 1])].repeat(int(scenes_len[i]))
            for i in range(len(scenes_len))
        ])

    @cuda_cast
    def loss(self, ann_ids, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape, feats, superpoints, batch_offsets, object_idss, gt_pmasks, gt_spmasks, sp_ref_masks, lang_tokenss, lang_masks, coords_float, sp_ins_labels, dense_maps, scenes_len=None, meta_datas=None, view_dependents=None, feats_2d=None, completion_targets=None, teacher_masks=None):
        batch_size = len(batch_offsets) - 1
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

        sp_feats = self.extract_feat(input, superpoints, p2v_map)
        sp_coords_float = scatter_mean(coords_float, superpoints, dim=0)

        point_mass = self.point_masses(superpoints, batch_offsets, scenes_len)
        original_offsets = batch_offsets
        sp_feats, sp_coords_float, fps_seed_sp, batch_offsets, sp_ins_labels, feats_2d = self.expand_and_fps(sp_feats, sp_coords_float, batch_offsets, sp_ins_labels, scenes_len, feats_2d, mode='tr')
        lang_feats = self.text_encoder(lang_tokenss, attention_mask=lang_masks)[0]

        out = self.dec(sp_feats, fps_seed_sp, batch_offsets, lang_feats, lang_masks, sp_coords_float, feats_2d, point_mass)
        if teacher_masks is not None:
            completion_targets = []
            expression = 0
            metadata = []
            with torch.no_grad():
                for scene, count in enumerate(scenes_len):
                    start, end = int(original_offsets[scene]), int(original_offsets[scene+1])
                    point_ids = superpoints[(superpoints >= start) & (superpoints < end)] - start
                    mass = torch.bincount(point_ids, minlength=end-start).float()
                    for _ in range(int(count)):
                        truth_mass = torch.bincount(
                            point_ids, weights=gt_pmasks[expression].gt(.5).float(),
                            minlength=end-start)
                        student = out["masks"][expression, :out["completion_index"], :end-start]
                        teacher = teacher_masks[expression]
                        if teacher.shape[-1] != end-start or teacher.requires_grad:
                            raise ValueError("Current-batch teacher alignment invalid")
                        def oracle(logits):
                            masks = logits.gt(0).float()
                            intersection = masks @ truth_mass
                            union = masks @ mass + truth_mass.sum()-intersection
                            ious = intersection / union.clamp_min(1)
                            return ious.max(0)
                        student_iou, _ = oracle(student.detach())
                        teacher_iou, teacher_index = oracle(teacher)
                        material = bool(teacher_iou-student_iou > .05)
                        target = teacher[teacher_index].gt(0).float() if material else gt_spmasks[expression].float()
                        completion_targets.append(target.detach())
                        metadata.append({
                            "scan_id": scan_ids[scene], "ann_id": str(ann_ids[expression]),
                            "object_ids": [int(obj) for obj in object_idss[expression]],
                            "material": material, "superpoints": end-start,
                            "teacher_iou": float(teacher_iou),
                            "student_iou": float(student_iou),
                            "target_detached": not target.requires_grad,
                        })
                        expression += 1
            self.last_teacher_targets = metadata
        
        loss, loss_dict = self.criterion(
            out, gt_spmasks, sp_ref_masks, object_idss, sp_ins_labels,
            dense_maps, lang_masks, fps_seed_sp, sp_coords_float, batch_offsets,
            completion_targets=completion_targets,
        )

        return loss, loss_dict
    
    @cuda_cast
    def predict(self, ann_ids, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape, feats, superpoints, batch_offsets, object_idss, gt_pmasks, gt_spmasks, sp_ref_masks, lang_tokenss, lang_masks, coords_float, sp_ins_labels, dense_maps, scenes_len=None, meta_datas=None, view_dependents=None, feats_2d=None):
        batch_size = len(batch_offsets) - 1
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        
        sp_feats = self.extract_feat(input, superpoints, p2v_map)
        sp_coords_float = scatter_mean(coords_float, superpoints, dim=0)

        point_mass = self.point_masses(superpoints, batch_offsets, scenes_len)
        sp_feats, sp_coords_float, fps_seed_sp, batch_offsets, sp_ins_labels, feats_2d = self.expand_and_fps(sp_feats, sp_coords_float, batch_offsets, sp_ins_labels, scenes_len, feats_2d, mode='pred')
        lang_feats = self.text_encoder(lang_tokenss, attention_mask=lang_masks)[0]
        
        out = self.dec(sp_feats, fps_seed_sp, batch_offsets, lang_feats, lang_masks, sp_coords_float, feats_2d, point_mass)
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
                sp_feats_expand = sp_feats[s:e].repeat(scenes_len[i],1)
                sp_coords_float_expand = sp_coords_float[s:e].repeat(scenes_len[i],1)
                feats_2d_expand = feats_2d[s:e].repeat(scenes_len[i],1)
                fps_seed_sp = FurthestPointSampling.apply(sp_coords_float[s:e].unsqueeze(0), self.fps_num)
                fps_seed_sp_expand = fps_seed_sp.squeeze(0).repeat(scenes_len[i],1)
            else:
                sp_feats_expand = torch.cat((sp_feats_expand, sp_feats[s:e].repeat(scenes_len[i],1)),dim=0)
                sp_coords_float_expand = torch.cat((sp_coords_float_expand, sp_coords_float[s:e].repeat(scenes_len[i],1)))
                feats_2d_expand = torch.cat((feats_2d_expand, feats_2d[s:e].repeat(scenes_len[i],1)),dim=0)
                fps_seed_sp = FurthestPointSampling.apply(sp_coords_float[s:e].unsqueeze(0), self.fps_num)
                fps_seed_sp_expand = torch.cat((fps_seed_sp_expand, fps_seed_sp.squeeze(0).repeat(scenes_len[i],1)),dim=0)
            for j in range(scenes_len[i]):
                batch_offsets_expand = torch.cat((batch_offsets_expand, batch_offsets_expand[-1:]+batch_offsets[i+1:i+2]-batch_offsets[i:i+1]), dim=0)
                if sp_ins_labels:
                    sp_ins_labels_expand.append(sp_ins_labels[i])
        return sp_feats_expand, sp_coords_float_expand, fps_seed_sp_expand, batch_offsets_expand, sp_ins_labels_expand, feats_2d_expand
    
    
