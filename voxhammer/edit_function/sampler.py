import os
import torch
import numpy as np
from typing import *
from tqdm import tqdm
from trellis.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler
from voxhammer.edit_function.dict_and_utils import match_coords_chunked
import sys
from voxhammer.edit_function.dict_and_utils import DiskDict
class InversionFlowEulerGuidanceIntervalSampler(FlowEulerGuidanceIntervalSampler):

    def _inference_model(self, model, sample, t, cond, kv, self_kv_mask, cross_kv_mask, t_latent, order, pos, is_text):
        t = torch.tensor([1000 * t] * sample.shape[0], device=sample.device, dtype=torch.float32)
        if cond is not None and cond.shape[0] == 1 and (sample.shape[0] > 1):
            cond = cond.repeat(sample.shape[0], *[1] * (len(cond.shape) - 1))
        return model(sample, t, cond, kv, self_kv_mask, cross_kv_mask, t_latent, order, pos, is_text)

    def inference_model(self, model, sample, t, cond, cfg_strength, kv, self_kv_mask, cross_kv_mask, t_latent, order, is_text):
        cfg_interval = [0.5, 1.0]
        if cfg_interval[0] <= t_latent <= cfg_interval[1]:
            pos = 1
            pred = self._inference_model(model, sample, t, cond['cond'], kv, self_kv_mask, cross_kv_mask, t_latent, order, pos, is_text)
            pos = 0
            neg_pred = self._inference_model(model, sample, t, cond['neg_cond'], kv, self_kv_mask, cross_kv_mask, t_latent, order, pos, is_text)
            return (1 + cfg_strength) * pred - cfg_strength * neg_pred
        else:
            pos = 1
            return self._inference_model(model, sample, t, cond['cond'], kv, self_kv_mask, cross_kv_mask, t_latent, order, pos, is_text)

    def sample_once(self, model, sample, t_curr, t_prev, cond, cfg_strength, kv, self_kv_mask, cross_kv_mask, t_latent, is_text):
        order = 1
        pred = self.inference_model(model, sample, t_curr, cond, cfg_strength, kv, self_kv_mask, cross_kv_mask, t_latent, order, is_text)
        sample_mid = sample + (t_prev - t_curr) / 2 * pred
        t_mid = t_curr + (t_prev - t_curr) / 2
        order = 2
        pred_mid = self.inference_model(model, sample_mid, t_mid, cond, cfg_strength, kv, self_kv_mask, cross_kv_mask, t_latent, order, is_text)
        first_order = (pred_mid - pred) / ((t_prev - t_curr) / 2)
        sample = sample + (t_prev - t_curr) * pred - 0.5 * (t_prev - t_curr) ** 2 * first_order
        return sample

    @torch.no_grad()
    def sample(self, model, stage, noise, cond, cfg_strength, latent=None, latent_mask=None, kv=None, self_kv_mask=None, cross_kv_mask=None, skip_step=None, noise_init=None, is_text=False):
        steps = 25
        rescale_t = 3.0
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        inverse_bool = latent_mask is None
        if skip_step is not None:
            t_seq = t_seq[skip_step:]
            steps = steps - skip_step
        if noise_init is not None:
            noise_randn = torch.randn_like(noise)
            t_init = t_seq[0]
            sample = noise_init * (1 - t_init) + noise_randn * t_init
        if inverse_bool:
            t_seq = t_seq[::-1]
            desc = 'Inversing'
            root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if root_path not in sys.path:
                sys.path.append(root_path)
            ssd_path = f'{root_path}/cache'
            os.makedirs(ssd_path, exist_ok=True)
            latent = {}
            kv = {}
            latent[f'{t_seq[0]}'] = sample.cpu()
        else:
            desc = 'Sampling'
        if not inverse_bool:
            if stage == 1:
                dst_idx = latent_mask.get('dst_idx_tgt', None)
                src_idx_of_dst = latent_mask.get('src_idx_of_dst_tgt', None)
                inp_idx = latent_mask.get('inp_idx', None)
                if dst_idx is not None and src_idx_of_dst is not None and (dst_idx.numel() > 0):
                    dst_idx = dst_idx.to(device=sample.device, dtype=torch.long)
                    src_idx_of_dst = src_idx_of_dst.to(device=sample.device, dtype=torch.long)
                    src_idx = src_idx_of_dst.index_select(0, dst_idx)
                    (B, C) = (sample.shape[0], sample.shape[1])
                    sample_flat = sample.view(B, C, -1)
                    sample_flat[:, :, dst_idx] = sample_flat[:, :, src_idx]
                    sample = sample_flat.view_as(sample)
                if inp_idx is not None and inp_idx.numel() > 0:
                    inp_idx = inp_idx.to(device=sample.device, dtype=torch.long)
                    (B, C) = (sample.shape[0], sample.shape[1])
                    sample_flat = sample.view(B, C, -1)
                    eps = torch.randn(B, C, inp_idx.numel(), device=sample.device, dtype=sample.dtype)
                    sample_flat[:, :, inp_idx] = eps
                    sample = sample_flat.view_as(sample)
            elif stage == 2:
                dst_coords = latent_mask.get('dst_coords_tgt', None)
                src_coords = latent_mask.get('src_coords_tgt', None)
                inp_coords = latent_mask.get('inp_coords', None)
                keep_coords = latent_mask.get('keep_coords', None)
                sparse_A = latent[f'{t_seq[0]}'].cuda()
                sparse_B = sample
                feats = sparse_B.feats.clone()
                coords_B = sparse_B.coords
                coords_A = sparse_A.coords
                device = feats.device
                if keep_coords is not None and keep_coords.numel() > 0:
                    keep_coords = keep_coords.to(device=device, dtype=torch.long)
                    (matched_idx_B, matched_idx_keep) = match_coords_chunked(coords_B, keep_coords)
                    is_bg_in_B = torch.zeros(len(coords_B), dtype=torch.bool, device=device)
                    is_bg_in_B[matched_idx_B] = True
                    bg_idx_B = torch.where(is_bg_in_B)[0]
                    if len(bg_idx_B) > 0:
                        (valid_bg_idx_A, matched_B_subset_idx) = match_coords_chunked(coords_A, coords_B[bg_idx_B])
                        valid_bg_idx_B = bg_idx_B[matched_B_subset_idx]
                        feats[valid_bg_idx_B] = sparse_A.feats[valid_bg_idx_A].to(device)
                        print(f'✅ Only replaced {len(valid_bg_idx_B)} background point features')
                (matched_B_idx, matched_dst_idx) = match_coords_chunked(coords_B, dst_coords)
                print(f'Found {len(matched_B_idx)} matched points at B')
                (matched_A_idx, matched_src_idx) = match_coords_chunked(coords_A, src_coords)
                print(f'Found {len(matched_A_idx)} matched points at A')
                equal_matrix = matched_dst_idx.unsqueeze(1).eq(matched_src_idx.unsqueeze(0))
                equal_pairs = equal_matrix.nonzero(as_tuple=True)
                i_tensor = equal_pairs[0]
                j_tensor = equal_pairs[1]
                replace_B_idx = matched_B_idx[i_tensor].to(device)
                replace_A_idx = matched_A_idx[j_tensor].to(device)
                print(f'\nFound {len(replace_B_idx)} replaceable pairs:')
                if len(replace_B_idx) == 0 or len(replace_A_idx) == 0:
                    print('⚠️ No valid replacement indices, skipping feature replacement')
                else:
                    assert len(replace_B_idx) == len(replace_A_idx), f'Replacement indices length mismatch! B indices count: {len(replace_B_idx)}, A indices count: {len(replace_A_idx)}'
                    assert replace_B_idx.max() < len(sparse_B.feats), f'B index out of bounds! Max index {replace_B_idx.max()} >= B features count {len(sparse_B.feats)}'
                    assert replace_A_idx.max() < len(sparse_A.feats), f'A index out of bounds! Max index {replace_A_idx.max()} >= A features count {len(sparse_A.feats)}'
                    feats[replace_B_idx] = sparse_A.feats[replace_A_idx].to(device)
                    print(f'\n✅ Feature replacement completed! Replaced {len(replace_B_idx)} point features in total')
                sample = sample.replace(feats=feats, coords=coords_B)
                print('===== SLAT stage: Region replacement completed, no changes in other regions =====\n')
        t_pairs = list(((t_seq[i], t_seq[i + 1]) for i in range(steps)))
        for (t_curr, t_prev) in tqdm(t_pairs, desc=desc, disable=False, position=0):
            if inverse_bool:
                t_latent = t_prev
            else:
                t_latent = t_curr
                if stage == 1:
                    keep_mask = latent_mask.get('keep_mask', None)
                    if keep_mask is not None:
                        pass
                elif stage == 2:
                    keep_coords = latent_mask.get('keep_coords', None)
                    if keep_coords is not None:
                        pass
            sample = self.sample_once(model, sample, t_curr, t_prev, cond, cfg_strength, kv, self_kv_mask, cross_kv_mask, t_latent, is_text)
            if inverse_bool:
                latent[f'{t_latent}'] = sample.cpu()
        return (sample, latent, kv)