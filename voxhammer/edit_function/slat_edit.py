import math
import torch
import torch.nn.functional as F
from typing import *
import trellis.modules.sparse as sp
from voxhammer.edit_function.sampler import InversionFlowEulerGuidanceIntervalSampler

def slat_attn_forward(self, x, context=None, slat_kv=None, kv_mask=None, t_latent=None, order=None, pos=None, layer=None, is_text=False):
    dst_coords_all = None
    src_coords_all = None
    if self._type == 'self':
        qkv = self._linear(self.to_qkv, x)
        qkv = self._fused_pre(qkv, num_fused=3)
        if kv_mask is None:
            if self.use_rope:
                k_pre = qkv.unbind(dim=1)[1]
                slat_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_k_pre'] = k_pre.cpu()
                qkv = self._rope(qkv)
            (q, k, v) = qkv.unbind(dim=1)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
            slat_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_q'] = q.cpu()
            slat_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_k'] = k.cpu()
            slat_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_v'] = v.cpu()
            k = k.feats.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
            v = v.feats.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
        else:
            base_mask = kv_mask.get('mask', None)
            dst_coords_all = kv_mask.get('dst_coords_all', None)
            src_coords_all = kv_mask.get('src_coords_all', None)
            keep_coords = kv_mask.get('keep_coords', base_mask)
            A_tgt = kv_mask.get('A_tgt', None)
            sparse_A_q = slat_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_q'].cuda()
            sparse_A_k = slat_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_k'].cuda()
            sparse_A_v = slat_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_v'].cuda()
            sparse_A_k_pre = slat_kv.get(f'{t_latent}_{order}_{pos}_{layer}_{self._type}_k_pre', None)
            if sparse_A_k_pre is not None:
                sparse_A_k_pre = sparse_A_k_pre.cuda()
            if self.use_rope:
                qkv = self._rope(qkv)
            (q, k, v) = qkv.unbind(dim=1)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
            if keep_coords is not None and keep_coords.numel() > 0:
                device = q.coords.device
                keep_coords = keep_coords.to(device=device, dtype=torch.long)
                match_1_q = (q.coords.unsqueeze(1) == keep_coords.unsqueeze(0)).all(dim=-1)
                match_2_q = (sparse_A_q.coords.unsqueeze(1) == keep_coords.unsqueeze(0)).all(dim=-1)
                idx_1_q = match_1_q.float().argmax(0)
                idx_2_q = match_2_q.float().argmax(0)
                valid_q = match_2_q[idx_2_q, torch.arange(len(keep_coords))]
                idx_1_valid = idx_1_q[valid_q]
                idx_2_valid = idx_2_q[valid_q]
                if len(idx_1_valid) > 0:
                    feats_q = q.feats.clone()
                    feats_q[idx_1_valid] = sparse_A_q.feats[idx_2_valid].to(device)
                    q = q.replace(feats_q).type(q.dtype)
                if len(idx_1_valid) > 0:
                    feats_k = k.feats.clone()
                    feats_k[idx_1_valid] = sparse_A_k.feats[idx_2_valid].to(device)
                    k = k.replace(feats_k).type(q.dtype)
                if len(idx_1_valid) > 0:
                    feats_v = v.feats.clone()
                    feats_v[idx_1_valid] = sparse_A_v.feats[idx_2_valid].to(device)
                    v = v.replace(feats_v).type(q.dtype)
            if dst_coords_all is not None and src_coords_all is not None:
                device = q.coords.device
                dst_coords_all = dst_coords_all.to(device=device, dtype=torch.long)
                src_coords_all = src_coords_all.to(device=device, dtype=torch.long)
                v_cache = sparse_A_v
                match_src_v = (v_cache.coords.unsqueeze(1) == src_coords_all.unsqueeze(0)).all(dim=-1)
                idx_src_v = match_src_v.float().argmax(0)
                valid_src_v = match_src_v[idx_src_v, torch.arange(len(src_coords_all))]
                idx_src_v_valid = idx_src_v[valid_src_v]
                v_drag_feats = v_cache.feats[idx_src_v_valid]
                if self.use_rope and sparse_A_k_pre is not None:
                    k_pre_cache = sparse_A_k_pre
                    match_src_k = (k_pre_cache.coords.unsqueeze(1) == src_coords_all.unsqueeze(0)).all(dim=-1)
                    idx_src_k = match_src_k.float().argmax(0)
                    valid_src_k = match_src_k[idx_src_k, torch.arange(len(src_coords_all))]
                    idx_src_k_valid = idx_src_k[valid_src_k]
                    k_drag_feats = k_pre_cache.feats[idx_src_k_valid]
                    q_dummy = torch.zeros_like(k_drag_feats)
                    v_dummy = torch.zeros_like(k_drag_feats)
                    qkv_drag = type(q)(feats=torch.stack([q_dummy, k_drag_feats, v_dummy], dim=1), coords=dst_coords_all[valid_src_v]).cuda()
                    qkv_drag = self._rope(qkv_drag)
                    k_drag_feats = qkv_drag.unbind(dim=1)[1].feats
                    if self.qk_rms_norm:
                        k_drag_feats = self.k_rms_norm(k_drag_feats)
                else:
                    k_cache = sparse_A_k
                    match_src_k2 = (k_cache.coords.unsqueeze(1) == src_coords_all.unsqueeze(0)).all(dim=-1)
                    idx_src_k2 = match_src_k2.float().argmax(0)
                    valid_src_k2 = match_src_k2[idx_src_k2, torch.arange(len(src_coords_all))]
                    idx_src_k2_valid = idx_src_k2[valid_src_k2]
                    k_drag_feats = k_cache.feats[idx_src_k2_valid]
                if len(k_drag_feats) > 0 and len(v_drag_feats) > 0:
                    k_drag = k_drag_feats.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
                    v_drag = v_drag_feats.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
                    k = k.feats.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
                    v = v.feats.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
                    k = torch.cat([k, k_drag.to(dtype=k.dtype, device=k.device)], dim=2)
                    v = torch.cat([v, v_drag.to(dtype=v.dtype, device=v.device)], dim=2)
                else:
                    k = k.feats.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
                    v = v.feats.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
            else:
                k = k.feats.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
                v = v.feats.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
        h = q
        q = q.feats.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1, 3)[0]
        h = h.replace(out)
        h = self._reshape_chs(h, (-1,))
        h = self._linear(self.to_out, h)
        if kv_mask is None:
            slat_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_y'] = h.cpu()
        if kv_mask is not None and self._type == 'self':
            dst_coords = kv_mask.get('dst_coords_all')
            src_coords = kv_mask.get('src_coords_all')
            A = kv_mask.get('A_tgt')
            cache_key = f'{t_latent}_{order}_{pos}_{layer}_{self._type}_y'
            if not (dst_coords is not None and src_coords is not None and (A is not None) and (cache_key in slat_kv)):
                return
            device = h.coords.device
            dst_coords = dst_coords.to(device, torch.long)
            src_coords = src_coords.to(device, torch.long)
            A = A.to(device, h.feats.dtype)
            y_cache = slat_kv[cache_key].to(device)
            match_dst = (h.coords.unsqueeze(1) == dst_coords.unsqueeze(0)).all(-1)
            (b_idx, dst_idx) = match_dst.nonzero(as_tuple=True)
            match_src = (y_cache.coords.unsqueeze(1) == src_coords.unsqueeze(0)).all(-1)
            (a_idx, src_idx) = match_src.nonzero(as_tuple=True)
            (replace_B_idx, replace_A_idx, gamma_idx) = ([], [], [])
            if len(dst_idx) > 0 and len(src_idx) > 0:
                equal_mask = dst_idx.unsqueeze(1) == src_idx.unsqueeze(0)
                (i, j) = equal_mask.nonzero(as_tuple=True)
                valid = (b_idx[i] < len(h.feats)) & (b_idx[i] >= 0) & (a_idx[j] < len(y_cache.feats)) & (a_idx[j] >= 0) & (dst_idx[i] < len(A)) & (dst_idx[i] >= 0)
                if valid.any():
                    replace_B_idx = b_idx[i[valid]]
                    replace_A_idx = a_idx[j[valid]]
                    gamma_idx = dst_idx[i[valid]]
            if len(replace_B_idx) > 0:
                replace_B_idx = replace_B_idx.clamp(0, len(h.feats) - 1)
                replace_A_idx = replace_A_idx.clamp(0, len(y_cache.feats) - 1)
                gamma_idx = gamma_idx.clamp(0, len(A) - 1)
                y_src = y_cache.feats[replace_A_idx]
                y_dst = h.feats[replace_B_idx]
                gamma = (float(t_latent) * A).view(-1, 1)[gamma_idx]
                y_new = (1.0 - gamma) * y_dst + gamma * y_src
                feats = h.feats.clone()
                feats[replace_B_idx] = y_new
                h = h.replace(feats)
            else:
                print('No valid pairing index, skipping gated fusion.')
    else:
        q = self._linear(self.to_q, x)
        q = self._reshape_chs(q, (self.num_heads, -1))
        kv = self._linear(self.to_kv, context)
        kv = self._fused_pre(kv, num_fused=2)
        (k, v) = kv.unbind(dim=2)
        h = q
        q = q.feats.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        if not is_text:
            if kv_mask is None:
                slat_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_k'] = k.cpu()
                slat_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_v'] = v.cpu()
            else:
                mask = kv_mask.get('mask', None)
                if mask is not None:
                    k = k * mask + slat_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_k'].cuda() * (1 - mask)
                    v = v * mask + slat_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_v'].cuda() * (1 - mask)
                    k = k.type(q.dtype)
                    v = v.type(q.dtype)
    out = F.scaled_dot_product_attention(q, k, v)
    out = out.permute(0, 2, 1, 3)[0]
    h = h.replace(out)
    h = self._reshape_chs(h, (-1,))
    h = self._linear(self.to_out, h)
    return h

def slat_attn_logit_forward(self, x, context=None, slat_kv=None, kv_mask=None, t_latent=None, order=None, pos=None, layer=None, is_text=False):
    dst_coords_all = None
    src_coords_all = None
    coord_to_idx = {}
    if self._type == 'self':
        qkv = self._linear(self.to_qkv, x)
        qkv = self._fused_pre(qkv, num_fused=3)
        if kv_mask is None:
            (q, k, v) = qkv.unbind(dim=1)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
            slat_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_q'] = q.cpu()
            slat_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_k'] = k.cpu()
            slat_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_v'] = v.cpu()
            k = k.feats.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
            v = v.feats.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
        else:
            base_mask = kv_mask.get('mask', None)
            dst_coords_all = kv_mask.get('dst_coords_all', None)
            src_coords_all = kv_mask.get('src_coords_all', None)
            keep_coords = kv_mask.get('keep_coords', base_mask)
            A_tgt = kv_mask.get('A_tgt', None)
            sparse_A_q = slat_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_q'].cuda()
            sparse_A_k = slat_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_k'].cuda()
            sparse_A_v = slat_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_v'].cuda()
            (q, k, v) = qkv.unbind(dim=1)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
            if keep_coords is not None and keep_coords.numel() > 0:
                device = q.coords.device
                keep_coords = keep_coords.to(device=device, dtype=torch.long)
                if 'keep' not in coord_to_idx:
                    match_1_q = (q.coords.unsqueeze(1) == keep_coords.unsqueeze(0)).all(dim=-1)
                    match_2_q = (sparse_A_q.coords.unsqueeze(1) == keep_coords.unsqueeze(0)).all(dim=-1)
                    idx_1_q = match_1_q.float().argmax(0)
                    idx_2_q = match_2_q.float().argmax(0)
                    valid_q = match_2_q[idx_2_q, torch.arange(len(keep_coords))]
                    idx_1_valid = idx_1_q[valid_q]
                    idx_2_valid = idx_2_q[valid_q]
                    coord_to_idx['keep'] = {'curr_idx': idx_1_valid, 'cache_idx': idx_2_valid}
                keep_curr_idx = coord_to_idx['keep']['curr_idx']
                keep_cache_idx = coord_to_idx['keep']['cache_idx']
                if len(keep_curr_idx) > 0:
                    feats_q = q.feats.clone()
                    feats_q[keep_curr_idx] = sparse_A_q.feats[keep_cache_idx].to(device)
                    q = q.replace(feats_q).type(q.dtype)
                    feats_k = k.feats.clone()
                    feats_k[keep_curr_idx] = sparse_A_k.feats[keep_cache_idx].to(device)
                    k = k.replace(feats_k).type(q.dtype)
                    feats_v = v.feats.clone()
                    feats_v[keep_curr_idx] = sparse_A_v.feats[keep_cache_idx].to(device)
                    v = v.replace(feats_v).type(q.dtype)
            if dst_coords_all is not None and src_coords_all is not None:
                device = q.coords.device
                dst_coords_all = dst_coords_all.to(device=device, dtype=torch.long)
                src_coords_all = src_coords_all.to(device=device, dtype=torch.long)
                if 'dst_src' not in coord_to_idx:
                    match_1_dstsrc = (q.coords.unsqueeze(1) == dst_coords_all.unsqueeze(0)).all(dim=-1)
                    match_2_dstsrc = (sparse_A_v.coords.unsqueeze(1) == src_coords_all.unsqueeze(0)).all(dim=-1)
                    idx_1_dstsrc = match_1_dstsrc.float().argmax(0)
                    idx_2_dstsrc = match_2_dstsrc.float().argmax(0)
                    valid_dstsrc = match_2_dstsrc[idx_2_dstsrc, torch.arange(len(src_coords_all))]
                    idx_1_valid_dstsrc = idx_1_dstsrc[valid_dstsrc]
                    idx_2_valid_dstsrc = idx_2_dstsrc[valid_dstsrc]
                    coord_to_idx['dst_src'] = {'dst_idx': idx_1_valid_dstsrc, 'src_idx': idx_2_valid_dstsrc, 'dst_coords': dst_coords_all[valid_dstsrc], 'src_coords': src_coords_all[valid_dstsrc]}
                dst_src_map = coord_to_idx['dst_src']
                if len(dst_src_map['dst_idx']) > 0:
                    v_cache = sparse_A_v
                    dst_idx = dst_src_map['dst_idx']
                    src_idx = dst_src_map['src_idx']
                    out = custom_attention_unfused_slat(q.feats, k.feats, v.feats, v_cache.feats, dst_idx, src_idx)
                else:
                    q_dense = q.feats.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
                    k = k.feats.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
                    v = v.feats.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
                    out = F.scaled_dot_product_attention(q_dense, k, v)
                    out = out.permute(0, 2, 1, 3)[0]
            else:
                k = k.feats.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
                v = v.feats.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
        h = q
        q_dense = q.feats.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
        if kv_mask is None:
            out = F.scaled_dot_product_attention(q_dense, k, v)
            out = out.permute(0, 2, 1, 3)[0]
        h = h.replace(out)
        h = self._reshape_chs(h, (-1,))
        h = self._linear(self.to_out, h)
        if kv_mask is None:
            slat_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_y'] = h.cpu()
        if kv_mask is not None and self._type == 'self':
            dst_coords = kv_mask.get('dst_coords_all')
            src_coords = kv_mask.get('src_coords_all')
            A = kv_mask.get('A_tgt')
            cache_key = f'{t_latent}_{order}_{pos}_{layer}_{self._type}_y'
            if not (dst_coords is not None and src_coords is not None and (A is not None) and (cache_key in slat_kv)):
                return h
            if 'dst_src' in coord_to_idx:
                dst_src_map = coord_to_idx['dst_src']
                device = h.coords.device
                A = A.to(device, h.feats.dtype)
                y_cache = slat_kv[cache_key].to(device)
                replace_B_idx = dst_src_map['dst_idx']
                replace_A_idx = dst_src_map['src_idx']
                gamma_idx = torch.arange(len(replace_B_idx)).to(device)
                if len(replace_B_idx) > 0:
                    replace_B_idx = replace_B_idx.clamp(0, len(h.feats) - 1)
                    replace_A_idx = replace_A_idx.clamp(0, len(y_cache.feats) - 1)
                    gamma_idx = gamma_idx.clamp(0, len(A) - 1)
                    y_src = y_cache.feats[replace_A_idx]
                    y_dst = h.feats[replace_B_idx]
                    gamma = (float(t_latent) * A).view(-1, 1)[gamma_idx]
                    y_new = (1.0 - gamma) * y_dst + gamma * y_src
                    feats = h.feats.clone()
                    feats[replace_B_idx] = y_new
                    h = h.replace(feats)
            else:
                print('No valid dst/src mapping, skipping gated fusion.')
    else:
        q = self._linear(self.to_q, x)
        q = self._reshape_chs(q, (self.num_heads, -1))
        kv = self._linear(self.to_kv, context)
        kv = self._fused_pre(kv, num_fused=2)
        (k, v) = kv.unbind(dim=2)
        h = q
        q_dense = q.feats.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
        k_dense = k.permute(0, 2, 1, 3).contiguous()
        v_dense = v.permute(0, 2, 1, 3).contiguous()
        if not is_text:
            if kv_mask is None:
                slat_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_k'] = k_dense.cpu()
                slat_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_v'] = v_dense.cpu()
            else:
                mask = kv_mask.get('mask', None)
                if mask is not None:
                    k_dense = k_dense * mask + slat_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_k'].cuda() * (1 - mask)
                    v_dense = v_dense * mask + slat_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_v'].cuda() * (1 - mask)
                    k_dense = k_dense.type(q_dense.dtype)
                    v_dense = v_dense.type(q_dense.dtype)
        out = F.scaled_dot_product_attention(q_dense, k_dense, v_dense)
        out = out.permute(0, 2, 1, 3)[0]
        h = h.replace(out)
        h = self._reshape_chs(h, (-1,))
        h = self._linear(self.to_out, h)
    return h

def custom_attention_unfused(q, k, v, v_cache, dst_idx, src_idx):
    (B, H, L, D) = q.shape
    scale = 1.0 / math.sqrt(D)
    attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale
    copied_logits = attn_logits[:, :, :, dst_idx]
    attn_logits_new = torch.cat([attn_logits, copied_logits], dim=-1)
    attn_weights = F.softmax(attn_logits_new, dim=-1)
    v_src_subset = v_cache[:, :, src_idx, :]
    v_cat = torch.cat([v, v_src_subset], dim=2)
    h = torch.matmul(attn_weights, v_cat)
    return h

def custom_attention_unfused_slat(q, k, v, v_cache, dst_idx, src_idx):
    head_dim = q.shape[-1]
    scale = 1.0 / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32))
    q_trans = q.transpose(0, 1)
    k_trans = k.transpose(0, 1)
    attn_logits = torch.matmul(q_trans, k_trans.transpose(-2, -1)) * scale
    copied_logits = attn_logits[:, :, dst_idx]
    attn_logits_new = torch.cat([attn_logits, copied_logits], dim=-1)
    attn_weights = F.softmax(attn_logits_new, dim=-1)
    v_src_subset = v_cache[src_idx, :, :]
    v_src_subset = v_src_subset.permute(1, 0, 2)
    v_cat = torch.cat([v.transpose(0, 1), v_src_subset], dim=1)
    h = torch.matmul(attn_weights, v_cat)
    h = h.transpose(0, 1)
    return h

def custom_attention_unfused_slat2(q, k, v, v_cache, dst_idx, src_idx):
    head_dim = q.shape[-1]
    scale = 1.0 / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32))
    q_trans = q.transpose(0, 1)
    k_trans = k.transpose(0, 1)
    attn_logits = torch.matmul(q_trans, k_trans.transpose(-2, -1)) * scale
    copied_logits = attn_logits[:, :, dst_idx]
    N = q_trans.shape[1]
    bg_mask = torch.ones(N, dtype=torch.bool, device=q.device)
    bg_mask[dst_idx] = False
    copied_logits[:, bg_mask, :] = -10000.0
    attn_logits_new = torch.cat([attn_logits, copied_logits], dim=-1)
    attn_weights = F.softmax(attn_logits_new, dim=-1)
    v_src_subset = v_cache[src_idx, :, :].permute(1, 0, 2)
    v_cat = torch.cat([v.transpose(0, 1), v_src_subset], dim=1)
    h = torch.matmul(attn_weights, v_cat)
    return h.transpose(0, 1)

def slat_trsfmr_forward(self, x, mod, context, slat_kv, self_kv_mask, cross_kv_mask, t_latent, order, pos, layer, is_text):
    if self.share_mod:
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = mod.chunk(6, dim=1)
    else:
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(mod).chunk(6, dim=1)
    h = x.replace(self.norm1(x.feats))
    h = h * (1 + scale_msa) + shift_msa
    h = self.self_attn(h, slat_kv=slat_kv, kv_mask=self_kv_mask, t_latent=t_latent, order=order, pos=pos, layer=layer, is_text=is_text)
    h = h * gate_msa
    x = x + h
    h = x.replace(self.norm2(x.feats))
    h = self.cross_attn(h, context, slat_kv=slat_kv, kv_mask=None, t_latent=t_latent, order=order, pos=pos, layer=layer, is_text=is_text)
    x = x + h
    h = x.replace(self.norm3(x.feats))
    h = h * (1 + scale_mlp) + shift_mlp
    h = self.mlp(h)
    h = h * gate_mlp
    x = x + h
    return x

def slat_flow_forward(self, x, t, cond, slat_kv, self_kv_mask, cross_kv_mask, t_latent, order, pos, is_text):
    h = self.input_layer(x).type(self.dtype)
    t_emb = self.t_embedder(t)
    if self.share_mod:
        t_emb = self.adaLN_modulation(t_emb)
    t_emb = t_emb.type(self.dtype)
    cond = cond.type(self.dtype)
    skips = []
    for block in self.input_blocks:
        h = block(h, t_emb)
        skips.append(h.feats)
    if self.pe_mode == 'ape':
        h = h + self.pos_embedder(h.coords[:, 1:]).type(self.dtype)
    for (layer, block) in enumerate(self.blocks):
        h = block(h, t_emb, cond, slat_kv, self_kv_mask, cross_kv_mask, t_latent, order, pos, layer, is_text)
    for (block, skip) in zip(self.out_blocks, reversed(skips)):
        if self.use_skip_connection:
            h = block(h.replace(torch.cat([h.feats, skip], dim=1)), t_emb)
        else:
            h = block(h, t_emb)
    h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
    h = self.out_layer(h.type(x.dtype))
    return h

def sample_slat_inverse(pipeline, cond_src, slat_src, cfg_strength_stage_2_inverse, is_text):
    stage = 2
    slat_inverse = slat_src.replace(slat_src.feats, slat_src.coords)
    flow_model = pipeline.models['slat_flow_model']
    std = torch.tensor(pipeline.slat_normalization['std'], device=pipeline.device)[None]
    mean = torch.tensor(pipeline.slat_normalization['mean'], device=pipeline.device)[None]
    slat_inverse = (slat_inverse - mean) / std
    sigma_min = pipeline.slat_sampler.sigma_min
    slat_sampler = InversionFlowEulerGuidanceIntervalSampler(sigma_min)
    if cfg_strength_stage_2_inverse is None:
        cfg_strength = pipeline.sparse_structure_sampler_params['cfg_strength']
    else:
        cfg_strength = cfg_strength_stage_2_inverse
    (noise, slat_latent, slat_kv) = slat_sampler.sample(flow_model, stage, slat_inverse, cond_src, cfg_strength, is_text=is_text)
    return (slat_latent, slat_kv)

def sample_slat_denoise(pipeline, cond_tgt, coords_tgt, slat_src, coords_mask, slat_latent, slat_kv, slat_self_kv_mask, slat_cross_kv_mask, cfg_strength_stage_2_forward, is_text):
    stage = 2
    flow_model = pipeline.models['slat_flow_model']
    noise = sp.SparseTensor(feats=torch.randn(coords_tgt.shape[0], flow_model.in_channels).to(pipeline.device), coords=coords_tgt)
    sigma_min = pipeline.slat_sampler.sigma_min
    slat_sampler = InversionFlowEulerGuidanceIntervalSampler(sigma_min)
    if cfg_strength_stage_2_forward is None:
        cfg_strength = pipeline.sparse_structure_sampler_params['cfg_strength']
    else:
        cfg_strength = cfg_strength_stage_2_forward
    (slat, slat_latent, slat_kv) = slat_sampler.sample(flow_model, stage, noise, cond_tgt, cfg_strength, slat_latent, coords_mask, slat_kv, slat_self_kv_mask, slat_cross_kv_mask, is_text=is_text)
    if slat == None:
        return None
    std = torch.tensor(pipeline.slat_normalization['std'], device=pipeline.device)[None]
    mean = torch.tensor(pipeline.slat_normalization['mean'], device=pipeline.device)[None]
    slat = slat * std + mean
    return slat