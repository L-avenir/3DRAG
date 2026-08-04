import math
import torch
import torch.nn.functional as F
from typing import *
from trellis.modules.spatial import patchify, unpatchify
from voxhammer.edit_function.sampler import InversionFlowEulerGuidanceIntervalSampler

def ss_attn_forward(self, x, context=None, indices=None, ss_kv=None, kv_mask=None, t_latent=None, order=None, pos=None, layer=None, is_text=False):
    (B, L, C) = x.shape
    if self._type == 'self':
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, -1)
        (q, k, v) = qkv.unbind(dim=2)
    else:
        Lkv = context.shape[1]
        q = self.to_q(x)
        kv = self.to_kv(context)
        q = q.reshape(B, L, self.num_heads, -1)
        kv = kv.reshape(B, Lkv, 2, self.num_heads, -1)
        (k, v) = kv.unbind(dim=2)
    if self.qk_rms_norm:
        q = self.q_rms_norm(q)
        k = self.k_rms_norm(k)
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    if not (is_text and self._type != 'self'):
        if kv_mask is None:
            ss_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_k'] = k.cpu()
            ss_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_v'] = v.cpu()
            ss_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_q'] = q.cpu()
        else:
            if self._type != 'self':
                mask = kv_mask.get('mask', None)
                if mask is not None:
                    mask = mask.to(device=k.device)
                    bm = mask.to(dtype=torch.bool)
                    k_cache = ss_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_k'].to(device=k.device, dtype=k.dtype)
                    v_cache = ss_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_v'].to(device=v.device, dtype=v.dtype)
                    k = torch.where(bm, k, k_cache)
                    v = torch.where(bm, v, v_cache)
            else:
                base_mask = kv_mask.get('mask', None)
                if base_mask is not None:
                    base_mask = base_mask.to(device=q.device)
                    bm = base_mask.to(dtype=torch.bool)
                    q_cache = ss_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_q'].to(device=q.device, dtype=q.dtype)
                    k_cache = ss_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_k'].to(device=q.device, dtype=q.dtype)
                    v_cache = ss_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_v'].to(device=q.device, dtype=q.dtype)
                    q = torch.where(bm, q, q_cache)
                    k = torch.where(bm, k, k_cache)
                    v = torch.where(bm, v, v_cache)
                    assert q.dim() == 4 and q_cache.dim() == 4
                    assert q.shape == q_cache.shape, (q.shape, q_cache.shape)
                    assert base_mask.shape[:3] == q.shape[:3], (base_mask.shape, q.shape)
            k = k.type(q.dtype)
            v = v.type(q.dtype)
    assert q.dtype == k.dtype == v.dtype == torch.float16, f'Dtype mismatch among QKV: {q.dtype}, {k.dtype}, {v.dtype}'
    h = F.scaled_dot_product_attention(q, k, v)
    h = h.permute(0, 2, 1, 3)
    h = h.reshape(B, L, -1)
    h = self.to_out(h)
    if not (is_text and self._type != 'self') and kv_mask is None:
        ss_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_y'] = h.cpu()
    if not (is_text and self._type != 'self') and kv_mask is not None and (self._type == 'self'):
        src_idx_of_dst = kv_mask.get('src_idx_of_dst_tgt', None)
        dst_idx = kv_mask.get('dst_idx_tgt', None)
        A = kv_mask.get('A_tgt', None)
        if src_idx_of_dst is not None and dst_idx is not None and (f'{t_latent}_{order}_{pos}_{layer}_{self._type}_y' in ss_kv):
            dst_idx = dst_idx.to(device=h.device, dtype=torch.long)
            src_idx = src_idx_of_dst.index_select(0, dst_idx).to(device=h.device, dtype=torch.long)
            y_cache = ss_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_y'].to(device=h.device, dtype=h.dtype)
            y_src = y_cache.index_select(1, src_idx)
            ht = float(t_latent)
            A = A.to(device=h.device, dtype=h.dtype)
            gamma = (ht * A).view(1, -1, 1).expand(h.shape[0], -1, 1)
            y_dst = h.index_select(1, dst_idx)
            y_new = (1.0 - gamma) * y_dst + gamma * y_src
            h[:, dst_idx, :] = y_new
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

def ss_attn_logit_forward(self, x, context=None, indices=None, ss_kv=None, kv_mask=None, t_latent=None, order=None, pos=None, layer=None, is_text=False):
    (B, L, C) = x.shape
    if self._type == 'self':
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, -1)
        (q, k, v) = qkv.unbind(dim=2)
    else:
        Lkv = context.shape[1]
        q = self.to_q(x)
        kv = self.to_kv(context)
        q = q.reshape(B, L, self.num_heads, -1)
        kv = kv.reshape(B, Lkv, 2, self.num_heads, -1)
        (k, v) = kv.unbind(dim=2)
    if self.qk_rms_norm:
        q = self.q_rms_norm(q)
        k = self.k_rms_norm(k)
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    v_cache = None
    if not (is_text and self._type != 'self'):
        if kv_mask is None:
            ss_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_k'] = k.cpu()
            ss_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_v'] = v.cpu()
            ss_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_q'] = q.cpu()
        else:
            if self._type != 'self':
                mask = kv_mask.get('mask', None)
                if mask is not None:
                    mask = mask.to(device=k.device)
                    bm = mask.to(dtype=torch.bool)
                    k_cache = ss_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_k'].to(device=k.device, dtype=k.dtype)
                    v_cache = ss_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_v'].to(device=v.device, dtype=v.dtype)
                    k = torch.where(bm, k, k_cache)
                    v = torch.where(bm, v, v_cache)
            else:
                base_mask = kv_mask.get('mask', None)
                if base_mask is not None:
                    base_mask = base_mask.to(device=q.device)
                    bm = base_mask.to(dtype=torch.bool)
                    q_cache = ss_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_q'].to(device=q.device, dtype=q.dtype)
                    k_cache = ss_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_k'].to(device=q.device, dtype=q.dtype)
                    v_cache = ss_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_v'].to(device=q.device, dtype=q.dtype)
                    q = torch.where(bm, q, q_cache)
                    k = torch.where(bm, k, k_cache)
                    v = torch.where(bm, v, v_cache)
                    assert q.dim() == 4 and q_cache.dim() == 4
                    assert q.shape == q_cache.shape, (q.shape, q_cache.shape)
                    assert base_mask.shape[:3] == q.shape[:3], (base_mask.shape, q.shape)
            k = k.type(q.dtype)
            v = v.type(q.dtype)
    assert q.dtype == k.dtype == v.dtype == torch.float16, f'Dtype mismatch among QKV: {q.dtype}, {k.dtype}, {v.dtype}'
    if not (is_text and self._type != 'self') and kv_mask is not None:
        src_idx_of_dst_all = kv_mask.get('src_idx_of_dst_all', None)
        dst_idx_all = kv_mask.get('dst_idx_all', None)
        dst_idx_all = dst_idx_all.to(device=q.device, dtype=torch.long)
        src_idx_all = src_idx_of_dst_all.index_select(0, dst_idx_all).to(device=q.device, dtype=torch.long)
        h = custom_attention_unfused(q, k, v, v_cache, dst_idx=dst_idx_all, src_idx=src_idx_all)
    else:
        h = F.scaled_dot_product_attention(q, k, v)
    h = h.permute(0, 2, 1, 3)
    h = h.reshape(B, L, -1)
    h = self.to_out(h)
    if not (is_text and self._type != 'self') and kv_mask is None:
        ss_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_y'] = h.cpu()
    if not (is_text and self._type != 'self') and kv_mask is not None and (self._type == 'self'):
        src_idx_of_dst = kv_mask.get('src_idx_of_dst_tgt', None)
        dst_idx = kv_mask.get('dst_idx_tgt', None)
        A = kv_mask.get('A_tgt', None)
        if src_idx_of_dst is not None and dst_idx is not None and (f'{t_latent}_{order}_{pos}_{layer}_{self._type}_y' in ss_kv):
            dst_idx = dst_idx.to(device=h.device, dtype=torch.long)
            src_idx = src_idx_of_dst.index_select(0, dst_idx).to(device=h.device, dtype=torch.long)
            y_cache = ss_kv[f'{t_latent}_{order}_{pos}_{layer}_{self._type}_y'].to(device=h.device, dtype=h.dtype)
            y_src = y_cache.index_select(1, src_idx)
            ht = float(t_latent)
            A = A.to(device=h.device, dtype=h.dtype)
            gamma = (ht * A).view(1, -1, 1).expand(h.shape[0], -1, 1)
            y_dst = h.index_select(1, dst_idx)
            y_new = (1.0 - gamma) * y_dst + gamma * y_src
            h[:, dst_idx, :] = y_new
    return h

def ss_trsfmr_forward(self, x, mod, context, ss_kv, self_kv_mask, cross_kv_mask, t_latent, order, pos, layer, is_text, indices=None):
    if self.share_mod:
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = mod.chunk(6, dim=1)
    else:
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(mod).chunk(6, dim=1)
    h = self.norm1(x)
    h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
    h = self.self_attn(h, ss_kv=ss_kv, kv_mask=self_kv_mask, t_latent=t_latent, order=order, pos=pos, layer=layer, is_text=is_text, indices=indices)
    h = h * gate_msa.unsqueeze(1)
    x = x + h
    h = self.norm2(x)
    cross_kv_mask_dict = None if cross_kv_mask is None else {'mask': cross_kv_mask}
    h = self.cross_attn(h, context, ss_kv=ss_kv, kv_mask=cross_kv_mask_dict, t_latent=t_latent, order=order, pos=pos, layer=layer, is_text=is_text, indices=indices)
    x = x + h
    h = self.norm3(x)
    h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
    h = self.mlp(h)
    h = h * gate_mlp.unsqueeze(1)
    x = x + h
    return x

def ss_flow_forward(self, x, t, cond, ss_kv, self_kv_mask, cross_kv_mask, t_latent, order, pos, is_text):
    assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], f'Input shape mismatch, got {x.shape}, expected {[x.shape[0], self.in_channels, *[self.resolution] * 3]}'
    h = patchify(x, self.patch_size)
    h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()
    h = self.input_layer(h)
    open_debug = False
    if open_debug:
        key_prepos = f'{t_latent}_{order}_{pos}_h_cache'
        if self_kv_mask is None:
            ss_kv[key_prepos] = h.cpu()
        elif key_prepos in ss_kv:
            h_cache = ss_kv[key_prepos].to(device=h.device, dtype=h.dtype)
            dst_idx_all = self_kv_mask.get('dst_idx_all', None)
            src_map = self_kv_mask.get('src_idx_of_dst_all', None)
            A = self_kv_mask.get('A_all', None)
            if dst_idx_all is not None and src_map is not None and (dst_idx_all.numel() > 0):
                dst_idx_all = dst_idx_all.to(device=h.device, dtype=torch.long)
                src_idx_all = src_map.index_select(0, dst_idx_all).to(device=h.device, dtype=torch.long)
                h_src = h_cache.index_select(1, src_idx_all)
                h_dst = h.index_select(1, dst_idx_all)
                if A is not None:
                    A = A.to(device=h.device, dtype=h.dtype)
                    ht = float(t_latent)
                    gamma = (ht * A).clamp(0.0, 1.0).view(1, -1, 1)
                    h_new = (1.0 - gamma) * h_dst + gamma * h_src
                    h[:, dst_idx_all, :] = h_new
                else:
                    h[:, dst_idx_all, :] = h_src
    h = h + self.pos_emb[None]
    t_emb = self.t_embedder(t)
    if self.share_mod:
        t_emb = self.adaLN_modulation(t_emb)
    t_emb = t_emb.type(self.dtype)
    h = h.type(self.dtype)
    cond = cond.type(self.dtype)
    B = h.shape[0]
    G = self.resolution // self.patch_size
    L = G * G * G
    device = h.device
    zs = torch.arange(G, device=device)
    ys = torch.arange(G, device=device)
    xs = torch.arange(G, device=device)
    (grid_x, grid_y, grid_z) = torch.meshgrid(xs, ys, zs, indexing='ij')
    coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(L, 3)
    indices = coords.unsqueeze(0).expand(B, -1, -1).contiguous()
    for (layer, block) in enumerate(self.blocks):
        h = block(h, t_emb, cond, ss_kv, self_kv_mask, cross_kv_mask, t_latent, order, pos, layer, is_text, indices=indices)
    h = h.type(x.dtype)
    h = F.layer_norm(h, h.shape[-1:])
    h = self.out_layer(h)
    h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3)
    h = unpatchify(h, self.patch_size).contiguous()
    return h

def sample_sparse_structure_inverse(pipeline, cond_src, voxel_src, cfg_strength_stage_1_inverse, skip_step, is_text):
    stage = 1
    flow_model = pipeline.models['sparse_structure_flow_model']
    encoder = pipeline.models['sparse_structure_encoder']
    z_s = encoder(voxel_src)
    sigma_min = pipeline.sparse_structure_sampler.sigma_min
    sparse_structure_sampler = InversionFlowEulerGuidanceIntervalSampler(sigma_min)
    if cfg_strength_stage_1_inverse is None:
        cfg_strength = pipeline.sparse_structure_sampler_params['cfg_strength']
    else:
        cfg_strength = cfg_strength_stage_1_inverse
    (noise, ss_latent, ss_kv) = sparse_structure_sampler.sample(flow_model, stage, z_s, cond_src, cfg_strength, skip_step=skip_step, is_text=is_text)
    return (noise, ss_latent, ss_kv)

def sample_sparse_structure_denoise(pipeline, cond_tgt, noise, voxel_src, voxel_mask, ss_latent, ss_latent_mask, ss_kv, ss_self_kv_mask, ss_cross_kv_mask, cfg_strength_stage_1_forward, skip_step, re_init, is_text):
    stage = 1
    flow_model = pipeline.models['sparse_structure_flow_model']
    sigma_min = pipeline.sparse_structure_sampler.sigma_min
    sparse_structure_sampler = InversionFlowEulerGuidanceIntervalSampler(sigma_min)
    if cfg_strength_stage_1_forward is None:
        cfg_strength = pipeline.sparse_structure_sampler_params['cfg_strength']
    else:
        cfg_strength = cfg_strength_stage_1_forward
    if re_init:
        encoder = pipeline.models['sparse_structure_encoder']
        noise_init = encoder(voxel_src)
    else:
        noise_init = None
    (z_s, ss_latent, ss_kv) = sparse_structure_sampler.sample(flow_model, stage, noise, cond_tgt, cfg_strength, ss_latent, ss_latent_mask, ss_kv, ss_self_kv_mask, ss_cross_kv_mask, skip_step, noise_init, is_text=is_text)
    decoder = pipeline.models['sparse_structure_decoder']
    voxel = decoder(z_s)
    return voxel