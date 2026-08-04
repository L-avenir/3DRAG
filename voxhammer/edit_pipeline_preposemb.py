import os
os.environ['ATTN_BACKEND'] = 'sdpa'
os.environ['SPCONV_ALGO'] = 'native'
import gc
import torch
import utils3d
import numpy as np
import torchvision.transforms as transforms
from typing import *
from types import MethodType
import sys
import plotly.graph_objects as go
root_path = os.path.dirname(os.path.abspath(__file__))
if root_path not in sys.path:
    sys.path.append(root_path)
PROJECT_ROOT = root_path
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
    print(f'✅ Added project path: {PROJECT_ROOT}')
from voxhammer.edit_function.dict_and_utils import match_src_dst_coords
from trellis.utils import postprocessing_utils
from voxhammer.edit_function.dict_and_utils import ply_to_coords, coords_to_voxel, feats_to_slat, ply_to_ss_mask, preprocess_image, load_map_std_from_json, ply_to_slat_mask
from voxhammer.edit_function.slat_edit import slat_flow_forward, slat_trsfmr_forward, slat_attn_forward, slat_attn_logit_forward, sample_slat_inverse, sample_slat_denoise
from voxhammer.edit_function.ss_edit import ss_flow_forward, ss_trsfmr_forward, ss_attn_logit_forward, ss_attn_forward, ss_attn_logit_forward, ss_attn_forward, sample_sparse_structure_inverse, sample_sparse_structure_denoise

def visualize_voxels_3d_plotly(coords_dict, grid_size=64):
    color_palette = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
    fig = go.Figure()
    for (i, (label, coords)) in enumerate(coords_dict.items()):
        if coords is None or len(coords) == 0:
            continue
        if isinstance(coords, torch.Tensor):
            coords = coords.detach().cpu().numpy()
        if coords.shape[1] == 4:
            pts = coords[:, 1:4]
        elif coords.shape[1] == 3:
            pts = coords
        else:
            raise ValueError(f'Shape {coords.shape} of {label} is invalid! Expected (N, 3) or (N, 4)')
        fig.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode='markers', name=label, marker=dict(size=3, color=color_palette[i % len(color_palette)], opacity=0.6)))
    fig.update_layout(scene=dict(xaxis=dict(range=[0, grid_size], title='X Axis'), yaxis=dict(range=[0, grid_size], title='Y Axis'), zaxis=dict(range=[0, grid_size], title='Z Axis'), aspectmode='cube'), title='3D Voxel Alignment Debug (Interactive)', margin=dict(l=0, r=0, b=0, t=40), legend=dict(x=0.8, y=0.9))
    fig.show()

def run_edit(pipeline, render_dir, output_path, image_dir, is_text, source_prompt, target_prompt, map_path=None, skip_step=0, re_init=False, cfg=[5.0, 6.0, 0.0, 0.0], use_logit_blend=False, use_logit_slat=True):
    ss_flow = pipeline.models['sparse_structure_flow_model']
    ss_flow.forward = MethodType(ss_flow_forward, ss_flow)
    for block in ss_flow.blocks:
        trsfmr_obj = block
        trsfmr_obj.forward = MethodType(ss_trsfmr_forward, trsfmr_obj)
        self_attn_obj = block.self_attn
        if use_logit_blend:
            self_attn_obj.forward = MethodType(ss_attn_logit_forward, self_attn_obj)
        else:
            self_attn_obj.forward = MethodType(ss_attn_forward, self_attn_obj)
        cross_attn_obj = block.cross_attn
        if use_logit_blend:
            cross_attn_obj.forward = MethodType(ss_attn_logit_forward, cross_attn_obj)
        else:
            cross_attn_obj.forward = MethodType(ss_attn_forward, cross_attn_obj)
    slat_flow = pipeline.models['slat_flow_model']
    slat_flow.forward = MethodType(slat_flow_forward, slat_flow)
    for block in slat_flow.blocks:
        trsfmr_obj = block
        trsfmr_obj.forward = MethodType(slat_trsfmr_forward, trsfmr_obj)
        self_attn_obj = block.self_attn
        if use_logit_slat and use_logit_blend:
            self_attn_obj.forward = MethodType(slat_attn_logit_forward, self_attn_obj)
        else:
            self_attn_obj.forward = MethodType(slat_attn_forward, self_attn_obj)
        cross_attn_obj = block.cross_attn
        if use_logit_slat and use_logit_blend:
            cross_attn_obj.forward = MethodType(slat_attn_logit_forward, cross_attn_obj)
        else:
            cross_attn_obj.forward = MethodType(slat_attn_forward, cross_attn_obj)
    cfg_strength_stage_1_inverse = cfg[0]
    cfg_strength_stage_1_forward = cfg[1]
    cfg_strength_stage_2_inverse = cfg[2]
    cfg_strength_stage_2_forward = cfg[3]
    coords_src = ply_to_coords(os.path.join(render_dir, 'voxels.ply'))
    voxel_src = coords_to_voxel(coords_src)
    slat_src = feats_to_slat(pipeline, os.path.join(render_dir, 'features.npz'))
    if not is_text:
        img_src_path = os.path.join(image_dir, '2d_render.png')
        img_tgt_path = os.path.join(image_dir, '2d_edit.png')
        img_mask_path = os.path.join(image_dir, '2d_mask.png')
        (pre_src, pre_tgt, pre_mask) = preprocess_image(img_src_path, img_tgt_path, img_mask_path)
        cond_src = pipeline.get_cond([pre_src])
        cond_tgt = pipeline.get_cond([pre_tgt])
    else:
        pre_mask = None
        cond_src = pipeline.get_cond([source_prompt])
        cond_tgt = pipeline.get_cond([target_prompt])
    map_std = load_map_std_from_json(map_path, ss_flow, device='cuda')
    coords_preserve = map_std['keep_voxel_xyz']
    (voxel_mask, ss_latent_mask, ss_self_kv_mask, cross_kv_mask) = ply_to_ss_mask(coords_preserve, pre_mask, map_std['ss'])
    (noise, ss_latent, ss_kv) = sample_sparse_structure_inverse(pipeline, cond_src, voxel_src, cfg_strength_stage_1_inverse, skip_step, is_text)
    voxel_tgt = sample_sparse_structure_denoise(pipeline, cond_tgt, noise, voxel_src, voxel_mask, ss_latent, ss_latent_mask, ss_kv, ss_self_kv_mask, cross_kv_mask, cfg_strength_stage_1_forward, skip_step, re_init, is_text)
    if hasattr(ss_latent, 'cleanup'):
        ss_latent.cleanup()
    if hasattr(ss_kv, 'cleanup'):
        ss_kv.cleanup()
    del ss_latent, ss_kv
    coords_tgt = torch.argwhere(voxel_tgt > 0)[:, [0, 2, 3, 4]].int()
    save_data = {'coords_src': coords_src.cpu().numpy() if isinstance(coords_src, torch.Tensor) else coords_src, 'coords_tgt': coords_tgt.cpu().numpy(), 'coords_preserve': coords_preserve.cpu().numpy() if isinstance(coords_preserve, torch.Tensor) else coords_preserve, 'cfg_strength_stage_1_inverse': cfg_strength_stage_1_inverse, 'cfg_strength_stage_1_forward': cfg_strength_stage_1_forward, 'cfg_strength_stage_2_inverse': cfg_strength_stage_2_inverse, 'cfg_strength_stage_2_forward': cfg_strength_stage_2_forward, 'voxel_mask_shape': voxel_mask.shape, 'pre_mask': pre_mask.cpu().numpy() if pre_mask is not None else None, 'is_text': is_text, 'skip_step': skip_step, 're_init': re_init, 'map_path': map_path, 'source_prompt': source_prompt, 'target_prompt': target_prompt}
    from pathlib import Path
    out = Path(output_path)
    stem = out.with_suffix('') if out.suffix else out
    print(f'Output directory: {out}')
    print(f'File prefix: {stem}')
    os.makedirs(stem, exist_ok=True)
    npz_save_path = os.path.join(stem, 'key_results.npz')
    np.savez(npz_save_path, **{k: v for (k, v) in save_data.items() if v is not None})
    print(f'Core tensors/arrays saved to: {npz_save_path}')
    stage1_save_dir = str(stem)
    os.makedirs(stage1_save_dir, exist_ok=True)
    save_dir = stage1_save_dir
    torch.save(voxel_tgt.detach().cpu(), os.path.join(save_dir, 'st_drag_voxel_tgt.pt'))
    coords_xyz = coords_tgt[:, 1:].detach().cpu().numpy()
    try:
        utils3d.io.write_ply(os.path.join(save_dir, 'st_drag_points_int.ply'), coords_xyz)
    except Exception:
        import trimesh
        trimesh.PointCloud(coords_xyz).export(os.path.join(save_dir, 'st_drag_points_int.ply'))
    pts = coords_xyz.astype('float32') / 64.0 - 0.5
    try:
        utils3d.io.write_ply(os.path.join(save_dir, 'st_drag_points.ply'), pts)
    except Exception:
        import trimesh
        trimesh.PointCloud(pts).export(os.path.join(save_dir, 'st_drag_points.ply'))
    print(f"[Stage1 only] Saved: {os.path.join(save_dir, 'st_drag_voxel_tgt.pt')} and st_drag_points.ply")
    (slat_self_kv_mask, slat_latent_mask) = ply_to_slat_mask(map_std['slat'])
    (slat_self_kv_mask_64, slat_latent_mask_64) = ply_to_slat_mask(map_std['slat_64'])
    dst_coords_scaled = slat_latent_mask_64['dst_coords_tgt'].clone()
    src_coords_scaled = slat_latent_mask_64['src_coords_tgt'].clone()
    match_src_dst_coords(slat_src, coords_tgt, dst_coords_scaled, src_coords_scaled)
    (slat_latent, slat_kv) = sample_slat_inverse(pipeline, cond_src, slat_src, cfg_strength_stage_2_inverse, is_text)
    slat_tgt = sample_slat_denoise(pipeline, cond_tgt, coords_tgt, slat_src, slat_latent_mask_64, slat_latent, slat_kv, slat_self_kv_mask, None, cfg_strength_stage_2_forward, is_text)
    if slat_tgt is None:
        print('SLAT stage output is empty, exiting')
        return
    torch.cuda.empty_cache()
    with torch.no_grad():
        assets_tgt = pipeline.decode_slat(slat_tgt, ['gaussian', 'mesh'])
    if 'gaussian' in assets_tgt and assets_tgt['gaussian'][0] is not None:
        gaussian_asset = assets_tgt['gaussian'][0]
        if hasattr(gaussian_asset, 'xyz') and gaussian_asset.xyz.requires_grad:
            gaussian_asset.xyz = gaussian_asset.xyz.detach()
        if hasattr(gaussian_asset, 'features') and gaussian_asset.features.requires_grad:
            gaussian_asset.features = gaussian_asset.features.detach()
    if 'mesh' in assets_tgt and assets_tgt['mesh'][0] is not None:
        mesh_asset = assets_tgt['mesh'][0]
        if hasattr(mesh_asset, 'vertices') and mesh_asset.vertices.requires_grad:
            mesh_asset.vertices = mesh_asset.vertices.detach()
        if hasattr(mesh_asset, 'faces') and mesh_asset.faces.requires_grad:
            mesh_asset.faces = mesh_asset.faces.detach()
        if hasattr(mesh_asset, 'textures') and mesh_asset.textures.requires_grad:
            mesh_asset.textures = mesh_asset.textures.detach()
    torch.set_grad_enabled(True)
    print('Deleting cuda cache')
    del pipeline
    del slat_tgt, slat_src, slat_latent, slat_kv
    del cond_tgt, cond_src, coords_tgt
    del slat_latent_mask_64, slat_self_kv_mask
    gc.collect()
    torch.cuda.empty_cache()
    glb_tgt = postprocessing_utils.to_glb(assets_tgt['gaussian'][0], assets_tgt['mesh'][0], simplify=0.95, texture_size=1024)
    out = Path(output_path)
    glb_tgt.export(out)
    print(f'SLAT stage completed! Final GLB file saved to: {out}')
if __name__ == '__main__':
    os.environ['ATTN_BACKEND'] = 'sdpa'
    os.environ['SPCONV_ALGO'] = 'native'
    run_edit()