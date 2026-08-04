import numpy as np
import torch
from typing import List, Dict, Tuple
from utils.utils import CUBE_SIZE, MASK_SHAPE
from utils.vector_get import generate_explicit_correspondence_map
import open3d as o3d
from pathlib import Path
import os
confirmed_spheres: List[Dict[str, float]] = []
confirmed_cubes_center: List[Dict[str, float]] = []
confirmed_cubes_corner: List[Dict[str, float]] = []
global_mask: np.ndarray = np.zeros(MASK_SHAPE, dtype=np.uint8)
confirmed_arrows: List[Dict] = []
current_handle_point = [32, 32, 32]
current_target_point = [48, 48, 48]
has_unsaved_arrow = False
vector_field_data = None
region_masks_data = None
M_map_data = None
A_map_data = None
vector_display_percentage = 20
Model_input = np.zeros((CUBE_SIZE, CUBE_SIZE, CUBE_SIZE), dtype=np.float32)
Model_was_loaded = False
Model_output = None
Model_was_out = False
Model_voxel_pcd = None
Model_name = ''
Model_suffix = ''

def update_global_mask() -> None:
    global global_mask, confirmed_spheres, confirmed_cubes_center, confirmed_cubes_corner
    global_mask = np.zeros(MASK_SHAPE, dtype=np.uint8)
    if not confirmed_spheres and (not confirmed_cubes_center) and (not confirmed_cubes_corner):
        return
    for cube in confirmed_cubes_center:
        (x1, y1, z1) = (int(cube['x1']) - int(cube['x2']), int(cube['y1']) - int(cube['y2']), int(cube['z1']) - int(cube['z2']))
        (x2, y2, z2) = (int(cube['x1']) + int(cube['x2']), int(cube['y1']) + int(cube['y2']), int(cube['z1']) + int(cube['z2']))
        x1 = np.clip(x1, 0, CUBE_SIZE)
        y1 = np.clip(y1, 0, CUBE_SIZE)
        z1 = np.clip(z1, 0, CUBE_SIZE)
        x2 = np.clip(x2, 0, CUBE_SIZE)
        y2 = np.clip(y2, 0, CUBE_SIZE)
        z2 = np.clip(z2, 0, CUBE_SIZE)
        global_mask[x1:x2 + 1, y1:y2 + 1, z1:z2 + 1] = 1
    for cube in confirmed_cubes_corner:
        (x1, y1, z1) = (int(cube['x1']), int(cube['y1']), int(cube['z1']))
        (x2, y2, z2) = (int(cube['x2']), int(cube['y2']), int(cube['z2']))
        x1 = np.clip(min(x1, x2), 0, CUBE_SIZE)
        y1 = np.clip(min(y1, y2), 0, CUBE_SIZE)
        z1 = np.clip(min(z1, z2), 0, CUBE_SIZE)
        x2 = np.clip(max(x1, x2), 0, CUBE_SIZE)
        y2 = np.clip(max(y1, y2), 0, CUBE_SIZE)
        z2 = np.clip(max(z1, z2), 0, CUBE_SIZE)
        global_mask[x1:x2 + 1, y1:y2 + 1, z1:z2 + 1] = 1
    for sphere in confirmed_spheres:
        (sx, sy, sz) = (sphere['x'], sphere['y'], sphere['z'])
        sr = sphere['radius']
        x = np.arange(0, 64, 1, dtype=np.float32)
        y = np.arange(0, 64, 1, dtype=np.float32)
        z = np.arange(0, 64, 1, dtype=np.float32)
        (X, Y, Z) = np.meshgrid(x, y, z, indexing='ij')
        (X, Y, Z) = (X.flatten(), Y.flatten(), Z.flatten())
        dist_sq = (X - sx) ** 2 + (Y - sy) ** 2 + (Z - sz) ** 2
        sphere_mask = (dist_sq <= sr ** 2).astype(np.uint8)
        sphere_mask_3d = sphere_mask.reshape(MASK_SHAPE)
        global_mask = np.logical_or(global_mask, sphere_mask_3d).astype(np.uint8)

def add_sphere_core(x: float, y: float, z: float, radius: float) -> str:
    global confirmed_spheres
    x = np.clip(x, 0, CUBE_SIZE)
    y = np.clip(y, 0, CUBE_SIZE)
    z = np.clip(z, 0, CUBE_SIZE)
    radius = np.clip(radius, 1, 32)
    confirmed_spheres.append({'x': x, 'y': y, 'z': z, 'radius': radius})
    update_global_mask()
    return f'✅ Added voxelized sphere {len(confirmed_spheres)}: radius {radius:.1f} voxels'

def clear_core() -> str:
    global confirmed_spheres, vector_field_data, region_masks_data, M_map_data, A_map_data, confirmed_cubes_center, confirmed_cubes_corner
    confirmed_cubes_center = []
    confirmed_cubes_corner = []
    confirmed_spheres = []
    vector_field_data = None
    region_masks_data = None
    M_map_data = None
    A_map_data = None
    update_global_mask()
    return '🗑️ Cleared all voxelized spheres, Mask reset to all 0s'

def start_new_arrow_core() -> str:
    global current_handle_point, current_target_point, has_unsaved_arrow
    current_handle_point = [32, 32, 32]
    current_target_point = [48, 48, 48]
    has_unsaved_arrow = True
    return '🆕 Start creating a new arrow, please adjust the start and end positions'

def confirm_arrow_core() -> str:
    global confirmed_arrows, current_handle_point, current_target_point, has_unsaved_arrow
    handle = tuple(current_handle_point)
    target = tuple(current_target_point)
    if not all((0 <= coord <= CUBE_SIZE for coord in handle + target)):
        return '❌ Coordinates out of range (0-63)'
    new_arrow = {'handle': handle, 'target': target, 'index': len(confirmed_arrows)}
    confirmed_arrows.append(new_arrow)
    has_unsaved_arrow = False
    return f'✅ Added arrow {len(confirmed_arrows)}: {handle}→{target}'

def update_handle_point_core(x: int, y: int, z: int) -> None:
    global current_handle_point, has_unsaved_arrow
    current_handle_point = [int(x), int(y), int(z)]
    has_unsaved_arrow = True

def update_target_point_core(x: int, y: int, z: int) -> None:
    global current_target_point, has_unsaved_arrow
    current_target_point = [int(x), int(y), int(z)]
    has_unsaved_arrow = True

def run_vector_field_generation_core() -> Tuple[bool, str, any, any, any, any]:
    global vector_field_data, region_masks_data, M_map_data, A_map_data, confirmed_arrows, global_mask
    if len(confirmed_arrows) == 0:
        return (False, '❌ Please add at least one arrow first', None, None, None, None)
    if np.sum(global_mask) == 0:
        return (False, '❌ Please add at least one sphere first', None, None, None, None)
    handle_points_list = [[arrow['handle'][2], arrow['handle'][1], arrow['handle'][0]] for arrow in confirmed_arrows]
    target_points_list = [[arrow['target'][2], arrow['target'][1], arrow['target'][0]] for arrow in confirmed_arrows]
    mask_tensor = torch.from_numpy(global_mask).unsqueeze(0).unsqueeze(0).long()
    if Model_was_loaded:
        device = mask_tensor.device
        invert_code = torch.from_numpy(Model_input).unsqueeze(0).unsqueeze(0).to(device)
    else:
        invert_code = torch.zeros(mask_tensor.shape, dtype=torch.float32)
    print('Starting to generate vector field...')
    (M_map, A_map, region_masks) = generate_explicit_correspondence_map(invert_code=invert_code, handle_points=handle_points_list, target_points=target_points_list, mask_cp_handle=mask_tensor)
    for key in region_masks:
        if isinstance(region_masks[key], torch.Tensor):
            region_masks[key] = region_masks[key].to(torch.long)
    vector_field_data = 'generated'
    region_masks_data = region_masks
    M_map_data = M_map
    A_map_data = A_map
    return (True, '✅ Vector field generated successfully!', M_map, A_map, region_masks, vector_field_data)

def add_cube_core_center(p1, p2):
    global confirmed_cubes_center
    (x1, y1, z1) = p1
    (x2, y2, z2) = p2
    x1 = np.clip(x1, 0, CUBE_SIZE)
    y1 = np.clip(y1, 0, CUBE_SIZE)
    z1 = np.clip(z1, 0, CUBE_SIZE)
    x2 = np.clip(x2, 0, CUBE_SIZE)
    y2 = np.clip(y2, 0, CUBE_SIZE)
    z2 = np.clip(z2, 0, CUBE_SIZE)
    confirmed_cubes_center.append({'x1': x1, 'y1': y1, 'z1': z1, 'x2': x2, 'y2': y2, 'z2': z2})
    update_global_mask()
    return f'✅ Added voxelized cube {len(confirmed_cubes_center)}'

def add_cube_core_corner(p1, p2):
    global confirmed_cubes_corner
    (x1, y1, z1) = p1
    (x2, y2, z2) = p2
    x1 = np.clip(x1, 0, CUBE_SIZE)
    y1 = np.clip(y1, 0, CUBE_SIZE)
    z1 = np.clip(z1, 0, CUBE_SIZE)
    x2 = np.clip(x2, 0, CUBE_SIZE)
    y2 = np.clip(y2, 0, CUBE_SIZE)
    z2 = np.clip(z2, 0, CUBE_SIZE)
    confirmed_cubes_corner.append({'x1': x1, 'y1': y1, 'z1': z1, 'x2': x2, 'y2': y2, 'z2': z2})
    update_global_mask()
    return f'✅ Added voxelized cube {len(confirmed_cubes_corner)}'

def load_voxels_ply_to_global(ply_file_path):
    global Model_input, Model_was_loaded
    try:
        if not os.path.exists(ply_file_path):
            return (False, f'❌ PLY file does not exist: {ply_file_path}')
        print(f'📖 Reading voxel PLY file: {ply_file_path}')
        voxel_pcd = o3d.io.read_point_cloud(ply_file_path)
        if len(voxel_pcd.points) == 0:
            return (False, '❌ PLY file is empty or failed to read')
        voxel_points = np.asarray(voxel_pcd.points)
        num_voxels = len(voxel_points)
        print(f'🎯 Read {num_voxels} voxel points')
        print(f'📊 Coordinate range: {voxel_points.min(axis=0)} -> {voxel_points.max(axis=0)}')
        Model_input = np.zeros((64, 64, 64), dtype=np.float32)
        processed_points = [tuple((num * 64 + 32 - 0.5 for num in point)) for point in voxel_points]
        voxel_indices = np.round(processed_points).astype(int)
        voxel_indices = np.clip(voxel_indices, 0, CUBE_SIZE - 1)
        valid_mask = np.all((voxel_indices >= 0) & (voxel_indices < 64), axis=1)
        valid_indices = voxel_indices[valid_mask]
        print(f'✅ Valid voxel indices: {len(valid_indices)}/{num_voxels}')
        for idx in valid_indices:
            Model_input[idx[2], idx[1], idx[0]] = 1.0
        Model_was_loaded = True
        actual_voxel_count = np.sum(Model_input)
        print(f'🎉 Voxel loading completed!')
        print(f'📋 Model name: {Model_name}')
        print(f'🧩 Actual voxel count: {actual_voxel_count}')
        print(f'📊 Model_input shape: {Model_input.shape}')
        status_text = f'✅ Successfully loaded voxel PLY file\n🎯 Voxel count: {actual_voxel_count}\n📁 File: {Path(ply_file_path).name}'
        file_path_obj = Path(ply_file_path)
        model_info_text = f'Reference model: {Model_name}\nFormat: {Model_suffix}\nVoxel file: {file_path_obj.name}\nVoxel count: {actual_voxel_count}\nCoordinate range: {voxel_points.min(axis=0).round(2)} -> {voxel_points.max(axis=0).round(2)}\nStatus: Loaded and converted to voxel Mask\nUsage: Only as a visual reference basemap\nOriginal points: {num_voxels}\nValid indices: {len(valid_indices)}'
        return (status_text, model_info_text)
    except Exception as e:
        error_msg = f'❌ Failed to load voxel PLY file: {str(e)}'
        print(error_msg)

def convert_ply_to_voxel(ply_file_path):
    global Model_output, Model_was_out
    try:
        if not os.path.exists(ply_file_path):
            return (False, f'❌ PLY file does not exist: {ply_file_path}')
        print(f'📖 Reading voxel PLY file: {ply_file_path}')
        voxel_pcd = o3d.io.read_point_cloud(ply_file_path)
        if len(voxel_pcd.points) == 0:
            return (False, '❌ PLY file is empty or failed to read')
        voxel_points = np.asarray(voxel_pcd.points)
        num_voxels = len(voxel_points)
        print(f'🎯 Read {num_voxels} voxel points')
        print(f'📊 Coordinate range: {voxel_points.min(axis=0)} -> {voxel_points.max(axis=0)}')
        Model_output = np.zeros((64, 64, 64), dtype=np.float32)
        processed_points = [tuple((num * 64 + 32 for num in point)) for point in voxel_points]
        voxel_indices = np.round(processed_points).astype(int)
        voxel_indices = np.clip(voxel_indices, 0, CUBE_SIZE - 1)
        valid_mask = np.all((voxel_indices >= 0) & (voxel_indices < 64), axis=1)
        valid_indices = voxel_indices[valid_mask]
        print(f'✅ Valid voxel indices: {len(valid_indices)}/{num_voxels}')
        for idx in valid_indices:
            Model_output[idx[2], idx[1], idx[0]] = 1.0
        Model_was_out = True
        actual_voxel_count = np.sum(Model_output)
        print(f'🎉 Voxel loading completed!')
        print(f'🧩 Actual voxel count: {actual_voxel_count}')
        print(f'📊 Model_output shape: {Model_output.shape}')
        status_text = f'✅ Successfully loaded voxel PLY file\n🎯 Voxel count: {actual_voxel_count}\n📁 File: {Path(ply_file_path).name}'
        file_path_obj = Path(ply_file_path)
        model_info_text = f'\nVoxel file: {file_path_obj.name}\nVoxel count: {actual_voxel_count}\nCoordinate range: {voxel_points.min(axis=0).round(2)} -> {voxel_points.max(axis=0).round(2)}\nStatus: Loaded and converted to voxel Mask\nUsage: Only as a visual reference basemap\nOriginal points: {num_voxels}\nValid indices: {len(valid_indices)}'
        return (status_text, model_info_text)
    except Exception as e:
        error_msg = f'❌ Failed to load voxel PLY file: {str(e)}'
        print(error_msg)