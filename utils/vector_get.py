import torch
from tqdm import tqdm
import numpy as np
import trimesh
from collections import defaultdict
import scipy.ndimage as ndimage
pdist = torch.nn.PairwiseDistance(p=2)
global destination_mask_dict_use
destination_mask_dict_use = None

def get_sphere(mask: torch.Tensor):
    device = mask.device
    single_mask = mask[0, 0]
    voxel_coords = torch.nonzero(single_mask)
    if voxel_coords.numel() == 0:
        center = torch.zeros(3, device=device)
        radius = torch.tensor(0.0, device=device)
    else:
        try:
            points_np = voxel_coords.cpu().numpy()
            point_cloud = trimesh.points.PointCloud(points_np)
            sphere = point_cloud.bounding_sphere
            radius_val = sphere.primitive.radius
            center_np = sphere.center
            center = torch.from_numpy(center_np).to(device, dtype=torch.float32)
            radius = torch.tensor(radius_val, device=device, dtype=torch.float32)
        except Exception as e:
            center = torch.zeros(3, device=device)
            radius = torch.tensor(0.0, device=device)
    return (center, radius)

def get_scale_factor(C, A, AO, d_AO, R, O):
    AC = C - A
    d_AC = torch.norm(AC)
    e_AC = AC / d_AC
    if d_AO < 1e-05:
        e_AO = torch.tensor([0.0, 0.0, 0.0], device=C.device)
    else:
        e_AO = AO / d_AO
    COS = torch.dot(e_AC, e_AO)
    SIN = torch.sqrt(1 - COS ** 2)
    h = d_AO * SIN
    s = torch.sqrt(R ** 2 - h ** 2)
    L0 = d_AO * COS + s
    d_AP = L0
    d_PC = d_AP - d_AC
    scale_factor = d_PC / d_AP
    return scale_factor

def fastdrag_elastic_displacement_vectorized(A_points, B_points, O, R, C_points, mask_cp_handle):
    move_vectors = []
    move_vectors_radio = []
    device = mask_cp_handle.device
    for point_i in range(len(A_points)):
        A = A_points[point_i].to(device=device)
        B = B_points[point_i].to(device=device)
        shift_xyz = B - A
        print('shift_xyz:', shift_xyz)
        AO = O - A
        d_AO = torch.norm(AO)
        for (j, index) in enumerate(tqdm(C_points, desc='get factor')):
            C = index[-3:]
            if torch.norm(C - A) < 1e-05:
                scale_factor = torch.tensor(1.0, device=device)
            else:
                scale_factor = get_scale_factor(C, A, AO, d_AO, R, O)
            move_vector = scale_factor * shift_xyz
            if len(move_vectors) <= j:
                move_vectors.append([move_vector])
                move_vectors_radio.append([1 / (torch.norm(C - A) + 0.0001)])
            else:
                move_vectors[j].append(move_vector)
                move_vectors_radio[j].append(1 / (torch.norm(C - A) + 0.0001))
    displacements_matrix = torch.stack([torch.stack(row, dim=0) for row in move_vectors], dim=0)
    radio_matrix = torch.tensor(move_vectors_radio, device=device, dtype=torch.float32)
    return (displacements_matrix, radio_matrix)

def generate_explicit_correspondence_map(invert_code, handle_points, target_points, mask_cp_handle):
    device = invert_code.device
    (batch_size, channels, d, h, w) = invert_code.shape
    updated_latent = torch.randn_like(invert_code)
    M_map = torch.zeros((d, h, w, 3), device=device, dtype=torch.long)
    A_map = torch.zeros((d, h, w), device=device, dtype=torch.float32)
    voxel_presence = (invert_code[0, 0] != 0).float()
    mask_np = mask_cp_handle[0, 0].detach().cpu().numpy()
    struct = ndimage.generate_binary_structure(3, 3)
    (labeled_array, num_features) = ndimage.label(mask_np, structure=struct)
    print(f'Connected components analysis complete: detected {num_features} independent editable regions.')
    if num_features == 0:
        return (M_map, A_map, {})
    handle_points_tensor = torch.stack([torch.tensor(p, device=device, dtype=torch.float32) for p in handle_points])
    target_points_tensor = torch.stack([torch.tensor(p, device=device, dtype=torch.float32) for p in target_points])
    destination_sources = {}
    global_expanded_circle = torch.zeros((d, h, w), device=device, dtype=torch.bool)
    editable_mask = torch.zeros((d, h, w), device=device, dtype=torch.bool)
    all_editable_points = []
    all_radio_matrices = []
    all_displacements_matrices = []
    for comp_idx in range(1, num_features + 1):
        comp_mask_np = labeled_array == comp_idx
        comp_mask_tensor = torch.from_numpy(comp_mask_np).to(device)
        comp_mask_5d = torch.zeros_like(mask_cp_handle)
        comp_mask_5d[0, 0] = comp_mask_tensor
        (O, R) = get_sphere(comp_mask_5d)
        (z_grid, y_grid, x_grid) = torch.meshgrid(torch.arange(d, device=device), torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        distances_to_center = torch.sqrt((z_grid - O[0]) ** 2 + (y_grid - O[1]) ** 2 + (x_grid - O[2]) ** 2)
        comp_expanded_circle = distances_to_center <= R + 1
        global_expanded_circle = global_expanded_circle | comp_expanded_circle
        comp_editable_indices = torch.nonzero(comp_mask_5d).to(device=device)
        m = comp_editable_indices.shape[0]
        if m == 0:
            continue
        comp_coords_3d = comp_editable_indices[:, 2:5]
        editable_mask[comp_coords_3d[:, 0], comp_coords_3d[:, 1], comp_coords_3d[:, 2]] = True
        valid_handle_indices = []
        num_total_handles = len(handle_points_tensor)
        for (h_idx, h_pt) in enumerate(handle_points_tensor):
            (hz, hy, hx) = torch.round(h_pt).long()
            hz = torch.clamp(hz, 0, d - 1)
            hy = torch.clamp(hy, 0, h - 1)
            hx = torch.clamp(hx, 0, w - 1)
            if comp_mask_5d[0, 0, hz, hy, hx]:
                valid_handle_indices.append(h_idx)
        (displacements_matrix, radio_matrix) = fastdrag_elastic_displacement_vectorized(A_points=handle_points_tensor, B_points=target_points_tensor, O=O, R=R, C_points=comp_editable_indices, mask_cp_handle=comp_mask_5d)
        invalid_handle_indices = [i for i in range(num_total_handles) if i not in valid_handle_indices]
        if len(invalid_handle_indices) > 0:
            radio_matrix[:, invalid_handle_indices] = 0.0
        if len(valid_handle_indices) == 0:
            winner_vectors = torch.zeros((m, 3), device=device, dtype=torch.float32)
            winner_indices = torch.zeros(m, device=device, dtype=torch.long)
            winner_weights = torch.zeros(m, device=device, dtype=torch.float32)
        else:
            max_weight_idx = torch.argmax(radio_matrix, dim=1)
            winner_vectors = displacements_matrix[torch.arange(m), max_weight_idx]
            winner_vectors = winner_vectors.to(device=device, dtype=torch.float32)
            winner_indices = max_weight_idx
            winner_weights = radio_matrix[torch.arange(m), max_weight_idx]
            winner_weights = torch.clamp(winner_weights, max=1.0)
        editable_points = comp_editable_indices[:, -3:].float().to(device=device)
        all_editable_points.append(editable_points)
        all_radio_matrices.append(radio_matrix)
        all_displacements_matrices.append(displacements_matrix)
        destination_coords = torch.round(editable_points + winner_vectors).long()
        destination_coords[:, 0] = torch.clamp(destination_coords[:, 0], 0, d - 1)
        destination_coords[:, 1] = torch.clamp(destination_coords[:, 1], 0, h - 1)
        destination_coords[:, 2] = torch.clamp(destination_coords[:, 2], 0, w - 1)
        z_coords = comp_coords_3d[:, 0]
        y_coords = comp_coords_3d[:, 1]
        x_coords = comp_coords_3d[:, 2]
        for j in range(m):
            source_coord = (z_coords[j].item(), y_coords[j].item(), x_coords[j].item())
            dest_coord = (destination_coords[j, 0].item(), destination_coords[j, 1].item(), destination_coords[j, 2].item())
            has_voxel = voxel_presence[source_coord].item() > 0.0
            if dest_coord not in destination_sources:
                destination_sources[dest_coord] = []
            destination_sources[dest_coord].append({'source': source_coord, 'weight': winner_weights[j].item(), 'handle_idx': winner_indices[j].item(), 'has_voxel': has_voxel})
    destination_mask = torch.zeros((d, h, w), device=device, dtype=torch.bool)
    src_to_dsts = defaultdict(list)
    dst_to_src = {}
    print('start to check and resolve WTA...')
    for (dest_coord, sources) in destination_sources.items():
        (dest_z, dest_y, dest_x) = dest_coord
        if not editable_mask[dest_z, dest_y, dest_x]:
            continue
        voxel_sources = [s for s in sources if s['has_voxel']]
        candidate_sources = voxel_sources if len(voxel_sources) > 0 else sources
        if len(candidate_sources) == 1:
            winner = candidate_sources[0]
        else:
            max_weight = -1
            winner = None
            for source in candidate_sources:
                if source['weight'] > max_weight:
                    max_weight = source['weight']
                    winner = source
        (src_z, src_y, src_x) = winner['source']
        M_map[dest_z, dest_y, dest_x, 0] = src_z
        M_map[dest_z, dest_y, dest_x, 1] = src_y
        M_map[dest_z, dest_y, dest_x, 2] = src_x
        src_coord = (src_z, src_y, src_x)
        dst_coord = (dest_z, dest_y, dest_x)
        dst_to_src[dst_coord] = src_coord
        src_to_dsts[src_coord].append(dst_coord)
        A_map[dest_z, dest_y, dest_x] = min(1.0, winner['weight'])
        destination_mask[dest_z, dest_y, dest_x] = True
    transition_mask = global_expanded_circle & ~editable_mask
    region_masks = {'destination': destination_mask, 'inpainting': editable_mask & ~destination_mask, 'transition': transition_mask, 'background': ~global_expanded_circle}
    print('Region counts - Destination: {}, Inpainting: {}, Transition: {}, Background: {}'.format(torch.sum(region_masks['destination']).item(), torch.sum(region_masks['inpainting']).item(), torch.sum(region_masks['transition']).item(), torch.sum(region_masks['background']).item()))
    print("Constructing updated latent with paper's rules...")
    bg_trans_mask = region_masks['background'] | region_masks['transition']
    updated_latent[:, :, bg_trans_mask] = invert_code[:, :, bg_trans_mask]
    dest_indices = torch.where(destination_mask)
    if len(dest_indices[0]) > 0:
        (dest_z, dest_y, dest_x) = dest_indices
        src_z = M_map[dest_z, dest_y, dest_x, 0]
        src_y = M_map[dest_z, dest_y, dest_x, 1]
        src_x = M_map[dest_z, dest_y, dest_x, 2]
        updated_latent[:, :, dest_z, dest_y, dest_x] = invert_code[:, :, src_z, src_y, src_x]
    print('Corrected LazyDrag implementation completed!')
    global destination_mask_dict_use
    if len(all_editable_points) > 0:
        global_editable_points = torch.cat(all_editable_points, dim=0)
        global_radio_matrix = torch.cat(all_radio_matrices, dim=0)
        global_displacements_matrix = torch.cat(all_displacements_matrices, dim=0)
        destination_mask_dict_use = destination_mask_dict(destination_mask, handle_points_tensor, global_editable_points, global_radio_matrix.to(device=device), global_displacements_matrix.to(device=device))
    return (M_map, A_map, region_masks)
import json

def save_lazydrag_maps_to_json(M_map, A_map, region_masks, json_save_path, handle_points, target_points):

    def _tensor2list(tensor):
        return tensor.detach().cpu().numpy().tolist()

    def _pts_to_list(pts):
        if pts is None:
            return None
        out = []
        for p in pts:
            if hasattr(p, 'detach'):
                p = p.detach().cpu().numpy()
            if hasattr(p, 'tolist'):
                p = p.tolist()
            p = list(p)
            if len(p) != 3:
                raise ValueError(f'Point must be length-3, got {p}')
            out.append([int(p[0]), int(p[1]), int(p[2])])
        return out
    hp = _pts_to_list(handle_points)
    tp = _pts_to_list(target_points)
    if (hp is None) ^ (tp is None):
        raise ValueError('handle_points and target_points must be both provided or both None.')
    if hp is not None and len(hp) != len(tp):
        raise ValueError(f'len(handle_points) != len(target_points): {len(hp)} vs {len(tp)}')
    json_data = {'M_map': _tensor2list(M_map), 'A_map': _tensor2list(A_map), 'region_masks': {k: _tensor2list(v) for (k, v) in region_masks.items()}, 'handle_points': hp, 'target_points': tp, 'meta': {'d': M_map.shape[0], 'h': M_map.shape[1], 'w': M_map.shape[2], 'M_map_dtype': 'long', 'A_map_dtype': 'float32', 'region_masks_dtype': 'bool'}}
    with open(json_save_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f'✅ Data successfully saved to JSON: {json_save_path}')
    return json_save_path

def destination_mask_dict(destination_mask, handle_points, editable_points, radio_matrix, displacements_matrix):
    winner_handle_indices = torch.argmax(radio_matrix, dim=1)
    num_handles = len(handle_points)
    destination_masks_by_handle = []
    (d, h, w) = destination_mask.shape
    device = destination_mask.device
    for handle_idx in range(num_handles):
        handle_mask = torch.zeros((d, h, w), device=device, dtype=torch.bool)
        handle_point_indices = torch.where(winner_handle_indices == handle_idx)[0]
        for point_idx in handle_point_indices:
            editable_point = editable_points[point_idx]
            winner_vector = displacements_matrix[point_idx, handle_idx]
            destination_coord = torch.round(editable_point + winner_vector).long()
            dest_z = torch.clamp(destination_coord[0], 0, d - 1).item()
            dest_y = torch.clamp(destination_coord[1], 0, h - 1).item()
            dest_x = torch.clamp(destination_coord[2], 0, w - 1).item()
            if destination_mask[dest_z, dest_y, dest_x]:
                handle_mask[dest_z, dest_y, dest_x] = True
        destination_masks_by_handle.append(handle_mask)
    return destination_masks_by_handle
import json
import numpy as np
from collections import defaultdict

def downsample_lazydrag_json(input_json_path, output_json_path):
    print('=' * 60)
    print(f'🚀 Start downsampling process: {input_json_path}')
    print('=' * 60)
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    (D, H, W) = (data['meta']['d'], data['meta']['h'], data['meta']['w'])
    (trunc_D, trunc_H, trunc_W) = (D - D % 2, H - H % 2, W - W % 2)
    (new_D, new_H, new_W) = (trunc_D // 2, trunc_H // 2, trunc_W // 2)
    M_map_orig = np.array(data['M_map'])[:trunc_D, :trunc_H, :trunc_W]
    A_map_orig = np.array(data['A_map'])[:trunc_D, :trunc_H, :trunc_W]
    bg_orig = np.array(data['region_masks']['background'])[:trunc_D, :trunc_H, :trunc_W]
    tr_orig = np.array(data['region_masks']['transition'])[:trunc_D, :trunc_H, :trunc_W]
    inp_orig = np.array(data['region_masks']['inpainting'])[:trunc_D, :trunc_H, :trunc_W]
    dest_orig = np.array(data['region_masks']['destination'])[:trunc_D, :trunc_H, :trunc_W]
    edit_orig = inp_orig | dest_orig
    print('\n[Stage 1: Original high-dimensional space statistics]')
    print(f'  => Original resolution: {D}x{H}x{W} (After truncation: {trunc_D}x{trunc_H}x{trunc_W})')
    print(f'  => Original Dst: {dest_orig.sum()}, Original Inp: {inp_orig.sum()}')
    print(f'  => Original total edit regions (Inp + Dest): {edit_orig.sum()}')
    print('\n[Stage 2: Merge 2x2x2 sub-grids to establish new region attributes]')
    edit_reshaped = edit_orig.reshape(new_D, 2, new_H, 2, new_W, 2)
    tr_reshaped = tr_orig.reshape(new_D, 2, new_H, 2, new_W, 2)
    edit_down = edit_reshaped.sum(axis=(1, 3, 5)) > 0
    tr_down = (tr_reshaped.sum(axis=(1, 3, 5)) > 0) & ~edit_down
    bg_down = ~(edit_down | tr_down)
    print(f'  => Total downsampled edit region (Edit) grids: {edit_down.sum()}')
    print('\n[Stage 3: Extract mapping relationships of 8 sub-grids (induce all Src)]')
    valid_orig_dsts = np.argwhere(dest_orig)
    valid_orig_srcs = M_map_orig[valid_orig_dsts[:, 0], valid_orig_dsts[:, 1], valid_orig_dsts[:, 2]]
    valid_orig_weights = A_map_orig[valid_orig_dsts[:, 0], valid_orig_dsts[:, 1], valid_orig_dsts[:, 2]]
    down_dsts = valid_orig_dsts // 2
    down_srcs = valid_orig_srcs // 2
    dst_candidates = defaultdict(list)
    for i in range(len(valid_orig_dsts)):
        d_d = tuple(down_dsts[i])
        s_d = tuple(down_srcs[i])
        w = valid_orig_weights[i]
        dst_candidates[d_d].append({'src': s_d, 'weight': w})
    print(f'  => Processed {len(valid_orig_dsts)} original correspondences, distributed in lower-dimensional grids.')
    print('\n[Stage 4: Traverse and determine Dst and Inp within the new edit region]')
    M_map_down = np.zeros((new_D, new_H, new_W, 3), dtype=int)
    A_map_down = np.zeros((new_D, new_H, new_W), dtype=float)
    dest_mask_down = np.zeros((new_D, new_H, new_W), dtype=bool)
    inp_mask_down = np.zeros((new_D, new_H, new_W), dtype=bool)
    edit_indices = np.argwhere(edit_down)
    count_inp = 0
    count_dst = 0
    count_multi_src = 0
    for idx in edit_indices:
        (dz, dy, dx) = tuple(idx)
        cands = dst_candidates.get((dz, dy, dx), [])
        if len(cands) == 0:
            inp_mask_down[dz, dy, dx] = True
            count_inp += 1
        else:
            if len(cands) > 1:
                count_multi_src += 1
            best_src = None
            max_dist = -1
            best_weight = 0
            for cand in cands:
                (s_z, s_y, s_x) = cand['src']
                dist = np.linalg.norm(np.array([dz, dy, dx]) - np.array([s_z, s_y, s_x]))
                if dist > max_dist:
                    max_dist = dist
                    best_src = [s_z, s_y, s_x]
                    best_weight = cand['weight']
            dest_mask_down[dz, dy, dx] = True
            M_map_down[dz, dy, dx] = best_src
            A_map_down[dz, dy, dx] = best_weight
            count_dst += 1
    print(f'  => Traversed {len(edit_indices)} downsampled edit grids:')
    print(f'     - Grids with 0 Src, determined as Inp: {count_inp}')
    print(f'     - Grids with >=1 Src, determined as Dst: {count_dst}')
    print(f'     - (Among which multi-Src distance competition occurred: {count_multi_src})')
    print('\n[Stage 5: Final output]')
    hp = [[int(c // 2) for c in p] for p in data.get('handle_points', [])]
    tp = [[int(c // 2) for c in p] for p in data.get('target_points', [])]
    json_data = {'M_map': M_map_down.tolist(), 'A_map': A_map_down.tolist(), 'region_masks': {'destination': dest_mask_down.tolist(), 'inpainting': inp_mask_down.tolist(), 'transition': tr_down.tolist(), 'background': bg_down.tolist()}, 'handle_points': hp, 'target_points': tp, 'meta': {'d': new_D, 'h': new_H, 'w': new_W, 'M_map_dtype': 'long', 'A_map_dtype': 'float32', 'region_masks_dtype': 'bool'}}
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f'✅ Data successfully saved to: {output_json_path}')
    print('=' * 60)
    return output_json_path