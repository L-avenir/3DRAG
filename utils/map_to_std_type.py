import numpy as np
from typing import *
import json

def map_to_std(raw_json_path, out_json_path, ss_patch_size=4, slat_factor=2):
    with open(raw_json_path, 'r') as f:
        raw = json.load(f)
    M_raw = np.array(raw['M_map'], dtype=np.int64)
    A_raw = np.array(raw['A_map'], dtype=np.float32)
    rm_raw = {k: np.array(v, dtype=bool) for (k, v) in raw['region_masks'].items()}
    G_hi = A_raw.shape[0]
    G_lo = G_hi // ss_patch_size
    L_lo = G_lo ** 3
    bg_coords_hi = np.argwhere(rm_raw['background'])
    keep_voxel_xyz = bg_coords_hi[:, [2, 1, 0]].tolist()

    def get_flat_idx(zyx_coords, G):
        if len(zyx_coords) == 0:
            return []
        return [int((x * G + y) * G + z) for (z, y, x) in zyx_coords]
    region_labels = np.zeros((G_lo, G_lo, G_lo), dtype=int)
    prio_list = ['background', 'transition', 'inpainting', 'destination']
    for z in range(G_lo):
        for y in range(G_lo):
            for x in range(G_lo):
                block = [rm_raw[k][z * 4:(z + 1) * 4, y * 4:(y + 1) * 4, x * 4:(x + 1) * 4] for k in prio_list]
                counts = [np.sum(b) for b in block]
                region_labels[z, y, x] = np.argmax(counts)
    dest_lo_zyx = np.argwhere(region_labels == 3)
    inp_lo_zyx = np.argwhere(region_labels == 2)
    trans_lo_zyx = np.argwhere(region_labels == 1)
    edit_lo_zyx = np.argwhere(region_labels > 0)
    (dst_idx_tgt, src_idx_tgt, A_tgt) = ([], [], [])
    for (z, y, x) in dest_lo_zyx:
        sub_A = A_raw[z * 4:(z + 1) * 4, y * 4:(y + 1) * 4, x * 4:(x + 1) * 4]
        flat_max = np.argmax(sub_A)
        (dz, dy, dx) = np.unravel_index(flat_max, (4, 4, 4))
        (hz, hy, hx) = (z * 4 + dz, y * 4 + dy, x * 4 + dx)
        src_hi = M_raw[hz, hy, hx]
        src_lo = src_hi // 4
        dst_idx_tgt.append(int((x * G_lo + y) * G_lo + z))
        src_idx_tgt.append(int((src_lo[2] * G_lo + src_lo[1]) * G_lo + src_lo[0]))
        A_tgt.append(float(A_raw[hz, hy, hx]))
    dst_idx_trans = get_flat_idx(trans_lo_zyx, G_lo)
    dst_idx_all = dst_idx_tgt + dst_idx_trans
    src_idx_all = src_idx_tgt + dst_idx_trans
    A_all = A_tgt + [0.0] * len(dst_idx_trans)
    mul = ss_patch_size // slat_factor

    def to_slat_coords(zyx_arr):
        if len(zyx_arr) == 0:
            return []
        coords = []
        for (z, y, x) in zyx_arr:
            coords.append([0, int(x * mul), int(y * mul), int(z * mul)])
        return coords
    out = {'keep_voxel_xyz': keep_voxel_xyz, 'ss': {'G': G_lo, 'L': L_lo, 'dst_idx_tgt': dst_idx_tgt, 'src_idx_tgt': src_idx_tgt, 'A_tgt': A_tgt, 'dst_idx_all': dst_idx_all, 'src_idx_all': src_idx_all, 'A_all': A_all, 'inp_idx': get_flat_idx(inp_lo_zyx, G_lo), 'editable_idx': get_flat_idx(edit_lo_zyx, G_lo), 'background_idx': get_flat_idx(np.argwhere(region_labels == 0), G_lo)}, 'slat': {'dst_coords_tgt': to_slat_coords(dest_lo_zyx), 'src_coords_tgt': [], 'A_tgt': A_tgt, 'dst_coords_all': to_slat_coords(np.concatenate([dest_lo_zyx, trans_lo_zyx])) if len(edit_lo_zyx) > 0 else [], 'src_coords_all': [], 'A_all': A_all, 'inp_coords': to_slat_coords(inp_lo_zyx)}}

    def get_slat_src_coords(dest_zyx_list, M_raw_map, mul):
        coords = []
        for (z, y, x) in dest_zyx_list:
            sub_A = A_raw[z * 4:(z + 1) * 4, y * 4:(y + 1) * 4, x * 4:(x + 1) * 4]
            (dz, dy, dx) = np.unravel_index(np.argmax(sub_A), (4, 4, 4))
            src_hi = M_raw_map[z * 4 + dz, y * 4 + dy, x * 4 + dx]
            src_lo = src_hi // 4
            coords.append([0, int(int(src_lo[2]) * mul), int(int(src_lo[1]) * mul), int(int(src_lo[0]) * mul)])
        return coords
    out['slat']['src_coords_tgt'] = get_slat_src_coords(dest_lo_zyx, M_raw, mul)
    trans_src = [[0, int(int(p[2]) * mul), int(int(p[1]) * mul), int(int(p[0]) * mul)] for p in trans_lo_zyx]
    out['slat']['src_coords_all'] = out['slat']['src_coords_tgt'] + trans_src
    with open(out_json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Done! Saved to {out_json_path}')
    return out_json_path

def map_to_std_16(raw_json_path, out_json_path, ss_patch_size=4, slat_factor=2):
    with open(raw_json_path, 'r') as f:
        raw = json.load(f)
    M_raw = np.array(raw['M_map'], dtype=np.int64)
    A_raw = np.array(raw['A_map'], dtype=np.float32)
    rm_raw = {k: np.array(v, dtype=bool) for (k, v) in raw['region_masks'].items()}
    G_hi = A_raw.shape[0]
    G_lo = G_hi
    L_lo = G_lo ** 3
    print(G_lo, L_lo)
    bg_coords_hi = np.argwhere(rm_raw['background'])
    keep_voxel_xyz = bg_coords_hi[:, [2, 1, 0]].tolist()

    def get_flat_idx_16(zyx_coords, G):
        if len(zyx_coords) == 0:
            return []
        return [int((x * G + y) * G + z) for (z, y, x) in zyx_coords]
    region_labels = np.zeros((G_lo, G_lo, G_lo), dtype=int)
    region_labels[rm_raw['destination']] = 3
    region_labels[rm_raw['inpainting']] = 2
    region_labels[rm_raw['transition']] = 1
    region_labels[rm_raw['background']] = 0
    dest_lo_zyx = np.argwhere(rm_raw['destination'])
    inp_lo_zyx = np.argwhere(rm_raw['inpainting'])
    trans_lo_zyx = np.argwhere(rm_raw['transition'])
    edit_lo_zyx = np.argwhere(rm_raw['destination'] | rm_raw['inpainting'] | rm_raw['transition'])
    (dst_idx_tgt, src_idx_tgt, A_tgt) = ([], [], [])
    for (z, y, x) in dest_lo_zyx:
        src_hi = M_raw[z, y, x]
        dst_idx_tgt.append(int((x * G_lo + y) * G_lo + z))
        src_idx_tgt.append(int((src_hi[2] * G_lo + src_hi[1]) * G_lo + src_hi[0]))
        A_tgt.append(float(A_raw[z, y, x]))
    dst_idx_trans = get_flat_idx_16(trans_lo_zyx, G_lo)
    dst_idx_all = dst_idx_tgt + dst_idx_trans
    src_idx_all = src_idx_tgt + dst_idx_trans
    A_all = A_tgt + [0.0] * len(dst_idx_trans)
    mul = ss_patch_size // slat_factor

    def to_slat_coords(zyx_arr):
        if len(zyx_arr) == 0:
            return []
        coords = []
        for (z, y, x) in zyx_arr:
            coords.append([0, int(x * mul), int(y * mul), int(z * mul)])
        return coords
    print(G_lo, L_lo)
    out = {'keep_voxel_xyz': keep_voxel_xyz, 'ss': {'G': G_lo, 'L': L_lo, 'dst_idx_tgt': dst_idx_tgt, 'src_idx_tgt': src_idx_tgt, 'A_tgt': A_tgt, 'dst_idx_all': dst_idx_all, 'src_idx_all': src_idx_all, 'A_all': A_all, 'inp_idx': get_flat_idx_16(inp_lo_zyx, G_lo), 'editable_idx': get_flat_idx_16(edit_lo_zyx, G_lo), 'background_idx': get_flat_idx_16(np.argwhere(region_labels == 0), G_lo)}, 'slat': {'dst_coords_tgt': to_slat_coords(dest_lo_zyx), 'src_coords_tgt': [], 'A_tgt': A_tgt, 'dst_coords_all': to_slat_coords(np.concatenate([dest_lo_zyx, trans_lo_zyx])) if len(edit_lo_zyx) > 0 else [], 'src_coords_all': [], 'A_all': A_all, 'inp_coords': to_slat_coords(inp_lo_zyx)}}

    def get_slat_src_coords_16(dest_zyx_list, M_raw_map, mul):
        coords = []
        for (z, y, x) in dest_zyx_list:
            src_hi = M_raw_map[z, y, x]
            coords.append([0, int(int(src_hi[2]) * mul), int(int(src_hi[1]) * mul), int(int(src_hi[0]) * mul)])
        return coords
    out['slat']['src_coords_tgt'] = get_slat_src_coords_16(dest_lo_zyx, M_raw, mul)
    trans_src = [[0, int(int(p[2]) * mul), int(int(p[1]) * mul), int(int(p[0]) * mul)] for p in trans_lo_zyx]
    out['slat']['src_coords_all'] = out['slat']['src_coords_tgt'] + trans_src
    with open(out_json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Done! Saved to {out_json_path}')
    return out_json_path
import json
import numpy as np
import json
import numpy as np

def map_to_std_16_with_32_and_64(raw_json_path1, raw_json_path2, raw_json_path3, out_json_path):
    with open(raw_json_path1, 'r') as f:
        raw_16 = json.load(f)
    with open(raw_json_path2, 'r') as f:
        raw_32 = json.load(f)
    with open(raw_json_path3, 'r') as f:
        raw_64 = json.load(f)
    M_raw_16 = np.array(raw_16['M_map'], dtype=np.int64)
    A_raw_16 = np.array(raw_16['A_map'], dtype=np.float32)
    rm_raw_16 = {k: np.array(v, dtype=bool) for (k, v) in raw_16['region_masks'].items()}
    G_lo = 16
    L_lo = G_lo ** 3
    bg_coords_hi = np.argwhere(rm_raw_16['background'])
    keep_voxel_xyz = bg_coords_hi[:, [2, 1, 0]].tolist()

    def get_flat_idx_16(zyx_coords, G):
        if len(zyx_coords) == 0:
            return []
        return [int((x * G + y) * G + z) for (z, y, x) in zyx_coords]
    region_labels = np.zeros((G_lo, G_lo, G_lo), dtype=int)
    region_labels[rm_raw_16['destination']] = 3
    region_labels[rm_raw_16['inpainting']] = 2
    region_labels[rm_raw_16['transition']] = 1
    region_labels[rm_raw_16['background']] = 0
    dest_lo_zyx = np.argwhere(rm_raw_16['destination'])
    inp_lo_zyx = np.argwhere(rm_raw_16['inpainting'])
    trans_lo_zyx = np.argwhere(rm_raw_16['transition'])
    edit_lo_zyx = np.argwhere(rm_raw_16['destination'] | rm_raw_16['inpainting'] | rm_raw_16['transition'])
    (dst_idx_tgt, src_idx_tgt, A_tgt) = ([], [], [])
    for (z, y, x) in dest_lo_zyx:
        src_hi = M_raw_16[z, y, x]
        dst_idx_tgt.append(int((x * G_lo + y) * G_lo + z))
        src_idx_tgt.append(int((src_hi[2] * G_lo + src_hi[1]) * G_lo + src_hi[0]))
        A_tgt.append(float(A_raw_16[z, y, x]))
    dst_idx_trans = get_flat_idx_16(trans_lo_zyx, G_lo)
    dst_idx_all = dst_idx_tgt + dst_idx_trans
    src_idx_all = src_idx_tgt + dst_idx_trans
    A_all = A_tgt + [0.0] * len(dst_idx_trans)
    M_raw_32 = np.array(raw_32['M_map'], dtype=np.int64)
    A_raw_32 = np.array(raw_32['A_map'], dtype=np.float32)
    rm_raw_32 = {k: np.array(v, dtype=bool) for (k, v) in raw_32['region_masks'].items()}
    G_32 = 32
    dest_32_zyx = np.argwhere(rm_raw_32['destination'])
    trans_32_zyx = np.argwhere(rm_raw_32['transition'])
    inp_32_zyx = np.argwhere(rm_raw_32['inpainting'])
    back_32_zyx = np.argwhere(rm_raw_32['background'])

    def to_slat_coords_32(zyx_arr):
        if len(zyx_arr) == 0:
            return []
        coords = []
        for (z, y, x) in zyx_arr:
            coords.append([0, int(x), int(y), int(z)])
        return coords

    def get_slat_src_coords_32(dest_zyx_list, M_raw_map):
        coords = []
        for (z, y, x) in dest_zyx_list:
            src_hi = M_raw_map[z, y, x]
            coords.append([0, int(src_hi[2]), int(src_hi[1]), int(src_hi[0])])
        return coords
    slat_dst_coords_tgt = to_slat_coords_32(dest_32_zyx)
    slat_src_coords_tgt = get_slat_src_coords_32(dest_32_zyx, M_raw_32)
    trans_src_32 = [[0, int(p[2]), int(p[1]), int(p[0])] for p in trans_32_zyx]
    slat_dst_coords_all = to_slat_coords_32(np.concatenate([dest_32_zyx, trans_32_zyx])) if len(dest_32_zyx) + len(trans_32_zyx) > 0 else []
    slat_src_coords_all = slat_src_coords_tgt + trans_src_32
    A_tgt_32 = []
    for (z, y, x) in dest_32_zyx:
        A_tgt_32.append(float(A_raw_32[z, y, x]))
    A_all_32 = A_tgt_32 + [0.0] * len(trans_32_zyx)
    M_raw_64 = np.array(raw_64['M_map'], dtype=np.int64)
    A_raw_64 = np.array(raw_64['A_map'], dtype=np.float32)
    rm_raw_64 = {k: np.array(v, dtype=bool) for (k, v) in raw_64['region_masks'].items()}
    G_64 = 64
    dest_64_zyx = np.argwhere(rm_raw_64['destination'])
    trans_64_zyx = np.argwhere(rm_raw_64['transition'])
    inp_64_zyx = np.argwhere(rm_raw_64['inpainting'])
    back_64_zyx = np.argwhere(rm_raw_64['background'])

    def to_slat_coords_64(zyx_arr):
        if len(zyx_arr) == 0:
            return []
        coords = []
        for (z, y, x) in zyx_arr:
            coords.append([0, int(x), int(y), int(z)])
        return coords

    def get_slat_src_coords_64(dest_zyx_list, M_raw_map):
        coords = []
        for (z, y, x) in dest_zyx_list:
            src_hi = M_raw_map[z, y, x]
            coords.append([0, int(src_hi[2]), int(src_hi[1]), int(src_hi[0])])
        return coords
    slat64_dst_coords_tgt = to_slat_coords_64(dest_64_zyx)
    slat64_src_coords_tgt = get_slat_src_coords_64(dest_64_zyx, M_raw_64)
    trans_src_64 = [[0, int(p[2]), int(p[1]), int(p[0])] for p in trans_64_zyx]
    slat64_dst_coords_all = to_slat_coords_64(np.concatenate([dest_64_zyx, trans_64_zyx])) if len(dest_64_zyx) + len(trans_64_zyx) > 0 else []
    slat64_src_coords_all = slat64_src_coords_tgt + trans_src_64
    A_tgt_64 = []
    for (z, y, x) in dest_64_zyx:
        A_tgt_64.append(float(A_raw_64[z, y, x]))
    A_all_64 = A_tgt_64 + [0.0] * len(trans_64_zyx)
    out = {'keep_voxel_xyz': keep_voxel_xyz, 'ss': {'G': G_lo, 'L': L_lo, 'dst_idx_tgt': dst_idx_tgt, 'src_idx_tgt': src_idx_tgt, 'A_tgt': A_tgt, 'dst_idx_all': dst_idx_all, 'src_idx_all': src_idx_all, 'A_all': A_all, 'inp_idx': get_flat_idx_16(inp_lo_zyx, G_lo), 'editable_idx': get_flat_idx_16(edit_lo_zyx, G_lo), 'background_idx': get_flat_idx_16(np.argwhere(region_labels == 0), G_lo)}, 'slat': {'dst_coords_tgt': slat_dst_coords_tgt, 'src_coords_tgt': slat_src_coords_tgt, 'A_tgt': A_tgt_32, 'dst_coords_all': slat_dst_coords_all, 'src_coords_all': slat_src_coords_all, 'A_all': A_all_32, 'inp_coords': to_slat_coords_32(inp_32_zyx), 'back_coords': to_slat_coords_32(back_32_zyx)}, 'slat_64': {'dst_coords_tgt': slat64_dst_coords_tgt, 'src_coords_tgt': slat64_src_coords_tgt, 'A_tgt': A_tgt_64, 'dst_coords_all': slat64_dst_coords_all, 'src_coords_all': slat64_src_coords_all, 'A_all': A_all_64, 'inp_coords': to_slat_coords_64(inp_64_zyx), 'back_coords': to_slat_coords_64(back_64_zyx)}}
    with open(out_json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Done! Saved to {out_json_path}')
    return out_json_path