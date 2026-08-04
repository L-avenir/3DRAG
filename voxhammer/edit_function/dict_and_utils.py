import os
import rembg
import torch
import utils3d
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import *
from PIL import Image
from tqdm import tqdm
import json
import tempfile
import atexit
import shutil
from collections.abc import MutableMapping
import trellis.modules.sparse as sp

class DiskDict(MutableMapping):

    def __init__(self, cache_dir=None, prefix='vram_cache_', desc='Disk IO', pos=1):
        self._temp_dir = tempfile.mkdtemp(dir=cache_dir, prefix=prefix)
        self._keys = set()
        self.desc_name = desc
        self.pbar = tqdm(desc=self.desc_name, position=pos, unit=' tensor', leave=True)
        atexit.register(self.cleanup)

    def __setitem__(self, key, value):
        short_key = key[-30:] if len(key) > 30 else key
        self.pbar.set_description(f'{self.desc_name} [Write: {short_key}]')
        file_path = os.path.join(self._temp_dir, f'{key}.pt')
        if hasattr(value, 'feats') and hasattr(value, 'coords'):
            torch.save({'feats': value.feats.cpu(), 'coords': value.coords.cpu()}, file_path)
        else:
            torch.save(value.cpu(), file_path)
        self._keys.add(key)
        self.pbar.update(1)
        del value

    def __getitem__(self, key):
        if key not in self._keys:
            raise KeyError(key)
        short_key = key[-30:] if len(key) > 30 else key
        self.pbar.set_description(f'{self.desc_name} [Read: {short_key}]')
        file_path = os.path.join(self._temp_dir, f'{key}.pt')
        data = torch.load(file_path, weights_only=False)
        if isinstance(data, dict) and 'feats' in data and ('coords' in data):
            data = sp.SparseTensor(feats=data['feats'], coords=data['coords'])
        self.pbar.update(1)
        return data

    def __delitem__(self, key):
        if key in self._keys:
            file_path = os.path.join(self._temp_dir, f'{key}.pt')
            if os.path.exists(file_path):
                os.remove(file_path)
            self._keys.remove(key)
        else:
            raise KeyError(key)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def cleanup(self):
        self.pbar.close()
        if os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir)

def _phi_xyz_to_idx(xyz: torch.Tensor, G: int) -> torch.Tensor:
    xyz = xyz.to(torch.long)
    return (xyz[:, 0] * G + xyz[:, 1]) * G + xyz[:, 2]

def _voxel_xyz_to_ss_token_xyz(xyz_voxel: torch.Tensor, ss_patch_size: int) -> torch.Tensor:
    return (xyz_voxel.to(torch.long) // int(ss_patch_size)).clamp(min=0)

def _voxel_xyz_to_slat_xyz(xyz_voxel: torch.Tensor, slat_factor=(2, 2, 2)) -> torch.Tensor:
    xyz = xyz_voxel.to(torch.long)
    (fx, fy, fz) = map(int, slat_factor)
    return torch.stack([xyz[:, 0] // fx, xyz[:, 1] // fy, xyz[:, 2] // fz], dim=-1)

def load_map_std_from_json(map_path: str, ss_flow, device='cuda', slat_factor=(2, 2, 2)):
    data = json.loads(open(map_path, 'r').read())
    ss_patch = int(ss_flow.patch_size)
    G_expected = int(ss_flow.resolution // ss_patch)
    L_expected = G_expected * G_expected * G_expected
    keep_list = data.get('keep_voxel_xyz', [])
    keep_xyz = torch.tensor(keep_list, dtype=torch.long, device=device)
    if keep_xyz.numel() == 0:
        keep_xyz = keep_xyz.reshape(0, 3)
    ss = data['ss']
    G = int(ss['G'])
    L = int(ss['L'])
    assert G == G_expected, f'SS G mismatch: json G={G}, expected {G_expected} from ss_flow'
    assert L == L_expected, f'SS L mismatch: json L={L}, expected {L_expected} from ss_flow'
    dst_idx_tgt = torch.tensor(ss.get('dst_idx_tgt', []), dtype=torch.long, device=device)
    src_idx_tgt = torch.tensor(ss.get('src_idx_tgt', []), dtype=torch.long, device=device)
    A_tgt = torch.tensor(ss.get('A_tgt', []), dtype=torch.float32, device=device)
    dst_idx_all = torch.tensor(ss.get('dst_idx_all', []), dtype=torch.long, device=device)
    src_idx_all = torch.tensor(ss.get('src_idx_all', []), dtype=torch.long, device=device)
    A_all = torch.tensor(ss.get('A_all', []), dtype=torch.float32, device=device)
    inp_idx = torch.tensor(ss.get('inp_idx', []), dtype=torch.long, device=device)
    editable_idx = torch.tensor(ss.get('editable_idx', []), dtype=torch.long, device=device)
    if dst_idx_tgt.numel() == 0:
        dst_idx_tgt = dst_idx_tgt.reshape(0)
    if src_idx_tgt.numel() == 0:
        src_idx_tgt = src_idx_tgt.reshape(0)
    if A_tgt.numel() == 0:
        A_tgt = A_tgt.reshape(0)
    if dst_idx_all.numel() == 0:
        dst_idx_all = dst_idx_all.reshape(0)
    if src_idx_all.numel() == 0:
        src_idx_all = src_idx_all.reshape(0)
    if A_all.numel() == 0:
        A_all = A_all.reshape(0)
    if inp_idx.numel() == 0:
        inp_idx = inp_idx.reshape(0)
    if editable_idx.numel() == 0:
        editable_idx = editable_idx.reshape(0)
    assert dst_idx_tgt.numel() == src_idx_tgt.numel(), f'SS tgt length mismatch: dst_idx_tgt={dst_idx_tgt.numel()} vs src_idx_tgt={src_idx_tgt.numel()}'
    assert dst_idx_tgt.numel() == A_tgt.numel(), f'SS tgt A length mismatch: dst_idx_tgt={dst_idx_tgt.numel()} vs A_tgt={A_tgt.numel()}'
    assert dst_idx_all.numel() == src_idx_all.numel(), f'SS all length mismatch: dst_idx_all={dst_idx_all.numel()} vs src_idx_all={src_idx_all.numel()}'
    assert dst_idx_all.numel() == A_all.numel(), f'SS all A length mismatch: dst_idx_all={dst_idx_all.numel()} vs A_all={A_all.numel()}'
    src_idx_of_dst_tgt = torch.full((L,), -1, dtype=torch.long, device=device)
    if dst_idx_tgt.numel() > 0:
        src_idx_of_dst_tgt[dst_idx_tgt] = src_idx_tgt
    src_idx_of_dst_all = torch.full((L,), -1, dtype=torch.long, device=device)
    if dst_idx_all.numel() > 0:
        src_idx_of_dst_all[dst_idx_all] = src_idx_all
    if dst_idx_tgt.numel() > 0 and inp_idx.numel() > 0:
        inter = torch.isin(dst_idx_tgt, inp_idx).any().item()
        assert not inter, 'SS invariant violated: dst_idx_tgt intersects inp_idx'
    slat = data['slat']

    def _bcoords_tensor(key: str):
        arr = slat.get(key, [])
        t = torch.tensor(arr, dtype=torch.long, device=device)
        if t.numel() == 0:
            t = t.reshape(0, 4)
        else:
            assert t.dim() == 2 and t.shape[1] == 4, f'{key} must have shape (N,4), got {tuple(t.shape)}'
        return t
    dst_coords_tgt = _bcoords_tensor('dst_coords_tgt')
    src_coords_tgt = _bcoords_tensor('src_coords_tgt')
    dst_coords_all = _bcoords_tensor('dst_coords_all')
    src_coords_all = _bcoords_tensor('src_coords_all')
    inp_coords = _bcoords_tensor('inp_coords')
    back_coords = _bcoords_tensor('back_coords')
    A_tgt_slat = torch.tensor(slat.get('A_tgt', []), dtype=torch.float32, device=device)
    A_all_slat = torch.tensor(slat.get('A_all', []), dtype=torch.float32, device=device)
    if A_tgt_slat.numel() == 0:
        A_tgt_slat = A_tgt_slat.reshape(0)
    if A_all_slat.numel() == 0:
        A_all_slat = A_all_slat.reshape(0)
    assert dst_coords_tgt.shape[0] == src_coords_tgt.shape[0], f'SLAT tgt coords mismatch: dst={dst_coords_tgt.shape[0]} vs src={src_coords_tgt.shape[0]}'
    assert dst_coords_tgt.shape[0] == A_tgt_slat.numel(), f'SLAT tgt A mismatch: dst={dst_coords_tgt.shape[0]} vs A_tgt={A_tgt_slat.numel()}'
    assert dst_coords_all.shape[0] == src_coords_all.shape[0], f'SLAT all coords mismatch: dst={dst_coords_all.shape[0]} vs src={src_coords_all.shape[0]}'
    assert dst_coords_all.shape[0] == A_all_slat.numel(), f'SLAT all A mismatch: dst={dst_coords_all.shape[0]} vs A_all={A_all_slat.numel()}'
    if data['slat_64'] == None:
        slat64_dst_coords_tgt = None
        slat64_src_coords_tgt = None
        A_tgt_slat64 = None
        slat64_dst_coords_all = None
        slat64_src_coords_all = None
        A_all_slat64 = None
        slat64_inp_coords = None
        slat64_back_coords = None
    else:
        slat_64 = data['slat_64']

        def _bcoords_tensor_64(key: str):
            arr = slat_64.get(key, [])
            t = torch.tensor(arr, dtype=torch.long, device=device)
            if t.numel() == 0:
                t = t.reshape(0, 4)
            else:
                assert t.dim() == 2 and t.shape[1] == 4, f'slat_64.{key} must have shape (N,4), got {tuple(t.shape)}'
            return t
        slat64_dst_coords_tgt = _bcoords_tensor_64('dst_coords_tgt')
        slat64_src_coords_tgt = _bcoords_tensor_64('src_coords_tgt')
        slat64_dst_coords_all = _bcoords_tensor_64('dst_coords_all')
        slat64_src_coords_all = _bcoords_tensor_64('src_coords_all')
        slat64_inp_coords = _bcoords_tensor_64('inp_coords')
        slat64_back_coords = _bcoords_tensor_64('back_coords')
        A_tgt_slat64 = torch.tensor(slat_64.get('A_tgt', []), dtype=torch.float32, device=device)
        A_all_slat64 = torch.tensor(slat_64.get('A_all', []), dtype=torch.float32, device=device)
        if A_tgt_slat64.numel() == 0:
            A_tgt_slat64 = A_tgt_slat64.reshape(0)
        if A_all_slat64.numel() == 0:
            A_all_slat64 = A_all_slat64.reshape(0)
        assert slat64_dst_coords_tgt.shape[0] == slat64_src_coords_tgt.shape[0], f'SLAT_64 tgt coords mismatch: dst={slat64_dst_coords_tgt.shape[0]} vs src={slat64_src_coords_tgt.shape[0]}'
        assert slat64_dst_coords_tgt.shape[0] == A_tgt_slat64.numel(), f'SLAT_64 tgt A mismatch: dst={slat64_dst_coords_tgt.shape[0]} vs A_tgt={A_tgt_slat64.numel()}'
        assert slat64_dst_coords_all.shape[0] == slat64_src_coords_all.shape[0], f'SLAT_64 all coords mismatch: dst={slat64_dst_coords_all.shape[0]} vs src={slat64_src_coords_all.shape[0]}'
        assert slat64_dst_coords_all.shape[0] == A_all_slat64.numel(), f'SLAT_64 all A mismatch: dst={slat64_dst_coords_all.shape[0]} vs A_all={A_all_slat64.numel()}'
    return {'keep_voxel_xyz': keep_xyz, 'ss': {'G': G, 'L': L, 'dst_idx_tgt': dst_idx_tgt, 'src_idx_of_dst_tgt': src_idx_of_dst_tgt, 'A_tgt': A_tgt, 'dst_idx_all': dst_idx_all, 'src_idx_of_dst_all': src_idx_of_dst_all, 'A_all': A_all, 'inp_idx': inp_idx, 'editable_idx': editable_idx}, 'slat': {'dst_coords_tgt': dst_coords_tgt, 'src_coords_tgt': src_coords_tgt, 'A_tgt': A_tgt_slat, 'dst_coords_all': dst_coords_all, 'src_coords_all': src_coords_all, 'A_all': A_all_slat, 'inp_coords': inp_coords, 'back_coords': back_coords}, 'slat_64': {'dst_coords_tgt': slat64_dst_coords_tgt, 'src_coords_tgt': slat64_src_coords_tgt, 'A_tgt': A_tgt_slat64, 'dst_coords_all': slat64_dst_coords_all, 'src_coords_all': slat64_src_coords_all, 'A_all': A_all_slat64, 'inp_coords': slat64_inp_coords, 'back_coords': slat64_back_coords}}

def ply_to_coords(ply_path):
    position = utils3d.io.read_ply(ply_path)[0]
    coords = ((torch.tensor(position) + 0.5) * 64).int().contiguous().cuda()
    return coords

def coords_to_voxel(coords):
    voxel = torch.zeros(1, 1, 64, 64, 64, dtype=torch.float)
    voxel[:, 0, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return voxel.cuda()

def voxel_to_point_positions(voxel: torch.Tensor, thresh: float=0.0) -> np.ndarray:
    if voxel.dim() == 5:
        v = voxel[0, 0]
    elif voxel.dim() == 3:
        v = voxel
    else:
        raise ValueError(f'Unexpected voxel shape: {tuple(voxel.shape)}')
    coords = torch.argwhere(v > thresh).to(torch.float32)
    pos = coords / 64.0 - 0.5
    return pos.cpu().numpy()

def feats_to_slat(pipeline, feats_path):
    feats = np.load(feats_path)
    feats_tensor = sp.SparseTensor(feats=torch.from_numpy(feats['patchtokens']).float(), coords=torch.cat([torch.zeros(feats['patchtokens'].shape[0], 1).int(), torch.from_numpy(feats['indices']).int()], dim=1)).cuda()
    feats_encoder = pipeline.models['slat_encoder']
    slat = feats_encoder(feats_tensor, sample_posterior=False)
    return slat

def image_rgb(img_path):
    image = Image.open(img_path)
    if image.mode == 'RGB':
        image_rgb = image
    else:
        image = image.convert('RGBA')
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image_rgb = background
    return image_rgb

def preprocess_image(img_src_path, img_tgt_path, img_mask_path):
    img_src = image_rgb(img_src_path)
    img_tgt = image_rgb(img_tgt_path)
    img_mask = Image.open(img_mask_path).convert('L')
    max_size = max(img_src.size)
    scale = min(1, 1024 / max_size)
    resize_size = (int(img_src.width * scale), int(img_src.height * scale))
    if scale < 1:
        img_src = img_src.resize(resize_size, Image.Resampling.LANCZOS)
        img_tgt = img_tgt.resize(resize_size, Image.Resampling.LANCZOS)
        img_mask = img_mask.resize(resize_size, Image.Resampling.LANCZOS)
    pre_img_src = rembg.remove(img_src, session=rembg.new_session('u2net'))
    pre_img_tgt = rembg.remove(img_tgt, session=rembg.new_session('u2net'))
    pre_img_src_np = np.array(pre_img_src)
    pre_img_tgt_np = np.array(pre_img_tgt)
    alpha_src = pre_img_src_np[:, :, 3]
    alpha_tgt = pre_img_tgt_np[:, :, 3]
    bbox_src = np.argwhere(alpha_src > 0.8 * 255)
    bbox_tgt = np.argwhere(alpha_tgt > 0.8 * 255)
    bbox = (min(np.min(bbox_src[:, 1]), np.min(bbox_tgt[:, 1])), min(np.min(bbox_src[:, 0]), np.min(bbox_tgt[:, 0])), max(np.max(bbox_src[:, 1]), np.max(bbox_tgt[:, 1])), max(np.max(bbox_src[:, 0]), np.max(bbox_tgt[:, 0])))
    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1.2)
    bbox = (center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2)
    pre_img_src = pre_img_src.crop(bbox)
    pre_img_tgt = pre_img_tgt.crop(bbox)
    pre_img_mask = img_mask.crop(bbox)
    pre_img_src = pre_img_src.resize((518, 518), Image.Resampling.LANCZOS)
    pre_img_tgt = pre_img_tgt.resize((518, 518), Image.Resampling.LANCZOS)
    pre_img_mask = pre_img_mask.resize((518, 518), Image.Resampling.LANCZOS)
    pre_img_src = np.array(pre_img_src).astype(np.float32) / 255
    pre_img_tgt = np.array(pre_img_tgt).astype(np.float32) / 255
    pre_img_src = pre_img_src[:, :, :3] * pre_img_src[:, :, 3:4]
    pre_img_tgt = pre_img_tgt[:, :, :3] * pre_img_tgt[:, :, 3:4]
    pre_img_src = Image.fromarray((pre_img_src * 255).astype(np.uint8))
    pre_img_tgt = Image.fromarray((pre_img_tgt * 255).astype(np.uint8))
    return (pre_img_src, pre_img_tgt, pre_img_mask)

def ply_to_ss_mask(coords_preserve, pre_mask, map_std_ss):
    device = 'cuda'
    G = int(map_std_ss['G'])
    L = int(map_std_ss['L'])
    voxel_mask = torch.ones(1, 1, 64, 64, 64, dtype=torch.float32, device=device)
    voxel_mask[:, 0, coords_preserve[:, 0], coords_preserve[:, 1], coords_preserve[:, 2]] = 0.0
    editable_idx = torch.tensor(map_std_ss.get('editable_idx', []), dtype=torch.long, device=device)
    assert editable_idx is not None, "map_std_ss must contain 'editable_idx' for SS token-level editable mask"
    editable_idx = torch.tensor(editable_idx, dtype=torch.long, device=device)
    token_edit = torch.zeros(L, dtype=torch.float32, device=device)
    token_edit[editable_idx] = 1.0
    token_edit = token_edit.view(1, 1, G, G, G)
    ss_latent_mask = token_edit.repeat(1, 8, 1, 1, 1).contiguous()
    ss_self_kv_mask = token_edit.view(1, 1, L, 1).repeat(1, 16, 1, 64).contiguous()
    cross_kv_mask = None
    if pre_mask is not None:
        img_mask = transforms.ToTensor()(pre_mask)
        img_mask = (img_mask > 0).float()
        cross_kv_mask = img_mask.reshape(37, 14, 37, 14)
        cross_kv_mask = cross_kv_mask.permute(1, 3, 0, 2)
        cross_kv_mask = cross_kv_mask.reshape(1, 196, 37, 37)
        cross_kv_mask = cross_kv_mask.any(dim=1, keepdim=True)
        cross_kv_mask = cross_kv_mask.repeat(1, 1024, 1, 1)
        cross_kv_mask = cross_kv_mask.reshape(1, 1024, 1369)
        cross_kv_mask = cross_kv_mask.permute(0, 2, 1)
        cross_kv_mask = torch.cat((torch.ones(1, 5, 1024, device=device), cross_kv_mask.to(device=device)), dim=1)
        cross_kv_mask = cross_kv_mask.reshape(1, 1374, 16, 64)
        cross_kv_mask = cross_kv_mask.permute(0, 2, 1, 3).contiguous()
    ss_latent_mask_dict = {'keep_mask': ss_latent_mask, 'dst_idx_tgt': map_std_ss.get('dst_idx_tgt', torch.zeros(0, dtype=torch.long, device=device)), 'src_idx_of_dst_tgt': map_std_ss.get('src_idx_of_dst_tgt', None), 'inp_idx': map_std_ss.get('inp_idx', torch.zeros(0, dtype=torch.long, device=device)), 'A_tgt': map_std_ss.get('A_tgt', None)}
    ss_self_kv_mask_dict = {'mask': ss_self_kv_mask, 'dst_idx_all': map_std_ss.get('dst_idx_all', None), 'src_idx_of_dst_all': map_std_ss.get('src_idx_of_dst_all', None), 'A_all': map_std_ss.get('A_all', None), 'dst_idx_tgt': map_std_ss.get('dst_idx_tgt', None), 'src_idx_of_dst_tgt': map_std_ss.get('src_idx_of_dst_tgt', None), 'A_tgt': map_std_ss.get('A_tgt', None)}
    print('[DBG] voxel preserve ratio:', float((voxel_mask == 0).float().mean().item()))
    print('[DBG] token editable ratio:', float(token_edit.mean().item()))
    return (voxel_mask, ss_latent_mask_dict, ss_self_kv_mask_dict, cross_kv_mask)

def ply_to_slat_mask(map_std_slat):
    device = 'cuda'
    slat_self_kv_mask_dict = {'mask': map_std_slat.get('back_coords', torch.zeros(0, 4, dtype=torch.long, device=device)), 'keep_coords': map_std_slat.get('back_coords', torch.zeros(0, 4, dtype=torch.long, device=device)), 'dst_coords_all': map_std_slat.get('dst_coords_all', torch.zeros(0, 4, dtype=torch.long, device=device)), 'src_coords_all': map_std_slat.get('src_coords_all', torch.zeros(0, 4, dtype=torch.long, device=device)), 'A_all': map_std_slat.get('A_all', None), 'dst_coords_tgt': map_std_slat.get('dst_coords_tgt', torch.zeros(0, 4, dtype=torch.long, device=device)), 'src_coords_tgt': map_std_slat.get('src_coords_tgt', torch.zeros(0, 4, dtype=torch.long, device=device)), 'A_tgt': map_std_slat.get('A_tgt', None)}
    slat_latent_mask_dict = {'keep_coords': map_std_slat.get('back_coords', torch.zeros(0, 4, dtype=torch.long, device=device)), 'dst_coords_tgt': map_std_slat.get('dst_coords_all', torch.zeros(0, 4, dtype=torch.long, device=device)), 'src_coords_tgt': map_std_slat.get('src_coords_all', torch.zeros(0, 4, dtype=torch.long, device=device)), 'inp_coords': map_std_slat.get('inp_coords', torch.zeros(0, 4, dtype=torch.long, device=device)), 'back_coords': map_std_slat.get('back_coords', torch.zeros(0, 4, dtype=torch.long, device=device))}
    return (slat_self_kv_mask_dict, slat_latent_mask_dict)

def get_rope_attention_bias(src_coords, dst_coords, dim=128, theta=10000.0):
    src_xyz = src_coords[:, 1:].float()
    dst_xyz = dst_coords[:, 1:].float()
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim)).to(src_xyz.device)
    src_pos = torch.einsum('md,d->m', src_xyz, inv_freq[:3])
    dst_pos = torch.einsum('md,d->m', dst_xyz, inv_freq[:3])
    attn_bias = torch.mm(dst_pos.unsqueeze(1), src_pos.unsqueeze(0))
    attn_bias = torch.zeros_like(attn_bias)
    idx = torch.arange(attn_bias.shape[0])
    attn_bias[idx, idx] = 1.0
    attn_bias = attn_bias * 10.0
    return attn_bias

def match_src_dst_coords(sparse_A, sparse_B, dst_coords, src_coords):
    device = sparse_A.coords.device
    coords_A = sparse_A.coords.to(device)
    coords_B = sparse_B
    dst_coords = dst_coords.to(device, dtype=torch.long)
    src_coords = src_coords.to(device, dtype=torch.long)
    if src_coords.shape[0] != dst_coords.shape[0]:
        raise ValueError(f'Length of src_coords ({src_coords.shape[0]}) and dst_coords ({dst_coords.shape[0]}) mismatch!')
    M = src_coords.shape[0]
    match_B_dst = (coords_B.unsqueeze(1) == dst_coords.unsqueeze(0)).all(dim=-1)
    is_B_in_dst = match_B_dst.any(dim=1)
    B_dst_idx = torch.where(is_B_in_dst)[0]
    K = len(B_dst_idx)
    B_dst_coords = coords_B[B_dst_idx]
    repair_coords_B = coords_B[~is_B_in_dst]
    valid_dst_coords_B = torch.empty(0, 4, device=device)
    valid_src_coords_A = torch.empty(0, 4, device=device)
    total_valid = 0
    total_invalid = 0
    if K > 0:
        match_dst_map = (B_dst_coords.unsqueeze(1) == dst_coords.unsqueeze(0)).all(dim=-1)
        dst_map_idx = match_dst_map.float().argmax(1)
        dst_map_valid = match_dst_map[torch.arange(K), dst_map_idx]
        B_dst_valid_idx = B_dst_idx[dst_map_valid]
        K_valid = len(B_dst_valid_idx)
        B_dst_valid_coords = B_dst_coords[dst_map_valid]
        dst_map_idx = dst_map_idx[dst_map_valid]
        if K_valid > 0:
            src_coords_for_B_dst = src_coords[dst_map_idx]
            match_A_src = (coords_A.unsqueeze(1) == src_coords_for_B_dst.unsqueeze(0)).all(dim=-1)
            src_exists_in_A = match_A_src.any(dim=0)
            src_idx_A = match_A_src.float().argmax(0)
            src_exists_valid = match_A_src[src_idx_A, torch.arange(K_valid)]
            src_exists_in_A = src_exists_in_A & src_exists_valid
            total_valid = torch.sum(src_exists_in_A).item()
            total_invalid = K_valid - total_valid
            valid_mask = src_exists_in_A
            valid_dst_coords_B = B_dst_valid_coords[valid_mask]
            valid_src_coords_A = coords_A[src_idx_A[valid_mask]]
    return (valid_dst_coords_B, valid_src_coords_A, repair_coords_B)

def match_coords_chunked(coords_A, coords_B):
    chunk_size = 1000
    device = coords_A.device
    matched_idx_A_list = []
    matched_idx_B_list = []
    for start_idx in range(0, coords_A.shape[0], chunk_size):
        end_idx = min(start_idx + chunk_size, coords_A.shape[0])
        chunk_A = coords_A[start_idx:end_idx].to(device)
        match_chunk = (chunk_A.unsqueeze(1) == coords_B.unsqueeze(0)).all(dim=-1)
        match_positions = match_chunk.nonzero(as_tuple=False)
        if len(match_positions) > 0:
            chunk_inner_A_idx = match_positions[:, 0]
            B_global_idx = match_positions[:, 1]
            A_global_idx = chunk_inner_A_idx + start_idx
            matched_idx_A_list.append(A_global_idx.to(device))
            matched_idx_B_list.append(B_global_idx.to(device))
        del match_chunk
    if matched_idx_A_list:
        matched_idx_A = torch.cat(matched_idx_A_list, dim=0)
        matched_idx_B = torch.cat(matched_idx_B_list, dim=0)
    else:
        matched_idx_A = torch.tensor([], dtype=torch.long, device=device)
        matched_idx_B = torch.tensor([], dtype=torch.long, device=device)
    return (matched_idx_A, matched_idx_B)