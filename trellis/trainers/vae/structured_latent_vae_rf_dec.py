from typing import *
import copy
import torch
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict
import utils3d.torch
from ..basic import BasicTrainer
from ...representations import Strivec
from ...renderers import OctreeRenderer
from ...modules.sparse import SparseTensor
from ...utils.loss_utils import l1_loss, l2_loss, ssim, lpips

class SLatVaeRadianceFieldDecoderTrainer(BasicTrainer):

    def __init__(self, *args, loss_type: str='l1', lambda_ssim: float=0.2, lambda_lpips: float=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips
        self._init_renderer()

    def _init_renderer(self):
        rendering_options = {'near': 0.8, 'far': 1.6, 'bg_color': 'random'}
        self.renderer = OctreeRenderer(rendering_options)
        self.renderer.pipe.primitive = 'trivec'

    def _render_batch(self, reps: List[Strivec], extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        ret = None
        for (i, representation) in enumerate(reps):
            render_pack = self.renderer.render(representation, extrinsics[i], intrinsics[i])
            if ret is None:
                ret = {k: [] for k in list(render_pack.keys()) + ['bg_color']}
            for (k, v) in render_pack.items():
                ret[k].append(v)
            ret['bg_color'].append(self.renderer.bg_color)
        for (k, v) in ret.items():
            ret[k] = torch.stack(v, dim=0)
        return ret

    def training_losses(self, latents: SparseTensor, image: torch.Tensor, alpha: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor, return_aux: bool=False, **kwargs) -> Tuple[Dict, Dict]:
        reps = self.training_models['decoder'](latents)
        self.renderer.rendering_options.resolution = image.shape[-1]
        render_results = self._render_batch(reps, extrinsics, intrinsics)
        terms = edict(loss=0.0, rec=0.0)
        rec_image = render_results['color']
        gt_image = image * alpha[:, None] + (1 - alpha[:, None]) * render_results['bg_color'][..., None, None]
        if self.loss_type == 'l1':
            terms['l1'] = l1_loss(rec_image, gt_image)
            terms['rec'] = terms['rec'] + terms['l1']
        elif self.loss_type == 'l2':
            terms['l2'] = l2_loss(rec_image, gt_image)
            terms['rec'] = terms['rec'] + terms['l2']
        else:
            raise ValueError(f'Invalid loss type: {self.loss_type}')
        if self.lambda_ssim > 0:
            terms['ssim'] = 1 - ssim(rec_image, gt_image)
            terms['rec'] = terms['rec'] + self.lambda_ssim * terms['ssim']
        if self.lambda_lpips > 0:
            terms['lpips'] = lpips(rec_image, gt_image)
            terms['rec'] = terms['rec'] + self.lambda_lpips * terms['lpips']
        terms['loss'] = terms['loss'] + terms['rec']
        if return_aux:
            return (terms, {}, {'rec_image': rec_image, 'gt_image': gt_image})
        return (terms, {})

    @torch.no_grad()
    def run_snapshot(self, num_samples: int, batch_size: int, verbose: bool=False) -> Dict:
        dataloader = DataLoader(copy.deepcopy(self.dataset), batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None)
        ret_dict = {}
        gt_images = []
        exts = []
        ints = []
        reps = []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            args = {k: v[:batch].cuda() for (k, v) in data.items()}
            gt_images.append(args['image'] * args['alpha'][:, None])
            exts.append(args['extrinsics'])
            ints.append(args['intrinsics'])
            reps.extend(self.models['decoder'](args['latents']))
        gt_images = torch.cat(gt_images, dim=0)
        ret_dict.update({f'gt_image': {'value': gt_images, 'type': 'image'}})
        exts = torch.cat(exts, dim=0)
        ints = torch.cat(ints, dim=0)
        self.renderer.rendering_options.bg_color = (0, 0, 0)
        self.renderer.rendering_options.resolution = gt_images.shape[-1]
        render_results = self._render_batch(reps, exts, ints)
        ret_dict.update({f'rec_image': {'value': render_results['color'], 'type': 'image'}})
        self.renderer.rendering_options.resolution = 512
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]
        miltiview_images = []
        for (yaw, pitch) in zip(yaws, pitch):
            orig = torch.tensor([np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(pitch), np.sin(pitch)]).float().cuda() * 2
            fov = torch.deg2rad(torch.tensor(30)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            extrinsics = extrinsics.unsqueeze(0).expand(num_samples, -1, -1)
            intrinsics = intrinsics.unsqueeze(0).expand(num_samples, -1, -1)
            render_results = self._render_batch(reps, extrinsics, intrinsics)
            miltiview_images.append(render_results['color'])
        miltiview_images = torch.cat([torch.cat(miltiview_images[:2], dim=-2), torch.cat(miltiview_images[2:], dim=-2)], dim=-1)
        ret_dict.update({f'miltiview_image': {'value': miltiview_images, 'type': 'image'}})
        self.renderer.rendering_options.bg_color = 'random'
        return ret_dict