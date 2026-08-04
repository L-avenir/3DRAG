from typing import *
import copy
import torch
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict
import utils3d.torch
from ..basic import BasicTrainer
from ...representations import Gaussian
from ...renderers import GaussianRenderer
from ...modules.sparse import SparseTensor
from ...utils.loss_utils import l1_loss, l2_loss, ssim, lpips

class SLatVaeGaussianTrainer(BasicTrainer):

    def __init__(self, *args, loss_type: str='l1', lambda_ssim: float=0.2, lambda_lpips: float=0.2, lambda_kl: float=1e-06, regularizations: Dict={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips
        self.lambda_kl = lambda_kl
        self.regularizations = regularizations
        self._init_renderer()

    def _init_renderer(self):
        rendering_options = {'near': 0.8, 'far': 1.6, 'bg_color': 'random'}
        self.renderer = GaussianRenderer(rendering_options)
        self.renderer.pipe.kernel_size = self.models['decoder'].rep_config['2d_filter_kernel_size']

    def _render_batch(self, reps: List[Gaussian], extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
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

    @torch.no_grad()
    def _get_status(self, z: SparseTensor, reps: List[Gaussian]) -> Dict:
        xyz = torch.cat([g.get_xyz for g in reps], dim=0)
        xyz_base = (z.coords[:, 1:].float() + 0.5) / self.models['decoder'].resolution - 0.5
        offset = xyz - xyz_base.unsqueeze(1).expand(-1, self.models['decoder'].rep_config['num_gaussians'], -1).reshape(-1, 3)
        status = {'xyz': xyz, 'offset': offset, 'scale': torch.cat([g.get_scaling for g in reps], dim=0), 'opacity': torch.cat([g.get_opacity for g in reps], dim=0)}
        for k in list(status.keys()):
            status[k] = {'mean': status[k].mean().item(), 'max': status[k].max().item(), 'min': status[k].min().item()}
        return status

    def _get_regularization_loss(self, reps: List[Gaussian]) -> Tuple[torch.Tensor, Dict]:
        loss = 0.0
        terms = {}
        if 'lambda_vol' in self.regularizations:
            scales = torch.cat([g.get_scaling for g in reps], dim=0)
            volume = torch.prod(scales, dim=1)
            terms[f'reg_vol'] = volume.mean()
            loss = loss + self.regularizations['lambda_vol'] * terms[f'reg_vol']
        if 'lambda_opacity' in self.regularizations:
            opacity = torch.cat([g.get_opacity for g in reps], dim=0)
            terms[f'reg_opacity'] = (opacity - 1).pow(2).mean()
            loss = loss + self.regularizations['lambda_opacity'] * terms[f'reg_opacity']
        return (loss, terms)

    def training_losses(self, feats: SparseTensor, image: torch.Tensor, alpha: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor, return_aux: bool=False, **kwargs) -> Tuple[Dict, Dict]:
        (z, mean, logvar) = self.training_models['encoder'](feats, sample_posterior=True, return_raw=True)
        reps = self.training_models['decoder'](z)
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
        terms['kl'] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
        terms['loss'] = terms['loss'] + self.lambda_kl * terms['kl']
        (reg_loss, reg_terms) = self._get_regularization_loss(reps)
        terms.update(reg_terms)
        terms['loss'] = terms['loss'] + reg_loss
        status = self._get_status(z, reps)
        if return_aux:
            return (terms, status, {'rec_image': rec_image, 'gt_image': gt_image})
        return (terms, status)

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
            z = self.models['encoder'](args['feats'], sample_posterior=True, return_raw=False)
            reps.extend(self.models['decoder'](z))
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