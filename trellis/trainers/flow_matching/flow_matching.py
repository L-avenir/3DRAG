from typing import *
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict
from ..basic import BasicTrainer
from ...pipelines import samplers
from ...utils.general_utils import dict_reduce
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.text_conditioned import TextConditionedMixin
from .mixins.image_conditioned import ImageConditionedMixin

class FlowMatchingTrainer(BasicTrainer):

    def __init__(self, *args, t_schedule: dict={'name': 'logitNormal', 'args': {'mean': 0.0, 'std': 1.0}}, sigma_min: float=1e-05, **kwargs):
        super().__init__(*args, **kwargs)
        self.t_schedule = t_schedule
        self.sigma_min = sigma_min

    def diffuse(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor]=None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_0)
        assert noise.shape == x_0.shape, 'noise must have same shape as x_0'
        t = t.view(-1, *[1 for _ in range(len(x_0.shape) - 1)])
        x_t = (1 - t) * x_0 + (self.sigma_min + (1 - self.sigma_min) * t) * noise
        return x_t

    def reverse_diffuse(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        assert noise.shape == x_t.shape, 'noise must have same shape as x_t'
        t = t.view(-1, *[1 for _ in range(len(x_t.shape) - 1)])
        x_0 = (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * noise) / (1 - t)
        return x_0

    def get_v(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (1 - self.sigma_min) * noise - x_0

    def get_cond(self, cond, **kwargs):
        return cond

    def get_inference_cond(self, cond, **kwargs):
        return {'cond': cond, **kwargs}

    def get_sampler(self, **kwargs) -> samplers.FlowEulerSampler:
        return samplers.FlowEulerSampler(self.sigma_min)

    def vis_cond(self, **kwargs):
        return {}

    def sample_t(self, batch_size: int) -> torch.Tensor:
        if self.t_schedule['name'] == 'uniform':
            t = torch.rand(batch_size)
        elif self.t_schedule['name'] == 'logitNormal':
            mean = self.t_schedule['args']['mean']
            std = self.t_schedule['args']['std']
            t = torch.sigmoid(torch.randn(batch_size) * std + mean)
        else:
            raise ValueError(f"Unknown t_schedule: {self.t_schedule['name']}")
        return t

    def training_losses(self, x_0: torch.Tensor, cond=None, **kwargs) -> Tuple[Dict, Dict]:
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **kwargs)
        pred = self.training_models['denoiser'](x_t, t * 1000, cond, **kwargs)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)
        terms = edict()
        terms['mse'] = F.mse_loss(pred, target)
        terms['loss'] = terms['mse']
        mse_per_instance = np.array([F.mse_loss(pred[i], target[i]).item() for i in range(x_0.shape[0])])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f'bin_{i}'] = {'mse': mse_per_instance[time_bin == i].mean()}
        return (terms, {})

    @torch.no_grad()
    def run_snapshot(self, num_samples: int, batch_size: int, verbose: bool=False) -> Dict:
        dataloader = DataLoader(copy.deepcopy(self.dataset), batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None)
        sampler = self.get_sampler()
        sample_gt = []
        sample = []
        cond_vis = []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            data = {k: v[:batch].cuda() if isinstance(v, torch.Tensor) else v[:batch] for (k, v) in data.items()}
            noise = torch.randn_like(data['x_0'])
            sample_gt.append(data['x_0'])
            cond_vis.append(self.vis_cond(**data))
            del data['x_0']
            args = self.get_inference_cond(**data)
            res = sampler.sample(self.models['denoiser'], noise=noise, **args, steps=50, cfg_strength=3.0, verbose=verbose)
            sample.append(res.samples)
        sample_gt = torch.cat(sample_gt, dim=0)
        sample = torch.cat(sample, dim=0)
        sample_dict = {'sample_gt': {'value': sample_gt, 'type': 'sample'}, 'sample': {'value': sample, 'type': 'sample'}}
        sample_dict.update(dict_reduce(cond_vis, None, {'value': lambda x: torch.cat(x, dim=0), 'type': lambda x: x[0]}))
        return sample_dict

class FlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, FlowMatchingTrainer):
    pass

class TextConditionedFlowMatchingCFGTrainer(TextConditionedMixin, FlowMatchingCFGTrainer):
    pass

class ImageConditionedFlowMatchingCFGTrainer(ImageConditionedMixin, FlowMatchingCFGTrainer):
    pass