from typing import *
import os
import copy
import functools
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict
from ...modules import sparse as sp
from ...utils.general_utils import dict_reduce
from ...utils.data_utils import cycle, BalancedResumableSampler
from .flow_matching import FlowMatchingTrainer
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.text_conditioned import TextConditionedMixin
from .mixins.image_conditioned import ImageConditionedMixin

class SparseFlowMatchingTrainer(FlowMatchingTrainer):

    def prepare_dataloader(self, **kwargs):
        self.data_sampler = BalancedResumableSampler(self.dataset, shuffle=True, batch_size=self.batch_size_per_gpu)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size_per_gpu, num_workers=int(np.ceil(os.cpu_count() / torch.cuda.device_count())), pin_memory=True, drop_last=True, persistent_workers=True, collate_fn=functools.partial(self.dataset.collate_fn, split_size=self.batch_split), sampler=self.data_sampler)
        self.data_iterator = cycle(self.dataloader)

    def training_losses(self, x_0: sp.SparseTensor, cond=None, **kwargs) -> Tuple[Dict, Dict]:
        noise = x_0.replace(torch.randn_like(x_0.feats))
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **kwargs)
        pred = self.training_models['denoiser'](x_t, t * 1000, cond, **kwargs)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)
        terms = edict()
        terms['mse'] = F.mse_loss(pred.feats, target.feats)
        terms['loss'] = terms['mse']
        mse_per_instance = np.array([F.mse_loss(pred.feats[x_0.layout[i]], target.feats[x_0.layout[i]]).item() for i in range(x_0.shape[0])])
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
            data = {k: v[:batch].cuda() if not isinstance(v, list) else v[:batch] for (k, v) in data.items()}
            noise = data['x_0'].replace(torch.randn_like(data['x_0'].feats))
            sample_gt.append(data['x_0'])
            cond_vis.append(self.vis_cond(**data))
            del data['x_0']
            args = self.get_inference_cond(**data)
            res = sampler.sample(self.models['denoiser'], noise=noise, **args, steps=50, cfg_strength=3.0, verbose=verbose)
            sample.append(res.samples)
        sample_gt = sp.sparse_cat(sample_gt)
        sample = sp.sparse_cat(sample)
        sample_dict = {'sample_gt': {'value': sample_gt, 'type': 'sample'}, 'sample': {'value': sample, 'type': 'sample'}}
        sample_dict.update(dict_reduce(cond_vis, None, {'value': lambda x: torch.cat(x, dim=0), 'type': lambda x: x[0]}))
        return sample_dict

class SparseFlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, SparseFlowMatchingTrainer):
    pass

class TextConditionedSparseFlowMatchingCFGTrainer(TextConditionedMixin, SparseFlowMatchingCFGTrainer):
    pass

class ImageConditionedSparseFlowMatchingCFGTrainer(ImageConditionedMixin, SparseFlowMatchingCFGTrainer):
    pass