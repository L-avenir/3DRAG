from typing import *
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
from ..basic import BasicTrainer

class SparseStructureVaeTrainer(BasicTrainer):

    def __init__(self, *args, loss_type='bce', lambda_kl=1e-06, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.lambda_kl = lambda_kl

    def training_losses(self, ss: torch.Tensor, **kwargs) -> Tuple[Dict, Dict]:
        (z, mean, logvar) = self.training_models['encoder'](ss.float(), sample_posterior=True, return_raw=True)
        logits = self.training_models['decoder'](z)
        terms = edict(loss=0.0)
        if self.loss_type == 'bce':
            terms['bce'] = F.binary_cross_entropy_with_logits(logits, ss.float(), reduction='mean')
            terms['loss'] = terms['loss'] + terms['bce']
        elif self.loss_type == 'l1':
            terms['l1'] = F.l1_loss(F.sigmoid(logits), ss.float(), reduction='mean')
            terms['loss'] = terms['loss'] + terms['l1']
        elif self.loss_type == 'dice':
            logits = F.sigmoid(logits)
            terms['dice'] = 1 - (2 * (logits * ss.float()).sum() + 1) / (logits.sum() + ss.float().sum() + 1)
            terms['loss'] = terms['loss'] + terms['dice']
        else:
            raise ValueError(f'Invalid loss type {self.loss_type}')
        terms['kl'] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
        terms['loss'] = terms['loss'] + self.lambda_kl * terms['kl']
        return (terms, {})

    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=64, batch_size=1, verbose=False):
        super().snapshot(suffix=suffix, num_samples=num_samples, batch_size=batch_size, verbose=verbose)

    @torch.no_grad()
    def run_snapshot(self, num_samples: int, batch_size: int, verbose: bool=False) -> Dict:
        dataloader = DataLoader(copy.deepcopy(self.dataset), batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None)
        gts = []
        recons = []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            args = {k: v[:batch].cuda() if isinstance(v, torch.Tensor) else v[:batch] for (k, v) in data.items()}
            z = self.models['encoder'](args['ss'].float(), sample_posterior=False)
            logits = self.models['decoder'](z)
            recon = (logits > 0).long()
            gts.append(args['ss'])
            recons.append(recon)
        sample_dict = {'gt': {'value': torch.cat(gts, dim=0), 'type': 'sample'}, 'recon': {'value': torch.cat(recons, dim=0), 'type': 'sample'}}
        return sample_dict