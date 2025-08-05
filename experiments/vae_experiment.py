
from functools import cached_property
from typing import Callable, Dict, Any, List


import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch.nn as nn

from experiment_config import ExperimentConfig, DatasetConfig
from hyperparameter_config import HParamConfig
from experiment_util import compute_effective_rank, VAE, set_seed
from experiment_util import get_label_weights
from experiment_util import get_weighted_sampler

class VAEExperiment():
    def __init__(self, config: ExperimentConfig, hparams: HParamConfig, device, logger: Callable) -> None:
        assert hparams.seed is not None
        assert hparams.structure is not None
        assert hparams.structure.latent_dim is not None

        self.config = config
        self.hparams = hparams
        self.epoch = 0
        self.device = device
        self.logger = logger

        self.model = VAE(latent_dim=hparams.structure.latent_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.bce_loss = nn.BCELoss(reduction='none')

        def kl_to_unit_gaussian(mu, logvar):
            # KL between N(mu, sigma) and N(0, 1)
            return 0.5 * (torch.exp(logvar) + mu**2 - 1 - logvar)
        self.prior_loss = kl_to_unit_gaussian

        self._load_datasets()

        if config.dataset.drift:
            from experiment_util import get_label_weights
            # get initial drift weights
            init_w = get_label_weights(0, config.dataset.drift, hparams.seed).to(device)
            self.cum_label_weights = torch.zeros_like(init_w)
        else:
            self.static_train_loader = DataLoader(self.train_dataset, batch_size=config.dataset.train_batch_size, shuffle=True)

    def _load_datasets(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float())
        ])
        dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        cfg = self.config.dataset
        self.train_dataset = Subset(dataset, range(cfg.train_start, cfg.val_start ))
        self.val_dataset = Subset(dataset, range(cfg.val_start, len(dataset) - 1))
        self.static_train_loader = None
        self.static_val_loader = None

    def train(self, epoch: int):
        self.model.train()
        self._update_train_loader(epoch)

    def _update_train_loader(self, epoch:int):
        drift_cfg = self.config.dataset.drift
        if drift_cfg:
            label_weights = get_label_weights(epoch, drift_cfg, self.hparams.seed)
            # Update cumulative drift distribution
            self.cum_label_weights = (self.cum_label_weights * epoch + label_weights.to(self.device)) / (epoch + 1)

            # Sample train_size entries with a weighted sampler from the full train dataset
            train_size = self.config.dataset.train_size
            sampler = get_weighted_sampler(label_weights=label_weights, targets=self.train_dataset, num_samples=train_size)
            train_loader = DataLoader(self.train_dataset, batch_size=self.config.dataset.train_batch_size, sampler=sampler)
        else:
            if self.static_train_loader is None:
                static_train_subset = Subset(self.train_dataset, range(0, self.config.dataset.train_size))
                self.static_train_loader = DataLoader(static_train_subset, batch_size=self.config.dataset.train_batch_size)
            train_loader = self.static_train_loader
        self.train_loader = train_loader
    
    def train_batch_forward(self, x: torch.Tensor):
        x_recon, mu, logvar, z = self.model(x)
        
        eff_rank = compute_effective_rank(z)
        infos = {
            'eff_rank': eff_rank,
        }
        
        losses = {
            'recon': (self.bce_loss, [x_recon, x.detach()]),
            'prior': (self.prior_loss, [mu, logvar]),
        }
        return losses, infos

    def train_batch_backward1(self, shaped_losses:Dict):
        recon_loss = shaped_losses['recon']
        prior_loss = shaped_losses['prior']
        loss = recon_loss + prior_loss

        losses = loss.unbind(0)
        params = [p for p in self.model.encoder.parameters() if p.requires_grad]
        per_sample_grads = [
            torch.autograd.grad(
                l, params,
                retain_graph=True,
                allow_unused=True
            )
            for l in losses
        ]
        # per_sample_grads is a list of length B, each entry is a tuple of grads
        # now compute the norm for each sample
        per_sample_norms = torch.stack([
            torch.cat([g.reshape(-1) for g in grads if g is not None]).norm()
            for grads in per_sample_grads
        ], 0)        

        assert per_sample_norms.isfinite().all()
        return (loss, per_sample_norms)


        # This is how we need to find our grad mags for the ELBO envelope

    def train_batch_backward2(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def eval_dataloader(self, type:str):
        cfg = self.config.dataset
        val_size = cfg.val_size
        val_batch_size = cfg.val_batch_size
        if type=='static':
            if self.static_val_loader is None:
                # Static validation loader
                static_val_subset = Subset(self.val_dataset, range(0, val_size))
                self.static_val_loader = DataLoader(static_val_subset, batch_size=val_batch_size, shuffle=False)
            return self.static_val_loader
        elif type=='drift_current':
            assert cfg.drift # silences pylance warning on next line
            label_weights = get_label_weights(self.epoch, cfg.drift, self.hparams.seed)
            indices = torch.multinomial(
                label_weights,          
                num_samples=val_size,   
                replacement=True        
            ).tolist()
            val_subset_current = Subset(self.val_dataset, indices)
            val_loader_current = DataLoader(val_subset_current, batch_size=val_batch_size)
            return val_loader_current
        elif type=='drift_cumulative':
            # Cumulative drift validation loader
            indices = torch.multinomial(
                self.cum_label_weights,          
                num_samples=val_size,   
                replacement=True        
            ).tolist()
            val_subset_cumulative = Subset(self.val_dataset, indices)
            val_loader_cumulative = DataLoader(val_subset_cumulative, batch_size=val_batch_size)
            return val_loader_cumulative
        else:
            raise ValueError(f"Unknown val loader type {type}")

    def do_eval(self, type:str):
        suffixes = {
            'static':'',
            'drift_current':'_drift_current',
            'drift_cumulative':'_drift_cumulative',
        }
        sfx = suffixes[type] # Will raise on bad type. Desirable behavior.
        dataloader = self.eval_dataloader(type)

        # Static validation
        val_total_samples = 0
        val_total_bce = 0
        val_total_kl = 0
        val_eff_ranks = []

        for x, _ in dataloader:
            x = x.to(self.device)
            batch_size = x.size(0)
            val_total_samples += batch_size
            x_recon, mu, logvar, z = self.model(x)
            recon_loss = self.bce_loss(x_recon, x)
            kl = self.prior_loss(mu, logvar)
            val_eff_ranks.append(batch_size * compute_effective_rank(z))
            val_total_bce += recon_loss.sum().item()
            val_total_kl += kl.sum().item()

        self.logger({
            f'val_recon{sfx}': val_total_bce / val_total_samples,
            f'val_total_kl{sfx}': val_total_kl / val_total_samples,
            f'val_per_dim_kl{sfx}': val_total_kl / (val_total_samples * self.hparams.structure.latent_dim),
            f'val_eff_rank{sfx}': np.sum(val_eff_ranks) / val_total_samples
        })

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            self.do_eval('static')
            if self.config.dataset.drift:
                self.do_eval('drift_current')
                self.do_eval('drift_cumulative')


