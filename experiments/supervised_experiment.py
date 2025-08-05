# supervised_experiment.py
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from torchvision import datasets, transforms

from experiment_config import ExperimentConfig
from hyperparameter_config import HParamConfig


class _TrainLoaderWrapper:
    """
    Wrap a standard DataLoader so that each batch
    looks like:  for x, _ in train_loader:
      x == (inputs, targets)
    """
    def __init__(self, base_loader):
        self.base = base_loader

    def __iter__(self):
        for inputs, targets in self.base:
            # runner will unpack as: x == (inputs, targets), _ == None
            yield (inputs, targets), None

    def __len__(self):
        return len(self.base)


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class SupervisedExperiment:
    def __init__(
        self,
        config: ExperimentConfig,
        hparams: HParamConfig,
        device,
        logger,
    ):
        self.config = config
        self.hparams = hparams
        self.device = device
        self.logger = logger

        # --- dataset setup ---
        ds_cfg = config.dataset
        dataset_name = 'mnist' # TODO get from config
        if dataset_name == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            full = datasets.MNIST('./data',
                                  train=True,
                                  download=True,
                                  transform=transform)
            input_dim, num_classes = 28*28, 10

        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        rng = random.Random(hparams.seed)
        indices = list(range(len(full)))
        rng.shuffle(indices)

        train_indices = indices[:ds_cfg.train_size]
        val_indices = indices[ds_cfg.train_size:ds_cfg.train_size + ds_cfg.val_size]

        train_subset = Subset(full, train_indices)
        val_subset = Subset(full, val_indices)

        base_train_loader = DataLoader(
            train_subset,
            batch_size=ds_cfg.train_batch_size,
            shuffle=True,
            drop_last=False,
        )
        # wrap so runner sees x == (inputs, targets)
        self.train_loader = _TrainLoaderWrapper(base_train_loader)

        self.val_loader = DataLoader(
            val_subset,
            batch_size=ds_cfg.val_batch_size,
            shuffle=False,
        )

        # --- model, loss, optimizer ---
        self.model = SimpleMLP(
            input_dim=input_dim,
            hidden_dim=256,
            num_classes=num_classes
        ).to(self.device)

        # per‐sample CE
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self, epoch: int):
        self.model.train()

    def train_batch_forward(self, batch):
        """
        runner will call: losses, info = exp.train_batch_forward(x)
        where x == (inputs, targets)
        """
        (inputs, targets) = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        logits = self.model(inputs)
        # per_sample_loss = self.loss_fn(logits, targets)  # shape [B]

        # no extra info to log here
        infos = {}

        # The runner expects: losses = { name: (loss_fn, loss_args) }
        losses = {
            'classification': (self.loss_fn, [logits, targets.detach()])
        }

        return losses, infos

    def train_batch_backward(self, shaped_losses):
        """
        shaped_losses['classification'] is a 1‐D tensor of per-sample
        losses *after* envelope, but before sum().
        """
        loss = shaped_losses['classification']
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def eval(self):
        """
        Called by runner after each epoch: do a standard val‐set eval,
        log 'val_loss' and 'val_acc'.
        """
        self.model.eval()
        total, correct, sum_loss = 0, 0, 0.0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(inputs)
                losses = self.loss_fn(logits, targets)  # [B]
                sum_loss += losses.sum().item()

                preds = logits.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        self.logger({
            'val_loss': sum_loss / total,
            'val_acc':  correct / total,
        })