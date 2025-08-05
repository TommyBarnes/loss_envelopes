# supervised_ca_exp.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

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


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)   # single output for regression
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # shape [B]


class CaliforniaExperiment:
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

        # --- load & preprocess Boston housing data ---
        housing = fetch_california_housing(as_frame=True)
        X = housing.data           # type: ignore # shape [N, D]
        y = housing.target         # type: ignore # shape [N]

        # standardize inputs and targets
        self.X_scaler = StandardScaler().fit(X)
        self.y_scaler = StandardScaler().fit(y.to_numpy().reshape(-1, 1))

        X = self.X_scaler.transform(X)
        y = self.y_scaler.transform(y.to_numpy().reshape(-1, 1)).flatten()

        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()

        # Shuffle the dataset based on seed
        rng = torch.Generator().manual_seed(hparams.seed)
        shuffled_indices = torch.randperm(len(X_tensor), generator=rng).tolist()
        X_tensor = X_tensor[shuffled_indices]
        y_tensor = y_tensor[shuffled_indices]
        full_dataset = TensorDataset(X_tensor, y_tensor)

        ds_cfg = config.dataset
        train_subset = Subset(
            full_dataset,
            range(ds_cfg.train_start,
                  ds_cfg.train_start + ds_cfg.train_size)
        )
        val_subset = Subset(
            full_dataset,
            range(ds_cfg.val_start,
                  ds_cfg.val_start + ds_cfg.val_size)
        )

        base_train_loader = DataLoader(
            train_subset,
            batch_size=ds_cfg.train_batch_size,
            shuffle=True,
            drop_last=False,
        )
        self.train_loader = _TrainLoaderWrapper(base_train_loader)

        self.val_loader = DataLoader(
            val_subset,
            batch_size=ds_cfg.val_batch_size,
            shuffle=False,
        )

        # --- model, loss, optimizer ---
        input_dim = X.shape[1]
        hidden_dim = getattr(hparams, "hidden_dim", 64)
        self.model = MLPRegressor(input_dim=input_dim,
                                  hidden_dim=hidden_dim).to(self.device)

        # per‐sample MSE
        self.loss_fn = nn.MSELoss(reduction='none')
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

        preds = self.model(inputs)
        # per-sample MSE
        # shaped / envelope logic handled by runner
        infos = {}
        losses = {
            'mse': (self.loss_fn, [preds, targets.detach()])
        }
        return losses, infos

    def train_batch_backward(self, shaped_losses):
        loss = shaped_losses['mse']
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def eval(self):
        """
        Called by runner after each epoch: do a standard val‐set eval,
        log 'val_mse' and 'val_rmse'.
        """
        self.model.eval()
        total, sum_loss = 0, 0.0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                preds = self.model(inputs)
                losses = self.loss_fn(preds, targets)  # [B]
                sum_loss += losses.sum().item()
                total += targets.size(0)

        # unscale MSE back to original target units
        # note: Var(scaled_y) = Var(y) * scale^2, so MSE in original space:
        mse_scaled = sum_loss / total
        mse_original = mse_scaled * (self.y_scaler.scale_[0] ** 2) # type: ignore

        self.logger({
            'val_mse': mse_original,
            'val_rmse': mse_original ** 0.5,
        })