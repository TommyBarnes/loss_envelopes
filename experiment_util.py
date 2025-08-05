import torch
import torch.nn as nn
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)

def compute_effective_rank(z, tol=1e-6):
    z_centered = z - z.mean(dim=0, keepdim=True)
    cov = z_centered.T @ z_centered / (z.shape[0] - 1)
    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = eigvals.clamp(min=tol)
    probs = eigvals / eigvals.sum()
    entropy = -(probs * probs.log()).sum()
    return entropy.exp().item()

# More expressive VAE
class VAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 800),
            nn.ReLU(),
            nn.Linear(800, 400),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 800),
            nn.ReLU(),
            nn.Linear(800, 28 * 28),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z


# Sinusoidal label weights for drifting class distributions
from experiment_config import DriftConfig
def get_label_weights(epoch: int, drift: DriftConfig, seed: int):
    """
    Returns a tensor of label weights that follow a sinusoidal pattern over epochs.
    The pattern is controlled by drift_cfg, which should include:
        - freq: frequency multiplier for the sine wave
        - amp: amplitude of the sine wave
        - shift: baseline shift for all weights

    The result is a normalized weight vector over 10 MNIST classes.

    """
    g = torch.Generator()
    g.manual_seed(seed * 10_000 + epoch)  # Ensure reproducibility across epochs

    num_classes = 10 # Assume MNIST
    phase = drift.phase_spread * 2 * torch.pi * torch.rand(num_classes, generator=g)
    shift_offsets = drift.shift + drift.shift_spread * torch.rand(num_classes, generator=g)
    num_frequencies = num_classes if drift.frequency_per_class else 1
    frequency_sample = drift.frequency + drift.frequency * drift.frequency_std * torch.randn(num_frequencies, generator=g)

    t = torch.tensor(epoch, dtype=torch.float32)
    raw_weights = drift.amplitude * torch.sin(frequency_sample * t + phase) + shift_offsets
    raw_weights = raw_weights.clamp(min=1e-3)
    return raw_weights / raw_weights.sum()


# Weighted sampler utility
from torch.utils.data import Dataset
from torch.utils.data import Subset
def get_weighted_sampler(label_weights: torch.Tensor, targets, num_samples:int):
    """
    Create a WeightedRandomSampler based on label weights and dataset targets.

    Args:
        label_weights (Tensor): A 1D tensor of length equal to the number of classes.
        targets (Tensor or Dataset): A 1D tensor of dataset labels or a Subset object.

    Returns:
        torch.utils.data.WeightedRandomSampler: Sampler for DataLoader.
    """
    # Handle case where targets is a Subset of a dataset with `.targets`
    if isinstance(targets, Subset):
        full_targets = targets.dataset.targets # type: ignore
        subset_indices = targets.indices
        targets = full_targets[subset_indices]

    sample_weights = label_weights[targets]
    sample_weights = sample_weights / sample_weights.sum()  # Normalize to sum to 1
    num_samples = len(targets)

    return torch.utils.data.WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=num_samples,
        replacement=True
    )
