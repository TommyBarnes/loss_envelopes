import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envelopes.envelope_shape import StepShape, SigmoidShape, LinearShape

# Unified plotting function for envelope shapes
def plot_envelope_shapes():
    T = 1.0
    invslope = 1.0
    loss = np.linspace(0, 2 * T + invslope, 500)

    shapes = {
        "Step Envelope": (StepShape, {"T": T}),
        "Linear Envelope": (LinearShape, {"T": T, "invslope": invslope}),
        "Sigmoid Envelope": (SigmoidShape, {"T": T, "invslope": invslope}),
    }

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    for ax, (title, (ShapeClass, params)) in zip(axs, shapes.items()):
        loss_tensor = torch.tensor(loss, dtype=torch.float32)
        weights = ShapeClass.compute_weight(loss_tensor, params).numpy()
        ax.plot(loss, weights, label="w(ℓ)", color='C0', linewidth=2)

        ax.axvline(T, color='gray', linestyle='--', label='T')
        if 'invslope' in params:
            ax.axvspan(T - invslope / 2, T + invslope / 2, color='gray', alpha=0.2, label='invslope region')

        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, 2 * T + invslope)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xlabel("Loss ℓ")
        ax.grid(True)
        ax.set_ylabel("Weight w(ℓ)")
        ax.set_yticklabels(["0", "0.5", "1"])
        ax.set_xticks([])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_envelope_shapes()