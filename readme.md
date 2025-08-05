# Loss Attenuation Envelopes (LAEs)

This repository accompanies the paper **"Loss Attenuation Envelopes: A General-Purpose Gradient Shaping Technique for Robust Optimization."**  
It includes code for all experiments, including supervised learning, unsupervised variational autoencoders (VAEs), and reinforcement learning with PPO.

## Overview

Loss Attenuation Envelopes (LAEs) are a general-purpose method for improving optimization by downweighting low-loss samples during training.

They act as a dynamic filter on gradient contributions, targeting low signal-to-noise ratio (SNR) samples or samples likely to lead to overfitting.

Unlike task-specific techniques like free bits (for VAEs) or KL penalties (for PPO), LAEs apply to any per-sample loss and are easy to plug into existing training loops.

### Key Contributions
- **Unified Framework**: LAEs generalize techniques like free bits and hard example mining under a shared gradient shaping lens.
- **SNR Perspective**: We provide a novel justification for LAEs via per-sample gradient SNR, rooted in model capacity and approximation theory.
- **Regularization Effect**: LAEs can also act as a form of dynamic regularization, suppressing overfitting by halting updates from low-loss samples that may encode spurious or overspecialized patterns.
- **Empirical Results**: Demonstrated effectiveness across supervised learning, unsupervised VAEs, and reinforcement learning with PPO — with minimal tuning.

## Paper

The paper is included as [`lae_paper.pdf`](./lae_paper.pdf).  
It is currently a preliminary draft to foster discussion. Feel free to reach out with cirtical feedback, or if you find it interesting!

## Code

Brief overview of the code.

run_ablation.py      # Entry point for running experiments
envelopes/           # LAE implementation - drop this into your own project!
analysis/            # Scripts to reproduce plots and metrics
experiments/         # YAML configs for each experiment
lae_paper.pdf        # The draft paper
README.md            # This file

## Author

**Tommy Barnes**  
Independent researcher
Feel free to connect or cite this repo if you find it useful.

## License

The paper text is © Tommy Barnes, released for non-commercial academic use.

MIT License