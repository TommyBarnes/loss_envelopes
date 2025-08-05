# train_experiment.py

import os
import json
from typing import Dict, Any
import torch
import numpy as np
from torch.func import grad, vmap

from experiment_config import ExperimentConfig
from hyperparameter_config import HParamConfig
from envelopes.envelope_config import build_loss_envelope

from experiment_util import set_seed

from experiments.vae_experiment import VAEExperiment
from experiments.supervised_experiment import SupervisedExperiment
from experiments.supervised_ca_exp import CaliforniaExperiment

def run_experiment_from_config(hparams: HParamConfig, 
                               experiment_config: ExperimentConfig, 
                               device, save_dir, save_name):
    if experiment_config.name == 'ppo_acrobot' or experiment_config.name == 'ppo_lunar' or experiment_config.name == 'ppo_lunar2':
        from experiments.ppo_experiment import run_experiment_from_config
        return run_experiment_from_config(experiment_config, hparams, device, save_dir, save_name)

    # Easier math when they are evenly divisible.
    assert 0==(experiment_config.dataset.train_size % experiment_config.dataset.train_batch_size),"Train size should be evenly divisible by batch size"
    assert 0==(experiment_config.dataset.val_size % experiment_config.dataset.val_batch_size),"Val size should be evenly divisible by batch size"
    
    set_seed(hparams.seed)

    logger_data: dict = {}
    def logger(metrics):
        for k, v in metrics.items():
            if k not in logger_data:
                logger_data[k] = []
            logger_data[k].append(v)


    EXPT_REGISTRY = {
        'vae_overparam': VAEExperiment,
        'vae_drift': VAEExperiment,
        'supervised': SupervisedExperiment,
        'supervised_ca': CaliforniaExperiment,
        'supervised_ca_loss_mass': CaliforniaExperiment,
    }
    exp_cls = EXPT_REGISTRY[experiment_config.name]
    exp = exp_cls(config=experiment_config, hparams=hparams, device=device, logger=logger)

    loss_envelopes = {name:build_loss_envelope(env_config) for name,env_config in hparams.envelopes.items()}

    for epoch in range(experiment_config.epochs):
        epoch_raw_losses = { }
        epoch_shaped_losses = { }
        batch_infos = {}
        batch_losses = {}
        exp.train(epoch)
        total_samples = 0
        for x, _ in exp.train_loader:
            if isinstance(x, tuple):
                batch_size = x[0].size(0)
            else:
                batch_size = x.size(0)
            total_samples += batch_size

            losses, info = exp.train_batch_forward(x)

            # Store the infos for later logging
            for name, data in info.items():
                if name not in batch_infos:
                    batch_infos[name] = 0
                batch_infos[name] = batch_infos[name] + data

            # Get the losses, shape them, and log them
            shaped_losses = {}
            def _process_loss(loss_name, loss_tuple):
                if loss_name not in batch_losses:
                    batch_losses[loss_name]=[]

                envelope = loss_envelopes[loss_name]
                loss_fn, loss_X = loss_tuple
                _loss = loss_fn(*loss_X)
                # collect raw loss vector
                epoch_raw_losses.setdefault(loss_name, []).append(_loss.detach().cpu().view(-1))
                raw_vec = _loss
                shaped_vec = envelope(raw_vec)

                # collect shaped loss vector
                epoch_shaped_losses.setdefault(loss_name, []).append(shaped_vec.detach().cpu().view(-1))
                # sum for backward
                _shaped_loss = shaped_vec.reshape(batch_size,-1).sum(dim=1)
                shaped_losses[loss_name] = _shaped_loss

                batch_losses[loss_name].append({
                    # 'grad_X': grad_norm,
                    'loss': _loss,
                    'shaped_loss': _shaped_loss,
                })
                return shaped_vec

            shaped_losses = {}
            for loss_name, loss_tuple in losses.items():
                shaped_losses[loss_name] = _process_loss(loss_name, loss_tuple).mean()

            exp.train_batch_backward(shaped_losses)

        # Call the logger for the info in batch_infos and batch_losses
        epoch_log: Dict[str,Any] = {'epoch': epoch + 1 }
        
        for name, info in batch_infos.items():
            # batch sizes are even so just take the mean
            epoch_log[name] = info / total_samples

        for name in epoch_raw_losses:
            raw_all = torch.cat(epoch_raw_losses[name]).numpy()
            epoch_log[f'{name}_raw_mean'] = float(raw_all.mean())
            epoch_log[f'{name}_raw_std']  = float(raw_all.std())

            shaped_all = torch.cat(epoch_shaped_losses[name]).numpy()
            epoch_log[f'{name}_shaped_mean'] = float(shaped_all.mean())
            epoch_log[f'{name}_shaped_std']  = float(shaped_all.std())
        
        logger(epoch_log)

        # Log info from the envelopes, prefixing keys manually
        for envelope_name, envelope in loss_envelopes.items():
            info_dict = envelope.logging_dict()
            for k, v in info_dict.items():
                logger({f'{envelope_name}_{k}': v})

        exp.eval() # Does full evals on validation set and logs results

    def round_floats(obj, sig_figs=5):
        if isinstance(obj, float):
            return round(obj, sig_figs)
        elif isinstance(obj, dict):
            return {k: round_floats(v, sig_figs) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [round_floats(v, sig_figs) for v in obj]
        else:
            return obj
    logger_data = round_floats(logger_data)  # type: ignore

    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, save_name)
    with open(file_path, 'w') as f:
        json.dump(logger_data, f)
