import time
from multiprocessing import Pool
from util import is_linux
from tqdm import tqdm
import os
from experiment_config import ExperimentConfig
from run_experiment import run_experiment_from_config
from hyperparameter_config import HParamConfig, HParamSweep
from functools import partial
from typing import List
import yaml

def run_sweep(experiment_config: ExperimentConfig, configs: List[HParamConfig], use_multiprocessing: bool = True):

    root_dir = os.path.join("results", experiment_config.name)
    data_dir = os.path.join(root_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Save config YAML into root_dir
    experiment_config.save_to(root_dir)

    run_cfg_partial = partial(_run_cfg, experiment_config=experiment_config, data_dir=data_dir)

    max_concurrent_processes = 8 
    if is_linux():
        import pickle
        for cfg in configs:
            pickle.dumps(cfg)

        os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads to avoid conflicts
        os.environ['MKL_NUM_THREADS'] = '1'  # Limit MKL threads to avoid conflicts
        max_concurrent_processes = 10  # Works on my machine ;) pick a number that works for you

    if use_multiprocessing:
        with Pool(processes=max_concurrent_processes) as pool:
            for result in tqdm(pool.imap_unordered(run_cfg_partial, configs), total=len(configs), desc="Experiments"):
                cfg, elapsed = result
                tqdm.write(f"Completed {cfg} in {elapsed:.2f} seconds")
    else:
        for cfg in tqdm(configs, desc="Experiments"):
            result = run_cfg_partial(cfg)
            cfg, elapsed = result
            tqdm.write(f"Completed {cfg} in {elapsed:.2f} seconds")

# This function has to be top level so multiprocessing.pool can pickle it
def _run_cfg(cfg: HParamConfig, experiment_config: ExperimentConfig, data_dir: str):
    start_time = time.time()
    save_name = cfg.to_string() + ".json"
    save_path = os.path.join(data_dir, save_name)

    if os.path.exists(save_path):
        return cfg, 0.0  # Skip if file already exists

    run_experiment_from_config(
        hparams=cfg,
        experiment_config=experiment_config,
        device='cpu',
        save_dir=data_dir,
        save_name=save_name,
    )
    elapsed = time.time() - start_time
    return cfg, elapsed

if __name__ == '__main__':
    if is_linux():
        import multiprocessing as mp
        mp.set_start_method('spawn', force=True)  # Ensure 'spawn' method is used for multiprocessing

    # List of experiment config files to run
    config_paths = ["experiments/ppo_lunar.yaml"]
    for cfg_path in config_paths:
        # Load experiment configuration
        exp_config = ExperimentConfig(cfg_path)
        # Load sweep dictionary from the same YAML file
        with open(cfg_path, 'r') as f:
            raw = yaml.safe_load(f)
        sweep = raw.get('sweep', {})
        # Expand hyperparameter configurations from sweep section
        hparam_configs = HParamSweep(sweep).expand()


        import sys
        def is_debugger_active():
            return sys.gettrace() is not None

        # Run the experiments
        run_sweep(exp_config, hparam_configs, use_multiprocessing=not is_debugger_active())