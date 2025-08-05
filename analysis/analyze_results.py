# analyze_results.py
import os
import json
import numpy as np
import pandas as pd
from typing import List

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hyperparameter_config import HParamConfig
# Import the ParamSpec types for envelope parameter unpacking
from envelopes.envelope_config import LossAdaptiveParam

# List of metrics to extract at final epoch
TRACKED_METRICS = [
    'val_recon',
    'val_recon_drift_current',
    'val_recon_drift_cumulative',
    'val_loss',
    'val_acc',
    'val_mse',
    'val_rmse',
    # add other metrics you need
]

def load_all_results(results_dir='results', final_epoch=False, epoch=None) -> pd.DataFrame:
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    rows = []
    for root, _, files in os.walk(results_dir):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            full_path = os.path.join(root, fname)

            try:
                config_str = os.path.splitext(fname)[0]
                hp = HParamConfig.from_string(config_str)
            except Exception:
                continue

            with open(full_path, 'r') as f:
                record = json.load(f)

            epochs = record.get('epoch', [])
            if not epochs:
                continue
            metrics = [m for m in record.keys() if m not in ['epoch']]

            # Start the row with seed
            row: dict = {'seed': hp.seed}
            row['hp'] = hp.to_string(exclude_seed=True)

            if final_epoch or epoch is not None:
                if final_epoch:
                    final_epoch_idx = len(epochs) - 1
                else:
                    final_epoch_idx = epoch
                for m in metrics:
                    vals = record.get(m, [])
                    row[m] = vals[final_epoch_idx] if final_epoch_idx < len(vals) else np.nan
                rows.append(row)
            else:
                for e in range(len(epochs)):
                    row = row.copy()
                    row['epoch'] = e
                    for m in metrics:
                        vals = record.get(m, [])
                        row[m] = vals[e] if e < len(vals) else np.nan
                    rows.append(row)
            
    df = pd.DataFrame(rows)

    def extract_structure_from_hpconfig(hp_string):
        hp = HParamConfig.from_string(hp_string)
        series_data = {'hp':hp_string}
        latent_dim = -1
        if hp.structure is not None:
            if hp.structure.latent_dim is not None:
                latent_dim = hp.structure.latent_dim
        series_data.update({'latent_dim':latent_dim})
        return pd.Series(data=series_data)
    structure_df = df.hp.apply(extract_structure_from_hpconfig)
    df = df.merge(structure_df, on='hp')

    groupby = ['hp']
    if not (final_epoch or epoch is not None):
        groupby.append('epoch')
    if df['latent_dim'].nunique()>1:
        groupby.append('latent_dim')

    summary = (
        df
        .groupby(groupby)[metrics]
        .agg(['mean', 'std'])
        .reset_index()
    )

    return summary

def best_joint_envelopes(df_summary: pd.DataFrame, metric: str, maximize: bool = False) -> pd.DataFrame:
    """
    From a tall summary DataFrame, return only the best row per envelope shape
    according to metric_mean.
    """
    metric_col = f"{metric}_mean"
    ascending = not maximize
    shape_keys = [c for c in df_summary.columns if c.endswith('_shape')]
    if not shape_keys:
        return pd.DataFrame(columns=df_summary.columns)
    df_short = (
        df_summary
        .sort_values(metric_col, ascending=ascending)
        .groupby(shape_keys, dropna=False)
        .head(1)
        .reset_index(drop=True)
    )

    # Replace sentinel -1 with dash for missing parameters
    df_short = df_short.replace({-1: '-'})

    return df_short

def topk_by_group(df, group_keys, metric, minimize):
    metric_col=(metric,'mean')
    metric_std_col=(metric,'std')
    show_keys = [metric_col,metric_std_col]
    def expand_hpconfig(hp_string):
        hp = HParamConfig.from_string(hp_string)
        series_data = {('hp',''):hp_string}
        for name, env in hp.envelopes.items():
            series_data.update({(name,'shape'):env.shape,(name,'type'):env.adaptivity})
        return pd.Series(data=series_data)
 
    env_infos = df.hp.apply(expand_hpconfig)
    env_cols = [col for col in env_infos.columns if col not in [('hp','')]]
    env_cols.extend(group_keys)
    df = df.merge(env_infos, on='hp')
    df = (
        df
        .sort_values(metric_col, ascending=minimize)
        .groupby(env_cols)
        .head(1)
        .reset_index()
    )

    return df

def display_table(df, metric, minimize, show_keys=[('hp','')]):
    metric_col=(metric,'mean')
    show_keys = [*show_keys, metric_col, (metric,'std')]

    df = df.sort_values(metric_col, ascending=minimize)[show_keys]

    # Replace sentinel -1 with dash for missing parameters
    df = df.replace({-1: '-'})

    print(df.to_string(
        index=False,
        columns=show_keys,
        float_format=lambda x: f"{x:g}"
    ))


def supervised_results():
    exp_name = 'supervised_ca'

    if True:
        results_dir = f"results/{exp_name}/data"
        df = load_all_results(results_dir, final_epoch=False, epoch=99)
        # df = df[df.epoch==20]

        metric, minimize = 'val_mse', True
        print(f"\n=== Rankings {exp_name}:{metric} ===")
        display_table(df, metric, minimize=minimize)

        val_loss_df = topk_by_group(df, [], metric, minimize )
        print(f"\n=== Topk {exp_name}:{metric} ===")
        display_table(val_loss_df, metric, minimize=minimize, show_keys=[('hp','')])

    else:
        results_dir = f"results/{exp_name}/data"
        df = load_all_results(results_dir, final_epoch=True)

        print(f"\n=== Rankings {exp_name}:val_loss ===")
        display_table(df, 'val_loss', minimize=True)
        print(f"\n=== Rankings {exp_name}:val_acc ===")
        display_table(df, 'val_acc', minimize=False)

        val_loss_df = topk_by_group(df, [], 'val_loss', True )
        print(f"\n=== Topk {exp_name}:{'val_loss'} ===")
        display_table(val_loss_df, 'val_loss', minimize=True, show_keys=[('hp','')])

        val_acc_df = topk_by_group(df, [], 'val_acc', False )
        print(f"\n=== Topk {exp_name}:{'val_acc'} ===")
        display_table(val_acc_df, 'val_acc', minimize=False, show_keys=[('hp','')])

def rl_results():
    # exp_name = 'ppo_lunar2'
    exp_name = 'ppo_acrobot'
    results_dir = f"results/{exp_name}/data"
    df = load_all_results(results_dir, final_epoch=True)

    # df = load_all_results(results_dir, final_epoch=False, epoch=500)
    # df = df[df.epoch == 500]

    metric, minimize = 'ep_return_ma100', False

    print(f"\n=== Rankings {exp_name}:{metric} ===")
    display_table(df, metric, minimize=minimize)

    val_loss_df = topk_by_group(df, [], metric, minimize )
    print(f"\n=== Topk {exp_name}:{metric} ===")
    display_table(val_loss_df, metric, minimize=minimize, show_keys=[('hp','')])

def vae_results():
    exp_name = 'vae_overparam'

    results_dir = f"results/{exp_name}/data"
    df = load_all_results(results_dir, final_epoch=True)
    df = df[df.hp.str.contains('total=identity')]

    metric, minimize = 'val_recon', True

    val_loss_df = topk_by_group(df, [], metric, minimize )
    print(f"\n=== Topk {exp_name}:{metric} ===")
    display_table(val_loss_df, metric, minimize=minimize, show_keys=[('hp','')])

if __name__ == "__main__":
    if True:
        # vae_results()
        rl_results()
        # supervised_results()
    else:
        exp_name = 'supervised'
        exp_name = 'vae_overparam'
        exp_name = 'ppo_lunar'
        exp_name = 'vae_drift'
        results_dir = f"results/{exp_name}/data"
        df = load_all_results(results_dir, final_epoch=True)

        metric, minimize = 'ep_return_ma100', False
        metric, minimize = 'val_recon', True
        print(f"\n=== Rankings {exp_name}:{metric} ===")
        display_table(df, metric, minimize=minimize, show_keys=[('hp',''),('latent_dim','')])

        df = topk_by_group(df, [('latent_dim','')], metric, minimize )
        print(f"\n=== Topk {exp_name}:{metric} ===")
        display_table(df, metric, minimize=minimize, show_keys=[('hp',''),('latent_dim','')])



        # full rankings
        print(f"\n=== Rankings {exp_name}:val_loss ===")
        display_table(df, 'val_loss', minimize=True)
        print(f"\n=== Rankings {exp_name}:val_acc ===")
        display_table(df, 'val_acc', minimize=False)

        df = topk_by_group(df, [], 'val_loss', True)
        print(f"\n=== test {exp_name}:val_acc ===")
        display_table(df, 'val_loss', minimize=True)

        # 'short' best-per-shape
        df_short_loss = best_joint_envelopes(df, 'val_loss', False)
        df_short_acc  = best_joint_envelopes(df, 'val_acc', True)

        print(f"\n=== Best per envelope shape for {exp_name}:val_loss ===")
        print(df_short_loss.to_string(index=False, float_format=lambda x: f"{x:g}"))

        print(f"\n=== Best per envelope shape for {exp_name}:val_acc ===")
        print(df_short_acc.to_string(index=False, float_format=lambda x: f"{x:g}"))