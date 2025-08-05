import json
from experiment_util import set_seed
from envelopes.envelope_config import build_loss_envelope
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import gymnasium as gym
from typing import Callable, Dict, Any

from experiment_config import ExperimentConfig
from hyperparameter_config import HParamConfig
from experiment_util import set_seed, compute_effective_rank

class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # separate actor network
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim),
        )
        # separate critic network
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        raise NotImplementedError("Call actor or critic separately")

    def action_distribution(self, x):
        logits = self.actor(x)
        return torch.distributions.Categorical(logits=logits)

    def value(self, x):
        return self.critic(x).squeeze(-1)

class PPOExperiment:
    def __init__(self, config: ExperimentConfig, hparams: HParamConfig, device, logger: Callable):
        assert hparams.seed is not None
        set_seed(hparams.seed)
        
        self.config = config
        self.hparams = hparams
        self.device = device
        self.logger = logger

        self.env = gym.make(config.rl.environment)
        obs_dim = self.env.observation_space.shape[0] # type: ignore
        act_dim = self.env.action_space.n # type: ignore

        self.model = PPOActorCritic(obs_dim, act_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)

        self.gamma = 0.99
        self.lam = 0.95
        self.clip_eps = 0.2
        self.max_steps = config.rl.max_steps
        self.ppo_epochs = config.rl.ppo_epochs
        self.minibatch_size = config.rl.minibatch_size
        # rollout size for each PPO update
        self.rollout_size = config.rl.rollout_size

        # build policy and value loss envelopes
        self.policy_envelope = build_loss_envelope(hparams.envelopes['policy'])
        self.value_envelope  = build_loss_envelope(hparams.envelopes['value'])

        from collections import deque  # ensure deque is imported (already is)
        self.return_deque = deque(maxlen=100)
        self.current_episode_return = 0.0

    def train(self):
        total_steps = 0
        rollout_count = 0
        obs, _ = self.env.reset(seed=self.hparams.seed)
        current_episode_done = False
        current_episode_len = 0
        while total_steps < self.max_steps:
            # buffers
            obs_buf, act_buf, logp_buf = [], [], []
            rewards, values_buf, dones_buf = [], [], []
            ep_lens = []

            rollout_steps = min(self.rollout_size, self.max_steps - total_steps)
            for _ in range(rollout_steps):
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
                dist = self.model.action_distribution(obs_tensor)
                action = dist.sample()
                logp = dist.log_prob(action)

                next_obs, reward, terminated, truncated, _ = self.env.step(action.item())
                done = terminated or truncated

                value = self.model.value(obs_tensor)
                # append to buffers
                obs_buf.append(obs)
                act_buf.append(action.item())
                logp_buf.append(logp.item())
                rewards.append(reward)
                self.current_episode_return += reward
                values_buf.append(value.item())
                dones_buf.append(done)

                current_episode_len += 1
                total_steps += 1

                if done:
                    self.return_deque.append(self.current_episode_return)
                    ep_lens.append(current_episode_len)
                    obs, _ = self.env.reset()
                    current_episode_len = 0
                    self.current_episode_return = 0.0
                else:
                    obs = next_obs

            # compute GAE advantages and returns
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                next_value = self.model.value(obs_tensor).item()
            advantages, returns = self._compute_gae(rewards, values_buf, dones_buf, next_value)

            # log rollout statistics
            rollout_count += 1
            self.logger({
                "epoch": rollout_count,
                "total_steps": total_steps,
                "ep_length_mean": float(np.mean(ep_lens)),
                "ep_return_ma100": float(np.mean(self.return_deque)) if self.return_deque else 0.0
            })

            # policy update
            self._train_policy(obs_buf, act_buf, logp_buf, advantages, returns)


    def _compute_gae(self, rewards, values, dones, next_value):
        """
        Compute generalized advantage estimation (GAE).
        rewards: list of rewards for the rollout
        values: list of value estimates for each state in rollout
        dones: list of boolean done flags
        next_value: value estimate of the state following the last rollout state
        Returns (advantages, returns) as numpy arrays.
        """
        advantages = []
        gae = 0.0
        values_extended = values + [next_value]
        for step in reversed(range(len(rewards))):
            mask = 0.0 if dones[step] else 1.0
            delta = rewards[step] + self.gamma * values_extended[step + 1] * mask - values_extended[step]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values)]
        return np.array(advantages), np.array(returns)

    def _train_policy(self, obs_buf, act_buf, logp_buf, advantages, returns):
        obs_tensor = torch.tensor(np.array(obs_buf), dtype=torch.float32).to(self.device)
        act_tensor = torch.tensor(np.array(act_buf), dtype=torch.int64).to(self.device)
        logp_old_tensor = torch.tensor(np.array(logp_buf), dtype=torch.float32).to(self.device)
        adv_tensor = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        # normalize advantages
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
        ret_tensor = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # PPO-style multi-epoch, mini-batch updates
        N = obs_tensor.size(0)
        indices = np.arange(N)
        for epoch in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, N, self.minibatch_size):
                mb_idx = indices[start:start + self.minibatch_size]
                mb_obs = obs_tensor[mb_idx]
                mb_act = act_tensor[mb_idx]
                mb_logp_old = logp_old_tensor[mb_idx]
                mb_adv = adv_tensor[mb_idx]
                mb_ret = ret_tensor[mb_idx]

                dist = self.model.action_distribution(mb_obs)
                logp = dist.log_prob(mb_act)
                ratio = torch.exp(logp - mb_logp_old)
                clip_adv = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_adv

                policy_loss = -torch.min(ratio * mb_adv, clip_adv)
                value_loss = ((self.model.value(mb_obs) - mb_ret) ** 2)

                policy_loss = self.policy_envelope(policy_loss)
                value_loss  = self.value_envelope(value_loss)

                total_loss = policy_loss + 0.5 * value_loss
                total_loss = total_loss.sum()

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

# Run experiment from config, modeled after other experiment files
def run_experiment_from_config(config: ExperimentConfig, hparams: HParamConfig, device:str, save_dir:str, save_name:str):
    set_seed(hparams.seed)

    logger_data: dict = {}
    def logger(metrics):
        for k, v in metrics.items():
            if k not in logger_data:
                logger_data[k] = []
            logger_data[k].append(v)

    experiment = PPOExperiment(config, hparams, device, logger)
    experiment.train()

    import os
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, save_name)
    with open(file_path, "w") as f:
        json.dump(logger_data, f)