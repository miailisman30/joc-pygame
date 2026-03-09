"""
Train a simple DQN agent on FlappySaila using the Gym-compatible wrapper.

Usage:
  python train_dqn.py

Notes:
- Requires numpy, gymnasium (or gym), and torch. If torch isn't installed, the
  script will exit with a helpful message.
"""

import os
import time
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as e:
    print("PyTorch is required to run this training script.\n"
          "Please install torch first, e.g.: pip install torch")
    raise SystemExit(1)

try:
    import gymnasium as gym
except Exception:
    import gym  # type: ignore

from src.envs.flappy_env import FlappyGymEnv


class QNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    replay_size: int = 50_000
    start_epsilon: float = 1.0
    end_epsilon: float = 0.05
    epsilon_decay_steps: int = 50_000
    target_sync_interval: int = 1000
    train_start_size: int = 1000
    max_episodes: int = 300
    max_steps_per_episode: int = 2000
    frame_skip: int = 2
    seed: int = 42


Transition = namedtuple("Transition", "s a r s1 done")


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        s = np.stack([b.s for b in batch], axis=0)
        a = np.array([b.a for b in batch], dtype=np.int64)
        r = np.array([b.r for b in batch], dtype=np.float32)
        s1 = np.stack([b.s1 for b in batch], axis=0)
        done = np.array([b.done for b in batch], dtype=np.float32)
        return s, a, r, s1, done

    def __len__(self):
        return len(self.buffer)


def linear_epsilon(step: int, cfg: DQNConfig) -> float:
    span = max(1, cfg.epsilon_decay_steps)
    frac = min(1.0, step / span)
    return cfg.start_epsilon + frac * (cfg.end_epsilon - cfg.start_epsilon)


def train():
    cfg = DQNConfig()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = FlappyGymEnv(headless=True, frame_skip=cfg.frame_skip, max_steps=cfg.max_steps_per_episode)
    obs, _ = env.reset(seed=cfg.seed)
    obs_dim = obs.shape[0]
    act_dim = env.action_space.n

    q = QNet(obs_dim, act_dim)
    q_target = QNet(obs_dim, act_dim)
    q_target.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=cfg.lr)
    loss_fn = nn.SmoothL1Loss()

    rb = ReplayBuffer(cfg.replay_size)

    global_step = 0
    best_mean = -1e9
    save_path = "dqn_flappy.pt"

    for ep in range(1, cfg.max_episodes + 1):
        obs, _ = env.reset(seed=cfg.seed + ep)
        ep_reward = 0.0
        for t in range(cfg.max_steps_per_episode):
            eps = linear_epsilon(global_step, cfg)
            if np.random.rand() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    qvals = q(torch.from_numpy(obs).float().unsqueeze(0))
                    action = int(torch.argmax(qvals, dim=1).item())

            next_obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            rb.push(obs, action, reward, next_obs, float(done))
            ep_reward += reward
            obs = next_obs
            global_step += 1

            # learn
            if len(rb) >= cfg.train_start_size:
                s, a, r, s1, done_batch = rb.sample(cfg.batch_size)
                s_t = torch.from_numpy(s).float()
                a_t = torch.from_numpy(a).long().unsqueeze(1)
                r_t = torch.from_numpy(r).float().unsqueeze(1)
                s1_t = torch.from_numpy(s1).float()
                d_t = torch.from_numpy(done_batch).float().unsqueeze(1)

                q_sa = q(s_t).gather(1, a_t)
                with torch.no_grad():
                    q_next = q_target(s1_t).max(dim=1, keepdim=True)[0]
                    target = r_t + cfg.gamma * (1.0 - d_t) * q_next
                loss = loss_fn(q_sa, target)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()

                if global_step % cfg.target_sync_interval == 0:
                    q_target.load_state_dict(q.state_dict())

            if done:
                break

        print(f"Episode {ep:4d} | steps={t+1:4d} | reward={ep_reward:7.3f} | eps={eps:5.3f} | score={info.get('score', 0)}")

        # simple moving average over last 10 eps
        if ep % 10 == 0:
            # This is a placeholder; in a full setup we'd store rewards history
            # For brevity, just save periodically
            torch.save(q.state_dict(), save_path)
            print(f"Saved model to {save_path}")

    env.close()


if __name__ == "__main__":
    train()
