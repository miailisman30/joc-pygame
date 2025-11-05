"""Train a PPO agent on FlappySaila using Stable-Baselines3.

Usage:
  python ppo_train.py --timesteps 200000 --frame-skip 2 --n-envs 8 --seed 42

TensorBoard:
  tensorboard --logdir runs
"""

import argparse
import os
from typing import Callable

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from flappy_env import FlappyGymEnv


def make_env(headless: bool, frame_skip: int, seed: int, shaping: bool) -> Callable[[], Monitor]:
    def _thunk():
        env = FlappyGymEnv(headless=headless, frame_skip=frame_skip, shaping=shaping)
        return Monitor(env, filename=None)
    return _thunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logdir", type=str, default="runs/ppo")
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--shaping", type=int, default=1)
    args = parser.parse_args()

    headless = True
    n_envs = max(1, args.n_envs)
    frame_skip = max(1, args.frame_skip)

    # Vectorized environments
    if n_envs > 1:
        env_fns = [make_env(headless, frame_skip, args.seed + i, bool(args.shaping)) for i in range(n_envs)]
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv([make_env(headless, frame_skip, args.seed, bool(args.shaping))])

    os.makedirs(args.logdir, exist_ok=True)
    checkpoint_cb = CheckpointCallback(save_freq=args.checkpoint_every // n_envs, save_path=args.logdir, name_prefix="ppo_flappy")

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        n_steps=2048 // n_envs,
        batch_size=64,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        learning_rate=3e-4,
        tensorboard_log=args.logdir,
        seed=args.seed,
    )

    model.learn(total_timesteps=args.timesteps, callback=checkpoint_cb)
    model.save(os.path.join(args.logdir, "ppo_flappy_final"))
    vec_env.close()


if __name__ == "__main__":
    main()
