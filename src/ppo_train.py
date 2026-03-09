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
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from flappy_env import FlappyGymEnv


def make_env(headless: bool, frame_skip: int, seed: int, shaping: bool, curriculum: bool) -> Callable[[], Monitor]:
    def _thunk():
        # Force headless mode by disabling pygame video/display in each subprocess
        import os
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        env = FlappyGymEnv(headless=headless, frame_skip=frame_skip, shaping=shaping, curriculum=curriculum)
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
    parser.add_argument("--curriculum", type=int, default=0, help="Enable curriculum gap/speed adjustment")
    parser.add_argument("--normalize", type=int, default=1, help="Use VecNormalize for obs/reward")
    parser.add_argument("--eval-freq", type=int, default=50_000, help="Evaluation frequency in env steps")
    parser.add_argument("--net-arch", type=str, default="256,256", help="Hidden layer sizes or pi/vf split. Examples: '256,256,128' or 'pi:256,256|vf:512,256' ")
    parser.add_argument("--activation", type=str, default="tanh", choices=["tanh", "relu"], help="Activation function for policy/value MLP")
    parser.add_argument("--from-model", type=str, default="", help="Path to PPO .zip to resume training from")
    parser.add_argument("--resume-stats", type=int, default=1, help="When resuming, try to load VecNormalize stats from same folder")
    args = parser.parse_args()

    headless = True
    n_envs = max(1, args.n_envs)
    frame_skip = max(1, args.frame_skip)

    # Build base env(s)
    if n_envs > 1:
        env_fns = [make_env(headless, frame_skip, args.seed + i, bool(args.shaping), bool(args.curriculum)) for i in range(n_envs)]
        base_env = SubprocVecEnv(env_fns)
    else:
        base_env = DummyVecEnv([make_env(headless, frame_skip, args.seed, bool(args.shaping), bool(args.curriculum))])

    # Optionally load VecNormalize stats when resuming, else create fresh normalize wrapper
    vec_env = base_env
    if bool(args.normalize):
        if args.from_model and bool(args.resume_stats):
            # Try to find vecnormalize.pkl alongside the model
            stats_candidates = [
                os.path.join(os.path.dirname(args.from_model), "vecnormalize.pkl"),
                os.path.join(args.logdir, "vecnormalize.pkl"),
            ]
            loaded_stats = False
            for sp in stats_candidates:
                if os.path.exists(sp):
                    try:
                        vec_env = VecNormalize.load(sp, base_env)
                        vec_env.training = True
                        vec_env.norm_reward = True
                        loaded_stats = True
                        print(f"[resume] Loaded VecNormalize stats from {sp}")
                        break
                    except Exception:
                        pass
            if not loaded_stats:
                vec_env = VecNormalize(base_env, training=True, norm_obs=True, norm_reward=True, clip_obs=10.0)
        else:
            vec_env = VecNormalize(base_env, training=True, norm_obs=True, norm_reward=True, clip_obs=10.0)

    os.makedirs(args.logdir, exist_ok=True)
    checkpoint_cb = CheckpointCallback(save_freq=max(1, args.checkpoint_every // n_envs), save_path=args.logdir, name_prefix="ppo_flappy")

    # Eval environment (single env)
    eval_env = DummyVecEnv([make_env(True, frame_skip, args.seed + 100000, bool(args.shaping), bool(args.curriculum))])
    if isinstance(vec_env, VecNormalize):
        eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)
        eval_env.obs_rms = vec_env.obs_rms
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.logdir,
        log_path=args.logdir,
        eval_freq=max(1, args.eval_freq // n_envs),
        n_eval_episodes=5,
        deterministic=True,
    )

    def linear_schedule(initial_value: float):
        def func(progress_remaining: float):
            return progress_remaining * initial_value
        return func

    # Parse network architecture
    def parse_net_arch(spec: str):
        spec = spec.strip()
        if 'pi:' in spec or 'vf:' in spec:
            # Separate policy/value specification
            pi_part = []
            vf_part = []
            for segment in spec.split('|'):
                segment = segment.strip()
                if segment.startswith('pi:'):
                    pi_part = [int(x) for x in segment[3:].split(',') if x]
                elif segment.startswith('vf:'):
                    vf_part = [int(x) for x in segment[3:].split(',') if x]
            if not pi_part:
                pi_part = [256, 256]
            if not vf_part:
                vf_part = pi_part
            return dict(pi=pi_part, vf=vf_part)
        layers = [int(x) for x in spec.split(',') if x]
        if not layers:
            layers = [256, 256]
        return dict(pi=layers, vf=layers)

    net_arch = parse_net_arch(args.net_arch)
    from torch import nn
    act = nn.Tanh if args.activation == 'tanh' else nn.ReLU

    if args.from_model:
        print(f"[resume] Loading model from {args.from_model}")
        model = PPO.load(args.from_model, env=vec_env)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            verbose=1,
            n_steps=2048 // n_envs,
            batch_size=128, 
            n_epochs=10,
            gamma=0.99,  
            gae_lambda=0.95,
            learning_rate=linear_schedule(5e-4), 
            clip_range=0.2,
            ent_coef=0.01,  
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=args.logdir,
            seed=args.seed,
            policy_kwargs=dict(
                net_arch=net_arch,
                activation_fn=act
            ),
        )

    model.learn(total_timesteps=args.timesteps, callback=[checkpoint_cb, eval_cb])
    final_path = os.path.join(args.logdir, "ppo_flappy_final")
    model.save(final_path)
    try:
        best_path = os.path.join(args.logdir, "best_model.zip")
        if not os.path.exists(best_path):
            model.save(best_path.replace('.zip', ''))
    except Exception:
        pass
    if isinstance(vec_env, VecNormalize):
        vec_env.save(os.path.join(args.logdir, "vecnormalize.pkl"))
    vec_env.close()


if __name__ == "__main__":
    main()
