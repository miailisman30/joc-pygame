"""Evaluate a trained PPO agent (Stable-Baselines3) and watch it play.

Usage examples:
  # Use final model if present, otherwise the latest checkpoint
  python eval_ppo.py --episodes 5 --frame-skip 1 --fps 60 --turbo

  # Explicitly pick a model path
  python eval_ppo.py --model runs/ppo/ppo_flappy_final.zip --episodes 3 --frame-skip 2 --turbo

  # Search all checkpoints in runs/ppo, do a quick headless eval to pick the best, then render it
  python eval_ppo.py --auto-best --episodes 5 --turbo
"""

import argparse
import glob
import os
from typing import List, Optional, Tuple

import numpy as np
import pygame
from stable_baselines3 import PPO

from flappy_env import FlappyGymEnv


def list_checkpoints(folder: str) -> List[str]:
    # Look for final and step checkpoints
    pats = [
        os.path.join(folder, "ppo_flappy_final.zip"),
        os.path.join(folder, "ppo_flappy_*_steps.zip"),
    ]
    files: List[str] = []
    for p in pats:
        files.extend(glob.glob(p))
    # Unique and sort by steps if possible, keeping final first
    files = list(dict.fromkeys(files))
    final_first = []
    rest = []
    for f in files:
        if f.endswith("ppo_flappy_final.zip"):
            final_first.append(f)
        else:
            rest.append(f)
    # Sort rest by numeric steps ascending
    def steps_key(fp: str) -> int:
        try:
            base = os.path.basename(fp)
            num = base.split("ppo_flappy_")[1].split("_steps.zip")[0]
            return int(num)
        except Exception:
            return -1
    rest.sort(key=steps_key)
    return final_first + rest


def pick_latest(folder: str) -> Optional[str]:
    files = list_checkpoints(folder)
    if not files:
        return None
    # Prefer final, otherwise highest steps (last in sorted list)
    if files[0].endswith("ppo_flappy_final.zip"):
        return files[0]
    return files[-1]


def quick_eval(model_path: str, episodes: int = 2, frame_skip: int = 2) -> float:
    # Headless quick evaluation to estimate mean reward
    env = FlappyGymEnv(headless=True, frame_skip=frame_skip)
    model = PPO.load(model_path)
    total = 0.0
    for ep in range(episodes):
        obs, _ = env.reset(seed=1234 + ep)
        ep_r = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(int(action))
            ep_r += r
            if term or trunc:
                break
        total += ep_r
    env.close()
    return total / max(1, episodes)


def pick_best(folder: str, probe_episodes: int = 2) -> Optional[Tuple[str, float]]:
    files = list_checkpoints(folder)
    if not files:
        return None
    best_fp = None
    best_score = -1e9
    for fp in files:
        try:
            score = quick_eval(fp, episodes=probe_episodes, frame_skip=2)
            if score > best_score:
                best_fp, best_score = fp, score
        except Exception as e:
            # Skip problematic files
            continue
    if best_fp is None:
        return None
    return best_fp, best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="", help="Path to .zip model (overrides auto selection)")
    parser.add_argument("--runs", type=str, default="runs/ppo", help="Folder to search for PPO checkpoints")
    parser.add_argument("--auto-best", action="store_true", help="Probe all checkpoints to pick the best")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--turbo", action="store_true")
    args = parser.parse_args()

    model_path: Optional[str] = None
    if args.model:
        model_path = args.model
    elif args.auto_best:
        picked = pick_best(args.runs, probe_episodes=2)
        if picked is None:
            print("No PPO checkpoints found in", args.runs)
            return
        model_path, est = picked
        print(f"Auto-selected best: {model_path} (est. mean reward {est:.2f})")
    else:
        model_path = pick_latest(args.runs)
        if model_path is None:
            print("No PPO checkpoints found in", args.runs)
            return
        print(f"Auto-selected latest: {model_path}")

    # Visible window to watch
    env = FlappyGymEnv(headless=False, frame_skip=max(1, args.frame_skip))
    model = PPO.load(model_path)

    clock = pygame.time.Clock()
    target_fps = max(1, int(args.fps))

    for ep in range(args.episodes):
        obs, _ = env.reset()
        ep_r = 0.0
        steps = 0
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(action))
            ep_r += r
            steps += 1

            if not args.turbo:
                clock.tick(target_fps)

            if term or trunc or not running:
                print(f"Episode {ep+1}/{args.episodes}: steps={steps}, reward={ep_r:.2f}, score={info.get('score', 0)}")
                break

    env.close()


if __name__ == "__main__":
    main()
