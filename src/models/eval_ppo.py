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
import re
import os
import sys
import shutil
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.flappy_env import FlappyGymEnv


def list_checkpoints(folder: str) -> List[str]:
    # Recursively search for checkpoints in runs/ppo and subfolders
    best: List[str] = []
    finals: List[str] = []
    steps: List[str] = []
    step_pat = re.compile(r"^ppo_flappy_(\d+)_steps\.zip$")
    for root, _, files in os.walk(folder):
        for name in files:
            if name == "best_model.zip":
                best.append(os.path.join(root, name))
            elif name == "ppo_flappy_final.zip":
                finals.append(os.path.join(root, name))
            else:
                m = step_pat.match(name)
                if m:
                    steps.append(os.path.join(root, name))
    # Sort step checkpoints by numeric steps
    def steps_key(fp: str) -> int:
        m = step_pat.match(os.path.basename(fp))
        return int(m.group(1)) if m else -1
    steps.sort(key=steps_key)
    # Return with priority: best, final, then step checkpoints
    ordered = best + finals + steps
    # Deduplicate while preserving order
    seen = set()
    result: List[str] = []
    for fp in ordered:
        if fp not in seen:
            seen.add(fp)
            result.append(fp)
    return result


def pick_latest(folder: str) -> Optional[str]:
    files = list_checkpoints(folder)
    if not files:
        return None
    if files[0].endswith("ppo_flappy_final.zip"):
        return files[0]
    return files[-1]


def _maybe_wrap_norm(venv, stats_dir: str):
    # Try model directory first, then parent directory as fallback
    candidates = [
        os.path.join(stats_dir, "vecnormalize.pkl"),
        os.path.join(os.path.dirname(stats_dir), "vecnormalize.pkl"),
    ]
    for stats_path in candidates:
        if os.path.exists(stats_path):
            try:
                venv = VecNormalize.load(stats_path, venv)
                venv.training = False
                venv.norm_reward = False
                break
            except Exception:
                continue
    return venv


def quick_eval(model_path: str, episodes: int = 2, frame_skip: int = 2) -> float:
    # Build VecEnv for compatibility with VecNormalize
    base_env = DummyVecEnv([lambda: FlappyGymEnv(headless=True, frame_skip=frame_skip)])
    stats_dir = os.path.dirname(model_path)
    env = _maybe_wrap_norm(base_env, stats_dir)
    model = PPO.load(model_path, env=env)
    total = 0.0
    for ep in range(episodes):
        # Try to seed underlying envs for determinism
        try:
            env.env_method("reset", seed=1234 + ep)
        except Exception:
            pass
        obs = env.reset()
        ep_r = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step([int(action)])
            # r, term, trunc may be vectorized (size 1)
            r_scalar = float(np.array(r).reshape(-1)[0])
            done = bool(np.array(term).reshape(-1)[0] or np.array(trunc).reshape(-1)[0])
            ep_r += r_scalar
            if done:
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
            continue
    if best_fp is None:
        return None
    return best_fp, best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="", help="Path to .zip model (overrides auto selection)")
    parser.add_argument("--runs", type=str, default="runs/ppo", help="Folder to search for PPO checkpoints")
    parser.add_argument("--auto-best", action="store_true", help="Probe all checkpoints to pick the best")
    parser.add_argument("--export-best", type=str, default="", help="Copy the selected model to this path (adds .zip if missing)")
    parser.add_argument("--include-stats", action="store_true", help="When exporting, also copy vecnormalize.pkl next to the export if found")
    parser.add_argument("--export-only", action="store_true", help="Export the selected model and exit without rendering")
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

    if args.export_best:
        dest = args.export_best
        if os.path.isdir(dest):
            dest = os.path.join(dest, os.path.basename(model_path))
        if not dest.lower().endswith(".zip"):
            dest = dest + ".zip"
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        shutil.copy2(model_path, dest)
        print(f"Exported model to {dest}")
        if args.include_stats:
            stats_candidates = [
                os.path.join(os.path.dirname(model_path), "vecnormalize.pkl"),
                os.path.join(os.path.dirname(args.runs.rstrip(os.sep)), "vecnormalize.pkl"),
                os.path.join(args.runs, "vecnormalize.pkl"),
            ]
            for sp in stats_candidates:
                if os.path.exists(sp):
                    try:
                        shutil.copy2(sp, os.path.join(os.path.dirname(dest), "vecnormalize.pkl"))
                        print(f"Exported VecNormalize stats to {os.path.join(os.path.dirname(dest), 'vecnormalize.pkl')}")
                    except Exception:
                        pass
                    break
        if args.export_only:
            return

    base_env = DummyVecEnv([lambda: FlappyGymEnv(headless=False, frame_skip=max(1, args.frame_skip))])
    stats_dir = os.path.dirname(model_path)
    env = _maybe_wrap_norm(base_env, stats_dir)
    model = PPO.load(model_path, env=env)

    clock = pygame.time.Clock()
    target_fps = max(1, int(args.fps))

    for ep in range(args.episodes):
        obs = env.reset()
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
            step_out = env.step([int(action)])
            if len(step_out) == 4:
                obs, r, done_vec, infos = step_out
                term_vec = done_vec
                trunc_vec = np.zeros_like(done_vec)
            else:
                obs, r, term_vec, trunc_vec, infos = step_out
            r_scalar = float(np.array(r).reshape(-1)[0])
            done_flag = bool(np.array(term_vec).reshape(-1)[0]) or bool(np.array(trunc_vec).reshape(-1)[0])
            info0 = infos[0] if isinstance(infos, (list, tuple)) and len(infos) > 0 else {}
            ep_r += r_scalar
            steps += 1

            if not args.turbo:
                clock.tick(target_fps)

            if done_flag or not running:
                print(f"Episode {ep+1}/{args.episodes}: steps={steps}, reward={ep_r:.2f}, score={info0.get('score', 0)}")
                break

    env.close()


if __name__ == "__main__":
    main()
