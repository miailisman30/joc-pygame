"""Evaluate a trained DQN agent and watch it play.

Usage:
  python eval_agent.py --model dqn_flappy.pt --episodes 5 --frame-skip 1 --fps 60

Close the window or press ESC to exit early.
"""

import argparse
import time
from typing import Optional

import numpy as np
import torch
import pygame

from flappy_env import FlappyGymEnv
from train_dqn import QNet 


def load_policy(model_path: str, obs_dim: int, act_dim: int, device: str = "cpu") -> QNet:
    net = QNet(obs_dim, act_dim).to(device)
    state = torch.load(model_path, map_location=device)
    net.load_state_dict(state)
    net.eval()
    return net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="dqn_flappy.pt", help="Path to trained weights")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--fps", type=int, default=60, help="Target render FPS")
    parser.add_argument("--turbo", action="store_true", help="No FPS cap; as fast as possible")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Exploration during eval (0=greedy)")
    args = parser.parse_args() 

    # Visible window for watching: headless=False
    env = FlappyGymEnv(headless=False, frame_skip=args.frame_skip, max_steps=10_000)
    obs, _ = env.reset()
    obs_dim = int(obs.shape[0])
    act_dim = int(env.action_space.n)
    device = "cpu"
    policy = load_policy(args.model, obs_dim, act_dim, device)

    clock = pygame.time.Clock()
    target_fps = max(1, int(args.fps))

    running = True
    for ep in range(args.episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        steps = 0
        while True:
            # Handle window events so the OS doesn't mark it unresponsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
            if not running:
                break

            if np.random.rand() < args.epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    qvals = policy(torch.from_numpy(obs).float().unsqueeze(0).to(device))
                    action = int(torch.argmax(qvals, dim=1).item())

            obs, r, term, trunc, info = env.step(action)
            ep_reward += r
            steps += 1

            # Cap the visual frame-rate unless turbo
            if not args.turbo:
                clock.tick(target_fps)

            if term or trunc:
                print(f"Episode {ep+1}/{args.episodes}: steps={steps}, reward={ep_reward:.2f}, score={info.get('score', 0)}")
                break

        if not running:
            break

    env.close()


if __name__ == "__main__":
    main()
