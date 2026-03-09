"""Quick smoke test for the Gym env.

Runs a few episodes with random actions to validate the wrapper.
"""

import numpy as np

try:
    import gymnasium as gym
except Exception:
    import gym  # type: ignore

from src.envs.flappy_env import FlappyGymEnv


def main():
    env = FlappyGymEnv(headless=True, frame_skip=2, max_steps=500)
    episodes = 3
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        total_r = 0.0
        for t in range(500):
            a = env.action_space.sample()
            obs, r, term, trunc, info = env.step(a)
            total_r += r
            if term or trunc:
                break
        print(f"Episode {ep} finished in {t+1} steps, total reward {total_r:.2f}, score {info.get('score', 0)}")
    env.close()


if __name__ == "__main__":
    main()
