import math
from typing import Tuple, Dict, Any, Optional

import numpy as np
import random
import pygame

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # fallback to classic gym if gymnasium not installed
    import gym  # type: ignore
    from gym import spaces  # type: ignore

from flappy_saila import FlappySailaGame


class FlappyGymEnv(gym.Env):
    """
    Gym/Gymnasium-compatible environment that wraps FlappySailaGame.

    Observation (5,):
      - player_y (normalized 0..1)
      - player_velocity_y (scaled by 1000, clipped, then normalized to 0..1)
      - next_pipe_gap_x_distance (normalized by screen width)
      - next_pipe_gap_y (normalized 0..1)
      - bias (constant 1.0, helps linear layers)

    Actions: Discrete(2)
      0 -> do nothing
      1 -> jump

    Reward:
      +0.1 living reward per step
      +1.0 when passing a pipe (score increases)
      -1.0 on collision/game over and episode terminates

    Episode ends when game over or max_steps reached.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        headless: bool = True,
        frame_skip: int = 1,
        max_steps: int = 2000,
        shaping: bool = True,
        shaping_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.game = FlappySailaGame(width=800, height=600, fps=60, headless=headless)
        self.frame_skip = max(1, int(frame_skip))
        self.max_steps = int(max_steps)
        self._steps = 0
        self._last_score = 0
        self._prev_dx_norm: float = 0.0
        self._prev_v_norm: float = 0.0
        self._prev_gap_y_norm: float = 0.5
        self._prev_player_y_norm: float = 0.5

        # Reward shaping configuration
        self.shaping = bool(shaping)
        defaults = {
            "alive": 0.05,          # living reward per step
            "pipe_pass": 1.0,       # reward when score increases
            "crash": -1.0,          # penalty on crash
            "align": 0.5,           # encourages staying near gap center
            "progress": 0.2,        # encourages reducing dx to gap
            "smooth": 0.05,         # discourages sudden velocity changes
        }
        if shaping_weights:
            defaults.update(shaping_weights)
        self.w = defaults

        # Observation: 5 floats in [0,1]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        # Action: 0/1
        self.action_space = spaces.Discrete(2)

    # --- helpers ---
    def _observe(self) -> np.ndarray:
        # Create a state vector normalized to 0..1
        state = self.game.step_ai(0.0, False)  # dt=0 to just read state
        player_y = float(state["player_y"]) / float(self.game.height)
        # velocity range rough: [-1000, 1000]; clip, rescale to 0..1
        v = np.clip(float(state["player_velocity_y"]) / 1000.0, -1.0, 1.0)
        v = (v + 1.0) / 2.0
        dx = np.clip(float(state["next_pipe_gap_x_distance"]) / float(self.game.width), 0.0, 1.0)
        gap_y = float(state["next_pipe_gap_y"]) / float(self.game.height)
        bias = 1.0
        obs = np.array([player_y, v, dx, gap_y, bias], dtype=np.float32)
        return obs

    def _advance(self, action: int) -> None:
        jump = bool(action == 1)
        # use a fixed dt for stability; step frame_skip times
        dt = 1.0 / 60.0
        for _ in range(self.frame_skip):
            self.game.step_ai(dt, jump)

    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        # reset underlying game
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % (2**32 - 1))
        self.game.reset_game()
        self._steps = 0
        self._last_score = 0
        obs = self._observe()
        # initialize previous terms for shaping
        self._prev_player_y_norm = float(obs[0])
        self._prev_v_norm = float(obs[1])
        self._prev_dx_norm = float(obs[2])
        self._prev_gap_y_norm = float(obs[3])
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: int):
        self._advance(int(action))
        self._steps += 1

        # observe new state for reward shaping
        obs = self._observe()
        player_y_norm = float(obs[0])
        v_norm = float(obs[1])
        dx_norm = float(obs[2])
        gap_y_norm = float(obs[3])

        # base rewards
        score = self.game.current_score()
        reward = self.w["alive"]
        if score > self._last_score:
            reward += self.w["pipe_pass"]
            self._last_score = score

        # shaping terms
        if self.shaping:
            # alignment to gap center: higher when closer
            align = 1.0 - abs(player_y_norm - gap_y_norm)  # in [0,1]
            reward += self.w["align"] * align
            # progress: positive when dx decreases
            progress = self._prev_dx_norm - dx_norm
            reward += self.w["progress"] * progress
            # smoothness: penalize large velocity change
            smooth = -abs(v_norm - self._prev_v_norm)
            reward += self.w["smooth"] * smooth

        terminated = self.game.is_game_over()
        if terminated:
            reward += self.w["crash"]

        truncated = self._steps >= self.max_steps
        # update prev terms for next step
        self._prev_player_y_norm = player_y_norm
        self._prev_v_norm = v_norm
        self._prev_dx_norm = dx_norm
        self._prev_gap_y_norm = gap_y_norm
        info: Dict[str, Any] = {"score": score}
        return obs, reward, terminated, truncated, info

    def render(self):
        # If not headless, pygame window is already shown by GameEnvironment
        if self.game.headless:
            # return an RGB array of the current frame
            # Convert pygame Surface to numpy array (H, W, 3)
            surf = self.game.screen
            arr = np.transpose(np.array(pygame.surfarray.pixels3d(surf)), (1, 0, 2))  # type: ignore
            return arr.copy()
        return None

    def close(self):
        try:
            pygame.quit()
        except Exception:
            pass
