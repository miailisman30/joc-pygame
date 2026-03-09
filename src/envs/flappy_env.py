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
        # Curriculum options
        curriculum: bool = False,
        curriculum_episodes: int = 300,
        start_gap: int = 240,
        end_gap: int = 140,
        start_speed: int = 120,
        end_speed: int = 200,
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
        # Curriculum state
        self.curriculum = bool(curriculum)
        self.curriculum_episodes = int(curriculum_episodes)
        self.curr_episode = 0
        self.start_gap = int(start_gap)
        self.end_gap = int(end_gap)
        self.start_speed = int(start_speed)
        self.end_speed = int(end_speed)

        self.shaping = bool(shaping)
        defaults = {
            "alive": 0.1,           
            "pipe_pass": 10.0,      
            "crash": -5.0,          
            "align": 0.8,           
            "progress": 0.5,        
            "smooth": 0.02,         
            "distance": 0.3,        
        }
        if shaping_weights:
            defaults.update(shaping_weights)
        self.w = defaults

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def _observe(self) -> np.ndarray:
        state = self.game.step_ai(0.0, False)  # dt=0 to just read state
        player_y = float(state["player_y"]) / float(self.game.height)
        
        v = np.clip(float(state["player_velocity_y"]) / 1000.0, -1.0, 1.0)
        v = (v + 1.0) / 2.0
        
        dx = np.clip(float(state["next_pipe_gap_x_distance"]) / float(self.game.width), 0.0, 1.0)
        gap_y = float(state["next_pipe_gap_y"]) / float(self.game.height)
        

        player_to_gap_dy = abs(player_y - gap_y) 
        
        dist_to_ground = 1.0 - player_y
        dist_to_ceiling = player_y
        
        bias = 1.0
        
        obs = np.array([
            player_y, 
            v, 
            dx, 
            gap_y, 
            player_to_gap_dy,
            dist_to_ground,
            dist_to_ceiling,
            bias
        ], dtype=np.float32)
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
        # Apply curriculum after reset for next episode
        if self.curriculum:
            self.curr_episode += 1
            frac = min(1.0, self.curr_episode / max(1, self.curriculum_episodes))
            gap = int(self.start_gap + frac * (self.end_gap - self.start_gap))
            speed = float(self.start_speed + frac * (self.end_speed - self.start_speed))
            # Set on pipes manager
            try:
                pm = self.game.player.pipes_manager
                pm.gap_height = gap
                pm.pipe_speed = speed
                # modestly adjust spawn interval as it speeds up
                pm.spawn_interval = max(1.5, 2.2 - 0.3 * frac)
            except Exception:
                pass
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

        obs = self._observe()
        player_y_norm = float(obs[0])
        v_norm = float(obs[1])
        dx_norm = float(obs[2])
        gap_y_norm = float(obs[3])

        score = self.game.current_score()
        reward = self.w["alive"]
        if score > self._last_score:
            reward += self.w["pipe_pass"]
            self._last_score = score

        if self.shaping:
            align = 1.0 - abs(player_y_norm - gap_y_norm)
            reward += self.w["align"] * align
            
            progress = self._prev_dx_norm - dx_norm
            progress = self._prev_dx_norm - dx_norm
            reward += self.w["progress"] * progress
            
            smooth = -abs(v_norm - self._prev_v_norm)
            reward += self.w["smooth"] * smooth
            
            distance_bonus = 1.0 - dx_norm  # closer = higher reward
            reward += self.w["distance"] * distance_bonus * 0.1

        terminated = self.game.is_game_over()
        if terminated:
            reward += self.w["crash"]

        truncated = self._steps >= self.max_steps
        self._prev_player_y_norm = player_y_norm
        self._prev_v_norm = v_norm
        self._prev_dx_norm = dx_norm
        self._prev_gap_y_norm = gap_y_norm
        info: Dict[str, Any] = {"score": score}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.game.headless:
            surf = self.game.screen
            arr = np.transpose(np.array(pygame.surfarray.pixels3d(surf)), (1, 0, 2))  # type: ignore
            return arr.copy()
        return None

    def close(self):
        try:
            pygame.quit()
        except Exception:
            pass
