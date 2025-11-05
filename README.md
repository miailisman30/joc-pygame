# Flappy Saila RL

This repo contains a simple Flappy Bird–like game (`flappy_saila.py`) and a Gymnasium-compatible environment wrapper (`flappy_env.py`) so you can train reinforcement learning agents.

## Files
- `flappy_saila.py` — Game objects and loop atop a small engine (`game_engine.py`).
- `flappy_env.py` — Gym/Gymnasium environment (`FlappyGymEnv`) exposing `reset/step` for RL.
- `random_rollout.py` — Smoke test that runs random actions for a few episodes.
- `train_dqn.py` — Minimal DQN trainer in PyTorch.
- `eval_agent.py` — Load a trained model and watch it play in a window.
- `requirements.txt` — Python dependencies.

## Install (macOS, zsh)

Create a virtual environment and install deps:

```zsh
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# If torch failed due to platform-specific wheels on Apple Silicon, try:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Note: The game runs headlessly by default in RL. You don't need a visible window.

## Quick checks

- Play/run the original loop (headless):

```zsh
python main.py
```

- Random smoke test for the RL env:

```zsh
python random_rollout.py
```

- Train a DQN agent (basic):

```zsh
python train_dqn.py
```

Models are saved to `dqn_flappy.pt` periodically.

- Watch a trained agent play:

```zsh
python eval_agent.py --model dqn_flappy.pt --episodes 5 --frame-skip 2 --fps 60 --turbo
```

Tips:
- Press ESC or close the window to stop.
- Use `--turbo` to remove the FPS cap; combine with `--frame-skip` for very fast playback.
- Increase `--frame-skip` to speed up; reduce it for smoother motion.

## PPO (Stable-Baselines3)

Install extra dependencies:

```zsh
pip install stable-baselines3[extra] tensorboard
```

Start training:

```zsh
python ppo_train.py --timesteps 200000 --frame-skip 2 --n-envs 4 --seed 42
```

Launch TensorBoard:

```zsh
tensorboard --logdir runs
```

Notes:
- The environment includes gentle reward shaping (alignment, progress, smoothness) to ease learning.
- Tune shaping weights in `flappy_env.py` or pass via constructor if you embed it elsewhere.

## Environment details

Observation (shape 5):
- player_y (0..1)
- player_velocity_y (~[-1,1] scaled to 0..1)
- next_pipe_gap_x_distance (0..1)
- next_pipe_gap_y (0..1)
- bias (1.0)

Action space: Discrete(2)
- 0: do nothing
- 1: jump

Rewards:
- +0.1 every step (living reward)
- +1.0 when a pipe pair is passed (score increases)
- -1.0 on collision (episode ends)

Truncation: episodes also end after `max_steps`.

## Notes
- The wrapper uses a fixed time step (1/60s) and optional `frame_skip` to speed up training.
- Images are optional; the logic works without rendering.
- For reproducibility, `reset(seed=...)` seeds Python and NumPy RNGs.
