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

Start training (headless, vectorized, with optional curriculum and normalization):

```zsh
python ppo_train.py \
	--timesteps 200000 \
	--n-envs 4 \
	--frame-skip 2 \
	--seed 42 \
	--curriculum 1 \
	--normalize 1 \
	--eval-freq 50000
```

Launch TensorBoard:

```zsh
tensorboard --logdir runs
```

Notes:
- The trainer now supports VecNormalize (normalizes observations/rewards and saves stats to `runs/ppo/vecnormalize.pkl`).
- An EvalCallback evaluates every `--eval-freq` steps and saves the best model to `runs/ppo/best_model.zip`.
- The environment includes gentle reward shaping (alignment, progress, smoothness, distance bonus) and optional curriculum (gap size/speed ramp).
- Tune shaping weights and curriculum defaults in `flappy_env.py`.

Evaluate PPO (with optional auto-best and turbo):

```zsh
python eval_ppo.py --episodes 5 --frame-skip 2 --fps 60 --turbo

# Or probe all checkpoints and pick the best one
python eval_ppo.py --auto-best --episodes 5 --turbo
```
The evaluator will automatically load `vecnormalize.pkl` if present to ensure consistent observation scaling.

## Environment details

Observation (shape 8):
- player_y (0..1)
- velocity_y (~[-1,1] scaled to 0..1)
- dx to next gap (0..1)
- next gap y (0..1)
- |dy| to gap center (0..1)
- distance to ground (0..1)
- distance to ceiling (0..1)
- bias (1.0)

Action space: Discrete(2)
- 0: do nothing
- 1: jump

Rewards (with shaping):
- +0.1 every step (living reward)
- +10 when a pipe pair is passed (score increases)
- -5 on collision (episode ends)
- + alignment/progress/smoothness/distance bonuses while alive

Truncation: episodes also end after `max_steps`.

## Notes
- The wrapper uses a fixed time step (1/60s) and optional `frame_skip` to speed up training.
- Images are optional; the logic works without rendering.
- For reproducibility, `reset(seed=...)` seeds Python and NumPy RNGs.
