"""Unified experiment harness for training algorithms on simple environments.

Usage:
    python -m experiments.run --config experiments/configs/rnn_ppo_pendulum.yaml
    python -m experiments.run --config ... --set train.learning_rate=3e-4 --video

Modules:
  wrappers  - MJX wrappers for mujoco_playground envs (vmap/episode/auto-reset)
  envs      - environment registry ('pendulum', 'pendulum_partial', 'cartpole', 'fast')
  algos     - algorithm registry ('ff_ppo', 'rnn_ppo', 'rnn_ppo_feature', 'rnn_sac')
  run       - config-driven CLI entry point
"""
