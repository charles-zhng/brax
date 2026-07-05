"""Config-driven training CLI for the experiment harness.

Usage:
    python -m experiments.run --config experiments/configs/rnn_ppo_pendulum.yaml
    python -m experiments.run --config <cfg> --set train.learning_rate=3e-4 \
        --set num_timesteps=1000000 --video

Config schema (YAML):
    algo: rnn_ppo                # one of experiments.algos.algo_names()
    env: pendulum                # one of experiments.envs.env_names()
    num_timesteps: 10000000
    num_envs: 512
    num_evals: 20
    episode_length: 1000
    action_repeat: 1             # optional, default 1
    seed: 42
    network:                     # kwargs for the algo's network factory
      cell_type: simple
      policy_rnn_hidden_size: 64
    train:                       # extra kwargs passed straight to train()
      learning_rate: 1.0e-4
      ...

Outputs (under --output-root, default ./outputs):
    <algo>_<env>_<timestamp>/
        config.json      run config + git commit + argv
        metrics.csv      one row per progress callback
        training_plots.png
        params.pkl       final params
        rollout.mp4      (with --video)
"""

import argparse
import csv
import json
import os
import pickle
import re
import subprocess
import time
from datetime import datetime

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import jax
import yaml

from experiments import algos as algos_lib
from experiments import envs as envs_lib


# ============================================================================
# Config handling
# ============================================================================

_SCI_NOTATION = re.compile(r'^[+-]?\d+(\.\d*)?[eE][+-]?\d+$')


def _parse_override_value(raw_value):
    value = yaml.safe_load(raw_value)
    # YAML 1.1 parses dotless scientific notation ('3e-4') as a string;
    # treat it as the float the user obviously meant.
    if isinstance(value, str) and _SCI_NOTATION.match(value):
        return float(value)
    return value


def load_config(path, overrides):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, _, raw_value = item.partition('=')
        if not _:
            raise ValueError(f'--set expects key=value, got {item!r}')
        node = cfg
        parts = key.split('.')
        for part in parts[:-1]:
            # A bare `section:` line parses as None — treat it as empty.
            if node.get(part) is None:
                node[part] = {}
            node = node[part]
        node[parts[-1]] = _parse_override_value(raw_value)
    for required in ('algo', 'env', 'num_timesteps'):
        if required not in cfg:
            raise ValueError(f'Config is missing required key {required!r}')
    return cfg


def git_commit():
    try:
        return subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, check=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'


# ============================================================================
# Metrics collection
# ============================================================================

class MetricsCollector:
    """Collects scalar training metrics; writes CSV and summary plots."""

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.rows = []

    def update(self, step, metrics):
        row = {'step': int(step)}
        for k, v in metrics.items():
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                continue
        self.rows.append(row)
        self._write_csv()

    def _write_csv(self):
        fields = ['step'] + sorted({k for r in self.rows for k in r} - {'step'})
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)

    def plot(self, save_path):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        steps = [r['step'] for r in self.rows]
        rewards = [r.get('eval/episode_reward', 0) for r in self.rows]
        stds = [r.get('eval/episode_reward_std', 0) for r in self.rows]
        sps = [r.get('training/sps', 0) for r in self.rows]
        loss_keys = sorted(
            {k for r in self.rows for k in r if 'loss' in k.lower()}
        )[:2]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0, 0]
        ax.plot(steps, rewards, 'b-', linewidth=2, label='Mean Reward')
        ax.fill_between(
            steps,
            np.array(rewards) - np.array(stds),
            np.array(rewards) + np.array(stds),
            alpha=0.3, color='blue',
        )
        ax.set_xlabel('Environment Steps')
        ax.set_ylabel('Episode Reward')
        ax.set_title('Episode Reward')
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[0, 1]
        ax.plot(steps, sps, 'm-', linewidth=2)
        ax.set_xlabel('Environment Steps')
        ax.set_ylabel('Steps Per Second')
        ax.set_title('Training Speed')
        ax.grid(True, alpha=0.3)

        for ax, key in zip((axes[1, 0], axes[1, 1]), loss_keys):
            ax.plot(steps, [r.get(key, 0) for r in self.rows], linewidth=2)
            ax.set_xlabel('Environment Steps')
            ax.set_ylabel(key)
            ax.set_title(key)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved training plots to: {save_path}')


# ============================================================================
# Rollout + video
# ============================================================================

def rollout_policy(reset_fn, step_fn, act_fn, initial_carry, num_steps, seed):
    """Single-env deterministic rollout; returns (trajectory, total_reward)."""
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    state = reset_fn(reset_key)
    trajectory = [state]
    carry = initial_carry
    total_reward = 0.0
    for _ in range(num_steps):
        key, action_key = jax.random.split(key)
        action, carry = act_fn(state.obs, carry, action_key)
        state = step_fn(state, action)
        trajectory.append(state)
        total_reward += float(state.reward)
        if state.done:
            break
    return trajectory, total_reward


def render_video(env, trajectory, save_path, fps=50, width=640, height=480):
    import imageio

    print(f'Rendering {len(trajectory)} frames...')
    frames = env.render(trajectory, height=height, width=width)
    imageio.mimsave(save_path, frames, fps=fps)
    print(f'Saved video to: {save_path}')


def generate_video(env, algo, network, make_policy, params, output_dir,
                   num_steps, seeds=(0, 42, 100, 200, 300)):
    # Only mujoco_playground envs are renderable here; brax-registry envs
    # (e.g. 'fast') define render() but lack the pipeline state it needs.
    if not hasattr(env, 'mj_model'):
        print(f'Env {type(env).__name__} is not renderable; skipping video.')
        return
    act_fn, initial_carry = algos_lib.make_act_fn(algo, network, make_policy, params)
    reset_fn, step_fn = jax.jit(env.reset), jax.jit(env.step)
    best_trajectory, best_reward = None, -float('inf')
    for seed in seeds:
        trajectory, reward = rollout_policy(
            reset_fn, step_fn, act_fn, initial_carry, num_steps, seed)
        print(f'Rollout seed {seed}: reward={reward:.2f} ({len(trajectory)} steps)')
        if reward > best_reward:
            best_reward, best_trajectory = reward, trajectory
    if best_trajectory is None:
        print('No rollout produced a finite reward; skipping video.')
        return
    print(f'Best rollout reward: {best_reward:.2f}')
    render_video(env, best_trajectory, os.path.join(output_dir, 'rollout.mp4'))


# ============================================================================
# Main
# ============================================================================

def run(cfg, output_root, video, video_steps, wandb_project):
    algo_name, env_name = cfg['algo'], cfg['env']
    algo = algos_lib.get_algo(algo_name)
    env, wrap_env_fn = envs_lib.make_env(env_name, cfg.get('env_config'))

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_root, f'{algo_name}_{env_name}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    run_info = {
        'config': cfg,
        'git_commit': git_commit(),
        'jax_devices': [str(d) for d in jax.devices()],
        'timestamp': timestamp,
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(run_info, f, indent=2, default=str)

    print('=' * 60)
    print(f'{algo_name} on {env_name}')
    print(f'Timesteps: {cfg["num_timesteps"]:,}  Output: {output_dir}')
    print(f'Commit: {run_info["git_commit"][:12]}  Devices: {run_info["jax_devices"]}')
    print('=' * 60)

    wandb_run = None
    if wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=wandb_project,
            name=f'{algo_name}_{env_name}_{timestamp}',
            config=cfg,
        )

    collector = MetricsCollector(os.path.join(output_dir, 'metrics.csv'))

    def progress_fn(step, metrics):
        collector.update(step, metrics)
        if wandb_run is not None:
            wandb_run.log({'step': step, **metrics}, step=step)
        reward = metrics.get('eval/episode_reward', 0)
        reward_std = metrics.get('eval/episode_reward_std', 0)
        sps = metrics.get('training/sps', 0)
        print(f'Step {step:>10,}: reward = {reward:>8.2f} +/- {reward_std:.2f}, '
              f'SPS = {sps:,.0f}')

    network_factory = algos_lib.make_network_factory(algo, cfg.get('network') or {})

    train_kwargs = dict(
        environment=env,
        num_timesteps=cfg['num_timesteps'],
        num_envs=cfg.get('num_envs', 512),
        num_evals=cfg.get('num_evals', 20),
        episode_length=cfg.get('episode_length', 1000),
        action_repeat=cfg.get('action_repeat', 1),
        seed=cfg.get('seed', 42),
        wrap_env=True,
        network_factory=network_factory,
        progress_fn=progress_fn,
        **(cfg.get('train') or {}),
    )
    if wrap_env_fn is not None:
        train_kwargs['wrap_env_fn'] = wrap_env_fn

    start = time.time()
    make_policy, params, final_metrics = algo.train_fn(**train_kwargs)
    elapsed = time.time() - start

    final_reward = float(final_metrics.get('eval/episode_reward', 0))
    print(f'\nTraining completed in {elapsed:.1f}s; final reward: {final_reward:.2f}')

    collector.plot(os.path.join(output_dir, 'training_plots.png'))
    with open(os.path.join(output_dir, 'params.pkl'), 'wb') as f:
        pickle.dump(params, f)

    if video:
        network = network_factory(env.observation_size, env.action_size)
        generate_video(env, algo, network, make_policy, params, output_dir,
                       num_steps=video_steps)

    if wandb_run is not None:
        wandb_run.summary['final_reward'] = final_reward
        wandb_run.finish()

    print(f'Done. Outputs in {output_dir}')
    return final_reward, output_dir


def main():
    parser = argparse.ArgumentParser(
        description='Train an algorithm on a simple environment.')
    parser.add_argument('--config', required=True, help='Path to YAML config.')
    parser.add_argument(
        '--set', action='append', metavar='KEY=VALUE', dest='overrides',
        help='Override a config value (dotted keys, YAML-parsed values), '
             'e.g. --set train.learning_rate=3e-4. Repeatable.')
    parser.add_argument('--output-root', default='outputs',
                        help='Directory that run output folders go under.')
    parser.add_argument('--video', action='store_true',
                        help='Render a rollout video after training.')
    parser.add_argument('--video-steps', type=int, default=500)
    parser.add_argument('--wandb-project', default=None,
                        help='If set, also log metrics to this wandb project.')
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    run(cfg, args.output_root, args.video, args.video_steps, args.wandb_project)


if __name__ == '__main__':
    main()
