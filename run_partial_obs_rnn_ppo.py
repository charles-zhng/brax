#!/usr/bin/env python3
"""
Final Partial Observation Experiment with OPTIMAL hyperparameters from EXPERIMENT_LOG.md.

Uses the same configs that achieved ~800 reward on full obs PendulumSwingup:
- FF PPO: LR=1e-3, reward_scaling=10, action_repeat=4
- RNN PPO: LR=5e-4, reward_scaling=10, action_repeat=4 (most stable from LR sweep)
"""

import os
import time
import pickle
from datetime import datetime
from typing import Dict, List

import jax
import jax.numpy as jp
import numpy as np
import matplotlib.pyplot as plt

from mujoco_playground import registry
from mujoco_playground._src.wrapper import wrap_for_brax_training
from mujoco_playground.config import dm_control_suite_params

from brax.training.agents.ppo import train as ppo
from brax.training.agents.recurrent_ppo import train as rnn_ppo
from brax.training.agents.recurrent_ppo import networks as rnn_networks

from pendulum_partial_obs import PendulumSwingupPartialObs


ENV_NAME = "PendulumSwingup"
NUM_TIMESTEPS = 200_000_000  # 200M steps
NUM_EVALS = 40
SEED = 0  # Seed 0 known to work well for FF PPO


def run_ff_ppo(env, name: str, rl_config, output_dir: str) -> Dict:
    """Run FF PPO with optimal hyperparameters."""
    print("\n" + "=" * 70)
    print(f"Experiment: {name}")
    print(f"  Algorithm: FF PPO")
    print(f"  Observation size: {env.observation_size}")
    print(f"  LR: {rl_config.learning_rate}, reward_scaling: {rl_config.reward_scaling}")
    print(f"  action_repeat: {rl_config.action_repeat}")
    print("=" * 70)

    metrics_list = []
    start_time = time.time()

    def progress_fn(step, metrics):
        r = metrics.get("eval/episode_reward", 0)
        r_std = metrics.get("eval/episode_reward_std", 0)
        sps = metrics.get("training/sps", 0)
        elapsed = time.time() - start_time
        print(f"[{name}] Step {step:>10,}: reward = {r:>7.2f} +/- {r_std:>5.2f}, SPS = {sps:,.0f}, time = {elapsed:.1f}s")
        metrics_list.append({
            "step": step,
            "reward": float(r),
            "reward_std": float(r_std),
            "time": elapsed,
        })

    t0 = time.time()
    make_policy, policy_params, final_metrics = ppo.train(
        environment=env,
        num_timesteps=NUM_TIMESTEPS,
        wrap_env=True,
        wrap_env_fn=wrap_for_brax_training,
        episode_length=rl_config.episode_length,
        num_evals=NUM_EVALS,
        reward_scaling=rl_config.reward_scaling,
        normalize_observations=rl_config.normalize_observations,
        action_repeat=rl_config.action_repeat,
        unroll_length=rl_config.unroll_length,
        num_updates_per_batch=rl_config.num_updates_per_batch,
        discounting=rl_config.discounting,
        learning_rate=rl_config.learning_rate,
        entropy_cost=rl_config.entropy_cost,
        normalize_advantage=True,
        clipping_epsilon=0.2,
        num_envs=rl_config.num_envs,
        batch_size=rl_config.batch_size,
        num_minibatches=rl_config.num_minibatches,
        seed=SEED,
        progress_fn=progress_fn,
    )
    elapsed = time.time() - t0
    final_reward = final_metrics.get("eval/episode_reward", 0)
    peak_reward = max(m['reward'] for m in metrics_list) if metrics_list else 0

    print(f"\n{name} completed in {elapsed:.1f}s, Final: {final_reward:.2f}, Peak: {peak_reward:.2f}")

    return {
        "name": name,
        "algorithm": "ff",
        "metrics": metrics_list,
        "time": elapsed,
        "final_reward": final_reward,
        "peak_reward": peak_reward,
    }


def run_rnn_ppo(env, name: str, rl_config, output_dir: str, learning_rate: float = 5e-4) -> Dict:
    """Run RNN PPO with optimal hyperparameters from EXPERIMENT_LOG.md."""
    # Optimal RNN config from experiment log (LR=5e-4 most stable)
    rnn_config = {
        'num_envs': 1024,
        'batch_size': 512,
        'num_minibatches': 16,
        'unroll_length': 64,
        'num_updates_per_batch': 8,
        'learning_rate': learning_rate,
        'entropy_cost': 5e-3,
        'discounting': 0.99,
        'max_grad_norm': 0.5,
        'policy_rnn_hidden_size': 64,
        'cell_type': 'gru',
    }

    print("\n" + "=" * 70)
    print(f"Experiment: {name}")
    print(f"  Algorithm: RNN PPO (GRU)")
    print(f"  Observation size: {env.observation_size}")
    print(f"  LR: {rnn_config['learning_rate']}, reward_scaling: {rl_config.reward_scaling}")
    print(f"  action_repeat: {rl_config.action_repeat}, unroll: {rnn_config['unroll_length']}")
    print("=" * 70)

    metrics_list = []
    start_time = time.time()

    def progress_fn(step, metrics):
        r = metrics.get("eval/episode_reward", 0)
        r_std = metrics.get("eval/episode_reward_std", 0)
        sps = metrics.get("training/sps", 0)
        elapsed = time.time() - start_time
        print(f"[{name}] Step {step:>10,}: reward = {r:>7.2f} +/- {r_std:>5.2f}, SPS = {sps:,.0f}, time = {elapsed:.1f}s")
        metrics_list.append({
            "step": step,
            "reward": float(r),
            "reward_std": float(r_std),
            "time": elapsed,
        })

    def network_factory(obs_size, action_size, **kw):
        return rnn_networks.make_rnn_ppo_networks(
            obs_size,
            action_size,
            cell_type=rnn_config['cell_type'],
            policy_rnn_hidden_size=rnn_config['policy_rnn_hidden_size'],
            policy_output_layer_sizes=(64,),
            **kw,
        )

    t0 = time.time()
    make_policy, policy_params, final_metrics = rnn_ppo.train(
        environment=env,
        num_timesteps=NUM_TIMESTEPS,
        wrap_env=True,
        wrap_env_fn=wrap_for_brax_training,
        episode_length=rl_config.episode_length,
        num_evals=NUM_EVALS,
        reward_scaling=rl_config.reward_scaling,
        normalize_observations=rl_config.normalize_observations,
        action_repeat=rl_config.action_repeat,
        unroll_length=rnn_config['unroll_length'],
        num_updates_per_batch=rnn_config['num_updates_per_batch'],
        discounting=rnn_config['discounting'],
        learning_rate=rnn_config['learning_rate'],
        entropy_cost=rnn_config['entropy_cost'],
        max_grad_norm=rnn_config['max_grad_norm'],
        normalize_advantage=True,
        clipping_epsilon=0.2,
        num_envs=rnn_config['num_envs'],
        batch_size=rnn_config['batch_size'],
        num_minibatches=rnn_config['num_minibatches'],
        network_factory=network_factory,
        seed=SEED,
        progress_fn=progress_fn,
    )
    elapsed = time.time() - t0
    final_reward = final_metrics.get("eval/episode_reward", 0)
    peak_reward = max(m['reward'] for m in metrics_list) if metrics_list else 0

    print(f"\n{name} completed in {elapsed:.1f}s, Final: {final_reward:.2f}, Peak: {peak_reward:.2f}")

    return {
        "name": name,
        "algorithm": "rnn",
        "metrics": metrics_list,
        "time": elapsed,
        "final_reward": final_reward,
        "peak_reward": peak_reward,
        "learning_rate": learning_rate,
    }


def plot_results(results: List[Dict], output_dir: str):
    """Generate comparison plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        'FF_full_obs': '#2ecc71',
        'FF_partial_obs': '#e74c3c',
        'RNN_partial_obs': '#3498db',
    }

    labels = {
        'FF_full_obs': 'FF PPO (Full Obs) - Control',
        'FF_partial_obs': 'FF PPO (Partial Obs)',
        'RNN_partial_obs': 'RNN PPO (Partial Obs)',
    }

    # Left: Training curves
    ax = axes[0]
    for result in results:
        name = result['name']
        steps = [m['step'] for m in result['metrics']]
        rewards = [m['reward'] for m in result['metrics']]
        stds = [m['reward_std'] for m in result['metrics']]
        ax.plot(steps, rewards, color=colors[name], lw=2, label=labels[name])
        ax.fill_between(steps, np.array(rewards) - np.array(stds),
                       np.array(rewards) + np.array(stds), alpha=0.2, color=colors[name])

    ax.set_xlabel("Environment Steps", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title("Partial Observation Experiment (Optimal HPs)", fontsize=13)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=800, color='gray', linestyle='--', alpha=0.5)

    # Right: Bar chart
    ax = axes[1]
    names = [r['name'] for r in results]
    final_rewards = [r['final_reward'] for r in results]
    peak_rewards = [r['peak_reward'] for r in results]
    bar_colors = [colors[n] for n in names]

    x = np.arange(len(names))
    width = 0.35
    bars1 = ax.bar(x - width/2, final_rewards, width, label='Final', color=bar_colors, alpha=0.7)
    bars2 = ax.bar(x + width/2, peak_rewards, width, label='Peak', color=bar_colors, edgecolor='black')

    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title("Final vs Peak Performance", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(['FF\nFull', 'FF\nPartial', 'RNN\nPartial'], fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=800, color='gray', linestyle='--', alpha=0.5)

    for bar, val in zip(bars2, peak_rewards):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{val:.0f}', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'partial_obs_final.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'partial_obs_final.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved plots to {output_dir}/partial_obs_final.png")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"partial_obs_final_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("PARTIAL OBSERVATION EXPERIMENT (OPTIMAL HYPERPARAMETERS)")
    print("=" * 70)
    print(f"Using configs from EXPERIMENT_LOG.md that achieved ~800 reward")
    print(f"Seed: {SEED}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Load environments
    full_obs_env = registry.load(ENV_NAME)
    partial_obs_env = PendulumSwingupPartialObs()
    rl_config = dm_control_suite_params.brax_ppo_config(ENV_NAME)

    print(f"\nConfig from dm_control_suite_params:")
    print(f"  reward_scaling: {rl_config.reward_scaling}")
    print(f"  action_repeat: {rl_config.action_repeat}")
    print(f"  learning_rate: {rl_config.learning_rate}")
    print(f"  episode_length: {rl_config.episode_length}")

    print(f"\nEnvironments:")
    print(f"  Full obs size: {full_obs_env.observation_size}")
    print(f"  Partial obs size: {partial_obs_env.observation_size}")

    results = []

    # 1. FF PPO with Full Obs (control)
    result = run_ff_ppo(full_obs_env, 'FF_full_obs', rl_config, output_dir)
    results.append(result)

    # 2. FF PPO with Partial Obs (expected to fail)
    result = run_ff_ppo(partial_obs_env, 'FF_partial_obs', rl_config, output_dir)
    results.append(result)

    # 3. RNN PPO with Partial Obs (expected to succeed)
    result = run_rnn_ppo(partial_obs_env, 'RNN_partial_obs', rl_config, output_dir, learning_rate=5e-4)
    results.append(result)

    # Save results
    results_to_save = [{k: v for k, v in r.items()} for r in results]
    with open(os.path.join(output_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results_to_save, f)

    # Plot
    plot_results(results, output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Condition':<20} {'Final':>10} {'Peak':>10} {'Time':>10}")
    print("-" * 55)
    for r in results:
        print(f"{r['name']:<20} {r['final_reward']:>10.2f} {r['peak_reward']:>10.2f} {r['time']:>9.1f}s")
    print("=" * 70)

    ff_full = next(r for r in results if r['name'] == 'FF_full_obs')
    ff_partial = next(r for r in results if r['name'] == 'FF_partial_obs')
    rnn_partial = next(r for r in results if r['name'] == 'RNN_partial_obs')

    print("\nKey Findings:")
    print(f"  FF full obs peak: {ff_full['peak_reward']:.0f}")
    print(f"  FF partial obs peak: {ff_partial['peak_reward']:.0f} (expected ~0)")
    print(f"  RNN partial obs peak: {rnn_partial['peak_reward']:.0f} (should match FF full)")
    print(f"  RNN advantage: {rnn_partial['peak_reward'] - ff_partial['peak_reward']:+.0f}")

    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
