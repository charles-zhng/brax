#!/usr/bin/env python3
"""
Partial Observation Experiment with Recurrent TD3.

Tests recurrent TD3 on the partial observation pendulum task where the
angular velocity is hidden, requiring memory to solve optimally.

Compares:
- FF TD3 with full observations (control)
- FF TD3 with partial observations (expected to struggle)
- RNN TD3 with partial observations (expected to succeed)
"""

import os
import time
import pickle
from datetime import datetime
from typing import Dict, List

import numpy as np

from mujoco_playground import registry
from mujoco_playground._src.wrapper import wrap_for_brax_training
from mujoco_playground.config import dm_control_suite_params

from brax.training.agents.sac import train as sac
from brax.training.agents.recurrent_td3 import train as rnn_td3
from brax.training.agents.recurrent_td3 import networks as rnn_td3_networks

from pendulum_partial_obs import PendulumSwingupPartialObs


ENV_NAME = "PendulumSwingup"
NUM_TIMESTEPS = 20_000_000  # 20M steps
NUM_EVALS = 40
SEED = 0


def run_sac(env, name: str, rl_config) -> Dict:
    """Run SAC (feedforward) as baseline off-policy algorithm."""
    print("\n" + "=" * 70)
    print(f"Experiment: {name}")
    print(f"  Algorithm: SAC (Feedforward)")
    print(f"  Observation size: {env.observation_size}")
    print(f"  reward_scaling: {rl_config.reward_scaling}")
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
    _, _, final_metrics = sac.train(
        environment=env,
        num_timesteps=NUM_TIMESTEPS,
        wrap_env=True,
        wrap_env_fn=wrap_for_brax_training,
        episode_length=rl_config.episode_length,
        num_evals=NUM_EVALS,
        reward_scaling=rl_config.reward_scaling,
        normalize_observations=rl_config.normalize_observations,
        action_repeat=rl_config.action_repeat,
        discounting=0.99,
        learning_rate=3e-4,
        num_envs=512,
        batch_size=256,
        grad_updates_per_step=64,
        min_replay_size=10000,
        max_replay_size=1000000,
        seed=SEED,
        progress_fn=progress_fn,
    )
    elapsed = time.time() - t0
    final_reward = final_metrics.get("eval/episode_reward", 0)
    peak_reward = max(m['reward'] for m in metrics_list) if metrics_list else 0

    print(f"\n{name} completed in {elapsed:.1f}s, Final: {final_reward:.2f}, Peak: {peak_reward:.2f}")

    return {
        "name": name,
        "algorithm": "sac",
        "metrics": metrics_list,
        "time": elapsed,
        "final_reward": final_reward,
        "peak_reward": peak_reward,
    }


def run_rnn_td3(
    env,
    name: str,
    rl_config,
    cell_type: str = 'simple',
    use_privileged_critic: bool = False,
    exploration_noise: float = 0.1,
) -> Dict:
    """Run Recurrent TD3 with configurable RNN cell type."""
    # TD3 hyperparameters
    td3_config = {
        'num_envs': 512,
        'batch_size': 256,
        'learning_rate': 3e-4,
        'discounting': 0.99,
        'exploration_noise': exploration_noise,
        'target_noise': 0.2,
        'noise_clip': 0.5,
        'policy_delay': 2,
        'tau': 0.005,
        'grad_updates_per_step': 64,
        'min_replay_size': 10000,
        'max_replay_size': 1000000,
        'max_grad_norm': 1.0,
        'actor_rnn_hidden_size': 64,
        'actor_output_layer_sizes': (64,),
        'q_hidden_layer_sizes': (256, 256),
    }

    critic_info = " (privileged critic)" if use_privileged_critic else ""
    print("\n" + "=" * 70)
    print(f"Experiment: {name}")
    print(f"  Algorithm: Recurrent TD3 ({cell_type.upper()}){critic_info}")
    print(f"  Observation size: {env.observation_size}")
    print(f"  reward_scaling: {rl_config.reward_scaling}")
    print(f"  action_repeat: {rl_config.action_repeat}")
    print(f"  exploration_noise: {td3_config['exploration_noise']}")
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

    # Configure asymmetric actor-critic if using privileged information
    policy_obs_key = 'state'
    q_obs_key = 'privileged_state' if use_privileged_critic else 'state'

    def network_factory(observation_size, action_size, **kw):
        return rnn_td3_networks.make_recurrent_td3_networks(
            observation_size,
            action_size,
            cell_type=cell_type,
            actor_rnn_hidden_size=td3_config['actor_rnn_hidden_size'],
            actor_output_layer_sizes=td3_config['actor_output_layer_sizes'],
            q_hidden_layer_sizes=td3_config['q_hidden_layer_sizes'],
            policy_obs_key=policy_obs_key,
            q_obs_key=q_obs_key,
            **kw,
        )

    t0 = time.time()
    _, _, final_metrics = rnn_td3.train(
        environment=env,
        num_timesteps=NUM_TIMESTEPS,
        wrap_env=True,
        wrap_env_fn=wrap_for_brax_training,
        episode_length=rl_config.episode_length,
        num_evals=NUM_EVALS,
        reward_scaling=rl_config.reward_scaling,
        normalize_observations=rl_config.normalize_observations,
        action_repeat=rl_config.action_repeat,
        num_envs=td3_config['num_envs'],
        batch_size=td3_config['batch_size'],
        learning_rate=td3_config['learning_rate'],
        discounting=td3_config['discounting'],
        exploration_noise=td3_config['exploration_noise'],
        target_noise=td3_config['target_noise'],
        noise_clip=td3_config['noise_clip'],
        policy_delay=td3_config['policy_delay'],
        tau=td3_config['tau'],
        grad_updates_per_step=td3_config['grad_updates_per_step'],
        min_replay_size=td3_config['min_replay_size'],
        max_replay_size=td3_config['max_replay_size'],
        max_grad_norm=td3_config['max_grad_norm'],
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
        "algorithm": f"rnn_td3_{cell_type}",
        "metrics": metrics_list,
        "time": elapsed,
        "final_reward": final_reward,
        "peak_reward": peak_reward,
        "cell_type": cell_type,
        "use_privileged_critic": use_privileged_critic,
        "exploration_noise": exploration_noise,
    }


def plot_results(results: List[Dict], output_dir: str):
    """Generate comparison plots."""
    import matplotlib.pyplot as plt

    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        'RNN_TD3_noise_0.05': '#2ecc71',
        'RNN_TD3_noise_0.1': '#3498db',
        'RNN_TD3_noise_0.2': '#e74c3c',
    }

    labels = {
        'RNN_TD3_noise_0.05': 'Noise=0.05',
        'RNN_TD3_noise_0.1': 'Noise=0.1',
        'RNN_TD3_noise_0.2': 'Noise=0.2',
    }

    # Left: Training curves
    ax = axes[0]
    for result in results:
        name = result['name']
        if name not in colors:
            continue
        steps = [m['step'] for m in result['metrics']]
        rewards = [m['reward'] for m in result['metrics']]
        stds = [m['reward_std'] for m in result['metrics']]
        ax.plot(steps, rewards, color=colors[name], lw=2, label=labels[name])
        ax.fill_between(steps, np.array(rewards) - np.array(stds),
                       np.array(rewards) + np.array(stds), alpha=0.2, color=colors[name])

    ax.set_xlabel("Environment Steps", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title("Partial Observation Experiment (Recurrent TD3)", fontsize=13)
    ax.axhline(y=780, color='gray', linestyle='--', alpha=0.7, label='SAC Full Obs (baseline)')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)

    # Right: Bar chart
    ax = axes[1]
    names = [r['name'] for r in results if r['name'] in colors]
    final_rewards = [r['final_reward'] for r in results if r['name'] in colors]
    peak_rewards = [r['peak_reward'] for r in results if r['name'] in colors]
    bar_colors = [colors[n] for n in names]

    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, final_rewards, width, label='Final', color=bar_colors, alpha=0.7)
    bars2 = ax.bar(x + width/2, peak_rewards, width, label='Peak', color=bar_colors, edgecolor='black')

    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title("Final vs Peak Performance", fontsize=13)
    ax.set_xticks(x)
    short_labels = ['Noise\n0.05', 'Noise\n0.1', 'Noise\n0.2']
    ax.set_xticklabels(short_labels[:len(names)], fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=780, color='gray', linestyle='--', alpha=0.7, label='SAC Full Obs')

    for bar, val in zip(bars2, peak_rewards):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{val:.0f}', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'partial_obs_td3.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'partial_obs_td3.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved plots to {output_dir}/partial_obs_td3.png")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"partial_obs_td3_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("PARTIAL OBSERVATION EXPERIMENT (RECURRENT TD3)")
    print("=" * 70)
    print(f"Testing recurrent TD3 on partial observation pendulum task")
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
    print(f"  episode_length: {rl_config.episode_length}")

    print(f"\nEnvironments:")
    print(f"  Full obs size: {full_obs_env.observation_size}")
    print(f"  Partial obs size: {partial_obs_env.observation_size}")

    results = []

    # Test 3 different exploration noise values with privileged critic
    exploration_noises = [0.05, 0.1, 0.2]

    for noise in exploration_noises:
        result = run_rnn_td3(
            partial_obs_env,
            f'RNN_TD3_noise_{noise}',
            rl_config,
            cell_type='simple',
            use_privileged_critic=True,
            exploration_noise=noise,
        )
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
    print(f"{'Condition':<30} {'Final':>10} {'Peak':>10} {'Time':>10}")
    print("-" * 65)
    for r in results:
        print(f"{r['name']:<30} {r['final_reward']:>10.2f} {r['peak_reward']:>10.2f} {r['time']:>9.1f}s")
    print("=" * 70)

    print("\nKey Findings:")
    print("  SAC full obs baseline (from previous run): ~780")
    best_result = max(results, key=lambda r: r['peak_reward'])
    print(f"  Best exploration noise: {best_result['name']} with peak {best_result['peak_reward']:.0f}")
    for r in sorted(results, key=lambda x: x['peak_reward'], reverse=True):
        pct = 100 * r['peak_reward'] / 780  # SAC baseline
        print(f"    {r['name']}: peak={r['peak_reward']:.0f} ({pct:.1f}% of SAC)")

    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
