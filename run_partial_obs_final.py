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
import dataclasses
import argparse
import pprint
import yaml
from datetime import datetime
from typing import Dict, List

# Must be set before importing JAX.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jp

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)
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
FF_TIMESTEPS = 150_000_000
RNN_TIMESTEPS = 1_500_000_000
NUM_EVALS = 40
SEED = 0


def _to_pretty_dict(obj):
    """Best-effort conversion of config objects to plain Python containers."""
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return {k: _to_pretty_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_pretty_dict(v) for v in obj]
    if hasattr(obj, "_asdict"):  # namedtuple
        return _to_pretty_dict(obj._asdict())
    if hasattr(obj, "__dict__"):
        return _to_pretty_dict(vars(obj))
    if hasattr(obj, "item"):  # numpy scalars
        return obj.item()
    if hasattr(obj, "tolist"):  # numpy arrays
        return obj.tolist()
    return obj


def print_config(title: str, cfg) -> None:
    cfg = _to_pretty_dict(cfg)
    print(f"\n{title}:")
    print(
        pprint.pformat(
            cfg,
            indent=2,
            width=110,
            sort_dicts=True,
            compact=False,
        )
    )


def run_ff_ppo(env, name: str, rl_config, output_dir: str, num_timesteps: int) -> Dict:
    """Run FF PPO with optimal hyperparameters."""
    print("\n" + "=" * 70)
    print(f"Experiment: {name}")
    print(f"  Algorithm: FF PPO")
    print(f"  Observation size: {env.observation_size}")
    print(f"  Timesteps: {num_timesteps}")
    print(
        f"  LR: {rl_config.learning_rate}, reward_scaling: {rl_config.reward_scaling}"
    )
    print(f"  action_repeat: {rl_config.action_repeat}")
    print("=" * 70)

    # Config passed directly to ppo.train() via **kwargs
    train_config = {
        "num_timesteps": num_timesteps,
        "num_evals": NUM_EVALS,
        "seed": SEED,
        "wrap_env": True,
        "wrap_env_fn": wrap_for_brax_training,
        "episode_length": rl_config.episode_length,
        "reward_scaling": rl_config.reward_scaling,
        "normalize_observations": rl_config.normalize_observations,
        "action_repeat": rl_config.action_repeat,
        "unroll_length": rl_config.unroll_length,
        "num_updates_per_batch": rl_config.num_updates_per_batch,
        "discounting": rl_config.discounting,
        "learning_rate": rl_config.learning_rate,
        "entropy_cost": rl_config.entropy_cost,
        "normalize_advantage": True,
        "clipping_epsilon": 0.2,
        "num_envs": 2048,
        "batch_size": rl_config.batch_size,
        "num_minibatches": rl_config.num_minibatches,
    }

    # Metadata for logging only (not passed to train)
    metadata = {
        "algorithm": "ppo",
        "observation_size": getattr(env, "observation_size", -1),
        "action_size": getattr(env, "action_size", -1),
    }

    metrics_list = []
    start_time = time.time()

    def progress_fn(step, metrics):
        r = metrics.get("eval/episode_reward", 0)
        r_std = metrics.get("eval/episode_reward_std", 0)
        sps = metrics.get("training/sps", 0)
        elapsed = time.time() - start_time
        print(
            f"[{name}] Step {step:>10,}: reward = {r:>7.2f} +/- {r_std:>5.2f}, SPS = {sps:,.0f}, time = {elapsed:.1f}s"
        )
        metrics_list.append(
            {
                "step": step,
                "reward": float(r),
                "reward_std": float(r_std),
                "time": elapsed,
            }
        )

    print_config(f"[{name}] Full training config", {**train_config, **metadata})

    # Save config to YAML
    full_config = _to_pretty_dict({**train_config, **metadata})
    with open(os.path.join(output_dir, f"config_{name}.yaml"), "w") as f:
        yaml.dump(full_config, f, default_flow_style=False)

    t0 = time.time()
    make_policy, policy_params, final_metrics = ppo.train(
        environment=env, progress_fn=progress_fn, **train_config
    )
    elapsed = time.time() - t0
    final_reward = final_metrics.get("eval/episode_reward", 0)
    peak_reward = max(m["reward"] for m in metrics_list) if metrics_list else 0

    print(
        f"\n{name} completed in {elapsed:.1f}s, Final: {final_reward:.2f}, Peak: {peak_reward:.2f}"
    )

    return {
        "name": name,
        "algorithm": "ff",
        "metrics": metrics_list,
        "time": elapsed,
        "final_reward": final_reward,
        "peak_reward": peak_reward,
        "num_timesteps": num_timesteps,
    }


def run_rnn_ppo(
    env,
    name: str,
    rl_config,
    output_dir: str,
    num_timesteps: int,
    learning_rate: float = 5e-4,
) -> Dict:
    """Run RNN PPO with optimal hyperparameters from EXPERIMENT_LOG.md."""
    # Network config (used to build network_factory, logged in metadata)
    network_config = {
        "cell_type": "simple",
        "policy_rnn_hidden_size": 64,
        "policy_output_layer_sizes": (),
        "policy_obs_key": "state",  # Partial obs for policy (2D)
        "value_obs_key": "privileged_state",  # Full obs for value function (3D)
    }

    def network_factory(obs_size, action_size, **kw):
        return rnn_networks.make_rnn_ppo_networks(
            obs_size, action_size, **network_config, **kw
        )

    # Config passed directly to rnn_ppo.train() via **kwargs
    train_config = {
        "num_timesteps": num_timesteps,
        "num_evals": NUM_EVALS,
        "seed": SEED,
        "wrap_env": True,
        "wrap_env_fn": wrap_for_brax_training,
        "episode_length": rl_config.episode_length,
        "reward_scaling": rl_config.reward_scaling,
        "normalize_observations": rl_config.normalize_observations,
        "action_repeat": rl_config.action_repeat,
        "unroll_length": 64,
        "num_updates_per_batch": 4,
        "discounting": 0.99,
        "learning_rate": 8e-4,
        "entropy_cost": 5e-3,
        "max_grad_norm": 0.5,
        "normalize_advantage": True,
        "clipping_epsilon": 0.1,
        "num_envs": 1024,
        "batch_size": 256,
        "num_minibatches": 8,
        "network_factory": network_factory,
    }

    # Metadata for logging only (not passed to train)
    metadata = {
        "algorithm": "recurrent_ppo",
        "observation_size": getattr(env, "observation_size", -1),
        "action_size": getattr(env, "action_size", -1),
        **network_config,
    }

    print("\n" + "=" * 70)
    print(f"Experiment: {name}")
    print(f"  Algorithm: recurrent PPO ({network_config['cell_type']})")
    print(f"  Observation size: {env.observation_size}")
    print(f"  Timesteps: {num_timesteps}")
    print(
        f"  Policy obs key: {network_config['policy_obs_key']}, Value obs key: {network_config['value_obs_key']}"
    )
    print(
        f"  LR: {train_config['learning_rate']}, reward_scaling: {rl_config.reward_scaling}"
    )
    print(
        f"  action_repeat: {rl_config.action_repeat}, unroll: {train_config['unroll_length']}"
    )
    print("=" * 70)

    metrics_list = []
    start_time = time.time()

    def progress_fn(step, metrics):
        r = metrics.get("eval/episode_reward", 0)
        r_std = metrics.get("eval/episode_reward_std", 0)
        sps = metrics.get("training/sps", 0)
        elapsed = time.time() - start_time
        print(
            f"[{name}] Step {step:>10,}: reward = {r:>7.2f} +/- {r_std:>5.2f}, SPS = {sps:,.0f}, time = {elapsed:.1f}s"
        )
        metrics_list.append(
            {
                "step": step,
                "reward": float(r),
                "reward_std": float(r_std),
                "time": elapsed,
            }
        )

    print_config(f"[{name}] Full training config", {**train_config, **metadata})

    # Save config to YAML
    full_config = _to_pretty_dict({**train_config, **metadata})
    with open(os.path.join(output_dir, f"config_{name}.yaml"), "w") as f:
        yaml.dump(full_config, f, default_flow_style=False)

    t0 = time.time()
    make_policy, policy_params, final_metrics = rnn_ppo.train(
        environment=env, progress_fn=progress_fn, **train_config
    )
    elapsed = time.time() - t0
    final_reward = final_metrics.get("eval/episode_reward", 0)
    peak_reward = max(m["reward"] for m in metrics_list) if metrics_list else 0

    print(
        f"\n{name} completed in {elapsed:.1f}s, Final: {final_reward:.2f}, Peak: {peak_reward:.2f}"
    )

    return {
        "name": name,
        "algorithm": "rnn",
        "metrics": metrics_list,
        "time": elapsed,
        "final_reward": final_reward,
        "peak_reward": peak_reward,
        "learning_rate": learning_rate,
        "num_timesteps": num_timesteps,
    }


def plot_results(results: List[Dict], output_dir: str):
    """Generate comparison plots."""
    if not results:
        print("No results to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        "FF_full_obs": "#2ecc71",
        "RNN_partial_obs": "#3498db",
    }

    labels = {
        "FF_full_obs": "FF PPO (Full Obs) - Control",
        "RNN_partial_obs": "RNN PPO (Partial Obs)",
    }

    # Left: Training curves
    ax = axes[0]
    for result in results:
        name = result["name"]
        steps = [m["step"] for m in result["metrics"]]
        rewards = [m["reward"] for m in result["metrics"]]
        stds = [m["reward_std"] for m in result["metrics"]]

        label = labels.get(name, name)
        if "num_timesteps" in result:
            label += f" ({result['num_timesteps']:,} steps)"

        color = colors.get(name)
        if color is None:
            color = f"C{len(ax.lines)}"
        ax.plot(steps, rewards, color=color, lw=2, label=label)
        ax.fill_between(
            steps,
            np.array(rewards) - np.array(stds),
            np.array(rewards) + np.array(stds),
            alpha=0.2,
            color=color,
        )

    ax.set_xlabel("Environment Steps", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title("Partial Observation Experiment (Optimal HPs)", fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=800, color="gray", linestyle="--", alpha=0.5)

    # Right: Bar chart
    ax = axes[1]
    names = [r["name"] for r in results]
    final_rewards = [r["final_reward"] for r in results]
    peak_rewards = [r["peak_reward"] for r in results]
    bar_colors = [colors.get(n, f"C{i}") for i, n in enumerate(names)]

    x = np.arange(len(names))
    width = 0.35
    bars1 = ax.bar(
        x - width / 2, final_rewards, width, label="Final", color=bar_colors, alpha=0.7
    )
    bars2 = ax.bar(
        x + width / 2,
        peak_rewards,
        width,
        label="Peak",
        color=bar_colors,
        edgecolor="black",
    )

    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title("Final vs Peak Performance", fontsize=13)
    ax.set_xticks(x)
    short_xticklabels = {
        "FF_full_obs": "FF\nFull",
        "RNN_partial_obs": "RNN\nPartial",
    }
    ax.set_xticklabels([short_xticklabels.get(n, n) for n in names], fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=800, color="gray", linestyle="--", alpha=0.5)

    for bar, val in zip(bars2, peak_rewards):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f"{val:.0f}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "partial_obs_final.png"), dpi=150, bbox_inches="tight"
    )
    plt.savefig(os.path.join(output_dir, "partial_obs_final.pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved plots to {output_dir}/partial_obs_final.png")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run partial observation experiments with optional FF PPO."
    )
    parser.add_argument(
        "--run-ff",
        action="store_true",
        help="Also run FF PPO with full observation control.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", f"partial_obs_final_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("PARTIAL OBSERVATION EXPERIMENT")
    print("=" * 70)
    print(f"Seed: {SEED}")
    print(f"Output: {output_dir}")
    print(f"Run FF PPO: {args.run_ff}")
    print("=" * 70)

    # Load environments
    full_obs_env = registry.load(ENV_NAME)
    partial_obs_env = PendulumSwingupPartialObs()
    rl_config = dm_control_suite_params.brax_ppo_config(ENV_NAME)
    # Override action_repeat to 4 as per optimal hyperparameters
    rl_config.action_repeat = 4
    rl_config.num_updates_per_batch = 4

    print_config("Config from dm_control_suite_params.brax_ppo_config", rl_config)

    print(f"\nEnvironments:")
    print(f"  Full obs size: {full_obs_env.observation_size}")
    print(f"  Partial obs size: {partial_obs_env.observation_size}")

    results = []

    # 1. FF PPO with Full Obs (control) - optional
    if args.run_ff:
        result = run_ff_ppo(
            full_obs_env,
            "FF_full_obs",
            rl_config,
            output_dir,
            num_timesteps=FF_TIMESTEPS,
        )
        results.append(result)

    # 2. RNN PPO with Partial Obs
    result = run_rnn_ppo(
        partial_obs_env,
        "RNN_partial_obs",
        rl_config,
        output_dir,
        num_timesteps=RNN_TIMESTEPS,
        learning_rate=3e-4,
    )
    results.append(result)

    # Save results
    results_to_save = [{k: v for k, v in r.items()} for r in results]
    with open(os.path.join(output_dir, "results.pkl"), "wb") as f:
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
        print(
            f"{r['name']:<20} {r['final_reward']:>10.2f} {r['peak_reward']:>10.2f} {r['time']:>9.1f}s"
        )
    print("=" * 70)

    rnn_partial = next(r for r in results if r["name"] == "RNN_partial_obs")
    print("\nKey Findings:")
    if args.run_ff:
        ff_full = next(r for r in results if r["name"] == "FF_full_obs")
        print(f"  FF full obs peak: {ff_full['peak_reward']:.0f}")
        print(
            f"  RNN partial obs peak: {rnn_partial['peak_reward']:.0f} (should match FF full)"
        )
    else:
        print(f"  RNN partial obs peak: {rnn_partial['peak_reward']:.0f}")

    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
