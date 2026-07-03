#!/usr/bin/env python3
"""Train RNN-PPO on CartpoleBalance - the simplest MuJoCo Playground task."""

import os
import time
from datetime import datetime

from mujoco_playground import registry
from brax.training.agents.recurrent_ppo import train, networks
from brax.envs.base import Wrapper
import jax
import jax.numpy as jp
import numpy as np
import matplotlib.pyplot as plt
import imageio


# ============================================================================
# MJX Environment Wrappers (needed for mujoco_playground compatibility)
# ============================================================================

class MjxVmapWrapper(Wrapper):
    """Vectorizes MjxEnv."""
    def reset(self, rng):
        return jax.vmap(self.env.reset)(rng)

    def step(self, state, action):
        return jax.vmap(self.env.step)(state, action)


class MjxEpisodeWrapper(Wrapper):
    """Maintains episode step count and sets done at episode end."""

    def __init__(self, env, episode_length, action_repeat):
        super().__init__(env)
        self.episode_length = episode_length
        self.action_repeat = action_repeat

    def reset(self, rng):
        state = self.env.reset(rng)
        state = state.replace(info={
            **state.info,
            'steps': jp.zeros(rng.shape[:-1]),
            'truncation': jp.zeros(rng.shape[:-1]),
            'episode_done': jp.zeros(rng.shape[:-1]),
            'episode_metrics': {
                'sum_reward': jp.zeros(rng.shape[:-1]),
                'length': jp.zeros(rng.shape[:-1]),
            },
        })
        return state

    def step(self, state, action):
        def f(state, _):
            nstate = self.env.step(state, action)
            return nstate, nstate.reward

        state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
        state = state.replace(reward=jp.sum(rewards, axis=0))
        steps = state.info['steps'] + self.action_repeat
        one = jp.ones_like(state.done)
        zero = jp.zeros_like(state.done)
        episode_length = jp.array(self.episode_length, dtype=jp.int32)
        done = jp.where(steps >= episode_length, one, state.done)
        truncation = jp.where(steps >= episode_length, 1 - state.done, zero)
        prev_done = state.info['episode_done']
        # Zero the previous episode's totals BEFORE adding this step's
        # contribution, so the new episode's first step is counted.
        sum_reward = state.info['episode_metrics']['sum_reward'] * (1 - prev_done) + jp.sum(rewards, axis=0)
        length = state.info['episode_metrics']['length'] * (1 - prev_done) + self.action_repeat
        state = state.replace(
            done=done,
            info={
                **state.info,
                'steps': steps,
                'truncation': truncation,
                'episode_done': done,
                'episode_metrics': {'sum_reward': sum_reward, 'length': length},
            }
        )
        return state


class MjxAutoResetWrapper(Wrapper):
    """Automatically resets MjxEnvs that are done."""

    def reset(self, rng):
        state = self.env.reset(rng)
        state = state.replace(info={
            **state.info,
            'first_data': state.data,
            'first_obs': state.obs,
        })
        return state

    def step(self, state, action):
        if 'steps' in state.info:
            steps = jp.where(
                state.done, jp.zeros_like(state.info['steps']), state.info['steps']
            )
            state = state.replace(info={**state.info, 'steps': steps})
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)

        def where_done_leaf(x, y):
            """Reset leaf arrays where done=True."""
            done = state.done
            if not hasattr(x, 'shape') or x.shape == ():
                return y
            if len(x.shape) == 0:
                return y
            if done.shape and done.shape[0] != x.shape[0]:
                return y
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jp.where(done, x, y)

        data = jax.tree_util.tree_map(
            where_done_leaf, state.info['first_data'], state.data
        )
        obs = jax.tree_util.tree_map(
            where_done_leaf, state.info['first_obs'], state.obs
        )
        return state.replace(data=data, obs=obs)


def wrap_mjx_env(env, episode_length, action_repeat, **kwargs):
    """Wrap MjxEnv for training."""
    env = MjxVmapWrapper(env)
    env = MjxEpisodeWrapper(env, episode_length, action_repeat)
    env = MjxAutoResetWrapper(env)
    return env


# ============================================================================
# Metrics Collection
# ============================================================================

class MetricsCollector:
    """Collects training metrics for plotting."""

    def __init__(self):
        self.steps = []
        self.rewards = []
        self.reward_stds = []
        self.policy_losses = []
        self.value_losses = []
        self.sps = []

    def update(self, step: int, metrics: dict):
        self.steps.append(step)
        self.rewards.append(metrics.get('eval/episode_reward', 0))
        self.reward_stds.append(metrics.get('eval/episode_reward_std', 0))
        self.policy_losses.append(metrics.get('training/policy_loss', 0))
        self.value_losses.append(metrics.get('training/v_loss', 0))
        self.sps.append(metrics.get('training/sps', 0))

    def plot(self, save_path: str):
        """Generate and save training plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Episode Reward
        ax = axes[0, 0]
        ax.plot(self.steps, self.rewards, 'b-', linewidth=2, label='Mean Reward')
        ax.fill_between(
            self.steps,
            np.array(self.rewards) - np.array(self.reward_stds),
            np.array(self.rewards) + np.array(self.reward_stds),
            alpha=0.3, color='blue'
        )
        ax.set_xlabel('Environment Steps', fontsize=12)
        ax.set_ylabel('Episode Reward', fontsize=12)
        ax.set_title('Episode Reward vs Training Steps', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Policy Loss
        ax = axes[0, 1]
        ax.plot(self.steps, self.policy_losses, 'r-', linewidth=2)
        ax.set_xlabel('Environment Steps', fontsize=12)
        ax.set_ylabel('Policy Loss', fontsize=12)
        ax.set_title('Policy Loss vs Training Steps', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Value Loss
        ax = axes[1, 0]
        ax.plot(self.steps, self.value_losses, 'g-', linewidth=2)
        ax.set_xlabel('Environment Steps', fontsize=12)
        ax.set_ylabel('Value Loss', fontsize=12)
        ax.set_title('Value Loss vs Training Steps', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Training Speed (SPS)
        ax = axes[1, 1]
        ax.plot(self.steps, self.sps, 'm-', linewidth=2)
        ax.set_xlabel('Environment Steps', fontsize=12)
        ax.set_ylabel('Steps Per Second', fontsize=12)
        ax.set_title('Training Speed vs Training Steps', fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved training plots to: {save_path}')


# ============================================================================
# Rollout and Video
# ============================================================================

def rollout_policy(env, make_policy_fn, params, rnn_ppo_network, num_steps=500, seed=0):
    """Rollout policy and collect trajectory for rendering."""
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)

    policy_fn = make_policy_fn(params, deterministic=True)
    state = env.reset(reset_key)
    trajectory = [state]

    # Policy hidden state only (batch size 1); the value network is
    # feedforward and policy_fn expects the bare policy hidden state.
    hidden_states = rnn_ppo_network.policy_network.init_hidden(1)

    total_reward = 0.0

    @jax.jit
    def policy_step(obs, hidden_states, key):
        obs_batched = jax.tree_util.tree_map(lambda x: x[None, ...], obs)
        action, extras, new_hidden = policy_fn(obs_batched, hidden_states, key)
        return action[0], new_hidden

    for step in range(num_steps):
        key, action_key = jax.random.split(key)
        action, hidden_states = policy_step(state.obs, hidden_states, action_key)
        state = env.step(state, action)
        trajectory.append(state)
        total_reward += float(state.reward)
        if state.done:
            break

    print(f'Rollout: {len(trajectory)} steps, reward: {total_reward:.2f}')
    return trajectory, total_reward


def render_video(env, trajectory, save_path, fps=30, width=640, height=480):
    """Render trajectory to video."""
    print(f'Rendering {len(trajectory)} frames...')
    frames = env.render(trajectory, height=height, width=width)
    imageio.mimsave(save_path, frames, fps=fps)
    print(f'Saved video to: {save_path}')


# ============================================================================
# Main
# ============================================================================

def main():
    # Configuration - train longer for better results
    NUM_TIMESTEPS = 10_000_000
    NUM_ENVS = 1024
    EPISODE_LENGTH = 500
    NUM_EVALS = 20
    SEED = 42

    # RNN architecture - larger network
    CELL_TYPE = 'gru'
    RNN_HIDDEN_SIZE = 64
    OUTPUT_LAYER_SIZES = (64,)

    # PPO hyperparameters - tuned for stability
    LEARNING_RATE = 1e-4
    ENTROPY_COST = 0.005
    DISCOUNTING = 0.99
    GAE_LAMBDA = 0.95
    UNROLL_LENGTH = 32
    BATCH_SIZE = 512
    NUM_MINIBATCHES = 16
    NUM_UPDATES_PER_BATCH = 8
    MAX_GRAD_NORM = 0.5

    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'rnn_ppo_cartpole_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 60)
    print('RNN-PPO Training on CartpoleBalance')
    print('=' * 60)
    print(f'Cell type: {CELL_TYPE}')
    print(f'RNN hidden size: {RNN_HIDDEN_SIZE}')
    print(f'Total timesteps: {NUM_TIMESTEPS:,}')
    print(f'Num envs: {NUM_ENVS}')
    print(f'Output directory: {output_dir}')
    print('=' * 60)

    # Load environment
    print('\nLoading CartpoleBalance environment...')
    base_env = registry.load('CartpoleBalance')
    print(f'Observation size: {base_env.observation_size}')
    print(f'Action size: {base_env.action_size}')

    # Metrics collector
    metrics_collector = MetricsCollector()

    def progress_fn(step: int, metrics: dict):
        metrics_collector.update(step, metrics)
        reward = metrics.get('eval/episode_reward', 0)
        reward_std = metrics.get('eval/episode_reward_std', 0)
        sps = metrics.get('training/sps', 0)
        print(f'Step {step:>8,}: reward = {reward:>8.2f} +/- {reward_std:.2f}, SPS = {sps:.0f}')

    # Network factory
    def network_factory(obs_size, action_size, **kw):
        return networks.make_rnn_ppo_networks(
            obs_size,
            action_size,
            cell_type=CELL_TYPE,
            policy_rnn_hidden_size=RNN_HIDDEN_SIZE,
            policy_output_layer_sizes=OUTPUT_LAYER_SIZES,
            **kw
        )

    # Train
    print('\nStarting training...')
    start_time = time.time()

    make_policy, params, final_metrics = train.train(
        environment=base_env,
        num_timesteps=NUM_TIMESTEPS,
        num_envs=NUM_ENVS,
        wrap_env=True,
        wrap_env_fn=wrap_mjx_env,
        episode_length=EPISODE_LENGTH,
        num_evals=NUM_EVALS,
        learning_rate=LEARNING_RATE,
        entropy_cost=ENTROPY_COST,
        discounting=DISCOUNTING,
        gae_lambda=GAE_LAMBDA,
        unroll_length=UNROLL_LENGTH,
        batch_size=BATCH_SIZE,
        num_minibatches=NUM_MINIBATCHES,
        num_updates_per_batch=NUM_UPDATES_PER_BATCH,
        normalize_observations=True,
        normalize_advantage=True,
        clipping_epsilon=0.2,
        max_grad_norm=MAX_GRAD_NORM,
        network_factory=network_factory,
        seed=SEED,
        progress_fn=progress_fn,
        deterministic_eval=True,
    )

    training_time = time.time() - start_time
    print(f'\nTraining completed in {training_time:.1f} seconds')
    print(f'Final reward: {final_metrics.get("eval/episode_reward", 0):.2f}')

    # Save training plots
    plot_path = os.path.join(output_dir, 'training_plots.png')
    metrics_collector.plot(save_path=plot_path)

    # Save params
    import pickle
    params_path = os.path.join(output_dir, 'params.pkl')
    with open(params_path, 'wb') as f:
        pickle.dump(params, f)
    print(f'Saved params to: {params_path}')

    # Generate rollout video
    print('\nGenerating rollout video...')
    rnn_ppo_network = network_factory(
        base_env.observation_size,
        base_env.action_size,
    )

    best_trajectory = None
    best_reward = -float('inf')
    for rollout_seed in [0, 42, 100]:
        trajectory, reward = rollout_policy(
            base_env,
            make_policy,
            params,
            rnn_ppo_network,
            num_steps=500,
            seed=rollout_seed,
        )
        if reward > best_reward:
            best_reward = reward
            best_trajectory = trajectory

    print(f'Best rollout reward: {best_reward:.2f}')

    video_path = os.path.join(output_dir, 'rollout.mp4')
    render_video(base_env, best_trajectory, save_path=video_path, fps=50)

    print('\n' + '=' * 60)
    print('Training Complete!')
    print('=' * 60)
    print(f'Output directory: {output_dir}')
    print(f'  - Training plots: training_plots.png')
    print(f'  - Rollout video: rollout.mp4')
    print(f'Final eval reward: {final_metrics.get("eval/episode_reward", 0):.2f}')
    print('=' * 60)


if __name__ == '__main__':
    main()
