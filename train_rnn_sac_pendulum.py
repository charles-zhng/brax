#!/usr/bin/env python3
"""Train recurrent SAC on PendulumSwingup with plotting and video rendering.

Analogue of train_rnn_pendulum.py but using the off-policy recurrent SAC
agent. Reuses the same MJX env wrappers so the env setup stays identical.
"""

import os
import pickle
import time
from datetime import datetime
from typing import Callable, List

import imageio
import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import numpy as np
from brax.envs.base import Wrapper
from brax.training.agents.recurrent_sac import networks, train
from mujoco_playground import registry


# ============================================================================
# MJX Environment Wrappers (same as train_rnn_pendulum.py)
# ============================================================================


class MjxVmapWrapper(Wrapper):
    def reset(self, rng):
        return jax.vmap(self.env.reset)(rng)

    def step(self, state, action):
        return jax.vmap(self.env.step)(state, action)


class MjxEpisodeWrapper(Wrapper):
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
    env = MjxVmapWrapper(env)
    env = MjxEpisodeWrapper(env, episode_length, action_repeat)
    env = MjxAutoResetWrapper(env)
    return env


# ============================================================================
# Metrics collection & plotting
# ============================================================================


class MetricsCollector:
    def __init__(self):
        self.steps = []
        self.rewards = []
        self.reward_stds = []
        self.critic_losses = []
        self.actor_losses = []
        self.alpha_losses = []
        self.alphas = []
        self.sps = []

    def update(self, step: int, metrics: dict):
        self.steps.append(step)
        self.rewards.append(metrics.get('eval/episode_reward', 0))
        self.reward_stds.append(metrics.get('eval/episode_reward_std', 0))
        self.critic_losses.append(metrics.get('training/critic_loss', 0))
        self.actor_losses.append(metrics.get('training/actor_loss', 0))
        self.alpha_losses.append(metrics.get('training/alpha_loss', 0))
        self.alphas.append(metrics.get('training/alpha', 0))
        self.sps.append(metrics.get('training/sps', 0))

    def plot(self, save_path: str = 'training_plots.png'):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        ax = axes[0, 0]
        ax.plot(self.steps, self.rewards, 'b-', linewidth=2, label='Mean Reward')
        ax.fill_between(
            self.steps,
            np.array(self.rewards) - np.array(self.reward_stds),
            np.array(self.rewards) + np.array(self.reward_stds),
            alpha=0.3, color='blue',
        )
        ax.set_xlabel('Environment Steps')
        ax.set_ylabel('Episode Reward')
        ax.set_title('Episode Reward vs Steps')
        ax.grid(True, alpha=0.3)
        ax.legend()

        for ax, data, title, color in zip(
            [axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]],
            [self.critic_losses, self.actor_losses, self.alpha_losses, self.alphas, self.sps],
            ['Critic Loss', 'Actor Loss', 'Alpha Loss', 'Alpha (entropy temp)', 'Steps Per Second'],
            ['r', 'g', 'm', 'c', 'orange'],
        ):
            ax.plot(self.steps, data, color=color, linewidth=2)
            ax.set_xlabel('Environment Steps')
            ax.set_ylabel(title)
            ax.set_title(f'{title} vs Steps')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved training plots to: {save_path}')


# ============================================================================
# Rollout + video
# ============================================================================


def rollout_policy(
    env,
    make_policy_fn: Callable,
    params,
    rsac_network,
    num_steps: int = 1000,
    seed: int = 0,
):
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)

    # Inference policy consumes (normalizer_params, policy_params). params is
    # (normalizer_params, policy_params, q_params) — drop q_params.
    policy_fn = make_policy_fn((params[0], params[1]), deterministic=True)

    state = env.reset(reset_key)
    trajectory = [state]

    policy_hidden = rsac_network.policy_network.init_hidden(1)
    total_reward = 0.0

    @jax.jit
    def policy_step(obs, hidden, key):
        obs_batched = jax.tree_util.tree_map(lambda x: x[None, ...], obs)
        action, _, new_hidden = policy_fn(obs_batched, hidden, key)
        return action[0], new_hidden

    for step in range(num_steps):
        key, action_key = jax.random.split(key)
        action, policy_hidden = policy_step(state.obs, policy_hidden, action_key)
        state = env.step(state, action)
        trajectory.append(state)
        total_reward += float(state.reward)
        if state.done:
            print(f'Episode done at step {step + 1}')
            break

    print(f'Rollout: {len(trajectory)} steps, total reward: {total_reward:.2f}')
    return trajectory, total_reward


def render_video(env, trajectory, save_path='rollout.mp4', fps=30, width=640, height=480):
    print(f'Rendering {len(trajectory)} frames...')
    frames = env.render(trajectory, height=height, width=width)
    imageio.mimsave(save_path, frames, fps=fps)
    print(f'Saved video to: {save_path}')


# ============================================================================
# Main
# ============================================================================


def main():
    # Training config.
    NUM_TIMESTEPS = 1_000_000
    NUM_ENVS = 64
    EPISODE_LENGTH = 1000
    NUM_EVALS = 20
    SEED = 42

    # RNN architecture.
    CELL_TYPE = 'gru'
    RNN_HIDDEN_SIZE = 128
    POLICY_LAYER_SIZES = (128,)
    Q_LAYER_SIZES = (128,)

    # SAC hyperparameters.
    LEARNING_RATE = 3e-4
    DISCOUNTING = 0.99
    TAU = 0.005
    BATCH_SIZE = 256
    UNROLL_LENGTH = 32
    BURN_IN = 0
    MIN_REPLAY_SIZE = 10_000
    MAX_REPLAY_SIZE = 500_000
    GRAD_UPDATES_PER_STEP = 1
    REWARD_SCALING = 1.0

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'rnn_sac_pendulum_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 60)
    print('Recurrent SAC Training on PendulumSwingup')
    print('=' * 60)
    print(f'Cell type: {CELL_TYPE}')
    print(f'RNN hidden size: {RNN_HIDDEN_SIZE}')
    print(f'Unroll length: {UNROLL_LENGTH}, burn_in: {BURN_IN}')
    print(f'Total timesteps: {NUM_TIMESTEPS:,}')
    print(f'Num envs: {NUM_ENVS}')
    print(f'Output directory: {output_dir}')
    print('=' * 60)

    print('\nLoading PendulumSwingup environment...')
    base_env = registry.load('PendulumSwingup')
    print(f'Observation size: {base_env.observation_size}')
    print(f'Action size: {base_env.action_size}')

    metrics_collector = MetricsCollector()

    def progress_fn(step: int, metrics: dict):
        metrics_collector.update(step, metrics)
        reward = metrics.get('eval/episode_reward', 0)
        reward_std = metrics.get('eval/episode_reward_std', 0)
        sps = metrics.get('training/sps', 0)
        alpha = metrics.get('training/alpha', 0)
        print(
            f'Step {step:>10,}: reward = {reward:>8.2f} +/- {reward_std:.2f}, '
            f'alpha = {alpha:.3f}, SPS = {sps:.0f}'
        )

    def network_factory(obs_size, action_size, **kw):
        return networks.make_recurrent_sac_networks(
            obs_size,
            action_size,
            rnn_hidden_size=RNN_HIDDEN_SIZE,
            policy_output_layer_sizes=POLICY_LAYER_SIZES,
            q_hidden_layer_sizes=Q_LAYER_SIZES,
            cell_type=CELL_TYPE,
            **kw,
        )

    print('\nStarting training...')
    start_time = time.time()

    make_policy, params, final_metrics = train.train(
        environment=base_env,
        num_timesteps=NUM_TIMESTEPS,
        episode_length=EPISODE_LENGTH,
        wrap_env=True,
        wrap_env_fn=wrap_mjx_env,
        num_envs=NUM_ENVS,
        num_evals=NUM_EVALS,
        learning_rate=LEARNING_RATE,
        discounting=DISCOUNTING,
        tau=TAU,
        batch_size=BATCH_SIZE,
        unroll_length=UNROLL_LENGTH,
        burn_in=BURN_IN,
        min_replay_size=MIN_REPLAY_SIZE,
        max_replay_size=MAX_REPLAY_SIZE,
        grad_updates_per_step=GRAD_UPDATES_PER_STEP,
        reward_scaling=REWARD_SCALING,
        normalize_observations=True,
        network_factory=network_factory,
        seed=SEED,
        progress_fn=progress_fn,
        deterministic_eval=True,
    )

    training_time = time.time() - start_time
    print(f'\nTraining completed in {training_time:.1f} seconds')
    print(f'Final reward: {final_metrics.get("eval/episode_reward", 0):.2f}')

    metrics_collector.plot(save_path=os.path.join(output_dir, 'training_plots.png'))

    params_path = os.path.join(output_dir, 'params.pkl')
    with open(params_path, 'wb') as f:
        pickle.dump(params, f)
    print(f'Saved params to: {params_path}')

    print('\nGenerating rollout video...')
    rsac_network = network_factory(base_env.observation_size, base_env.action_size)

    best_trajectory = None
    best_reward = -float('inf')
    for rollout_seed in [0, 42, 100, 200, 300]:
        trajectory, reward = rollout_policy(
            base_env, make_policy, params, rsac_network, num_steps=500, seed=rollout_seed
        )
        print(f'Seed {rollout_seed}: reward={reward:.2f}')
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
    print(f'Final eval reward: {final_metrics.get("eval/episode_reward", 0):.2f}')
    print(f'Best rollout reward: {best_reward:.2f}')
    print('=' * 60)


if __name__ == '__main__':
    main()
