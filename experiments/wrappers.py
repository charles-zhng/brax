"""MJX environment wrappers for mujoco_playground compatibility.

brax's training loops expect envs with batched auto-resetting semantics and
``truncation`` / ``episode_metrics`` info fields. mujoco_playground envs are
single-instance and unwrapped; these wrappers bridge the gap. Pass
``wrap_mjx_env`` as ``wrap_env_fn`` to any brax ``train()``.
"""

import jax
import jax.numpy as jp

from brax.envs.base import Wrapper


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
    """Wrap a mujoco_playground MjxEnv for brax training."""
    env = MjxVmapWrapper(env)
    env = MjxEpisodeWrapper(env, episode_length, action_repeat)
    env = MjxAutoResetWrapper(env)
    return env
