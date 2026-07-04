"""Warp-backend validation: load, reset, and step envs with impl='warp'.

Run on a CUDA GPU node. For each env: construct with impl='warp', wrap with
the harness wrappers, vmapped reset + jitted random-action rollout, and check
outputs are finite. Exits nonzero on any failure.

Usage: python -m experiments.validate_warp [--envs CartpoleBalance,...] [--num-envs 64]
"""

import argparse
import time

import jax
import jax.numpy as jp

from mujoco_playground import registry
from experiments.wrappers import wrap_mjx_env

DEFAULT_ENVS = ['PendulumSwingup', 'CartpoleSwingup', 'FingerSpin', 'CheetahRun']


def validate_env(name, num_envs, num_steps=50, episode_length=1000):
    t0 = time.time()
    env = registry.load(name, config_overrides={'impl': 'warp'})
    impl = env.mjx_model.impl
    assert 'warp' in str(impl).lower(), f'{name}: impl is {impl}, not warp'

    wrapped = wrap_mjx_env(env, episode_length=episode_length, action_repeat=1)
    keys = jax.random.split(jax.random.PRNGKey(0), num_envs)
    state = jax.jit(wrapped.reset)(keys)

    @jax.jit
    def rollout(state, key):
        def step(carry, _):
            state, key = carry
            key, akey = jax.random.split(key)
            action = jax.random.uniform(
                akey, (num_envs, env.action_size), minval=-1.0, maxval=1.0
            )
            state = wrapped.step(state, action)
            return (state, key), state.reward

        (state, _), rewards = jax.lax.scan(
            step, (state, key), (), length=num_steps
        )
        return state, rewards

    state, rewards = rollout(state, jax.random.PRNGKey(1))
    jax.block_until_ready(rewards)
    obs_leaves = jax.tree_util.tree_leaves(state.obs)
    assert all(bool(jp.isfinite(leaf).all()) for leaf in obs_leaves), f'{name}: non-finite obs'
    assert bool(jp.isfinite(rewards).all()), f'{name}: non-finite rewards'
    dt = time.time() - t0
    sps = num_envs * num_steps / dt
    print(f'{name}: OK impl={impl} obs_finite reward_mean={float(rewards.mean()):.4f} '
          f'({dt:.1f}s incl. compile, ~{sps:,.0f} env-steps/s)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--envs', default=','.join(DEFAULT_ENVS))
    parser.add_argument('--num-envs', type=int, default=64)
    args = parser.parse_args()

    print('jax devices:', jax.devices())
    failures = []
    for name in args.envs.split(','):
        try:
            validate_env(name, args.num_envs)
        except Exception as e:  # noqa: BLE001 - report every env, then fail
            print(f'{name}: FAILED — {type(e).__name__}: {e}')
            failures.append(name)
    if failures:
        raise SystemExit(f'Warp validation failed for: {failures}')
    print('ALL ENVS OK')


if __name__ == '__main__':
    main()
