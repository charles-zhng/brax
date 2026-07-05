"""Environment wrappers for the experiment harness.

Training-time wrapping of mujoco_playground envs (vmap / episode bookkeeping /
auto-reset) is delegated to ``mujoco_playground.wrapper.wrap_for_brax_training``
— verified step-for-step equivalent to the hand-rolled stack it replaced
(modulo playground's one-time reset-key split). Only the POMDP obs wrapper is
ours.
"""

import jax.numpy as jp

from brax.envs.base import Wrapper


class PartialObsWrapper(Wrapper):
    """Exposes only `visible_idx` of a flat-observation env's obs (POMDP).

    observation_mode 'dict' emits {'state': partial, 'privileged_state': full}
    for recurrent agents with a privileged critic/value network
    (policy_obs_key='state', q_obs_key/value_obs_key='privileged_state').
    observation_mode 'flat' emits only the partial array — for feedforward
    baselines whose networks cannot consume dict observations.

    Wrap the RAW (single-instance) playground env with this, then apply
    wrap_for_brax_training on top.
    """

    def __init__(self, env, visible_idx, observation_mode='dict'):
        super().__init__(env)
        if observation_mode not in ('dict', 'flat'):
            raise ValueError(f'Unknown observation_mode {observation_mode!r}')
        self._visible_idx = tuple(visible_idx)
        self._mode = observation_mode

    def _transform(self, obs):
        partial = obs[..., jp.asarray(self._visible_idx)]
        if self._mode == 'flat':
            return partial
        return {'state': partial, 'privileged_state': obs}

    def reset(self, rng):
        state = self.env.reset(rng)
        return state.replace(obs=self._transform(state.obs))

    def step(self, state, action):
        state = self.env.step(state, action)
        return state.replace(obs=self._transform(state.obs))

    @property
    def observation_size(self):
        full = self.env.observation_size
        if not isinstance(full, int):
            raise ValueError(
                'PartialObsWrapper requires a flat-observation base env, got '
                f'observation_size={full!r}'
            )
        if self._mode == 'flat':
            return len(self._visible_idx)
        return {'state': len(self._visible_idx), 'privileged_state': full}
