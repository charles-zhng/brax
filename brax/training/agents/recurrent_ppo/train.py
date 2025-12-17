# Copyright 2025 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Recurrent Proximal policy optimization training."""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Optional, Tuple, Union

from absl import logging
from brax import base
from brax import envs
from brax.training import gradients
from brax.training import logger as metric_logger
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.recurrent_ppo import losses as recurrent_losses
from brax.training.agents.recurrent_ppo import networks as recurrent_networks
from brax.training.agents.ppo import checkpoint
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import optimizer as ppo_optimizer
from brax.training.types import Params
from brax.training.types import PRNGKey
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

_PMAP_AXIS_NAME = 'i'


@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""

  optimizer_state: optax.OptState
  params: recurrent_losses.RecurrentPPONetworkParams
  normalizer_params: running_statistics.RunningStatisticsState
  env_steps: types.UInt64


def _unpmap(v):
  return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree):
  def f(leaf):
    leaf = jnp.asarray(leaf)
    return jnp.astype(leaf, leaf.dtype)

  return jax.tree_util.tree_map(f, tree)


def _maybe_wrap_env(
    env: envs.Env,
    wrap_env: bool,
    num_envs: int,
    episode_length: Optional[int],
    action_repeat: int,
    device_count: int,
    key_env: PRNGKey,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
):
  """Wraps the environment for training/eval if wrap_env is True."""
  if not wrap_env:
    return env
  if episode_length is None:
    raise ValueError('episode_length must be specified in recurrent_ppo.train')
  v_randomization_fn = None
  if randomization_fn is not None:
    randomization_batch_size = num_envs // device_count
    randomization_rng = jax.random.split(key_env, randomization_batch_size)
    v_randomization_fn = functools.partial(
        randomization_fn, rng=randomization_rng
    )
  if wrap_env_fn is not None:
    wrap_for_training = wrap_env_fn
  else:
    wrap_for_training = envs.training.wrap
  env = wrap_for_training(
      env,
      episode_length=episode_length,
      action_repeat=action_repeat,
      randomization_fn=v_randomization_fn,
  )  # pytype: disable=wrong-keyword-args
  return env


def _remove_pixels(
    obs: Union[jnp.ndarray, dict[str, jax.Array]],
) -> Union[jnp.ndarray, dict[str, jax.Array]]:
  if not isinstance(obs, dict):
    return obs
  return {k: v for k, v in obs.items() if not k.startswith('pixels/')}


def _mask_core_state(
    state: recurrent_networks.RecurrentState,
    mask: jax.Array,
    mask_fn: Callable[[recurrent_networks.RecurrentState, jax.Array], Any],
):
  return mask_fn(state, mask)


def _agg_fn(metric, fn, to_aggregate, to_normalize, episode_lengths):
  if not to_aggregate:
    return metric
  if to_normalize:
    return fn(metric / episode_lengths)
  return fn(metric)


class RecurrentEvaluator:
  """Evaluation helper that tracks recurrent state."""

  def __init__(
      self,
      eval_env: envs.Env,
      eval_policy_fn: Callable[[InferenceParams], Callable[..., Any]],
      num_eval_envs: int,
      episode_length: int,
      action_repeat: int,
      key: PRNGKey,
      ppo_network: recurrent_networks.RecurrentPPONetworks,
  ):
    self._key = key
    self._eval_walltime = 0.0
    eval_env = envs.training.EvalWrapper(eval_env)
    self._eval_state_to_donate = jax.jit(eval_env.reset)(
        jax.random.split(key, num_eval_envs)
    )
    self._core_state_to_donate = ppo_network.initial_state_fn(num_eval_envs)

    def generate_eval_unroll(
        eval_env_state_donated: envs.State,
        core_state_donated: recurrent_networks.RecurrentState,
        policy_params: InferenceParams,
        key: PRNGKey,
    ):
      reset_keys = jax.random.split(key, num_eval_envs)
      eval_first_state = eval_env.reset(reset_keys)
      init_core = ppo_network.initial_state_fn(num_eval_envs)
      policy = eval_policy_fn(policy_params)

      def step_fn(carry, unused_t):
        state, core, step_key = carry
        step_key, sample_key = jax.random.split(step_key)
        actions, _, next_core = policy(state.obs, core, sample_key)
        next_state = eval_env.step(state, actions)
        truncation = next_state.info.get(
            'truncation',
            jnp.zeros_like(next_state.discount, dtype=next_state.discount.dtype),
        )
        mask = (1.0 - truncation) * next_state.discount
        next_core = ppo_network.mask_state_fn(next_core, mask)
        return (next_state, next_core, step_key), None

      (final_state, final_core_state, _), _ = jax.lax.scan(
          step_fn,
          (eval_first_state, init_core, key),
          (),
          length=episode_length // action_repeat,
      )
      return final_state, final_core_state

    self._generate_eval_unroll = jax.jit(
        generate_eval_unroll, donate_argnums=(0, 1), keep_unused=True
    )
    self._steps_per_unroll = episode_length * num_eval_envs

  def run_evaluation(
      self,
      policy_params: InferenceParams,
      training_metrics: Metrics,
      aggregate_episodes: bool = True,
  ) -> Metrics:
    self._key, unroll_key = jax.random.split(self._key)
    t = time.time()
    eval_state, core_state = self._generate_eval_unroll(
        self._eval_state_to_donate, self._core_state_to_donate, policy_params, unroll_key
    )
    self._eval_state_to_donate = eval_state
    self._core_state_to_donate = core_state

    eval_metrics = eval_state.info['eval_metrics']
    eval_metrics.active_episodes.block_until_ready()
    epoch_eval_time = time.time() - t
    episode_lengths = np.maximum(eval_metrics.episode_steps, 1.0).astype(float)

    metrics = {}
    for fn in [np.mean, np.std]:
      suffix = '_std' if fn == np.std else ''
      for name, value in eval_metrics.episode_metrics.items():
        metrics[f'eval/episode_{name}{suffix}'] = _agg_fn(
            value,
            fn,
            aggregate_episodes,
            name.endswith('per_step'),
            episode_lengths,
        )

    metrics['eval/avg_episode_length'] = np.mean(eval_metrics.episode_steps)
    metrics['eval/std_episode_length'] = np.std(eval_metrics.episode_steps)
    metrics['eval/epoch_eval_time'] = epoch_eval_time
    metrics['eval/sps'] = self._steps_per_unroll / epoch_eval_time
    self._eval_walltime = self._eval_walltime + epoch_eval_time
    metrics = {
        'eval/walltime': self._eval_walltime,
        **training_metrics,
        **metrics,
    }
    return metrics


def _recurrent_actor_step(
    env: envs.Env,
    env_state: envs.State,
    policy: Callable[..., Any],
    key: PRNGKey,
    core_state: recurrent_networks.RecurrentState,
    mask_fn: Callable[[recurrent_networks.RecurrentState, jax.Array], Any],
    extra_fields: tuple[str, ...],
):
  actions, policy_extras, new_core_state = policy(env_state.obs, core_state, key)
  nstate = env.step(env_state, actions)
  truncation = nstate.info.get(
      'truncation', jnp.zeros_like(nstate.discount, dtype=nstate.discount.dtype)
  )
  continuation_mask = (1.0 - truncation) * nstate.discount
  masked_core_state = _mask_core_state(
      new_core_state, continuation_mask, mask_fn
  )
  state_extras = {x: nstate.info[x] for x in extra_fields}
  return nstate, masked_core_state, types.Transition(
      observation=env_state.obs,
      action=actions,
      reward=nstate.reward,
      discount=1 - nstate.done,
      next_observation=nstate.obs,
      extras={
          'policy_extras': policy_extras | {'core_state': core_state},
          'state_extras': state_extras,
      },
  )


def _recurrent_generate_unroll(
    env: envs.Env,
    env_state: envs.State,
    policy: Callable[..., Any],
    key: PRNGKey,
    core_state: recurrent_networks.RecurrentState,
    unroll_length: int,
    mask_fn: Callable[[recurrent_networks.RecurrentState, jax.Array], Any],
    extra_fields: tuple[str, ...],
):
  """Collect trajectories of given unroll_length."""

  def f(carry, unused_t):
    state, rnn_state, current_key = carry
    current_key, next_key = jax.random.split(current_key)
    nstate, next_rnn_state, transition = _recurrent_actor_step(
        env,
        state,
        policy,
        current_key,
        rnn_state,
        mask_fn=mask_fn,
        extra_fields=extra_fields,
    )
    return (nstate, next_rnn_state, next_key), (transition, rnn_state)

  f_jit = jax.jit(f, donate_argnums=(0,))
  (final_state, final_core_state, _), (data, core_states) = jax.lax.scan(
      f_jit, (env_state, core_state, key), (), length=unroll_length
  )
  return final_state, final_core_state, data, core_states


def train(
    environment: envs.Env,
    num_timesteps: int,
    max_devices_per_host: Optional[int] = None,
    wrap_env: bool = True,
    num_envs: int = 1,
    episode_length: Optional[int] = None,
    action_repeat: int = 1,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    learning_rate: float = 1e-4,
    entropy_cost: float = 1e-4,
    discounting: float = 0.99,
    unroll_length: int = 10,
    bptt_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    num_resets_per_eval: int = 0,
    normalize_observations: bool = False,
    normalize_observations_std_eps: float = 0.0,
    normalize_observations_mode: str = "welford",
    reward_scaling: float = 1.0,
    clipping_epsilon: float = 0.3,
    clipping_epsilon_value: float | None = None,
    gae_lambda: float = 0.95,
    max_grad_norm: Optional[float] = None,
    normalize_advantage: bool = True,
    vf_loss_coefficient: float = 0.5,
    bootstrap_on_timeout: bool = False,
    desired_kl: float = 0.01,
    learning_rate_schedule: Optional[
        Union[str, ppo_optimizer.LRSchedule]
    ] = None,
    network_factory: types.NetworkFactory[
        recurrent_networks.RecurrentPPONetworks
    ] = recurrent_networks.make_recurrent_ppo_networks,
    seed: int = 0,
    use_pmap_on_reset: bool = True,
    # eval
    num_evals: int = 1,
    eval_env: Optional[envs.Env] = None,
    num_eval_envs: int = 128,
    deterministic_eval: bool = False,
    # training metrics
    log_training_metrics: bool = False,
    training_metrics_steps: Optional[int] = None,
    # callbacks
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    # checkpointing
    save_checkpoint_path: Optional[str] = None,
    restore_checkpoint_path: Optional[str] = None,
    restore_params: Optional[Any] = None,
    restore_value_fn: bool = True,
    run_evals: bool = True,
):
  """Recurrent PPO training."""
  assert batch_size * num_minibatches % num_envs == 0
  if unroll_length % bptt_length != 0:
    raise ValueError('unroll_length must be divisible by bptt_length')

  xt = time.time()

  process_count = jax.process_count()
  process_id = jax.process_index()
  local_device_count = jax.local_device_count()
  local_devices_to_use = (
      min(local_device_count, max_devices_per_host)
      if max_devices_per_host
      else local_device_count
  )
  logging.info(
      'Device count: %d, process count: %d (id %d), local device count: %d, '
      'devices to be used count: %d',
      jax.device_count(),
      process_count,
      process_id,
      local_device_count,
      local_devices_to_use,
  )
  device_count = local_devices_to_use * process_count

  env_step_per_training_step = (
      batch_size * unroll_length * num_minibatches * action_repeat
  )
  num_evals_after_init = max(num_evals - 1, 1)
  num_training_steps_per_epoch = np.ceil(
      num_timesteps
      / (
          num_evals_after_init
          * env_step_per_training_step
          * max(num_resets_per_eval, 1)
      )
  ).astype(int)

  key = jax.random.PRNGKey(seed)
  global_key, local_key = jax.random.split(key)
  del key
  local_key = jax.random.fold_in(local_key, process_id)
  local_key, key_env, eval_key = jax.random.split(local_key, 3)
  key_policy = global_key
  del global_key

  assert num_envs % device_count == 0

  env = _maybe_wrap_env(
      environment,
      wrap_env,
      num_envs,
      episode_length,
      action_repeat,
      device_count,
      key_env,
      wrap_env_fn,
      randomization_fn,
  )

  def reset_fn_donated_env_state(env_state_donated, key_envs):
    return env.reset(key_envs)

  key_envs = jax.random.split(key_env, num_envs // process_count)
  key_envs = jnp.reshape(
      key_envs, (local_devices_to_use, -1) + key_envs.shape[1:]
  )
  if local_devices_to_use > 1 or use_pmap_on_reset:
    reset_fn_ = jax.pmap(env.reset, axis_name=_PMAP_AXIS_NAME)
    env_state = reset_fn_(key_envs)
    reset_fn = jax.pmap(
        reset_fn_donated_env_state,
        axis_name=_PMAP_AXIS_NAME,
        donate_argnums=(0,),
    )
  else:
    reset_fn_ = jax.jit(jax.vmap(env.reset))
    env_state = reset_fn_(key_envs)
    reset_fn = jax.jit(
        reset_fn_donated_env_state, donate_argnums=(0,), keep_unused=True
    )

  obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)
  normalize = lambda x, y: x
  if normalize_observations:
    normalize = running_statistics.normalize
  ppo_network = network_factory(
      obs_shape, env.action_size, preprocess_observations_fn=normalize
  )
  make_policy = recurrent_networks.make_inference_fn(
      ppo_network,
      compute_value=True,
  )

  base_optimizer = optax.adam(learning_rate=learning_rate)
  lr_schedule = learning_rate_schedule or ppo_optimizer.LRSchedule.NONE
  lr_schedule = ppo_optimizer.LRSchedule(lr_schedule)
  lr_is_adaptive_kl = lr_schedule == ppo_optimizer.LRSchedule.ADAPTIVE_KL
  if lr_is_adaptive_kl:
    base_optimizer = optax.inject_hyperparams(optax.adam)(
        learning_rate=learning_rate
    )
  if max_grad_norm is not None:
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        base_optimizer,
    )
  else:
    optimizer = base_optimizer

  loss_fn = functools.partial(
      recurrent_losses.compute_recurrent_ppo_loss,
      ppo_network=ppo_network,
      entropy_cost=entropy_cost,
      clipping_epsilon=clipping_epsilon,
      normalize_advantage=normalize_advantage,
      vf_coefficient=vf_loss_coefficient,
      clipping_epsilon_value=clipping_epsilon_value,
  )

  loss_and_pgrad_fn = gradients.loss_and_pgrad(
      loss_fn, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
  )

  steps_between_logging = training_metrics_steps or env_step_per_training_step
  metrics_aggregator = metric_logger.EpisodeMetricsLogger(
      steps_between_logging=steps_between_logging,
      progress_fn=progress_fn,
  )

  segments_per_unroll = unroll_length // bptt_length

  def minibatch_step(
      carry,
      data: types.Transition,
      normalizer_params: running_statistics.RunningStatisticsState,
  ):
    optimizer_state, params, key = carry
    key, key_loss = jax.random.split(key)
    (_, metrics), grads = loss_and_pgrad_fn(
        params, normalizer_params, data, key_loss
    )

    if lr_is_adaptive_kl:
      kl_mean = metrics['kl_mean']
      kl_mean = jax.lax.pmean(kl_mean, axis_name=_PMAP_AXIS_NAME)
      optimizer_state, lr = ppo_optimizer.adaptive_kl_learning_rate(
          optimizer_state, kl_mean, desired_kl
      )
    else:
      lr = jnp.array(learning_rate)
    metrics['learning_rate'] = lr

    params_update, optimizer_state = optimizer.update(grads, optimizer_state)
    params = optax.apply_updates(params, params_update)

    return (optimizer_state, params, key), metrics

  def sgd_step(
      carry,
      unused_t,
      data: types.Transition,
      normalizer_params: running_statistics.RunningStatisticsState,
  ):
    optimizer_state, params, key = carry
    key, key_perm, key_grad = jax.random.split(key, 3)

    def convert_data(x: jnp.ndarray):
      x = jax.random.permutation(key_perm, x)
      x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
      return x

    shuffled_data = jax.tree_util.tree_map(convert_data, data)
    (optimizer_state, params, _), metrics = jax.lax.scan(
        functools.partial(minibatch_step, normalizer_params=normalizer_params),
        (optimizer_state, params, key_grad),
        shuffled_data,
        length=num_minibatches,
    )

    return (optimizer_state, params, key), metrics

  def training_step(
      carry: Tuple[
          TrainingState, envs.State, recurrent_networks.RecurrentState, PRNGKey
      ],
      unused_t,
  ):
    training_state, state, core_state, key = carry
    key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

    policy = make_policy(
        (training_state.normalizer_params, training_state.params.model)
    )

    def f(carry, unused_t):
      current_state, current_core_state, current_key = carry
      current_key, next_key = jax.random.split(current_key)
      extra_fields = ['truncation', 'episode_metrics', 'episode_done']
      if bootstrap_on_timeout:
        extra_fields.append('time_out')
      next_state, next_core_state, data, core_history = (
          _recurrent_generate_unroll(
              env,
              current_state,
              policy,
              current_key,
              current_core_state,
              unroll_length,
              mask_fn=ppo_network.mask_state_fn,
              extra_fields=tuple(extra_fields),
          )
      )
      return (next_state, next_core_state, next_key), (data, core_history, next_core_state)

    (state, core_state, _), (data, core_history, final_core) = jax.lax.scan(
        f,
        (state, core_state, key_generate_unroll),
        (),
        length=batch_size * num_minibatches // num_envs,
    )
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
    data = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
    )
    core_history = jax.tree_util.tree_map(
        lambda x: jnp.swapaxes(x, 1, 2), core_history
    )
    core_history = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), core_history
    )
    final_core = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), final_core
    )

    truncation = data.extras['state_extras']['truncation']
    discounts = data.discount
    if bootstrap_on_timeout:
      time_out = data.extras['state_extras']['time_out']
      value = data.extras['policy_extras']['value']
      data = types.Transition(
          observation=data.observation,
          action=data.action,
          reward=data.reward + discounting * time_out * value,
          discount=data.discount,
          next_observation=data.next_observation,
          extras=data.extras,
      )

    terminal_obs = jax.tree_util.tree_map(lambda x: x[:, -1], data.next_observation)
    bootstrap_state = final_core
    bootstrap_value = ppo_network.apply_fn(
        training_state.normalizer_params,
        training_state.params.model,
        terminal_obs,
        bootstrap_state,
    )[1]

    rewards = data.reward * reward_scaling
    termination = (1 - discounts) * (1 - truncation)
    vs, advantages = ppo_losses.compute_gae(
        truncation=jnp.swapaxes(truncation, 0, 1),
        termination=jnp.swapaxes(termination, 0, 1),
        rewards=jnp.swapaxes(rewards, 0, 1),
        values=jnp.swapaxes(data.extras['policy_extras']['value'], 0, 1),
        bootstrap_value=bootstrap_value,
        lambda_=gae_lambda,
        discount=discounting,
    )
    vs = jnp.swapaxes(vs, 0, 1)
    advantages = jnp.swapaxes(advantages, 0, 1)

    # Attach advantages/values to extras.
    data = types.Transition(
        observation=data.observation,
        action=data.action,
        reward=data.reward,
        discount=data.discount,
        next_observation=data.next_observation,
        extras={
            **data.extras,
            'advantages': advantages,
            'target_values': vs,
        },
    )

    # Build per-segment dataset.
    def reshape_time(x: jnp.ndarray):
      x = jnp.reshape(
          x, (x.shape[0], segments_per_unroll, bptt_length) + x.shape[2:]
      )
      return jnp.reshape(x, (-1,) + x.shape[2:])

    segmented_data = jax.tree_util.tree_map(reshape_time, data)
    initial_states = jax.tree_util.tree_map(
        lambda s: jax.lax.stop_gradient(
            jnp.reshape(
                jnp.reshape(
                    s, (s.shape[0], segments_per_unroll, bptt_length) + s.shape[2:]
                )[:, :, 0],
                (-1,) + s.shape[2:],
            )
        ),
        core_history,
    )

    segmented_data = types.Transition(
        observation=segmented_data.observation,
        action=segmented_data.action,
        reward=segmented_data.reward,
        discount=segmented_data.discount,
        next_observation=segmented_data.next_observation,
        extras={
            **segmented_data.extras,
            'initial_core_state': initial_states,
        },
    )

    normalizer_params = training_state.normalizer_params
    if not lr_is_adaptive_kl:
      normalizer_params = running_statistics.update(
          normalizer_params,
          _remove_pixels(segmented_data.observation),
          pmap_axis_name=_PMAP_AXIS_NAME,
      )

    (optimizer_state, params, _), metrics = jax.lax.scan(
        functools.partial(
            sgd_step, data=segmented_data, normalizer_params=normalizer_params
        ),
        (training_state.optimizer_state, training_state.params, key_sgd),
        (),
        length=num_updates_per_batch,
    )

    if lr_is_adaptive_kl:
      normalizer_params = running_statistics.update(
          normalizer_params,
          _remove_pixels(segmented_data.observation),
          pmap_axis_name=_PMAP_AXIS_NAME,
      )

    new_training_state = TrainingState(
      optimizer_state=optimizer_state,
      params=params,
      normalizer_params=normalizer_params,
      env_steps=training_state.env_steps + env_step_per_training_step,
    )

    if log_training_metrics:
      jax.debug.callback(
          metrics_aggregator.update_episode_metrics,
          data.extras['state_extras']['episode_metrics'],
          data.extras['state_extras']['episode_done'],
          metrics,
      )

    return (new_training_state, state, core_state, new_key), metrics

  def training_epoch(
      training_state: TrainingState,
      state: envs.State,
      core_state: recurrent_networks.RecurrentState,
      key: PRNGKey,
  ):
    (training_state, state, core_state, _), loss_metrics = jax.lax.scan(
        training_step,
        (training_state, state, core_state, key),
        (),
        length=num_training_steps_per_epoch,
    )
    loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
    return training_state, state, core_state, loss_metrics

  training_epoch = jax.pmap(
      training_epoch,
      axis_name=_PMAP_AXIS_NAME,
      donate_argnums=(
          0,
          1,
          2,
      ),
  )

  def training_epoch_with_timing(
      training_state: TrainingState,
      env_state: envs.State,
      core_state: recurrent_networks.RecurrentState,
      key: PRNGKey,
  ):
    nonlocal training_walltime
    t = time.time()
    training_state, env_state, core_state = _strip_weak_type(
        (training_state, env_state, core_state)
    )
    result = training_epoch(training_state, env_state, core_state, key)
    training_state, env_state, core_state, metrics = _strip_weak_type(result)

    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

    epoch_training_time = time.time() - t
    training_walltime += epoch_training_time
    sps = (
        num_training_steps_per_epoch
        * env_step_per_training_step
        * max(num_resets_per_eval, 1)
    ) / epoch_training_time
    metrics = {
        'training/sps': sps,
        'training/walltime': training_walltime,
        **{f'training/{name}': value for name, value in metrics.items()},
    }
    return training_state, env_state, core_state, metrics  # pytype: disable=bad-return-type  # jax-ndarray

  init_model_params = recurrent_losses.RecurrentPPONetworkParams(
      model=ppo_network.init_fn(key_policy)
  )
  obs_specs = jax.tree_util.tree_map(
      lambda x: specs.Array(x.shape[-1:], jnp.dtype('float32')), env_state.obs
  )
  training_state = TrainingState(
      optimizer_state=optimizer.init(init_model_params),  # pytype: disable=wrong-arg-types  # numpy-scalars
      params=init_model_params,
      normalizer_params=running_statistics.init_state(
          _remove_pixels(obs_specs),
          std_eps=normalize_observations_std_eps,
          mode=normalize_observations_mode,
      ),
      env_steps=types.UInt64(hi=0, lo=0),
  )

  if restore_checkpoint_path is not None:
    params = checkpoint.load(restore_checkpoint_path)
    value_params = params[1] if restore_value_fn else init_model_params.model
    training_state = training_state.replace(
        normalizer_params=params[0],
        params=training_state.params.replace(model=value_params),
    )

  if restore_params is not None:
    logging.info('Restoring TrainingState from `restore_params`.')
    value_params = restore_params[1] if restore_value_fn else init_model_params.model
    training_state = training_state.replace(
        normalizer_params=restore_params[0],
        params=training_state.params.replace(model=value_params),
    )

  if num_timesteps == 0:
    return (
        make_policy,
        (
            training_state.normalizer_params,
            training_state.params.model,
        ),
        {},
    )

  sample_obs = jax.tree_util.tree_leaves(env_state.obs)[0]
  initial_core_state = ppo_network.initial_state_fn(sample_obs.shape[1])
  training_state = jax.device_put_replicated(
      training_state, jax.local_devices()[:local_devices_to_use]
  )
  core_state = jax.device_put_replicated(
      initial_core_state, jax.local_devices()[:local_devices_to_use]
  )

  eval_env = _maybe_wrap_env(
      eval_env or environment,
      wrap_env,
      num_eval_envs,
      episode_length,
      action_repeat,
      device_count=1,
      key_env=eval_key,
      wrap_env_fn=wrap_env_fn,
      randomization_fn=randomization_fn,
  )
  evaluator = RecurrentEvaluator(
      eval_env,
      functools.partial(make_policy, deterministic=deterministic_eval),
      num_eval_envs=num_eval_envs,
      episode_length=episode_length,
      action_repeat=action_repeat,
      key=eval_key,
      ppo_network=ppo_network,
  )

  training_metrics = {}
  training_walltime = 0
  current_step = 0

  metrics = {}
  if process_id == 0 and num_evals > 1 and run_evals:
    metrics = evaluator.run_evaluation(
        _unpmap((
            training_state.normalizer_params,
            training_state.params.model,
        )),
        training_metrics={},
    )
    logging.info(metrics)
    progress_fn(0, metrics)

  params = _unpmap((
      training_state.normalizer_params,
      training_state.params.model,
  ))
  policy_params_fn(current_step, make_policy, params)

  for it in range(num_evals_after_init):
    logging.info('starting iteration %s %s', it, time.time() - xt)

    for _ in range(max(num_resets_per_eval, 1)):
      epoch_key, local_key = jax.random.split(local_key)
      epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
      (training_state, env_state, core_state, training_metrics) = (
          training_epoch_with_timing(training_state, env_state, core_state, epoch_keys)
      )
      current_step = int(_unpmap(training_state.env_steps))

      key_envs = jax.vmap(
          lambda x, s: jax.random.split(x[0], s), in_axes=(0, None)
      )(key_envs, key_envs.shape[1])
      if num_resets_per_eval > 0:
        env_state = reset_fn(env_state, key_envs)
        core_state = jax.device_put_replicated(
            ppo_network.initial_state_fn(core_state.hidden.shape[1]),
            jax.local_devices()[:local_devices_to_use],
        )

    if process_id != 0:
      continue

    params = _unpmap((
        training_state.normalizer_params,
        training_state.params.model,
    ))

    policy_params_fn(current_step, make_policy, params)

    if save_checkpoint_path is not None:
      ckpt_config = checkpoint.network_config(
          observation_size=obs_shape,
          action_size=env.action_size,
          normalize_observations=normalize_observations,
          network_factory=network_factory,
      )
      checkpoint.save(
          save_checkpoint_path, current_step, params, ckpt_config
      )

    if num_evals > 0:
      metrics = training_metrics
      if run_evals:
        metrics = evaluator.run_evaluation(
            params,
            training_metrics,
        )
      logging.info(metrics)
      progress_fn(current_step, metrics)

  total_steps = current_step
  if not total_steps >= num_timesteps:
    raise AssertionError(
        f'Total steps {total_steps} is less than `num_timesteps`='
        f' {num_timesteps}.'
    )

  pmap.assert_is_replicated(training_state)
  params = _unpmap((
      training_state.normalizer_params,
      training_state.params.model,
  ))
  logging.info('total steps: %s', total_steps)
  pmap.synchronize_hosts()
  return (make_policy, params, metrics)
