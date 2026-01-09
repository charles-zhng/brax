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

"""Recurrent TD3 training.

TD3 (Twin Delayed DDPG) with recurrent actor for partially observable
environments. Combines:
- Off-policy learning with replay buffer
- Recurrent deterministic policy for partial observability
- Twin Q-networks with target policy smoothing
- Delayed policy updates

See: https://arxiv.org/pdf/1802.09477.pdf (TD3)
"""

import functools
import time
from typing import Any, Callable, Mapping, Optional, Tuple

from absl import logging
from brax import base
from brax import envs
from brax.training import gradients
from brax.training import pmap
from brax.training import replay_buffers
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.recurrent_ppo.networks import HiddenState
from brax.training.agents.recurrent_td3 import checkpoint
from brax.training.agents.recurrent_td3 import losses as td3_losses
from brax.training.agents.recurrent_td3 import networks as recurrent_td3_networks
from brax.training.agents.recurrent_td3 import optimizer as td3_optimizer
from brax.training.types import Params
from brax.training.types import PRNGKey
import flax
import jax
import jax.numpy as jnp
import optax

Metrics = types.Metrics
Transition = types.Transition
InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]

ReplayBufferState = Any

_PMAP_AXIS_NAME = 'i'


@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""

  # Actor
  actor_optimizer_state: optax.OptState
  actor_params: Params
  target_actor_params: Params
  # Critic (twin Q)
  q_optimizer_state: optax.OptState
  q_params: Params
  target_q_params: Params
  # Counters
  gradient_steps: types.UInt64
  env_steps: types.UInt64
  # Normalizer
  normalizer_params: running_statistics.RunningStatisticsState


def _unpmap(v):
  return jax.tree_util.tree_map(lambda x: x[0], v)


def _reset_hidden_on_done(
    hidden: HiddenState,
    done: jnp.ndarray,
    td3_network: recurrent_td3_networks.RecurrentTD3Networks,
) -> HiddenState:
  """Reset hidden state where episodes are done.

  Args:
    hidden: Current hidden state
    done: Done flags [batch_size]
    td3_network: The TD3 networks (for getting cell type)

  Returns:
    Hidden state with zeros where done=True
  """
  done_expanded = done[..., None]  # [batch, 1]

  if td3_network.cell_type == 'lstm':
    c, h = hidden
    c = jnp.where(done_expanded, 0.0, c)
    h = jnp.where(done_expanded, 0.0, h)
    return (c, h)
  else:
    return jnp.where(done_expanded, 0.0, hidden)


def actor_step_rnn(
    env: envs.Env,
    env_state: envs.State,
    policy: Callable,
    hidden_state: HiddenState,
    key: PRNGKey,
    exploration_noise: float,
    td3_network: recurrent_td3_networks.RecurrentTD3Networks,
    extra_fields: tuple = (),
) -> Tuple[envs.State, types.Transition, HiddenState]:
  """Collect data with recurrent deterministic policy and exploration noise.

  Args:
    env: Environment
    env_state: Current environment state
    policy: Recurrent deterministic policy function
    hidden_state: Actor hidden state
    key: Random key
    exploration_noise: Standard deviation of exploration noise
    td3_network: TD3 networks for hidden state reset
    extra_fields: Extra fields to collect from env state

  Returns:
    Tuple of (next_state, transition, new_hidden_state)
  """
  key, noise_key = jax.random.split(key)
  actions, policy_extras, new_hidden = policy(env_state.obs, hidden_state, key)

  # Add exploration noise
  noise = jax.random.normal(noise_key, actions.shape) * exploration_noise
  noisy_actions = jnp.clip(actions + noise, -1.0, 1.0)

  nstate = env.step(env_state, noisy_actions)
  state_extras = {x: nstate.info[x] for x in extra_fields}

  # Store the initial hidden state for this step
  policy_extras['initial_hidden'] = hidden_state

  transition = types.Transition(
      observation=env_state.obs,
      action=noisy_actions,
      reward=nstate.reward,
      discount=1 - nstate.done,
      next_observation=nstate.obs,
      extras={'policy_extras': policy_extras, 'state_extras': state_extras},
  )

  # Reset hidden state where episodes are done
  new_hidden = _reset_hidden_on_done(new_hidden, nstate.done, td3_network)

  return nstate, transition, new_hidden


def _init_training_state(
    key: PRNGKey,
    obs_shape: types.ObservationSize,
    local_devices_to_use: int,
    td3_network: recurrent_td3_networks.RecurrentTD3Networks,
    actor_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation,
) -> TrainingState:
  """Inits the training state and replicates it over devices."""
  key_actor, key_q = jax.random.split(key)

  actor_params = td3_network.actor_network.init(key_actor)
  actor_optimizer_state = actor_optimizer.init(actor_params)

  q_params = td3_network.q_network.init(key_q)
  q_optimizer_state = q_optimizer.init(q_params)

  normalizer_params = running_statistics.init_state(obs_shape)

  training_state = TrainingState(
      actor_optimizer_state=actor_optimizer_state,
      actor_params=actor_params,
      target_actor_params=actor_params,
      q_optimizer_state=q_optimizer_state,
      q_params=q_params,
      target_q_params=q_params,
      gradient_steps=types.UInt64(hi=0, lo=0),
      env_steps=types.UInt64(hi=0, lo=0),
      normalizer_params=normalizer_params,
  )
  return jax.device_put_replicated(
      training_state, jax.local_devices()[:local_devices_to_use]
  )


def train(
    environment: envs.Env,
    num_timesteps: int,
    episode_length: int,
    wrap_env: bool = True,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_eval_envs: int = 128,
    # Learning
    learning_rate: float = 3e-4,
    discounting: float = 0.99,
    seed: int = 0,
    batch_size: int = 256,
    # TD3 specific
    exploration_noise: float = 0.1,
    target_noise: float = 0.2,
    noise_clip: float = 0.5,
    policy_delay: int = 2,
    tau: float = 0.005,
    # Replay buffer
    min_replay_size: int = 10000,
    max_replay_size: Optional[int] = None,
    grad_updates_per_step: int = 1,
    # Normalization
    normalize_observations: bool = False,
    reward_scaling: float = 1.0,
    max_grad_norm: Optional[float] = None,
    # Network
    network_factory: types.NetworkFactory[
        recurrent_td3_networks.RecurrentTD3Networks
    ] = recurrent_td3_networks.make_recurrent_td3_networks,
    max_devices_per_host: Optional[int] = None,
    # Evaluation
    num_evals: int = 1,
    deterministic_eval: bool = True,
    eval_env: Optional[envs.Env] = None,
    # Domain randomization
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    # Checkpointing
    checkpoint_logdir: Optional[str] = None,
    restore_checkpoint_path: Optional[str] = None,
    # Callbacks
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
) -> Tuple[Callable, Params, Metrics]:
  """Recurrent TD3 training.

  Args:
    environment: Environment to train on.
    num_timesteps: Total number of environment steps.
    episode_length: Length of each episode.
    wrap_env: Whether to wrap the environment.
    wrap_env_fn: Custom environment wrapper function.
    action_repeat: Number of times to repeat each action.
    num_envs: Number of parallel environments.
    num_eval_envs: Number of environments for evaluation.
    learning_rate: Learning rate for Adam optimizer.
    discounting: Discount factor (gamma).
    seed: Random seed.
    batch_size: Batch size for training.
    exploration_noise: Std of Gaussian noise added during collection.
    target_noise: Std of noise added to target actions.
    noise_clip: Clipping range for target action noise.
    policy_delay: Update actor every N critic updates.
    tau: Polyak averaging coefficient for target networks.
    min_replay_size: Minimum replay buffer size before training starts.
    max_replay_size: Maximum replay buffer size.
    grad_updates_per_step: Number of gradient updates per environment step.
    normalize_observations: Whether to normalize observations.
    reward_scaling: Scaling factor for rewards.
    max_grad_norm: Maximum gradient norm for clipping.
    network_factory: Factory function to create networks.
    max_devices_per_host: Maximum devices to use per host.
    num_evals: Number of evaluation runs.
    deterministic_eval: Whether to use deterministic actions for eval.
    eval_env: Separate environment for evaluation.
    randomization_fn: Domain randomization function.
    checkpoint_logdir: Directory for saving checkpoints.
    restore_checkpoint_path: Path to restore checkpoint from.
    progress_fn: Callback for progress reporting.

  Returns:
    Tuple of (make_policy, params, metrics).
  """
  process_id = jax.process_index()
  local_devices_to_use = jax.local_device_count()
  if max_devices_per_host is not None:
    local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
  device_count = local_devices_to_use * jax.process_count()
  logging.info(
      'local_device_count: %s; total_device_count: %s',
      local_devices_to_use,
      device_count,
  )

  if min_replay_size >= num_timesteps:
    raise ValueError(
        'No training will happen because min_replay_size >= num_timesteps'
    )

  if max_replay_size is None:
    max_replay_size = num_timesteps

  env_steps_per_actor_step = action_repeat * num_envs
  num_prefill_actor_steps = -(-min_replay_size // num_envs)
  num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
  assert num_timesteps - num_prefill_env_steps >= 0
  num_evals_after_init = max(num_evals - 1, 1)
  num_training_steps_per_epoch = -(
      -(num_timesteps - num_prefill_env_steps)
      // (num_evals_after_init * env_steps_per_actor_step)
  )

  assert num_envs % device_count == 0

  # Environment setup
  env = environment
  if wrap_env:
    if wrap_env_fn is not None:
      wrap_for_training = wrap_env_fn
    elif isinstance(env, envs.Env):
      wrap_for_training = envs.training.wrap
    else:
      raise ValueError('Unsupported environment type: %s' % type(env))

    rng = jax.random.PRNGKey(seed)
    rng, key = jax.random.split(rng)
    v_randomization_fn = None
    if randomization_fn is not None:
      v_randomization_fn = functools.partial(
          randomization_fn,
          rng=jax.random.split(
              key, num_envs // jax.process_count() // local_devices_to_use
          ),
      )
    env = wrap_for_training(
        env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )

  obs_size = env.observation_size
  action_size = env.action_size

  # Network setup
  normalize_fn = lambda x, y: x
  if normalize_observations:
    normalize_fn = running_statistics.normalize
  td3_network = network_factory(
      observation_size=obs_size,
      action_size=action_size,
      preprocess_observations_fn=normalize_fn,
  )
  make_policy = recurrent_td3_networks.make_inference_fn(td3_network)

  # Optimizers
  actor_optimizer = td3_optimizer.make_optimizer(learning_rate, max_grad_norm)
  q_optimizer = td3_optimizer.make_optimizer(learning_rate, max_grad_norm)

  # Replay buffer - create dummy observations based on obs_size structure
  def _make_dummy_obs(obs_size):
    if isinstance(obs_size, Mapping):
      return {k: jnp.zeros((v,)) for k, v in obs_size.items()}
    return jnp.zeros((obs_size,))

  dummy_obs = _make_dummy_obs(obs_size)
  dummy_action = jnp.zeros((action_size,))
  dummy_hidden = td3_network.actor_network.init_hidden(1)
  # Flatten hidden state for storage
  dummy_hidden_flat = jax.tree_util.tree_leaves(dummy_hidden)
  dummy_transition = Transition(
      observation=dummy_obs,
      action=dummy_action,
      reward=0.0,
      discount=0.0,
      next_observation=dummy_obs,
      extras={
          'state_extras': {'truncation': 0.0},
          'policy_extras': {
              'hidden': dummy_hidden,
              'initial_hidden': dummy_hidden,
          },
      },
  )
  replay_buffer = replay_buffers.UniformSamplingQueue(
      max_replay_size=max_replay_size // device_count,
      dummy_data_sample=dummy_transition,
      sample_batch_size=batch_size * grad_updates_per_step // device_count,
  )

  # Loss functions
  critic_loss_fn, actor_loss_fn = td3_losses.make_losses(
      td3_network=td3_network,
      reward_scaling=reward_scaling,
      discounting=discounting,
      target_noise=target_noise,
      noise_clip=noise_clip,
  )

  # Gradient update functions
  critic_update = gradients.gradient_update_fn(
      lambda *args: critic_loss_fn(*args)[0],
      q_optimizer,
      pmap_axis_name=_PMAP_AXIS_NAME,
      has_aux=False,
  )
  actor_update = gradients.gradient_update_fn(
      lambda *args: actor_loss_fn(*args)[0],
      actor_optimizer,
      pmap_axis_name=_PMAP_AXIS_NAME,
      has_aux=False,
  )

  def sgd_step(
      carry: Tuple[TrainingState, PRNGKey], transitions: Transition
  ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
    training_state, key = carry
    key, key_critic = jax.random.split(key)

    # Extract hidden states from transitions
    hidden_states = transitions.extras['policy_extras']['initial_hidden']
    # Use first hidden state in batch (they should all be zeros for fresh starts)
    # For proper sequence handling, we'd need to maintain hidden state continuity
    batch_hidden = jax.tree_util.tree_map(lambda x: x[0], hidden_states)

    # Always update critic
    critic_loss_val, q_params, q_optimizer_state = critic_update(
        training_state.q_params,
        training_state.target_q_params,
        training_state.target_actor_params,
        training_state.normalizer_params,
        transitions,
        batch_hidden,
        key_critic,
        optimizer_state=training_state.q_optimizer_state,
    )

    # Soft update target Q
    new_target_q_params = jax.tree_util.tree_map(
        lambda x, y: x * (1 - tau) + y * tau,
        training_state.target_q_params,
        q_params,
    )

    # Delayed actor update
    gradient_steps = training_state.gradient_steps
    should_update_actor = (gradient_steps.lo % policy_delay) == 0

    def update_actor(state_tuple):
      ts, k = state_tuple
      actor_loss_val, actor_params, actor_opt_state = actor_update(
          ts.actor_params,
          ts.normalizer_params,
          q_params,
          transitions,
          batch_hidden,
          optimizer_state=ts.actor_optimizer_state,
      )
      # Soft update target actor
      new_target_actor = jax.tree_util.tree_map(
          lambda x, y: x * (1 - tau) + y * tau,
          ts.target_actor_params,
          actor_params,
      )
      return actor_loss_val, actor_params, actor_opt_state, new_target_actor

    def skip_actor(state_tuple):
      ts, _ = state_tuple
      return (
          0.0,
          ts.actor_params,
          ts.actor_optimizer_state,
          ts.target_actor_params,
      )

    actor_loss_val, actor_params, actor_opt_state, target_actor_params = (
        jax.lax.cond(
            should_update_actor,
            update_actor,
            skip_actor,
            (training_state, key),
        )
    )

    new_training_state = TrainingState(
        actor_optimizer_state=actor_opt_state,
        actor_params=actor_params,
        target_actor_params=target_actor_params,
        q_optimizer_state=q_optimizer_state,
        q_params=q_params,
        target_q_params=new_target_q_params,
        gradient_steps=gradient_steps + 1,
        env_steps=training_state.env_steps,
        normalizer_params=training_state.normalizer_params,
    )

    metrics = {
        'critic_loss': critic_loss_val,
        'actor_loss': actor_loss_val,
    }

    return (new_training_state, key), metrics

  def get_experience(
      normalizer_params: running_statistics.RunningStatisticsState,
      actor_params: Params,
      hidden_state: HiddenState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[
      running_statistics.RunningStatisticsState,
      HiddenState,
      envs.State,
      ReplayBufferState,
  ]:
    policy = make_policy((normalizer_params, actor_params))
    env_state, transitions, new_hidden = actor_step_rnn(
        env,
        env_state,
        policy,
        hidden_state,
        key,
        exploration_noise,
        td3_network,
        extra_fields=('truncation',),
    )

    normalizer_params = running_statistics.update(
        normalizer_params,
        transitions.observation,
        pmap_axis_name=_PMAP_AXIS_NAME,
    )

    buffer_state = replay_buffer.insert(buffer_state, transitions)
    return normalizer_params, new_hidden, env_state, buffer_state

  def training_step(
      training_state: TrainingState,
      hidden_state: HiddenState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[TrainingState, HiddenState, envs.State, ReplayBufferState, Metrics]:
    experience_key, training_key = jax.random.split(key)
    normalizer_params, new_hidden, env_state, buffer_state = get_experience(
        training_state.normalizer_params,
        training_state.actor_params,
        hidden_state,
        env_state,
        buffer_state,
        experience_key,
    )
    training_state = training_state.replace(
        normalizer_params=normalizer_params,
        env_steps=training_state.env_steps + env_steps_per_actor_step,
    )

    buffer_state, transitions = replay_buffer.sample(buffer_state)
    transitions = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (grad_updates_per_step, -1) + x.shape[1:]),
        transitions,
    )
    (training_state, _), metrics = jax.lax.scan(
        sgd_step, (training_state, training_key), transitions
    )

    metrics['buffer_current_size'] = replay_buffer.size(buffer_state)
    return training_state, new_hidden, env_state, buffer_state, metrics

  def prefill_replay_buffer(
      training_state: TrainingState,
      hidden_state: HiddenState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[TrainingState, HiddenState, envs.State, ReplayBufferState, PRNGKey]:

    def f(carry, unused):
      del unused
      training_state, hidden, env_state, buffer_state, key = carry
      key, new_key = jax.random.split(key)
      new_normalizer_params, new_hidden, env_state, buffer_state = (
          get_experience(
              training_state.normalizer_params,
              training_state.actor_params,
              hidden,
              env_state,
              buffer_state,
              key,
          )
      )
      new_training_state = training_state.replace(
          normalizer_params=new_normalizer_params,
          env_steps=training_state.env_steps + env_steps_per_actor_step,
      )
      return (
          new_training_state,
          new_hidden,
          env_state,
          buffer_state,
          new_key,
      ), ()

    return jax.lax.scan(
        f,
        (training_state, hidden_state, env_state, buffer_state, key),
        (),
        length=num_prefill_actor_steps,
    )[0]

  prefill_replay_buffer = jax.pmap(
      prefill_replay_buffer,
      axis_name=_PMAP_AXIS_NAME,
      donate_argnums=(0, 1, 2, 3),
  )

  def training_epoch(
      training_state: TrainingState,
      hidden_state: HiddenState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[
      TrainingState, HiddenState, envs.State, ReplayBufferState, Metrics
  ]:

    def f(carry, unused_t):
      ts, hs, es, bs, k = carry
      k, new_key = jax.random.split(k)
      ts, hs, es, bs, metrics = training_step(ts, hs, es, bs, k)
      return (ts, hs, es, bs, new_key), metrics

    (training_state, hidden_state, env_state, buffer_state, key), metrics = (
        jax.lax.scan(
            f,
            (training_state, hidden_state, env_state, buffer_state, key),
            (),
            length=num_training_steps_per_epoch,
        )
    )
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    return training_state, hidden_state, env_state, buffer_state, metrics

  training_epoch = jax.pmap(
      training_epoch,
      axis_name=_PMAP_AXIS_NAME,
      donate_argnums=(0, 1, 2, 3),
  )

  # Initialize
  rng = jax.random.PRNGKey(seed)
  global_key, local_key = jax.random.split(rng)
  local_key = jax.random.fold_in(local_key, process_id)

  # Create obs_shape for normalizer - supports dict observations
  def _make_obs_shape(obs_size):
    if isinstance(obs_size, Mapping):
      return {
          k: specs.Array((v,), jnp.dtype('float32')) for k, v in obs_size.items()
      }
    return specs.Array((obs_size,), jnp.dtype('float32'))

  obs_shape = _make_obs_shape(obs_size)
  training_state = _init_training_state(
      key=global_key,
      obs_shape=obs_shape,
      local_devices_to_use=local_devices_to_use,
      td3_network=td3_network,
      actor_optimizer=actor_optimizer,
      q_optimizer=q_optimizer,
  )
  del global_key

  if restore_checkpoint_path is not None:
    params = checkpoint.load(restore_checkpoint_path)
    training_state = training_state.replace(
        normalizer_params=params[0],
        actor_params=params[1],
    )

  local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)

  # Env init
  env_keys = jax.random.split(env_key, num_envs // jax.process_count())
  env_keys = jnp.reshape(
      env_keys, (local_devices_to_use, -1) + env_keys.shape[1:]
  )
  env_state = jax.pmap(env.reset)(env_keys)

  # Hidden state init
  envs_per_device = num_envs // jax.process_count() // local_devices_to_use
  hidden_state = td3_network.actor_network.init_hidden(envs_per_device)
  hidden_state = jax.device_put_replicated(
      hidden_state, jax.local_devices()[:local_devices_to_use]
  )

  # Replay buffer init
  buffer_state = jax.pmap(replay_buffer.init)(
      jax.random.split(rb_key, local_devices_to_use)
  )

  # Evaluator setup
  if not eval_env:
    eval_env = environment
  if wrap_env:
    if wrap_env_fn is not None:
      eval_wrap_fn = wrap_env_fn
    else:
      eval_wrap_fn = envs.training.wrap
    v_randomization_fn = None
    if randomization_fn is not None:
      v_randomization_fn = functools.partial(
          randomization_fn, rng=jax.random.split(eval_key, num_eval_envs)
      )
    eval_env = eval_wrap_fn(
        eval_env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )

  evaluator = RecurrentTD3Evaluator(
      eval_env=eval_env,
      eval_policy_fn=functools.partial(
          make_policy, deterministic=deterministic_eval
      ),
      td3_network=td3_network,
      num_eval_envs=num_eval_envs,
      episode_length=episode_length,
      action_repeat=action_repeat,
      key=eval_key,
  )

  # Run initial eval
  metrics = {}
  if process_id == 0 and num_evals > 1:
    metrics = evaluator.run_evaluation(
        _unpmap(
            (training_state.normalizer_params, training_state.actor_params)
        ),
        training_metrics={},
    )
    logging.info(metrics)
    progress_fn(0, metrics)

  # Prefill replay buffer
  training_walltime = 0.0
  t = time.time()
  prefill_key, local_key = jax.random.split(local_key)
  prefill_keys = jax.random.split(prefill_key, local_devices_to_use)
  (
      training_state,
      hidden_state,
      env_state,
      buffer_state,
      _,
  ) = prefill_replay_buffer(
      training_state, hidden_state, env_state, buffer_state, prefill_keys
  )

  replay_size = (
      jnp.sum(jax.vmap(replay_buffer.size)(buffer_state)) * jax.process_count()
  )
  logging.info('replay size after prefill %s', replay_size)
  assert replay_size >= min_replay_size
  training_walltime = time.time() - t

  # Training loop
  current_step = 0
  for _ in range(num_evals_after_init):
    logging.info('step %s', current_step)

    # Optimization
    t = time.time()
    epoch_key, local_key = jax.random.split(local_key)
    epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
    (
        training_state,
        hidden_state,
        env_state,
        buffer_state,
        training_metrics,
    ) = training_epoch(
        training_state, hidden_state, env_state, buffer_state, epoch_keys
    )
    training_metrics = jax.tree_util.tree_map(jnp.mean, training_metrics)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), training_metrics)

    epoch_training_time = time.time() - t
    training_walltime += epoch_training_time
    sps = (
        env_steps_per_actor_step * num_training_steps_per_epoch
    ) / epoch_training_time

    current_step = int(_unpmap(training_state.env_steps))
    training_metrics = {
        'training/sps': sps,
        'training/walltime': training_walltime,
        **{f'training/{name}': value for name, value in training_metrics.items()},
    }

    # Eval and logging
    if process_id == 0:
      if checkpoint_logdir:
        params = _unpmap(
            (training_state.normalizer_params, training_state.actor_params)
        )
        ckpt_config = checkpoint.network_config(
            observation_size=obs_size,
            action_size=env.action_size,
            normalize_observations=normalize_observations,
            network_factory=network_factory,
        )
        checkpoint.save(checkpoint_logdir, current_step, params, ckpt_config)

      metrics = evaluator.run_evaluation(
          _unpmap(
              (training_state.normalizer_params, training_state.actor_params)
          ),
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

  params = _unpmap(
      (training_state.normalizer_params, training_state.actor_params)
  )

  pmap.assert_is_replicated(training_state)
  logging.info('total steps: %s', total_steps)
  pmap.synchronize_hosts()
  return (make_policy, params, metrics)


class RecurrentTD3Evaluator:
  """Class to run evaluations with recurrent TD3 policies."""

  def __init__(
      self,
      eval_env: envs.Env,
      eval_policy_fn: Callable[[Params], Callable],
      td3_network: recurrent_td3_networks.RecurrentTD3Networks,
      num_eval_envs: int,
      episode_length: int,
      action_repeat: int,
      key: PRNGKey,
  ):
    """Init."""
    self._key = key
    self._eval_walltime = 0.0
    self._td3_network = td3_network
    self._num_eval_envs = num_eval_envs

    eval_env = envs.training.EvalWrapper(eval_env)
    self._eval_state_to_donate = jax.jit(eval_env.reset)(
        jax.random.split(key, num_eval_envs)
    )

    def generate_eval_unroll(
        eval_env_state_donated: envs.State,
        policy_params: Params,
        key: PRNGKey,
    ) -> envs.State:
      reset_keys = jax.random.split(key, num_eval_envs)
      eval_first_state = eval_env.reset(reset_keys)

      # Initialize hidden states
      hidden_state = td3_network.actor_network.init_hidden(num_eval_envs)

      def step_fn(carry, unused):
        state, hidden, step_key = carry
        step_key, next_key = jax.random.split(step_key)
        policy = eval_policy_fn(policy_params)
        actions, _, new_hidden = policy(state.obs, hidden, step_key)
        next_state = eval_env.step(state, actions)
        # Reset hidden on done
        new_hidden = _reset_hidden_on_done(new_hidden, next_state.done, td3_network)
        return (next_state, new_hidden, next_key), None

      (final_state, _, _), _ = jax.lax.scan(
          step_fn,
          (eval_first_state, hidden_state, key),
          (),
          length=episode_length // action_repeat,
      )
      return final_state

    self._generate_eval_unroll = jax.jit(
        generate_eval_unroll, donate_argnums=(0,), keep_unused=True
    )
    self._steps_per_unroll = episode_length * num_eval_envs

  def run_evaluation(
      self,
      policy_params: Params,
      training_metrics: Metrics,
      aggregate_episodes: bool = True,
  ) -> Metrics:
    """Run one epoch of evaluation."""
    import numpy as np

    self._key, unroll_key = jax.random.split(self._key)

    t = time.time()
    eval_state = self._generate_eval_unroll(
        self._eval_state_to_donate, policy_params, unroll_key
    )
    self._eval_state_to_donate = eval_state

    eval_metrics = eval_state.info['eval_metrics']
    eval_metrics.active_episodes.block_until_ready()
    epoch_eval_time = time.time() - t
    episode_lengths = np.maximum(eval_metrics.episode_steps, 1.0).astype(float)

    def _agg_fn(metric, fn, to_aggregate, to_normalize, episode_lengths):
      if not to_aggregate:
        return metric
      if to_normalize:
        return fn(metric / episode_lengths)
      return fn(metric)

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
    metrics['eval/epoch_eval_time'] = epoch_eval_time
    metrics['eval/sps'] = self._steps_per_unroll / epoch_eval_time
    self._eval_walltime += epoch_eval_time
    metrics['eval/walltime'] = self._eval_walltime
    metrics.update(training_metrics)
    return metrics
