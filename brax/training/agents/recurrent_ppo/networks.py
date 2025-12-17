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

"""Recurrent PPO networks."""

from typing import Any, Callable, Literal, Mapping, Sequence, Tuple

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen
import jax
import jax.numpy as jnp


@flax.struct.dataclass
class RecurrentState:
  """Container for recurrent hidden states."""

  hidden: jax.Array
  cell: jax.Array | None = None


@flax.struct.dataclass
class RecurrentPPONetworks:
  """Recurrent PPO network bundle."""

  apply_fn: Callable[
      [types.Params, types.Params, types.Observation, RecurrentState],
      Tuple[jax.Array, jax.Array, RecurrentState],
  ]
  init_fn: Callable[[jax.Array], types.Params]
  initial_state_fn: Callable[[int], RecurrentState]
  mask_state_fn: Callable[[RecurrentState, jax.Array], RecurrentState]
  parametric_action_distribution: distribution.ParametricDistribution


class VanillaRNNCell(linen.Module):
  """A simple tanh RNN cell."""

  hidden_size: int
  kernel_init: networks.Initializer
  activation: networks.ActivationFn = linen.tanh

  @linen.compact
  def __call__(self, carry: jax.Array, inputs: jax.Array):
    hidden = jnp.concatenate([carry, inputs], axis=-1)
    new_hidden = self.activation(
        linen.Dense(
            self.hidden_size, kernel_init=self.kernel_init, name='rnn_dense'
        )(hidden)
    )
    return new_hidden, new_hidden


class RecurrentActorCritic(linen.Module):
  """Shared recurrent core with separate policy/value heads."""

  core_type: Literal['gru', 'lstm', 'rnn']
  hidden_size: int
  param_size: int
  policy_hidden_layer_sizes: Sequence[int]
  value_hidden_layer_sizes: Sequence[int]
  activation: networks.ActivationFn = linen.swish
  kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
  layer_norm: bool = False
  distribution_type: Literal['normal', 'tanh_normal'] = 'tanh_normal'
  noise_std_type: Literal['scalar', 'log'] = 'scalar'
  init_noise_std: float = 1.0
  state_dependent_std: bool = False

  @linen.compact
  def __call__(
      self, obs: jax.Array, state: RecurrentState
  ) -> Tuple[jax.Array, jax.Array, RecurrentState]:
    feature = networks.MLP(
        layer_sizes=list(self.policy_hidden_layer_sizes),
        activation=self.activation,
        kernel_init=self.kernel_init,
        layer_norm=self.layer_norm,
        activate_final=True,
    )(obs)

    if self.core_type == 'gru':
      rnn_cell = linen.GRUCell(
          name='gru', hidden_size=self.hidden_size, kernel_init=self.kernel_init
      )
      hidden, encoded = rnn_cell(state.hidden, feature)
      new_state = RecurrentState(hidden=hidden, cell=None)
    elif self.core_type == 'lstm':
      lstm_cell = linen.LSTMCell(
          name='lstm', hidden_size=self.hidden_size, kernel_init=self.kernel_init
      )
      carry = (
          state.cell if state.cell is not None else jnp.zeros_like(state.hidden),
          state.hidden,
      )
      (cell, hidden), encoded = lstm_cell(carry, feature)
      new_state = RecurrentState(hidden=hidden, cell=cell)
    elif self.core_type == 'rnn':
      rnn_cell = VanillaRNNCell(
          hidden_size=self.hidden_size,
          kernel_init=self.kernel_init,
          activation=self.activation,
      )
      hidden, encoded = rnn_cell(state.hidden, feature)
      new_state = RecurrentState(hidden=hidden, cell=None)
    else:
      raise ValueError(
          f'Unsupported core_type: {self.core_type}. Expected one of '
          '"gru", "lstm", or "rnn".'
      )

    if self.distribution_type == 'tanh_normal':
      policy_head = networks.MLP(
          layer_sizes=[self.param_size],
          activation=self.activation,
          kernel_init=self.kernel_init,
          layer_norm=self.layer_norm,
      )
      policy_logits = policy_head(encoded)
    else:
      policy_module = networks.PolicyModuleWithStd(
          param_size=self.param_size,
          hidden_layer_sizes=(),
          activation=self.activation,
          kernel_init=self.kernel_init,
          noise_std_type=self.noise_std_type,
          init_noise_std=self.init_noise_std,
          state_dependent_std=self.state_dependent_std,
      )
      policy_logits = policy_module(encoded)

    value_body = networks.MLP(
        layer_sizes=list(self.value_hidden_layer_sizes) + [1],
        activation=self.activation,
        kernel_init=self.kernel_init,
        layer_norm=self.layer_norm,
    )
    value = jnp.squeeze(value_body(encoded), axis=-1)
    return policy_logits, value, new_state


def _mask_state(state: RecurrentState, mask: jax.Array) -> RecurrentState:
  """Masks recurrent state on reset."""
  mask = jnp.expand_dims(mask, axis=-1)
  hidden = state.hidden * mask
  cell = state.cell * mask if state.cell is not None else None
  return RecurrentState(hidden=hidden, cell=cell)


def _initial_state(
    batch_size: int, hidden_size: int, core_type: Literal['gru', 'lstm', 'rnn']
) -> RecurrentState:
  hidden = jnp.zeros((batch_size, hidden_size))
  if core_type == 'lstm':
    cell = jnp.zeros((batch_size, hidden_size))
  else:
    cell = None
  return RecurrentState(hidden=hidden, cell=cell)


def make_inference_fn(
    ppo_networks: RecurrentPPONetworks, compute_value: bool = True
):
  """Creates params and inference function for the recurrent PPO agent."""

  def make_policy(params: types.Params, deterministic: bool = False):
    normalizer_params, policy_value_params = params

    def policy(
        observations: types.Observation,
        core_state: RecurrentState,
        key_sample: PRNGKey,
    ):
      logits, value, new_core_state = ppo_networks.apply_fn(
          normalizer_params, policy_value_params, observations, core_state
      )
      if deterministic:
        actions = ppo_networks.parametric_action_distribution.mode(logits)
        log_prob = jnp.zeros(actions.shape[0:1])
        raw_actions = actions
      else:
        raw_actions = (
            ppo_networks.parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
        )
        log_prob = ppo_networks.parametric_action_distribution.log_prob(
            logits, raw_actions
        )
        actions = ppo_networks.parametric_action_distribution.postprocess(
            raw_actions
        )
      extras = {
          'log_prob': log_prob,
          'raw_action': raw_actions,
          'distribution_params': logits,
      }
      if compute_value:
        extras['value'] = value
      return actions, extras, new_core_state

    return policy

  return make_policy


def make_recurrent_ppo_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (128, 128),
    value_hidden_layer_sizes: Sequence[int] = (128, 128),
    activation: networks.ActivationFn = linen.swish,
    policy_obs_key: str = 'state',
    core_type: Literal['gru', 'lstm', 'rnn'] = 'gru',
    hidden_size: int = 128,
    distribution_type: Literal['normal', 'tanh_normal'] = 'tanh_normal',
    noise_std_type: Literal['scalar', 'log'] = 'scalar',
    init_noise_std: float = 1.0,
    state_dependent_std: bool = False,
    policy_network_kernel_init_fn: networks.Initializer = jax.nn.initializers.lecun_uniform,
    policy_network_kernel_init_kwargs: Mapping[str, Any] | None = None,
    value_network_kernel_init_fn: networks.Initializer = jax.nn.initializers.lecun_uniform,
    value_network_kernel_init_kwargs: Mapping[str, Any] | None = None,
) -> RecurrentPPONetworks:
  """Builds recurrent policy/value networks."""
  policy_kernel_init_kwargs = policy_network_kernel_init_kwargs or {}
  value_kernel_init_kwargs = value_network_kernel_init_kwargs or {}

  parametric_action_distribution: distribution.ParametricDistribution
  if distribution_type == 'normal':
    parametric_action_distribution = distribution.NormalDistribution(
        event_size=action_size
    )
  elif distribution_type == 'tanh_normal':
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
  else:
    raise ValueError(
        f'Unsupported distribution type: {distribution_type}. Must be one'
        ' of "normal" or "tanh_normal".'
    )

  def apply_fn(
      normalizer_params: types.Params,
      params: types.Params,
      observations: types.Observation,
      core_state: RecurrentState,
  ):
    if isinstance(observations, Mapping):
      obs = preprocess_observations_fn(
          observations[policy_obs_key],
          networks.normalizer_select(normalizer_params, policy_obs_key),
      )
    else:
      obs = preprocess_observations_fn(observations, normalizer_params)
    logits, value, new_state = RecurrentActorCritic(
        core_type=core_type,
        hidden_size=hidden_size,
        param_size=parametric_action_distribution.param_size,
        policy_hidden_layer_sizes=policy_hidden_layer_sizes,
        value_hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
        kernel_init=policy_network_kernel_init_fn(**policy_kernel_init_kwargs),
        layer_norm=False,
        distribution_type=distribution_type,
        noise_std_type=noise_std_type,
        init_noise_std=init_noise_std,
        state_dependent_std=state_dependent_std,
    ).apply(params, obs, core_state)
    return logits, value, new_state

  obs_size = networks._get_obs_state_size(observation_size, policy_obs_key)  # pylint: disable=protected-access
  dummy_obs = jnp.zeros((1, obs_size))
  dummy_state = _initial_state(1, hidden_size, core_type)

  def init_fn(key: jax.Array):
    model = RecurrentActorCritic(
        core_type=core_type,
        hidden_size=hidden_size,
        param_size=parametric_action_distribution.param_size,
        policy_hidden_layer_sizes=policy_hidden_layer_sizes,
        value_hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
        kernel_init=value_network_kernel_init_fn(**value_kernel_init_kwargs),
        layer_norm=False,
        distribution_type=distribution_type,
        noise_std_type=noise_std_type,
        init_noise_std=init_noise_std,
        state_dependent_std=state_dependent_std,
    )
    return model.init(key, dummy_obs, dummy_state)

  return RecurrentPPONetworks(
      apply_fn=apply_fn,
      init_fn=init_fn,
      initial_state_fn=lambda batch_size: _initial_state(
          batch_size, hidden_size, core_type
      ),
      mask_state_fn=lambda state, mask: _mask_state(state, mask),
      parametric_action_distribution=parametric_action_distribution,
  )
