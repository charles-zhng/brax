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

"""Recurrent TD3 networks with support for RNN, GRU, and LSTM cells."""

import dataclasses
from typing import Any, Callable, Mapping, Sequence, Tuple

from brax.training import networks
from brax.training import types
from brax.training.agents.recurrent_ppo.networks import get_rnn_cell
from brax.training.agents.recurrent_ppo.networks import HiddenState
from brax.training.agents.recurrent_ppo.networks import init_hidden_state
from brax.training.agents.recurrent_ppo.networks import RNNCellType
from brax.training.types import PRNGKey
import flax
from flax import linen
import jax
import jax.numpy as jnp


ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


class RecurrentDeterministicActor(linen.Module):
  """Recurrent deterministic actor for TD3.

  Outputs tanh-bounded actions in [-1, 1].
  """

  action_size: int
  rnn_hidden_size: int
  output_layer_sizes: Sequence[int]
  cell_type: RNNCellType = 'gru'
  activation: ActivationFn = linen.relu
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform()

  def setup(self):
    self.rnn_cell = get_rnn_cell(
        self.cell_type, self.rnn_hidden_size, self.kernel_init
    )
    self.output_mlp = networks.MLP(
        layer_sizes=list(self.output_layer_sizes) + [self.action_size],
        activation=self.activation,
        kernel_init=self.kernel_init,
        activate_final=False,
    )

  def __call__(
      self, obs: jnp.ndarray, hidden: HiddenState
  ) -> Tuple[jnp.ndarray, HiddenState]:
    """Forward pass for single timestep.

    Args:
      obs: Observation of shape [batch, obs_dim]
      hidden: Hidden state from previous timestep

    Returns:
      Tuple of (action in [-1, 1], new_hidden_state)
    """
    new_hidden, _ = self.rnn_cell(hidden, obs)
    # For LSTM, the output is the second element (hidden state h)
    if self.cell_type == 'lstm':
      rnn_output = new_hidden[1]
    else:
      rnn_output = new_hidden
    action = jnp.tanh(self.output_mlp(rnn_output))
    return action, new_hidden

  def scan_forward(
      self,
      obs_seq: jnp.ndarray,
      initial_hidden: HiddenState,
      done_mask: jnp.ndarray | None = None,
  ) -> Tuple[jnp.ndarray, HiddenState]:
    """Forward pass over a sequence using scan.

    Args:
      obs_seq: Observations of shape [time, batch, obs_dim]
      initial_hidden: Initial hidden state
      done_mask: Optional done flags [time, batch] to reset hidden on episode
        boundaries

    Returns:
      Tuple of (actions [time, batch, action_dim], final_hidden_state)
    """

    def step_with_reset(hidden, inputs):
      obs, done = inputs
      # Reset hidden on episode boundary
      if self.cell_type == 'lstm':
        c, h = hidden
        c = jnp.where(done[..., None], 0.0, c)
        h = jnp.where(done[..., None], 0.0, h)
        hidden = (c, h)
      else:
        hidden = jnp.where(done[..., None], 0.0, hidden)
      action, new_hidden = self(obs, hidden)
      return new_hidden, action

    def step_no_reset(hidden, obs):
      action, new_hidden = self(obs, hidden)
      return new_hidden, action

    if done_mask is not None:
      final_hidden, actions = jax.lax.scan(
          step_with_reset, initial_hidden, (obs_seq, done_mask)
      )
    else:
      final_hidden, actions = jax.lax.scan(
          step_no_reset, initial_hidden, obs_seq
      )
    return actions, final_hidden


@dataclasses.dataclass
class RecurrentNetwork:
  """Recurrent network with init, apply, and hidden state initialization."""

  init: Callable[..., Any]
  apply: Callable[..., Any]
  apply_sequence: Callable[..., Any]
  init_hidden: Callable[[int], HiddenState]


@flax.struct.dataclass
class RecurrentTD3Networks:
  """Recurrent TD3 networks container."""

  actor_network: RecurrentNetwork
  q_network: networks.FeedForwardNetwork
  rnn_hidden_size: int
  cell_type: RNNCellType


def _get_obs_state_size(obs_size: types.ObservationSize, obs_key: str) -> int:
  obs_size = obs_size[obs_key] if isinstance(obs_size, Mapping) else obs_size
  return jax.tree_util.tree_flatten(obs_size)[0][-1]


def make_inference_fn(recurrent_td3_networks: RecurrentTD3Networks):
  """Creates params and inference function for the recurrent TD3 agent.

  Args:
    recurrent_td3_networks: The recurrent TD3 networks.

  Returns:
    A function that creates a policy from params.
  """

  def make_policy(
      params: types.PolicyParams, deterministic: bool = True
  ) -> Callable:
    """Creates a recurrent deterministic policy function.

    Args:
      params: Tuple of (normalizer_params, actor_params)
      deterministic: Unused for TD3 (always deterministic), kept for API
        compatibility

    Returns:
      A function that takes (observations, hidden_state, key) and returns
      (action, extras, new_hidden_state).
    """
    del deterministic  # TD3 policy is always deterministic
    actor_network = recurrent_td3_networks.actor_network

    def policy(
        observations: types.Observation,
        hidden_state: HiddenState,
        key_sample: PRNGKey,
    ) -> Tuple[types.Action, types.Extra, HiddenState]:
      """Recurrent deterministic policy function.

      Args:
        observations: Current observations
        hidden_state: Actor hidden state
        key_sample: Random key (unused for deterministic policy)

      Returns:
        Tuple of (actions, extras, new_hidden_state)
      """
      del key_sample  # Unused for deterministic policy
      normalizer_params, actor_params = params
      action, new_hidden = actor_network.apply(
          normalizer_params, actor_params, observations, hidden_state
      )
      extras = {'hidden': new_hidden}
      return action, extras, new_hidden

    return policy

  return make_policy


def make_recurrent_td3_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    actor_rnn_hidden_size: int = 64,
    actor_output_layer_sizes: Sequence[int] = (64,),
    q_hidden_layer_sizes: Sequence[int] = (256, 256),
    cell_type: RNNCellType = 'gru',
    activation: ActivationFn = linen.swish,
    policy_obs_key: str = 'state',
    q_obs_key: str = 'state',
    policy_network_kernel_init_fn: Initializer = jax.nn.initializers.lecun_uniform,
    policy_network_kernel_init_kwargs: Mapping[str, Any] | None = None,
    q_network_kernel_init_fn: Initializer = jax.nn.initializers.lecun_uniform,
    q_network_kernel_init_kwargs: Mapping[str, Any] | None = None,
) -> RecurrentTD3Networks:
  """Make recurrent TD3 networks.

  Actor architecture: obs -> RNN cell -> MLP output layers -> tanh(action)
  Q architecture: (obs, action) -> MLP -> Q-value (feedforward, twin critics)

  Args:
    observation_size: Size of observations
    action_size: Size of action space
    preprocess_observations_fn: Function to preprocess observations
    actor_rnn_hidden_size: Hidden size of actor RNN cell
    actor_output_layer_sizes: MLP layer sizes after RNN in actor network
    q_hidden_layer_sizes: Hidden layer sizes for Q-network MLP
    cell_type: Type of RNN cell ('simple', 'gru', 'lstm')
    activation: Activation function
    policy_obs_key: Key for actor observations (can be partial state)
    q_obs_key: Key for Q observations (can be privileged full state)
    policy_network_kernel_init_fn: Kernel initializer function for actor
    policy_network_kernel_init_kwargs: Kwargs for actor kernel initializer
    q_network_kernel_init_fn: Kernel initializer function for Q-networks
    q_network_kernel_init_kwargs: Kwargs for Q-network kernel initializer

  Returns:
    RecurrentTD3Networks containing actor and Q networks
  """
  policy_kernel_init_kwargs = policy_network_kernel_init_kwargs or {}
  q_kernel_init_kwargs = q_network_kernel_init_kwargs or {}
  actor_kernel_init = policy_network_kernel_init_fn(**policy_kernel_init_kwargs)
  q_kernel_init = q_network_kernel_init_fn(**q_kernel_init_kwargs)

  actor_obs_size = _get_obs_state_size(observation_size, policy_obs_key)
  q_obs_size = _get_obs_state_size(observation_size, q_obs_key)

  # Create actor module
  actor_module = RecurrentDeterministicActor(
      action_size=action_size,
      rnn_hidden_size=actor_rnn_hidden_size,
      output_layer_sizes=actor_output_layer_sizes,
      cell_type=cell_type,
      activation=activation,
      kernel_init=actor_kernel_init,
  )

  # Actor network functions
  def actor_apply(processor_params, actor_params, obs, hidden):
    if isinstance(obs, Mapping):
      obs = preprocess_observations_fn(
          obs[policy_obs_key],
          networks.normalizer_select(processor_params, policy_obs_key),
      )
    else:
      obs = preprocess_observations_fn(obs, processor_params)
    return actor_module.apply(actor_params, obs, hidden)

  def actor_apply_sequence(
      processor_params, actor_params, obs_seq, hidden, done_mask=None
  ):
    if isinstance(obs_seq, Mapping):
      obs_seq = preprocess_observations_fn(
          obs_seq[policy_obs_key],
          networks.normalizer_select(processor_params, policy_obs_key),
      )
    else:
      obs_seq = preprocess_observations_fn(obs_seq, processor_params)
    return actor_module.apply(
        actor_params,
        obs_seq,
        hidden,
        done_mask,
        method=actor_module.scan_forward,
    )

  dummy_obs = jnp.zeros((1, actor_obs_size))
  dummy_hidden = init_hidden_state(cell_type, actor_rnn_hidden_size, 1)

  def actor_init(key):
    return actor_module.init(key, dummy_obs, dummy_hidden)

  def actor_init_hidden(batch_size):
    return init_hidden_state(cell_type, actor_rnn_hidden_size, batch_size)

  actor_network = RecurrentNetwork(
      init=actor_init,
      apply=actor_apply,
      apply_sequence=actor_apply_sequence,
      init_hidden=actor_init_hidden,
  )

  # Q-network (feedforward, twin critics)
  q_network = networks.make_q_network(
      obs_size=q_obs_size,
      action_size=action_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=q_hidden_layer_sizes,
      activation=activation,
      n_critics=2,
      kernel_init=q_kernel_init,
  )

  return RecurrentTD3Networks(
      actor_network=actor_network,
      q_network=q_network,
      rnn_hidden_size=actor_rnn_hidden_size,
      cell_type=cell_type,
  )
