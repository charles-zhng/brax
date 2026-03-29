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

"""Checkpointing for Recurrent PPO."""

from typing import Any, Callable, Union

from brax.training import checkpoint
from brax.training import types
from brax.training.agents.recurrent_ppo_feature import networks as rnn_ppo_networks
from etils import epath
import jax
from ml_collections import config_dict

_CONFIG_FNAME = 'rnn_ppo_network_config.json'


def _infer_batch_size(
    observations: types.Observation,
    observation_size: types.ObservationSize,
) -> int:
  """Infers the observation batch size for hidden-state initialization."""
  obs_leaf = jax.tree_util.tree_leaves(observations)[0]
  size_leaf = jax.tree_util.tree_leaves(
      observation_size, is_leaf=lambda x: isinstance(x, (tuple, list))
  )[0]
  expected_shape = (size_leaf,) if isinstance(size_leaf, int) else tuple(size_leaf)
  if obs_leaf.shape == expected_shape:
    return 1
  if obs_leaf.shape[1:] == expected_shape:
    return obs_leaf.shape[0]
  raise ValueError(
      'Could not infer batch size from observations with shape '
      f'{obs_leaf.shape} and checkpoint observation size {expected_shape}.'
  )


def _unwrap_policy_hidden(policy_hidden: Any) -> Any:
  """Unwraps a `(policy_hidden, value_hidden)` tuple from legacy callers."""
  if (
      isinstance(policy_hidden, tuple)
      and len(policy_hidden) == 2
      and policy_hidden[1] is None
  ):
    return policy_hidden[0]
  return policy_hidden


class RecurrentCheckpointPolicy:
  """Callable recurrent policy loaded from a checkpoint.

  Supports both `policy(obs, key)` for one-off inference and
  `policy(obs, policy_hidden, key)` for explicit recurrent-state management.
  """

  def __init__(
      self,
      policy: Callable[..., Any],
      rnn_ppo_network: rnn_ppo_networks.RNNPPONetworks,
      observation_size: types.ObservationSize,
  ):
    self._policy = policy
    self._rnn_ppo_network = rnn_ppo_network
    self._observation_size = observation_size

  def init_hidden(self, batch_size: int) -> rnn_ppo_networks.HiddenState:
    return self._rnn_ppo_network.policy_network.init_hidden(batch_size)

  def __call__(
      self,
      observations: types.Observation,
      key_or_hidden: Any,
      key_sample: types.PRNGKey | None = None,
  ):
    if key_sample is None:
      key_sample = key_or_hidden
      policy_hidden = self.init_hidden(
          _infer_batch_size(observations, self._observation_size)
      )
    else:
      policy_hidden = _unwrap_policy_hidden(key_or_hidden)
      if policy_hidden is None:
        policy_hidden = self.init_hidden(
            _infer_batch_size(observations, self._observation_size)
        )

    return self._policy(observations, policy_hidden, key_sample)


def save(
    path: Union[str, epath.Path],
    step: int,
    params: Any,
    config: config_dict.ConfigDict,
):
  """Saves a checkpoint."""
  return checkpoint.save(path, step, params, config, _CONFIG_FNAME)


def load(
    path: Union[str, epath.Path],
):
  """Loads checkpoint."""
  return checkpoint.load(path)


def network_config(
    observation_size: types.ObservationSize,
    action_size: int,
    normalize_observations: bool,
    network_factory: types.NetworkFactory[rnn_ppo_networks.RNNPPONetworks],
) -> config_dict.ConfigDict:
  """Returns a config dict for re-creating a network from a checkpoint."""
  return checkpoint.network_config(
      observation_size, action_size, normalize_observations, network_factory
  )


def _get_rnn_ppo_network(
    config: config_dict.ConfigDict,
    network_factory: types.NetworkFactory[rnn_ppo_networks.RNNPPONetworks],
) -> rnn_ppo_networks.RNNPPONetworks:
  """Generates an RNN-PPO network given config."""
  return checkpoint.get_network(config, network_factory)


def load_config(
    path: Union[str, epath.Path],
) -> config_dict.ConfigDict:
  """Loads RNN-PPO config from checkpoint."""
  path = epath.Path(path)
  config_path = path / _CONFIG_FNAME
  return checkpoint.load_config(config_path)


def load_policy(
    path: Union[str, epath.Path],
    network_factory: types.NetworkFactory[
        rnn_ppo_networks.RNNPPONetworks
    ] = rnn_ppo_networks.make_rnn_ppo_networks,
    deterministic: bool = True,
):
  """Loads policy inference from an RNN-PPO checkpoint.

  The returned callable supports both `policy(observations, key)` and
  `policy(observations, policy_hidden, key)`. It also exposes
  `policy.init_hidden(batch_size)` for explicit hidden-state management.
  """
  path = epath.Path(path)
  config = load_config(path)
  params = load(path)
  rnn_ppo_network = _get_rnn_ppo_network(config, network_factory)
  make_inference_fn = rnn_ppo_networks.make_inference_fn(rnn_ppo_network)
  policy = make_inference_fn(params, deterministic=deterministic)

  return RecurrentCheckpointPolicy(
      policy,
      rnn_ppo_network,
      config.to_dict()['observation_size'],
  )
