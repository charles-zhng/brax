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

"""Checkpointing for Recurrent TD3."""

from typing import Any, Union

from brax.training import checkpoint
from brax.training import types
from brax.training.agents.recurrent_td3 import networks as recurrent_td3_networks
from etils import epath
from ml_collections import config_dict

_CONFIG_FNAME = 'recurrent_td3_network_config.json'


def save(
    path: Union[str, epath.Path],
    step: int,
    params: Any,
    config: config_dict.ConfigDict,
):
  """Saves a checkpoint.

  Args:
    path: Directory to save checkpoint.
    step: Training step number.
    params: Tuple of (normalizer_params, actor_params).
    config: Network configuration dict.
  """
  return checkpoint.save(path, step, params, config, _CONFIG_FNAME)


def load(
    path: Union[str, epath.Path],
):
  """Loads Recurrent TD3 checkpoint.

  Args:
    path: Directory containing checkpoint.

  Returns:
    Tuple of (normalizer_params, actor_params).
  """
  return checkpoint.load(path)


def network_config(
    observation_size: types.ObservationSize,
    action_size: int,
    normalize_observations: bool,
    network_factory: types.NetworkFactory[
        recurrent_td3_networks.RecurrentTD3Networks
    ],
) -> config_dict.ConfigDict:
  """Returns a config dict for re-creating a network from a checkpoint.

  Args:
    observation_size: Size of observations.
    action_size: Size of action space.
    normalize_observations: Whether observations are normalized.
    network_factory: Factory function used to create the network.

  Returns:
    Configuration dict for network recreation.
  """
  return checkpoint.network_config(
      observation_size, action_size, normalize_observations, network_factory
  )


def _get_network(
    config: config_dict.ConfigDict,
    network_factory: types.NetworkFactory[
        recurrent_td3_networks.RecurrentTD3Networks
    ],
) -> recurrent_td3_networks.RecurrentTD3Networks:
  """Generates a Recurrent TD3 network given config.

  Args:
    config: Network configuration dict.
    network_factory: Factory function to create network.

  Returns:
    RecurrentTD3Networks instance.
  """
  return checkpoint.get_network(config, network_factory)


def load_config(
    path: Union[str, epath.Path],
) -> config_dict.ConfigDict:
  """Loads Recurrent TD3 config from checkpoint.

  Args:
    path: Directory containing checkpoint.

  Returns:
    Network configuration dict.
  """
  path = epath.Path(path)
  config_path = path / _CONFIG_FNAME
  return checkpoint.load_config(config_path)


def load_policy(
    path: Union[str, epath.Path],
    network_factory: types.NetworkFactory[
        recurrent_td3_networks.RecurrentTD3Networks
    ] = recurrent_td3_networks.make_recurrent_td3_networks,
    deterministic: bool = True,
):
  """Loads policy inference function from Recurrent TD3 checkpoint.

  Args:
    path: Directory containing checkpoint.
    network_factory: Factory function to create network (must match training).
    deterministic: Unused for TD3 (always deterministic).

  Returns:
    A function that creates recurrent policy from params.
  """
  path = epath.Path(path)
  config = load_config(path)
  params = load(path)
  td3_network = _get_network(config, network_factory)
  make_inference_fn = recurrent_td3_networks.make_inference_fn(td3_network)

  return make_inference_fn(params, deterministic=deterministic)
