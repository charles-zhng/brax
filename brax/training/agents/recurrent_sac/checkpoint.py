# Copyright 2026 The Brax Authors.
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

"""Checkpointing for recurrent SAC.

Saved checkpoint pytree is ``(normalizer_params, policy_params, q_params)``.
The on-disk config captures the ``network_factory`` kwargs (``cell_type``,
``rnn_hidden_size``, layer sizes, distribution type, obs keys) so a restored
policy can be reconstructed without the caller re-specifying RNN geometry.
"""

from typing import Any, Callable, Union

from brax.training import checkpoint
from brax.training import types
from brax.training.agents.recurrent_sac import networks as recurrent_sac_networks
from etils import epath
import jax
from ml_collections import config_dict


_CONFIG_FNAME = "recurrent_sac_network_config.json"


def _infer_batch_size(
    observations: types.Observation,
    observation_size: types.ObservationSize,
) -> int:
    """Infers the observation batch size for hidden-state initialization."""
    obs_leaf = jax.tree_util.tree_leaves(observations)[0]
    size_leaf = jax.tree_util.tree_leaves(
        observation_size, is_leaf=lambda x: isinstance(x, (tuple, list))
    )[0]
    expected = (size_leaf,) if isinstance(size_leaf, int) else tuple(size_leaf)
    if obs_leaf.shape == expected:
        return 1
    if obs_leaf.shape[1:] == expected:
        return obs_leaf.shape[0]
    raise ValueError(
        f"Could not infer batch size from observations with shape {obs_leaf.shape} "
        f"and checkpoint observation size {expected}."
    )


class RecurrentCheckpointPolicy:
    """Callable recurrent SAC policy loaded from a checkpoint.

    Supports both ``policy(obs, key)`` (auto-initializes hidden state from the
    inferred batch size) and ``policy(obs, policy_hidden, key)`` for explicit
    state management.
    """

    def __init__(
        self,
        policy: Callable[..., Any],
        recurrent_sac_network: recurrent_sac_networks.RecurrentSACNetworks,
        observation_size: types.ObservationSize,
    ):
        self._policy = policy
        self._network = recurrent_sac_network
        self._observation_size = observation_size

    def init_hidden(self, batch_size: int):
        return self._network.policy_network.init_hidden(batch_size)

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
            policy_hidden = key_or_hidden
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
    """Save a recurrent SAC checkpoint."""
    return checkpoint.save(path, step, params, config, _CONFIG_FNAME)


def load(path: Union[str, epath.Path]):
    """Load a recurrent SAC checkpoint.

    Returns ``(normalizer_params, policy_params, q_params)``.
    """
    return checkpoint.load(path)


def network_config(
    observation_size: types.ObservationSize,
    action_size: int,
    normalize_observations: bool,
    network_factory: types.NetworkFactory[
        recurrent_sac_networks.RecurrentSACNetworks
    ],
) -> config_dict.ConfigDict:
    """Build a config dict for re-creating a network from checkpoint."""
    return checkpoint.network_config(
        observation_size, action_size, normalize_observations, network_factory
    )


def _get_network(
    config: config_dict.ConfigDict,
    network_factory: types.NetworkFactory[
        recurrent_sac_networks.RecurrentSACNetworks
    ],
) -> recurrent_sac_networks.RecurrentSACNetworks:
    return checkpoint.get_network(config, network_factory)


def load_config(path: Union[str, epath.Path]) -> config_dict.ConfigDict:
    """Load the saved network config."""
    path = epath.Path(path)
    return checkpoint.load_config(path / _CONFIG_FNAME)


def load_policy(
    path: Union[str, epath.Path],
    network_factory: types.NetworkFactory[
        recurrent_sac_networks.RecurrentSACNetworks
    ] = recurrent_sac_networks.make_recurrent_sac_networks,
    deterministic: bool = True,
):
    """Load an inference policy from a recurrent SAC checkpoint.

    The returned callable supports both ``policy(observations, key)`` and
    ``policy(observations, policy_hidden, key)``. Use
    ``policy.init_hidden(batch_size)`` to get a zeroed hidden state.
    """
    path = epath.Path(path)
    config = load_config(path)
    params = load(path)
    network = _get_network(config, network_factory)
    make_inference_fn = recurrent_sac_networks.make_inference_fn(network)
    # params is (normalizer_params, policy_params, q_params). Inference only
    # needs (normalizer_params, policy_params).
    policy = make_inference_fn(
        (params[0], params[1]), deterministic=deterministic
    )
    return RecurrentCheckpointPolicy(
        policy,
        network,
        config.to_dict()["observation_size"],
    )
