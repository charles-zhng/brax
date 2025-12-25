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

"""Test RNN-PPO checkpointing."""

import functools

from absl import flags
from absl.testing import absltest
from brax.training.acme import running_statistics
from brax.training.agents.rnn_ppo import checkpoint
from brax.training.agents.rnn_ppo import losses as rnn_ppo_losses
from brax.training.agents.rnn_ppo import networks as rnn_ppo_networks
from etils import epath
import jax
from jax import numpy as jp


class CheckpointTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    flags.FLAGS.mark_as_parsed()

  def test_rnn_ppo_params_config(self):
    """Test that network config is properly saved."""
    network_factory = functools.partial(
        rnn_ppo_networks.make_rnn_ppo_networks,
        policy_rnn_hidden_size=64,
        policy_output_layer_sizes=(32, 16),
    )
    config = checkpoint.network_config(
        action_size=3,
        observation_size=5,
        normalize_observations=True,
        network_factory=network_factory,
    )
    self.assertEqual(
        config.network_factory_kwargs.to_dict()['policy_rnn_hidden_size'],
        64,
    )
    self.assertEqual(
        config.network_factory_kwargs.to_dict()['policy_output_layer_sizes'],
        (32, 16),
    )
    self.assertEqual(config.action_size, 3)
    self.assertEqual(config.observation_size, 5)

  def test_save_and_load_checkpoint(self):
    """Test saving and loading RNN-PPO checkpoint."""
    path = self.create_tempdir('test')
    network_factory = functools.partial(
        rnn_ppo_networks.make_rnn_ppo_networks,
        policy_rnn_hidden_size=32,
        policy_output_layer_sizes=(16,),
        cell_type='gru',
    )
    config = checkpoint.network_config(
        observation_size=4,
        action_size=2,
        normalize_observations=True,
        network_factory=network_factory,
    )

    # Generate network params for saving a dummy checkpoint.
    normalize = lambda x, y: x
    if config.normalize_observations:
      normalize = running_statistics.normalize
    rnn_ppo_network = network_factory(
        config.observation_size,
        config.action_size,
        preprocess_observations_fn=normalize,
        **config.network_factory_kwargs,
    )
    dummy_key = jax.random.PRNGKey(0)
    network_params = rnn_ppo_losses.RNNPPONetworkParams(
        policy=rnn_ppo_network.policy_network.init(dummy_key),
        value=rnn_ppo_network.value_network.init(dummy_key),
    )
    normalizer_params = running_statistics.init_state(
        jax.tree_util.tree_map(jp.zeros, config.observation_size),
        std_eps=0.02,
    )
    params = (normalizer_params, network_params.policy, network_params.value)

    # Save and load a checkpoint.
    checkpoint.save(
        path.full_path,
        step=1,
        params=params,
        config=config,
    )

    policy_fn = checkpoint.load_policy(
        epath.Path(path.full_path) / '000000000001',
    )

    # Test that policy works with hidden states
    batch_size = 1
    policy_hidden = rnn_ppo_network.policy_network.init_hidden(batch_size)
    value_hidden = rnn_ppo_network.value_network.init_hidden(batch_size)
    hidden_states = (policy_hidden, value_hidden)

    out = policy_fn(jp.zeros((batch_size, 4)), hidden_states, jax.random.PRNGKey(0))
    # out is (action, extras, new_hidden_states)
    self.assertEqual(out[0].shape, (batch_size, 2))

    loaded_params = checkpoint.load(epath.Path(path.full_path) / '000000000001')
    loaded_normalizer = loaded_params[0]
    self.assertEqual(loaded_normalizer.std_eps, 0.02)


if __name__ == '__main__':
  absltest.main()
