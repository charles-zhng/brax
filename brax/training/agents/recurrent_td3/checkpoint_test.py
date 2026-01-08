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

"""Test Recurrent TD3 checkpointing."""

import functools

from absl import flags
from absl.testing import absltest
from brax.training.acme import running_statistics
from brax.training.agents.recurrent_td3 import checkpoint
from brax.training.agents.recurrent_td3 import networks as recurrent_td3_networks
from etils import epath
import jax
from jax import numpy as jp


class CheckpointTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    flags.FLAGS.mark_as_parsed()

  def test_recurrent_td3_params_config(self):
    """Test that network config is properly saved."""
    network_factory = functools.partial(
        recurrent_td3_networks.make_recurrent_td3_networks,
        actor_rnn_hidden_size=64,
        actor_output_layer_sizes=(32, 16),
        q_hidden_layer_sizes=(128, 128),
    )
    config = checkpoint.network_config(
        action_size=3,
        observation_size=5,
        normalize_observations=True,
        network_factory=network_factory,
    )
    self.assertEqual(
        config.network_factory_kwargs.to_dict()['actor_rnn_hidden_size'],
        64,
    )
    self.assertEqual(
        config.network_factory_kwargs.to_dict()['actor_output_layer_sizes'],
        (32, 16),
    )
    self.assertEqual(
        config.network_factory_kwargs.to_dict()['q_hidden_layer_sizes'],
        (128, 128),
    )
    self.assertEqual(config.action_size, 3)
    self.assertEqual(config.observation_size, 5)

  def test_save_and_load_checkpoint(self):
    """Test saving and loading Recurrent TD3 checkpoint."""
    path = self.create_tempdir('test')
    network_factory = functools.partial(
        recurrent_td3_networks.make_recurrent_td3_networks,
        actor_rnn_hidden_size=32,
        actor_output_layer_sizes=(16,),
        q_hidden_layer_sizes=(64, 64),
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
    td3_network = network_factory(
        config.observation_size,
        config.action_size,
        preprocess_observations_fn=normalize,
        **config.network_factory_kwargs,
    )
    dummy_key = jax.random.PRNGKey(0)
    actor_params = td3_network.actor_network.init(dummy_key)
    normalizer_params = running_statistics.init_state(
        jax.tree_util.tree_map(jp.zeros, config.observation_size),
        std_eps=0.02,
    )
    # TD3 saves (normalizer_params, actor_params)
    params = (normalizer_params, actor_params)

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
    hidden_state = td3_network.actor_network.init_hidden(batch_size)

    out = policy_fn(jp.zeros((batch_size, 4)), hidden_state, jax.random.PRNGKey(0))
    # out is (action, extras, new_hidden_state)
    self.assertEqual(out[0].shape, (batch_size, 2))

    # Verify action is bounded
    self.assertTrue(jp.all(out[0] >= -1.0))
    self.assertTrue(jp.all(out[0] <= 1.0))

    loaded_params = checkpoint.load(epath.Path(path.full_path) / '000000000001')
    loaded_normalizer = loaded_params[0]
    self.assertEqual(loaded_normalizer.std_eps, 0.02)

    # Verify actor params were loaded correctly
    loaded_actor_params = loaded_params[1]
    self.assertIn('params', loaded_actor_params)

  def test_load_policy_deterministic(self):
    """Test that loaded policy produces deterministic actions."""
    path = self.create_tempdir('test')
    network_factory = functools.partial(
        recurrent_td3_networks.make_recurrent_td3_networks,
        actor_rnn_hidden_size=16,
        actor_output_layer_sizes=(16,),
        cell_type='gru',
    )
    config = checkpoint.network_config(
        observation_size=4,
        action_size=2,
        normalize_observations=False,
        network_factory=network_factory,
    )

    td3_network = network_factory(
        config.observation_size,
        config.action_size,
        preprocess_observations_fn=lambda x, y: x,
        **config.network_factory_kwargs,
    )
    dummy_key = jax.random.PRNGKey(0)
    actor_params = td3_network.actor_network.init(dummy_key)
    normalizer_params = running_statistics.init_state(
        jax.tree_util.tree_map(jp.zeros, config.observation_size),
    )
    params = (normalizer_params, actor_params)

    checkpoint.save(path.full_path, step=1, params=params, config=config)

    policy_fn = checkpoint.load_policy(
        epath.Path(path.full_path) / '000000000001',
    )

    batch_size = 1
    hidden_state = td3_network.actor_network.init_hidden(batch_size)
    obs = jp.zeros((batch_size, 4))

    # TD3 policy should be deterministic - same result with different keys
    action1, _, _ = policy_fn(obs, hidden_state, jax.random.PRNGKey(0))
    action2, _, _ = policy_fn(obs, hidden_state, jax.random.PRNGKey(42))

    self.assertTrue(jp.allclose(action1, action2))


if __name__ == '__main__':
  absltest.main()
