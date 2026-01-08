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

"""Recurrent TD3 tests."""

import functools
import pickle
from absl.testing import absltest
from absl.testing import parameterized
from brax import envs
from brax.training.acme import running_statistics
from brax.training.agents.recurrent_td3 import networks as recurrent_td3_networks
from brax.training.agents.recurrent_td3 import train as recurrent_td3
import jax
import jax.numpy as jnp


class RecurrentTD3Test(parameterized.TestCase):
  """Tests for Recurrent TD3 module."""

  def testTrain(self):
    """Test Recurrent TD3 training with a simple env."""
    fast = envs.get_environment('fast')
    _, _, metrics = recurrent_td3.train(
        fast,
        num_timesteps=2**15,
        episode_length=128,
        num_envs=64,
        learning_rate=3e-4,
        discounting=0.99,
        batch_size=64,
        min_replay_size=1024,
        grad_updates_per_step=64,
        normalize_observations=True,
        seed=0,
        num_evals=3,
        reward_scaling=10,
        exploration_noise=0.1,
        target_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        tau=0.005,
    )
    # Off-policy methods need more time to learn, lower threshold
    self.assertGreater(metrics['eval/episode_reward'], 100)

  @parameterized.parameters('simple', 'gru', 'lstm')
  def testTrainWithCellTypes(self, cell_type):
    """Test Recurrent TD3 runs with different RNN cell types."""
    network_factory = functools.partial(
        recurrent_td3_networks.make_recurrent_td3_networks,
        cell_type=cell_type,
        actor_rnn_hidden_size=32,
        actor_output_layer_sizes=(32,),
        q_hidden_layer_sizes=(64, 64),
        activation=jax.nn.elu,
    )

    _, _, _ = recurrent_td3.train(
        envs.get_environment('inverted_pendulum', backend='spring'),
        num_timesteps=2**12,
        episode_length=50,
        num_envs=32,
        learning_rate=3e-4,
        discounting=0.95,
        batch_size=32,
        min_replay_size=256,
        normalize_observations=True,
        max_grad_norm=1.0,
        seed=0,
        reward_scaling=10,
        network_factory=network_factory,
    )

  @parameterized.parameters(1, 2, 4)
  def testDelayedPolicyUpdate(self, policy_delay):
    """Test Recurrent TD3 with different policy delay values."""
    _, _, _ = recurrent_td3.train(
        envs.get_environment('inverted_pendulum', backend='spring'),
        num_timesteps=2**12,
        episode_length=50,
        num_envs=32,
        learning_rate=3e-4,
        discounting=0.95,
        batch_size=32,
        min_replay_size=256,
        normalize_observations=True,
        seed=0,
        reward_scaling=10,
        policy_delay=policy_delay,
    )

  @parameterized.parameters(0.001, 0.005, 0.01)
  def testTargetNetworkUpdate(self, tau):
    """Test Recurrent TD3 with different tau (Polyak averaging) values."""
    _, _, _ = recurrent_td3.train(
        envs.get_environment('inverted_pendulum', backend='spring'),
        num_timesteps=2**12,
        episode_length=50,
        num_envs=32,
        learning_rate=3e-4,
        discounting=0.95,
        batch_size=32,
        min_replay_size=256,
        normalize_observations=True,
        seed=0,
        reward_scaling=10,
        tau=tau,
    )

  @parameterized.parameters(0.05, 0.1, 0.2)
  def testExplorationNoise(self, exploration_noise):
    """Test Recurrent TD3 with different exploration noise levels."""
    _, _, _ = recurrent_td3.train(
        envs.get_environment('inverted_pendulum', backend='spring'),
        num_timesteps=2**12,
        episode_length=50,
        num_envs=32,
        learning_rate=3e-4,
        discounting=0.95,
        batch_size=32,
        min_replay_size=256,
        normalize_observations=True,
        seed=0,
        reward_scaling=10,
        exploration_noise=exploration_noise,
    )

  @parameterized.parameters(True, False)
  def testNetworkEncoding(self, normalize_observations):
    """Test that network params can be pickled and restored."""
    env = envs.get_environment('fast')
    original_inference, params, _ = recurrent_td3.train(
        env,
        num_timesteps=2**10,
        episode_length=128,
        num_envs=64,
        min_replay_size=128,
        normalize_observations=normalize_observations,
    )
    normalize_fn = lambda x, y: x
    if normalize_observations:
      normalize_fn = running_statistics.normalize
    td3_network = recurrent_td3_networks.make_recurrent_td3_networks(
        env.observation_size, env.action_size, normalize_fn
    )
    inference = recurrent_td3_networks.make_inference_fn(td3_network)
    byte_encoding = pickle.dumps(params)
    decoded_params = pickle.loads(byte_encoding)

    # Compute one action with recurrent policy
    state = env.reset(jax.random.PRNGKey(0))

    # Initialize hidden state
    hidden_state = td3_network.actor_network.init_hidden(1)

    # Get action from original inference
    original_action, _, _ = original_inference(decoded_params)(
        state.obs, hidden_state, jax.random.PRNGKey(0)
    )
    # Get action from reconstructed inference
    action, _, _ = inference(decoded_params)(
        state.obs, hidden_state, jax.random.PRNGKey(0)
    )
    self.assertSequenceEqual(list(original_action), list(action))
    env.step(state, action)

  @parameterized.parameters('simple', 'gru', 'lstm')
  def testHiddenStateShapes(self, cell_type):
    """Test hidden state shapes for different cell types."""
    env = envs.get_environment('fast')
    hidden_size = 32
    batch_size = 8

    network_factory = functools.partial(
        recurrent_td3_networks.make_recurrent_td3_networks,
        cell_type=cell_type,
        actor_rnn_hidden_size=hidden_size,
    )

    td3_network = network_factory(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=lambda x, y: x,
    )

    hidden_state = td3_network.actor_network.init_hidden(batch_size)

    if cell_type == 'lstm':
      # LSTM has tuple state (carry, hidden)
      self.assertIsInstance(hidden_state, tuple)
      self.assertEqual(len(hidden_state), 2)
      self.assertEqual(hidden_state[0].shape, (batch_size, hidden_size))
      self.assertEqual(hidden_state[1].shape, (batch_size, hidden_size))
    else:
      # SimpleCell and GRU have single state
      self.assertEqual(hidden_state.shape, (batch_size, hidden_size))

  def testDeterministicAction(self):
    """Test that TD3 policy produces deterministic actions."""
    env = envs.get_environment('fast')
    td3_network = recurrent_td3_networks.make_recurrent_td3_networks(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=lambda x, y: x,
    )

    # Initialize params
    actor_params = td3_network.actor_network.init(jax.random.PRNGKey(0))
    normalizer_params = running_statistics.init_state(
        jax.tree_util.tree_map(jnp.zeros, env.observation_size)
    )

    # Create policy
    make_policy = recurrent_td3_networks.make_inference_fn(td3_network)
    policy = make_policy((normalizer_params, actor_params))

    # Get observation and hidden state
    state = env.reset(jax.random.PRNGKey(0))
    hidden_state = td3_network.actor_network.init_hidden(1)

    # Actions should be deterministic (same with different keys)
    action1, _, _ = policy(state.obs, hidden_state, jax.random.PRNGKey(0))
    action2, _, _ = policy(state.obs, hidden_state, jax.random.PRNGKey(42))

    # TD3 policy is deterministic, so actions should be identical
    self.assertTrue(jnp.allclose(action1, action2))

  def testActionBounds(self):
    """Test that TD3 policy produces actions in [-1, 1]."""
    env = envs.get_environment('fast')
    td3_network = recurrent_td3_networks.make_recurrent_td3_networks(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=lambda x, y: x,
    )

    # Initialize params
    actor_params = td3_network.actor_network.init(jax.random.PRNGKey(0))
    normalizer_params = running_statistics.init_state(
        jax.tree_util.tree_map(jnp.zeros, env.observation_size)
    )

    # Create policy
    make_policy = recurrent_td3_networks.make_inference_fn(td3_network)
    policy = make_policy((normalizer_params, actor_params))

    # Get observation and hidden state
    state = env.reset(jax.random.PRNGKey(0))
    hidden_state = td3_network.actor_network.init_hidden(1)

    # Check action bounds
    action, _, _ = policy(state.obs, hidden_state, jax.random.PRNGKey(0))
    self.assertTrue(jnp.all(action >= -1.0))
    self.assertTrue(jnp.all(action <= 1.0))

  def testTrainDomainRandomize(self):
    """Test Recurrent TD3 with domain randomization."""

    def rand_fn(sys, rng):
      @jax.vmap
      def get_offset(rng):
        offset = jax.random.uniform(rng, shape=(3,), minval=-0.1, maxval=0.1)
        pos = sys.link.transform.pos.at[0].set(offset)
        return pos

      sys_v = sys.tree_replace({'link.inertia.transform.pos': get_offset(rng)})
      in_axes = jax.tree.map(lambda x: None, sys)
      in_axes = in_axes.tree_replace({'link.inertia.transform.pos': 0})
      return sys_v, in_axes

    _, _, _ = recurrent_td3.train(
        envs.get_environment('inverted_pendulum', backend='spring'),
        num_timesteps=2**12,
        episode_length=50,
        num_envs=32,
        learning_rate=3e-4,
        discounting=0.95,
        batch_size=32,
        min_replay_size=256,
        normalize_observations=True,
        seed=0,
        reward_scaling=10,
        randomization_fn=rand_fn,
    )


if __name__ == '__main__':
  absltest.main()
