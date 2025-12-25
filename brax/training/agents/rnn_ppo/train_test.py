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

"""RNN-PPO tests."""

import functools
import pickle
from absl.testing import absltest
from absl.testing import parameterized
from brax import envs
from brax.training.acme import running_statistics
from brax.training.agents.rnn_ppo import networks as rnn_ppo_networks
from brax.training.agents.rnn_ppo import train as rnn_ppo
import jax
from jax import numpy as jnp


class RNNPPOTest(parameterized.TestCase):
  """Tests for RNN-PPO module."""

  @parameterized.parameters('ndarray', 'dict_state')
  def testTrain(self, obs_mode):
    """Test RNN-PPO with a simple env."""
    fast = envs.get_environment('fast', obs_mode=obs_mode)
    _, _, metrics = rnn_ppo.train(
        fast,
        num_timesteps=2**16,  # More steps for RNN to learn
        episode_length=128,
        num_envs=64,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        discounting=0.95,
        unroll_length=10,  # Longer unroll for recurrent network
        batch_size=64,
        num_minibatches=8,
        num_updates_per_batch=4,
        normalize_observations=True,
        seed=2,
        num_evals=3,
        reward_scaling=10,
        normalize_advantage=False,
    )
    # RNN-PPO may need more training to match feedforward PPO
    # The 'fast' env with these params reaches ~75-85 in this time
    self.assertGreater(metrics['eval/episode_reward'], 70)

  @parameterized.product(
      cell_type=['simple', 'gru', 'lstm'],
      normalize_mode=['welford', 'ema'],
  )
  def testTrainWithCellTypes(self, cell_type, normalize_mode):
    """Test RNN-PPO runs with different RNN cell types."""
    network_factory = functools.partial(
        rnn_ppo_networks.make_rnn_ppo_networks,
        cell_type=cell_type,
        policy_rnn_hidden_size=32,
        policy_output_layer_sizes=(32,),
        activation=jax.nn.elu,
    )

    _, _, _ = rnn_ppo.train(
        envs.get_environment('inverted_pendulum', backend='spring'),
        num_timesteps=2**13,
        episode_length=50,
        num_envs=64,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        discounting=0.95,
        unroll_length=5,
        batch_size=64,
        num_minibatches=8,
        num_updates_per_batch=4,
        normalize_observations=True,
        max_grad_norm=1.0,
        seed=2,
        reward_scaling=10,
        normalize_advantage=False,
        network_factory=network_factory,
        normalize_observations_mode=normalize_mode,
    )

  @parameterized.product(
      bootstrap_on_timeout=[True, False],
      clipping_epsilon_value=[None, 0.1],
  )
  def testTrainWithPPOParams(self, bootstrap_on_timeout, clipping_epsilon_value):
    """Test RNN-PPO runs with different PPO parameters."""
    _, _, _ = rnn_ppo.train(
        envs.get_environment('inverted_pendulum', backend='spring'),
        num_timesteps=2**13,
        episode_length=50,
        num_envs=64,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        discounting=0.95,
        unroll_length=5,
        batch_size=64,
        num_minibatches=8,
        num_updates_per_batch=4,
        normalize_observations=True,
        max_grad_norm=1.0,
        seed=2,
        reward_scaling=10,
        normalize_advantage=False,
        learning_rate_schedule='ADAPTIVE_KL',
        bootstrap_on_timeout=bootstrap_on_timeout,
        clipping_epsilon_value=clipping_epsilon_value,
    )

  def testTrainAsymmetricActorCritic(self):
    """Test RNN-PPO with asymmetric actor critic."""
    env = envs.get_environment(
        'fast', asymmetric_obs=True, obs_mode='dict_state'
    )

    network_factory = functools.partial(
        rnn_ppo_networks.make_rnn_ppo_networks,
        policy_rnn_hidden_size=32,
        policy_output_layer_sizes=(32,),
        value_hidden_layer_sizes=(32,),
        policy_obs_key='state',
        value_obs_key='privileged_state',
    )

    _, (_, policy_params, value_params), _ = rnn_ppo.train(
        env,
        num_timesteps=2**15,
        episode_length=1000,
        num_envs=64,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        discounting=0.95,
        unroll_length=5,
        batch_size=64,
        num_minibatches=8,
        num_updates_per_batch=4,
        normalize_observations=False,
        seed=2,
        reward_scaling=10,
        normalize_advantage=False,
        network_factory=network_factory,
    )

    # Check that value network uses privileged state size
    self.assertEqual(
        value_params['params']['hidden_0']['kernel'].shape[0],
        env.observation_size['privileged_state'],
    )

  @parameterized.parameters(True, False)
  def testNetworkEncoding(self, normalize_observations):
    """Test that network params can be pickled and restored."""
    env = envs.get_environment('fast')
    original_inference, params, _ = rnn_ppo.train(
        env,
        num_timesteps=128,
        episode_length=128,
        num_envs=128,
        normalize_observations=normalize_observations,
    )
    normalize_fn = lambda x, y: x
    if normalize_observations:
      normalize_fn = running_statistics.normalize
    rnn_ppo_network = rnn_ppo_networks.make_rnn_ppo_networks(
        env.observation_size, env.action_size, normalize_fn
    )
    inference = rnn_ppo_networks.make_inference_fn(rnn_ppo_network)
    byte_encoding = pickle.dumps(params)
    decoded_params = pickle.loads(byte_encoding)

    # Compute one action with recurrent policy
    state = env.reset(jax.random.PRNGKey(0))

    # Initialize hidden states
    policy_hidden = rnn_ppo_network.policy_network.init_hidden(1)
    value_hidden = rnn_ppo_network.value_network.init_hidden(1)
    hidden_states = (policy_hidden, value_hidden)

    # Get action from original inference
    original_action, _, _ = original_inference(decoded_params)(
        state.obs, hidden_states, jax.random.PRNGKey(0)
    )
    # Get action from reconstructed inference
    action, _, _ = inference(decoded_params)(
        state.obs, hidden_states, jax.random.PRNGKey(0)
    )
    self.assertSequenceEqual(list(original_action), list(action))
    env.step(state, action)

  def testTrainDomainRandomize(self):
    """Test RNN-PPO with domain randomization."""

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

    _, _, _ = rnn_ppo.train(
        envs.get_environment('inverted_pendulum', backend='spring'),
        num_timesteps=2**15,
        episode_length=1000,
        num_envs=64,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        discounting=0.95,
        unroll_length=5,
        batch_size=64,
        num_minibatches=8,
        num_updates_per_batch=4,
        normalize_observations=True,
        seed=2,
        reward_scaling=10,
        normalize_advantage=False,
        randomization_fn=rand_fn,
    )

  def testHiddenStateReset(self):
    """Test that hidden states are properly reset on episode boundaries."""
    env = envs.get_environment('fast')

    network_factory = functools.partial(
        rnn_ppo_networks.make_rnn_ppo_networks,
        cell_type='gru',
        policy_rnn_hidden_size=16,
    )

    rnn_ppo_network = network_factory(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=lambda x, y: x,
    )

    # Test hidden state initialization
    batch_size = 4
    policy_hidden = rnn_ppo_network.policy_network.init_hidden(batch_size)
    value_hidden = rnn_ppo_network.value_network.init_hidden(batch_size)

    self.assertEqual(policy_hidden.shape, (batch_size, 16))
    self.assertIsNone(value_hidden)  # Value network is non-recurrent

  @parameterized.parameters('simple', 'gru', 'lstm')
  def testHiddenStateShapes(self, cell_type):
    """Test hidden state shapes for different cell types."""
    env = envs.get_environment('fast')
    hidden_size = 32
    batch_size = 8

    network_factory = functools.partial(
        rnn_ppo_networks.make_rnn_ppo_networks,
        cell_type=cell_type,
        policy_rnn_hidden_size=hidden_size,
    )

    rnn_ppo_network = network_factory(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=lambda x, y: x,
    )

    policy_hidden = rnn_ppo_network.policy_network.init_hidden(batch_size)

    if cell_type == 'lstm':
      # LSTM has tuple state (carry, hidden)
      self.assertIsInstance(policy_hidden, tuple)
      self.assertEqual(len(policy_hidden), 2)
      self.assertEqual(policy_hidden[0].shape, (batch_size, hidden_size))
      self.assertEqual(policy_hidden[1].shape, (batch_size, hidden_size))
    else:
      # SimpleCell and GRU have single state
      self.assertEqual(policy_hidden.shape, (batch_size, hidden_size))


if __name__ == '__main__':
  jax.config.update('jax_threefry_partitionable', False)
  absltest.main()
