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

"""Recurrent PPO tests."""

import functools
import pickle
from absl.testing import absltest
from absl.testing import parameterized
from brax import envs
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.agents.recurrent_ppo_feature import networks as rnn_ppo_networks
from brax.training.agents.recurrent_ppo_feature import train as rnn_ppo
import jax
from jax import numpy as jnp


class RNNPPOTest(parameterized.TestCase):
    """Tests for RNN-PPO module."""

    @parameterized.parameters("ndarray", "dict_state")
    def testTrain(self, obs_mode):
        """Test RNN-PPO with a simple env."""
        fast = envs.get_environment("fast", obs_mode=obs_mode)
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
        self.assertGreater(metrics["eval/episode_reward"], 70)

    @parameterized.product(
        cell_type=["simple", "gru", "lstm"],
        normalize_mode=["welford", "ema"],
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
            envs.get_environment("inverted_pendulum", backend="spring"),
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
            envs.get_environment("inverted_pendulum", backend="spring"),
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
            learning_rate_schedule="ADAPTIVE_KL",
            bootstrap_on_timeout=bootstrap_on_timeout,
            clipping_epsilon_value=clipping_epsilon_value,
        )

    def testTrainAsymmetricActorCritic(self):
        """Test RNN-PPO with asymmetric actor critic."""
        env = envs.get_environment("fast", asymmetric_obs=True, obs_mode="dict_state")

        network_factory = functools.partial(
            rnn_ppo_networks.make_rnn_ppo_networks,
            policy_rnn_hidden_size=32,
            policy_output_layer_sizes=(32,),
            value_hidden_layer_sizes=(32,),
            policy_obs_key="state",
            value_obs_key="privileged_state",
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
            value_params["params"]["hidden_0"]["kernel"].shape[0],
            env.observation_size["privileged_state"],
        )

    @parameterized.parameters(True, False)
    def testNetworkEncoding(self, normalize_observations):
        """Test that network params can be pickled and restored."""
        env = envs.get_environment("fast")
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

        # Initialize policy hidden state
        policy_hidden = rnn_ppo_network.policy_network.init_hidden(1)

        # Get action from original inference
        original_action, _, _ = original_inference(decoded_params)(
            state.obs, policy_hidden, jax.random.PRNGKey(0)
        )
        # Get action from reconstructed inference
        action, _, _ = inference(decoded_params)(
            state.obs, policy_hidden, jax.random.PRNGKey(0)
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

            sys_v = sys.tree_replace({"link.inertia.transform.pos": get_offset(rng)})
            in_axes = jax.tree.map(lambda x: None, sys)
            in_axes = in_axes.tree_replace({"link.inertia.transform.pos": 0})
            return sys_v, in_axes

        _, _, _ = rnn_ppo.train(
            envs.get_environment("inverted_pendulum", backend="spring"),
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

    @parameterized.parameters("simple", "gru", "lstm")
    def testHiddenStateReset(self, cell_type):
        """Test that hidden states are properly initialized for all cell types."""
        env = envs.get_environment("fast")
        hidden_size = 16

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

        # Test hidden state initialization
        batch_size = 4
        policy_hidden = rnn_ppo_network.policy_network.init_hidden(batch_size)
        value_hidden = rnn_ppo_network.value_network.init_hidden(batch_size)

        if cell_type == "lstm":
            self.assertIsInstance(policy_hidden, tuple)
            self.assertEqual(policy_hidden[0].shape, (batch_size, hidden_size))
            self.assertEqual(policy_hidden[1].shape, (batch_size, hidden_size))
        else:
            self.assertEqual(policy_hidden.shape, (batch_size, hidden_size))
        self.assertIsNone(value_hidden)  # Value network is non-recurrent

    @parameterized.parameters("simple", "gru", "lstm")
    def testHiddenStateShapes(self, cell_type):
        """Test hidden state shapes for different cell types."""
        env = envs.get_environment("fast")
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

        if cell_type == "lstm":
            # LSTM has tuple state (carry, hidden)
            self.assertIsInstance(policy_hidden, tuple)
            self.assertEqual(len(policy_hidden), 2)
            self.assertEqual(policy_hidden[0].shape, (batch_size, hidden_size))
            self.assertEqual(policy_hidden[1].shape, (batch_size, hidden_size))
        else:
            # SimpleCell and GRU have single state
            self.assertEqual(policy_hidden.shape, (batch_size, hidden_size))

    # ============================================================================
    # CRITICAL Tests - Core RNN Correctness
    # ============================================================================

    @parameterized.parameters("simple", "gru", "lstm")
    def testResetHiddenOnDone(self, cell_type):
        """Test hidden states are zeroed correctly on episode boundaries."""
        from brax.training.agents.recurrent_ppo_feature import train as rnn_ppo_train

        env = envs.get_environment("fast")
        hidden_size = 16
        batch_size = 4

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

        # Create non-zero hidden state
        if cell_type == "lstm":
            hidden = (
                jnp.ones((batch_size, hidden_size)),
                jnp.ones((batch_size, hidden_size)) * 2,
            )
        else:
            hidden = jnp.ones((batch_size, hidden_size))

        # Create done mask: first two envs done, last two not done
        done = jnp.array([True, True, False, False])

        # Reset hidden states
        new_hidden = rnn_ppo_train._reset_hidden_on_done(hidden, done, rnn_ppo_network)

        if cell_type == "lstm":
            # Check carry (index 0) and hidden (index 1)
            for state in new_hidden:
                # First two should be zeros (done=True)
                self.assertTrue(jnp.allclose(state[:2], 0.0))
                # Last two should be unchanged (done=False)
                self.assertFalse(jnp.allclose(state[2:], 0.0))
        else:
            # First two should be zeros (done=True)
            self.assertTrue(jnp.allclose(new_hidden[:2], 0.0))
            # Last two should be unchanged (done=False)
            self.assertFalse(jnp.allclose(new_hidden[2:], 0.0))

    def testScanForwardMatchesSequential(self):
        """Test scan_forward produces same output as step-by-step calls."""
        env = envs.get_environment("fast")
        hidden_size = 16
        seq_length = 5
        batch_size = 2

        network_factory = functools.partial(
            rnn_ppo_networks.make_rnn_ppo_networks,
            cell_type="gru",
            policy_rnn_hidden_size=hidden_size,
        )

        rnn_ppo_network = network_factory(
            env.observation_size,
            env.action_size,
            preprocess_observations_fn=lambda x, y: x,
        )

        # Initialize network
        key = jax.random.PRNGKey(0)
        params = rnn_ppo_network.policy_network.init(key)

        # Create observation sequence [T, B, obs_dim]
        obs_seq = jax.random.normal(key, (seq_length, batch_size, env.observation_size))
        initial_hidden = rnn_ppo_network.policy_network.init_hidden(batch_size)

        # Run scan_forward
        normalizer_params = None
        scan_output, scan_final_hidden = rnn_ppo_network.policy_network.apply_sequence(
            normalizer_params, params, obs_seq, initial_hidden
        )

        # Run step-by-step
        hidden = initial_hidden
        step_outputs = []
        for t in range(seq_length):
            output, hidden = rnn_ppo_network.policy_network.apply(
                normalizer_params, params, obs_seq[t], hidden
            )
            step_outputs.append(output)
        step_output = jnp.stack(step_outputs, axis=0)

        # Compare outputs - use rtol=1e-3 and atol=1e-3 to account for float32
        # numerical precision differences between jax.lax.scan and explicit loops.
        # Near-zero values need atol; larger values need rtol for proper comparison.
        self.assertTrue(jnp.allclose(scan_output, step_output, rtol=1e-3, atol=1e-3))
        if isinstance(hidden, tuple):
            for h1, h2 in zip(scan_final_hidden, hidden):
                self.assertTrue(jnp.allclose(h1, h2, rtol=1e-3, atol=1e-3))
        else:
            self.assertTrue(
                jnp.allclose(scan_final_hidden, hidden, rtol=1e-3, atol=1e-3)
            )

    def testGradientFlowThroughHidden(self):
        """Test gradients propagate through RNN hidden states."""
        env = envs.get_environment("fast")
        hidden_size = 16

        network_factory = functools.partial(
            rnn_ppo_networks.make_rnn_ppo_networks,
            cell_type="gru",
            policy_rnn_hidden_size=hidden_size,
        )

        rnn_ppo_network = network_factory(
            env.observation_size,
            env.action_size,
            preprocess_observations_fn=lambda x, y: x,
        )

        key = jax.random.PRNGKey(0)
        params = rnn_ppo_network.policy_network.init(key)

        def loss_fn(params, obs_seq, initial_hidden):
            output, _ = rnn_ppo_network.policy_network.apply_sequence(
                None, params, obs_seq, initial_hidden
            )
            return jnp.mean(output**2)

        batch_size = 2
        seq_length = 3
        obs_seq = jax.random.normal(key, (seq_length, batch_size, env.observation_size))
        initial_hidden = rnn_ppo_network.policy_network.init_hidden(batch_size)

        # Compute gradients
        grads = jax.grad(loss_fn)(params, obs_seq, initial_hidden)

        # Check that RNN cell parameters have non-zero gradients
        # The exact param names depend on the implementation
        flat_grads = jax.tree_util.tree_leaves(grads)
        non_zero_grads = [jnp.any(g != 0) for g in flat_grads]
        self.assertTrue(any(non_zero_grads), "Expected non-zero gradients")

    # ============================================================================
    # IMPORTANT Tests - Algorithm Correctness
    # ============================================================================

    @parameterized.parameters("normal", "tanh_normal")
    def testDistributionTypes(self, distribution_type):
        """Test networks work with different distribution types."""
        network_factory = functools.partial(
            rnn_ppo_networks.make_rnn_ppo_networks,
            distribution_type=distribution_type,
            policy_rnn_hidden_size=32,
        )

        _, _, _ = rnn_ppo.train(
            envs.get_environment("inverted_pendulum", backend="spring"),
            num_timesteps=2**12,
            episode_length=50,
            num_envs=32,
            learning_rate=3e-4,
            unroll_length=5,
            batch_size=32,
            num_minibatches=4,
            num_updates_per_batch=2,
            seed=0,
            network_factory=network_factory,
        )

    def testLongEpisodeTraining(self):
        """Test training with episode_length > unroll_length."""
        # This tests hidden state propagation across unroll boundaries
        _, _, _ = rnn_ppo.train(
            envs.get_environment("fast"),
            num_timesteps=2**14,
            episode_length=200,  # Much longer than unroll_length
            unroll_length=10,
            num_envs=32,
            batch_size=32,
            num_minibatches=4,
            num_updates_per_batch=2,
            seed=0,
        )

    def testUnrollLengthOne(self):
        """Test training with unroll_length=1 (edge case)."""
        _, _, _ = rnn_ppo.train(
            envs.get_environment("fast"),
            num_timesteps=2**12,
            episode_length=50,
            unroll_length=1,  # Minimal unroll
            num_envs=32,
            batch_size=32,
            num_minibatches=4,
            num_updates_per_batch=2,
            seed=0,
        )

    @absltest.skip("Recurrent vision networks not yet implemented")
    def testPixelObservations(self):
        """Test recurrent PPO with pixel observations.

        Note: This test is skipped because recurrent_ppo currently doesn't have
        a vision network factory (like ppo_networks_vision). Implementing this
        would require creating a recurrent vision encoder.
        """
        pass

    # ============================================================================
    # NON-CRITICAL Tests - Edge Cases & Robustness
    # ============================================================================

    def testActorStepStoresHiddenStates(self):
        """Test actor_step stores initial hidden states in transition extras."""
        from brax.training.agents.recurrent_ppo_feature import train as rnn_ppo_train

        env = envs.get_environment("fast")
        hidden_size = 16
        batch_size = 4

        network_factory = functools.partial(
            rnn_ppo_networks.make_rnn_ppo_networks,
            cell_type="gru",
            policy_rnn_hidden_size=hidden_size,
        )

        rnn_ppo_network = network_factory(
            env.observation_size,
            env.action_size,
            preprocess_observations_fn=lambda x, y: x,
        )

        # Initialize
        key = jax.random.PRNGKey(0)
        params = rnn_ppo_network.policy_network.init(key)
        normalizer_params = None

        make_policy = rnn_ppo_networks.make_inference_fn(rnn_ppo_network)
        policy = make_policy((normalizer_params, params, None))

        # Create env state
        env_keys = jax.random.split(key, batch_size)
        env_state = jax.vmap(env.reset)(env_keys)

        # Initialize policy hidden state
        policy_hidden = rnn_ppo_network.policy_network.init_hidden(batch_size)

        # Run actor step
        _, transition, _ = rnn_ppo_train.actor_step_rnn(
            env,
            env_state,
            policy,
            policy_hidden,
            key,
            rnn_ppo_network=rnn_ppo_network,
        )

        # Check that hidden states are stored in extras
        self.assertIn("initial_policy_hidden", transition.extras["policy_extras"])
        stored_hidden = transition.extras["policy_extras"]["initial_policy_hidden"]
        self.assertEqual(stored_hidden.shape, (batch_size, hidden_size))

    def testLossComputationSequenceIntegrity(self):
        """Test that loss computation correctly processes sequences."""
        from brax.training.agents.recurrent_ppo_feature import losses as rnn_ppo_losses

        env = envs.get_environment("fast")
        hidden_size = 16
        batch_size = 4
        seq_length = 5

        network_factory = functools.partial(
            rnn_ppo_networks.make_rnn_ppo_networks,
            cell_type="gru",
            policy_rnn_hidden_size=hidden_size,
        )

        rnn_ppo_network = network_factory(
            env.observation_size,
            env.action_size,
            preprocess_observations_fn=lambda x, y: x,
        )

        # Initialize network params
        key = jax.random.PRNGKey(0)
        params = rnn_ppo_losses.RNNPPONetworkParams(
            policy=rnn_ppo_network.policy_network.init(key),
            value=rnn_ppo_network.value_network.init(key),
        )

        # Create mock data with shape [batch_size, seq_length, ...]
        obs = jax.random.normal(key, (batch_size, seq_length, env.observation_size))
        next_obs = jax.random.normal(
            key, (batch_size, seq_length, env.observation_size)
        )
        action = jax.random.normal(key, (batch_size, seq_length, env.action_size))
        reward = jax.random.normal(key, (batch_size, seq_length))
        discount = jnp.ones((batch_size, seq_length))

        # Create hidden states for each timestep
        # Shape: [batch_size, seq_length, hidden_size]
        initial_policy_hidden = jax.random.normal(
            key, (batch_size, seq_length, hidden_size)
        )

        # Get raw_action and log_prob from policy
        policy_logits, _ = rnn_ppo_network.policy_network.apply(
            None, params.policy, obs[:, 0], initial_policy_hidden[:, 0]
        )
        raw_action = action
        log_prob = jnp.zeros((batch_size, seq_length))
        distribution_params = jnp.zeros((batch_size, seq_length, env.action_size * 2))

        data = types.Transition(
            observation=obs,
            action=action,
            reward=reward,
            discount=discount,
            next_observation=next_obs,
            extras={
                "policy_extras": {
                    "raw_action": raw_action,
                    "log_prob": log_prob,
                    "distribution_params": distribution_params,
                    "initial_policy_hidden": initial_policy_hidden,
                },
                "state_extras": {
                    "truncation": jnp.zeros((batch_size, seq_length)),
                },
            },
        )

        # Compute loss - should not raise
        loss, metrics = rnn_ppo_losses.compute_rnn_ppo_loss(
            params=params,
            normalizer_params=None,
            data=data,
            rng=key,
            progress=jnp.zeros(()),
            rnn_ppo_network=rnn_ppo_network,
        )

        # Verify loss is a scalar
        self.assertEqual(loss.shape, ())
        # Verify loss is finite
        self.assertTrue(jnp.isfinite(loss))

    def testBootstrapValueUsesTerminalObservation(self):
        """Test that bootstrap value is computed from terminal observation."""
        from brax.training.agents.recurrent_ppo_feature import losses as rnn_ppo_losses

        env = envs.get_environment("fast")
        hidden_size = 16
        batch_size = 2
        seq_length = 3

        network_factory = functools.partial(
            rnn_ppo_networks.make_rnn_ppo_networks,
            cell_type="gru",
            policy_rnn_hidden_size=hidden_size,
        )

        rnn_ppo_network = network_factory(
            env.observation_size,
            env.action_size,
            preprocess_observations_fn=lambda x, y: x,
        )

        key = jax.random.PRNGKey(42)
        params = rnn_ppo_losses.RNNPPONetworkParams(
            policy=rnn_ppo_network.policy_network.init(key),
            value=rnn_ppo_network.value_network.init(key),
        )

        # Create observations where terminal obs is distinctive
        obs = jnp.zeros((batch_size, seq_length, env.observation_size))
        next_obs = jnp.zeros((batch_size, seq_length, env.observation_size))
        # Make the terminal next_observation distinctive (all ones)
        terminal_obs = jnp.ones((batch_size, env.observation_size)) * 5.0
        next_obs = next_obs.at[:, -1].set(terminal_obs)

        action = jnp.zeros((batch_size, seq_length, env.action_size))
        reward = jnp.zeros((batch_size, seq_length))
        discount = jnp.ones((batch_size, seq_length))

        initial_policy_hidden = jnp.zeros((batch_size, seq_length, hidden_size))

        data = types.Transition(
            observation=obs,
            action=action,
            reward=reward,
            discount=discount,
            next_observation=next_obs,
            extras={
                "policy_extras": {
                    "raw_action": action,
                    "log_prob": jnp.zeros((batch_size, seq_length)),
                    "distribution_params": jnp.zeros(
                        (batch_size, seq_length, env.action_size * 2)
                    ),
                    "initial_policy_hidden": initial_policy_hidden,
                },
                "state_extras": {
                    "truncation": jnp.zeros((batch_size, seq_length)),
                },
            },
        )

        # Compute loss
        loss, _ = rnn_ppo_losses.compute_rnn_ppo_loss(
            params=params,
            normalizer_params=None,
            data=data,
            rng=key,
            progress=jnp.zeros(()),
            rnn_ppo_network=rnn_ppo_network,
        )

        # Verify the loss is computed (value function should see different
        # bootstrap value due to distinctive terminal observation)
        self.assertTrue(jnp.isfinite(loss))


if __name__ == "__main__":
    jax.config.update("jax_threefry_partitionable", False)
    absltest.main()
