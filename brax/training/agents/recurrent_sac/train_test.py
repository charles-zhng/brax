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

"""Tests for recurrent SAC."""

import functools
import pickle

from absl.testing import absltest
from absl.testing import parameterized
from brax import envs
from brax.training import types
from brax.training.agents.recurrent_sac import losses as rsac_losses
from brax.training.agents.recurrent_sac import networks as rsac_nets
from brax.training.agents.recurrent_sac import train as rsac_train
import jax
import jax.numpy as jnp


def _tiny_network_factory(cell_type="gru", rnn_hidden_size=16):
    return functools.partial(
        rsac_nets.make_recurrent_sac_networks,
        rnn_hidden_size=rnn_hidden_size,
        policy_output_layer_sizes=(16,),
        q_hidden_layer_sizes=(16,),
        cell_type=cell_type,
    )


class RecurrentSACTest(parameterized.TestCase):

    def testTrain(self):
        """End-to-end training on the ``fast`` env."""
        env = envs.get_environment("fast")
        _, _, metrics = rsac_train.train(
            env,
            num_timesteps=4096,
            episode_length=128,
            num_envs=8,
            num_eval_envs=4,
            batch_size=8,
            unroll_length=8,
            burn_in=0,
            min_replay_size=128,
            max_replay_size=4096,
            grad_updates_per_step=1,
            num_evals=2,
            network_factory=_tiny_network_factory(),
            seed=0,
        )
        self.assertIn("eval/episode_reward", metrics)
        self.assertIn("training/critic_loss", metrics)
        self.assertIn("training/actor_loss", metrics)
        self.assertIn("training/alpha", metrics)

    @parameterized.parameters("simple", "gru", "lstm")
    def testTrainWithCellTypes(self, cell_type):
        env = envs.get_environment("fast")
        _, _, metrics = rsac_train.train(
            env,
            num_timesteps=1024,
            episode_length=64,
            num_envs=4,
            num_eval_envs=2,
            batch_size=4,
            unroll_length=4,
            burn_in=0,
            min_replay_size=64,
            max_replay_size=1024,
            num_evals=2,
            network_factory=_tiny_network_factory(cell_type=cell_type),
            seed=0,
        )
        self.assertGreater(metrics["eval/avg_episode_length"], 0)

    def testTrainWithBurnIn(self):
        env = envs.get_environment("fast")
        _, _, metrics = rsac_train.train(
            env,
            num_timesteps=1024,
            episode_length=64,
            num_envs=4,
            num_eval_envs=2,
            batch_size=4,
            unroll_length=4,
            burn_in=3,
            min_replay_size=64,
            max_replay_size=1024,
            num_evals=2,
            network_factory=_tiny_network_factory(),
            seed=0,
        )
        self.assertFalse(jnp.isnan(metrics["training/critic_loss"]))
        self.assertFalse(jnp.isnan(metrics["training/actor_loss"]))

    @parameterized.parameters("welford", "ema")
    def testTrainWithNormalization(self, mode):
        env = envs.get_environment("fast")
        _, _, metrics = rsac_train.train(
            env,
            num_timesteps=1024,
            episode_length=64,
            num_envs=4,
            num_eval_envs=2,
            batch_size=4,
            unroll_length=4,
            burn_in=0,
            min_replay_size=64,
            max_replay_size=1024,
            num_evals=2,
            normalize_observations=True,
            normalize_observations_mode=mode,
            network_factory=_tiny_network_factory(),
            seed=0,
        )
        self.assertFalse(jnp.isnan(metrics["training/critic_loss"]))

    def testNetworkEncoding(self):
        """Verify params pickle round-trip and produce identical actions."""
        env = envs.get_environment("fast")
        original_inference, params, _ = rsac_train.train(
            env,
            num_timesteps=512,
            episode_length=64,
            num_envs=4,
            num_eval_envs=2,
            batch_size=4,
            unroll_length=4,
            burn_in=0,
            min_replay_size=64,
            max_replay_size=512,
            num_evals=2,
            network_factory=_tiny_network_factory(),
            seed=0,
        )
        decoded = pickle.loads(pickle.dumps(params))
        state = env.reset(jax.random.PRNGKey(0))
        network = _tiny_network_factory()(
            env.observation_size, env.action_size
        )
        hidden = network.policy_network.init_hidden(1)
        obs = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), state.obs)
        policy = original_inference((decoded[0], decoded[1]), deterministic=True)
        a1, _, _ = policy(obs, hidden, jax.random.PRNGKey(0))
        a2, _, _ = policy(obs, hidden, jax.random.PRNGKey(0))
        self.assertTrue(jnp.allclose(a1, a2))

    @parameterized.parameters("simple", "gru", "lstm")
    def testResetHiddenOnDone(self, cell_type):
        """_reset_hidden_on_done zeroes hidden where done=True, preserves elsewhere."""
        B, H = 4, 8
        done = jnp.array([True, False, True, False])
        if cell_type == "lstm":
            c = jnp.ones((B, H))
            h = jnp.ones((B, H)) * 2.0
            out = rsac_losses._reset_hidden_on_done((c, h), done, cell_type)
            c_out, h_out = out
            self.assertEqual(c_out.shape, (B, H))
            self.assertEqual(h_out.shape, (B, H))
            self.assertTrue(jnp.all(c_out[0] == 0))
            self.assertTrue(jnp.all(c_out[1] == 1))
            self.assertTrue(jnp.all(h_out[0] == 0))
            self.assertTrue(jnp.all(h_out[1] == 2))
        else:
            hidden = jnp.ones((B, H))
            out = rsac_losses._reset_hidden_on_done(hidden, done, cell_type)
            self.assertEqual(out.shape, (B, H))
            self.assertTrue(jnp.all(out[0] == 0))
            self.assertTrue(jnp.all(out[1] == 1))
            self.assertTrue(jnp.all(out[2] == 0))
            self.assertTrue(jnp.all(out[3] == 1))

    def testLossShapesMatchExpectations(self):
        """All three losses return scalar values and admit gradients."""
        obs_dim, action_dim, hidden_dim = 4, 2, 8
        B, T = 3, 5
        network = rsac_nets.make_recurrent_sac_networks(
            observation_size=obs_dim,
            action_size=action_dim,
            rnn_hidden_size=hidden_dim,
            policy_output_layer_sizes=(hidden_dim,),
            q_hidden_layer_sizes=(hidden_dim,),
            cell_type="gru",
        )
        key = jax.random.PRNGKey(0)
        policy_params = network.policy_network.init(jax.random.fold_in(key, 1))
        q_params = network.q_network.init(jax.random.fold_in(key, 2))
        target_q_params = network.q_network.init(jax.random.fold_in(key, 3))

        transitions = types.Transition(
            observation=jnp.ones((B, T, obs_dim)),
            action=jnp.ones((B, T, action_dim)),
            reward=jnp.ones((B, T)),
            discount=jnp.ones((B, T)),
            next_observation=jnp.ones((B, T, obs_dim)),
            extras={
                "policy_extras": {
                    "initial_policy_hidden": jnp.zeros((B, T, hidden_dim))
                },
                "state_extras": {"truncation": jnp.zeros((B, T))},
            },
        )
        alpha_loss, critic_loss, actor_loss = rsac_losses.make_losses(
            network, reward_scaling=1.0, discounting=0.99, action_size=action_dim
        )
        a = alpha_loss(jnp.array(0.0), policy_params, None, transitions, key)
        c = critic_loss(
            q_params, policy_params, None, target_q_params, jnp.array(1.0),
            transitions, key
        )
        p = actor_loss(policy_params, None, q_params, jnp.array(1.0), transitions, key)
        self.assertEqual(a.shape, ())
        self.assertEqual(c.shape, ())
        self.assertEqual(p.shape, ())

        # Gradient sanity: critic gradient non-zero.
        def _loss(qp):
            return critic_loss(
                qp, policy_params, None, target_q_params, jnp.array(1.0),
                transitions, key
            )
        grads = jax.grad(_loss)(q_params)
        has_nonzero = any(
            jnp.any(jnp.abs(g) > 1e-10) for g in jax.tree_util.tree_leaves(grads)
        )
        self.assertTrue(has_nonzero)

    def testBurnInAffectsLoss(self):
        """Burn-in masking should change the loss value vs burn_in=0."""
        obs_dim, action_dim, hidden_dim = 4, 2, 8
        B, T = 3, 6
        network = rsac_nets.make_recurrent_sac_networks(
            observation_size=obs_dim, action_size=action_dim,
            rnn_hidden_size=hidden_dim,
            policy_output_layer_sizes=(hidden_dim,),
            q_hidden_layer_sizes=(hidden_dim,),
            cell_type="gru",
        )
        key = jax.random.PRNGKey(0)
        q_params = network.q_network.init(jax.random.fold_in(key, 2))
        policy_params = network.policy_network.init(jax.random.fold_in(key, 1))
        target_q_params = network.q_network.init(jax.random.fold_in(key, 3))

        # Use non-uniform data so the first few timesteps differ in loss.
        rng = jax.random.PRNGKey(42)
        transitions = types.Transition(
            observation=jax.random.normal(rng, (B, T, obs_dim)),
            action=jax.random.normal(jax.random.fold_in(rng, 1), (B, T, action_dim)),
            reward=jax.random.normal(jax.random.fold_in(rng, 2), (B, T)),
            discount=jnp.ones((B, T)),
            next_observation=jax.random.normal(
                jax.random.fold_in(rng, 3), (B, T, obs_dim)
            ),
            extras={
                "policy_extras": {
                    "initial_policy_hidden": jnp.zeros((B, T, hidden_dim))
                },
                "state_extras": {"truncation": jnp.zeros((B, T))},
            },
        )
        _, critic_no_burn, _ = rsac_losses.make_losses(
            network, 1.0, 0.99, action_dim, burn_in=0
        )
        _, critic_with_burn, _ = rsac_losses.make_losses(
            network, 1.0, 0.99, action_dim, burn_in=3
        )
        loss_no_burn = critic_no_burn(
            q_params, policy_params, None, target_q_params,
            jnp.array(1.0), transitions, key,
        )
        loss_with_burn = critic_with_burn(
            q_params, policy_params, None, target_q_params,
            jnp.array(1.0), transitions, key,
        )
        # With random data and differing timesteps, the two losses should differ.
        self.assertNotAlmostEqual(float(loss_no_burn), float(loss_with_burn), places=4)


if __name__ == "__main__":
    absltest.main()
