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

"""Tests for recurrent SAC checkpointing."""

import functools

from absl import flags
from absl.testing import absltest
from brax.training.acme import running_statistics
from brax.training.agents.recurrent_sac import checkpoint
from brax.training.agents.recurrent_sac import networks as rsac_networks
from etils import epath
import jax
from jax import numpy as jp


class CheckpointTest(absltest.TestCase):

    def setUp(self):
        super().setUp()
        flags.FLAGS.mark_as_parsed()

    def test_network_config_captures_factory_kwargs(self):
        network_factory = functools.partial(
            rsac_networks.make_recurrent_sac_networks,
            rnn_hidden_size=32,
            policy_output_layer_sizes=(16,),
            q_hidden_layer_sizes=(16,),
            cell_type="gru",
        )
        config = checkpoint.network_config(
            action_size=2,
            observation_size=4,
            normalize_observations=True,
            network_factory=network_factory,
        )
        kw = config.network_factory_kwargs.to_dict()
        self.assertEqual(kw["rnn_hidden_size"], 32)
        self.assertEqual(kw["policy_output_layer_sizes"], (16,))
        self.assertEqual(kw["q_hidden_layer_sizes"], (16,))
        self.assertEqual(kw["cell_type"], "gru")
        self.assertEqual(config.action_size, 2)
        self.assertEqual(config.observation_size, 4)

    def test_save_and_load_checkpoint(self):
        path = self.create_tempdir("test")
        network_factory = functools.partial(
            rsac_networks.make_recurrent_sac_networks,
            rnn_hidden_size=16,
            policy_output_layer_sizes=(16,),
            q_hidden_layer_sizes=(16,),
            cell_type="gru",
        )
        config = checkpoint.network_config(
            observation_size=4,
            action_size=2,
            normalize_observations=True,
            network_factory=network_factory,
        )

        normalize = running_statistics.normalize
        network = network_factory(
            config.observation_size,
            config.action_size,
            preprocess_observations_fn=normalize,
            **config.network_factory_kwargs,
        )
        key = jax.random.PRNGKey(0)
        policy_params = network.policy_network.init(jax.random.fold_in(key, 1))
        q_params = network.q_network.init(jax.random.fold_in(key, 2))
        normalizer_params = running_statistics.init_state(
            jax.tree_util.tree_map(jp.zeros, config.observation_size),
            std_eps=0.03,
        )
        params = (normalizer_params, policy_params, q_params)

        checkpoint.save(path.full_path, step=1, params=params, config=config)

        policy_fn = checkpoint.load_policy(
            epath.Path(path.full_path) / "000000000001",
        )

        batch_size = 1
        obs = jp.zeros((batch_size, 4))

        # Two-argument form: hidden is auto-initialized.
        out = policy_fn(obs, jax.random.PRNGKey(0))
        self.assertEqual(out[0].shape, (batch_size, 2))

        # Three-argument form: explicit hidden state.
        hidden = policy_fn.init_hidden(batch_size)
        out = policy_fn(obs, hidden, jax.random.PRNGKey(0))
        self.assertEqual(out[0].shape, (batch_size, 2))

        # Raw load returns (normalizer, policy, q).
        loaded = checkpoint.load(epath.Path(path.full_path) / "000000000001")
        self.assertEqual(loaded[0].std_eps, 0.03)
        self.assertIn("params", loaded[1])
        self.assertIn("params", loaded[2])


if __name__ == "__main__":
    absltest.main()
