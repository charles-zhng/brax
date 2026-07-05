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

"""Recurrent SAC networks.

Reuses the recurrent module primitives defined in ``recurrent_ppo.networks``:
  - ``RecurrentMLP`` / ``RecurrentPolicyModule`` for the policy,
  - ``RecurrentMLP`` with a 2-unit output head for the twin Q network.

Both policy and Q networks share ``cell_type`` and ``rnn_hidden_size``. The Q
network operates over ``concat(obs, action)`` at each timestep and returns a
``[..., 2]`` tensor (two critic heads, sharing the recurrent backbone).
"""

from typing import Any, Callable, Literal, Mapping, Sequence, Tuple

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.agents.recurrent_ppo import networks as rnn_shared
from brax.training.types import PRNGKey
import flax
from flax import linen
import jax
import jax.numpy as jnp


HiddenState = rnn_shared.HiddenState
RNNCellType = rnn_shared.RNNCellType
ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


@flax.struct.dataclass
class RecurrentSACNetworks:
    """Container for recurrent SAC networks."""

    policy_network: rnn_shared.RecurrentNetwork
    q_network: rnn_shared.RecurrentNetwork
    parametric_action_distribution: distribution.ParametricDistribution
    rnn_hidden_size: int
    cell_type: RNNCellType


def _get_obs_state_size(obs_size: types.ObservationSize, obs_key: str) -> int:
    obs_size = obs_size[obs_key] if isinstance(obs_size, Mapping) else obs_size
    return jax.tree_util.tree_flatten(obs_size)[0][-1]


def make_inference_fn(recurrent_sac_networks: RecurrentSACNetworks):
    """Creates an inference function for the recurrent SAC agent.

    The returned ``policy`` has signature
    ``(observations, policy_hidden, key_sample) -> (action, extras, new_policy_hidden)``.
    """

    def make_policy(
        params: types.Params, deterministic: bool = False
    ) -> Callable:
        policy_network = recurrent_sac_networks.policy_network
        parametric_action_distribution = (
            recurrent_sac_networks.parametric_action_distribution
        )

        def policy(
            observations: types.Observation,
            policy_hidden: HiddenState,
            key_sample: PRNGKey,
        ) -> Tuple[types.Action, types.Extra, HiddenState]:
            normalizer_params, policy_params = params[0], params[1]
            key_policy = jax.random.fold_in(key_sample, 1)
            logits, new_policy_hidden = policy_network.apply(
                normalizer_params,
                policy_params,
                observations,
                policy_hidden,
                rng=key_policy,
            )

            if deterministic:
                actions = parametric_action_distribution.mode(logits)
                return actions, {"policy_hidden": new_policy_hidden}, new_policy_hidden

            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            actions = parametric_action_distribution.postprocess(raw_actions)

            extras = {
                "log_prob": log_prob,
                "raw_action": raw_actions,
                "distribution_params": logits,
                "policy_hidden": new_policy_hidden,
            }
            return actions, extras, new_policy_hidden

        return policy

    return make_policy


def make_recurrent_sac_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    rnn_hidden_size: int = 256,
    policy_output_layer_sizes: Sequence[int] = (256,),
    q_hidden_layer_sizes: Sequence[int] = (256,),
    cell_type: RNNCellType = "gru",
    activation: ActivationFn = linen.relu,
    policy_obs_key: str = "state",
    q_obs_key: str = "state",
    distribution_type: Literal["normal", "tanh_normal", "sigmoid_normal"] = "tanh_normal",
    noise_std_type: Literal["scalar", "log"] = "scalar",
    init_noise_std: float = 1.0,
    state_dependent_std: bool = False,
    policy_network_kernel_init_fn: Initializer = jax.nn.initializers.lecun_uniform,
    policy_network_kernel_init_kwargs: Mapping[str, Any] | None = None,
    q_network_kernel_init_fn: Initializer = jax.nn.initializers.lecun_uniform,
    q_network_kernel_init_kwargs: Mapping[str, Any] | None = None,
    mean_kernel_init_fn: Initializer | None = None,
    mean_kernel_init_kwargs: Mapping[str, Any] | None = None,
    mean_bias_init_fn: Initializer | None = None,
    mean_bias_init_kwargs: Mapping[str, Any] | None = None,
    policy_network_layer_norm: bool = False,
    q_network_layer_norm: bool = False,
    shared_q_backbone: bool = True,
    policy_input_skip: bool = False,
    q_input_skip: bool = False,
) -> RecurrentSACNetworks:
    """Build recurrent SAC networks.

    Policy: obs -> RNN cell -> MLP -> action distribution parameters.
    Q network: concat(obs, action) -> RNN cell -> MLP -> [2] (twin heads).

    Args:
      observation_size: Observation shape / dict of shapes.
      action_size: Dimensionality of the action space.
      preprocess_observations_fn: Observation preprocessor (e.g. normalization).
      rnn_hidden_size: Hidden size shared by policy and Q RNN cells.
      policy_output_layer_sizes: MLP layer sizes after the policy RNN (excluding
        the final distribution-parameter layer, which is added automatically).
      q_hidden_layer_sizes: MLP layer sizes after the Q RNN (excluding the final
        2-unit twin-head output).
      cell_type: RNN cell type: ``"simple"``, ``"gru"``, or ``"lstm"``.
      activation: Activation function used in the MLP layers.
      policy_obs_key: Key to select from a dict observation for the policy.
      q_obs_key: Key to select from a dict observation for the Q network. Allows
        asymmetric actor-critic (privileged Q).
      distribution_type: ``"tanh_normal"`` (default, matches stock SAC) or
        ``"normal"``.
      noise_std_type: ``"scalar"`` or ``"log"`` when ``distribution_type``
        is ``"normal"``.
      init_noise_std: Initial std for non-tanh normal distributions.
      state_dependent_std: Whether the std is produced by the network.
      policy_network_kernel_init_fn: Kernel init factory for the policy.
      policy_network_kernel_init_kwargs: Kwargs for the policy kernel init.
      q_network_kernel_init_fn: Kernel init factory for the Q network.
      q_network_kernel_init_kwargs: Kwargs for the Q kernel init.
      shared_q_backbone: If True (current default), the twin Q heads share one
        recurrent backbone + MLP trunk and differ only in the final 2-unit
        layer. If False, two fully independent critics are built (stock-SAC
        style) and vmapped over stacked params; hidden states get a leading
        [2] axis, opaque to the losses.
      policy_input_skip: Concatenate the (normalized) observation to the RNN
        output before the policy MLP head — a direct feedforward path around
        the cell, making the policy a strict superset of an MLP policy.
      q_input_skip: Same for the Q network's concat(obs, action) input —
        notably gives the critic a direct dQ/da path around the cell.

    Returns:
      A ``RecurrentSACNetworks`` container.
    """
    policy_kernel_init_kwargs = policy_network_kernel_init_kwargs or {}
    q_kernel_init_kwargs = q_network_kernel_init_kwargs or {}
    policy_kernel_init = policy_network_kernel_init_fn(**policy_kernel_init_kwargs)
    q_kernel_init = q_network_kernel_init_fn(**q_kernel_init_kwargs)
    mean_kernel_init = (
        mean_kernel_init_fn(**(mean_kernel_init_kwargs or {}))
        if mean_kernel_init_fn is not None
        else None
    )
    mean_bias_init = (
        mean_bias_init_fn(**(mean_bias_init_kwargs or {}))
        if mean_bias_init_fn is not None
        else None
    )

    parametric_action_distribution: distribution.ParametricDistribution
    if distribution_type == "normal":
        parametric_action_distribution = distribution.NormalDistribution(
            event_size=action_size
        )
    elif distribution_type == "tanh_normal":
        parametric_action_distribution = distribution.NormalTanhDistribution(
            event_size=action_size
        )
    elif distribution_type == "sigmoid_normal":
        parametric_action_distribution = distribution.NormalSigmoidDistribution(
            event_size=action_size
        )
    else:
        raise ValueError(
            f"Unsupported distribution type: {distribution_type}. Must be one"
            ' of "normal", "tanh_normal", or "sigmoid_normal".'
        )

    policy_obs_size = _get_obs_state_size(observation_size, policy_obs_key)
    q_obs_size = _get_obs_state_size(observation_size, q_obs_key)

    # --- Policy module -----------------------------------------------------
    # tanh_normal and sigmoid_normal both have learned std (param_size = 2*event_size)
    # via softplus on the second half of the output. normal goes through
    # RecurrentPolicyModule which has a separately-parameterized scale head.
    if distribution_type in ("tanh_normal", "sigmoid_normal"):
        policy_module = rnn_shared.RecurrentMLP(
            rnn_hidden_size=rnn_hidden_size,
            output_layer_sizes=list(policy_output_layer_sizes)
            + [parametric_action_distribution.param_size],
            cell_type=cell_type,
            activation=activation,
            kernel_init=policy_kernel_init,
            activate_final=False,
            mean_kernel_init=mean_kernel_init,
            mean_bias_init=mean_bias_init,
            layer_norm=policy_network_layer_norm,
            input_skip=policy_input_skip,
        )
    else:
        policy_module = rnn_shared.RecurrentPolicyModule(
            param_size=parametric_action_distribution.param_size,
            rnn_hidden_size=rnn_hidden_size,
            output_layer_sizes=policy_output_layer_sizes,
            cell_type=cell_type,
            activation=activation,
            kernel_init=policy_kernel_init,
            noise_std_type=noise_std_type,
            init_noise_std=init_noise_std,
            state_dependent_std=state_dependent_std,
            mean_kernel_init=mean_kernel_init,
            mean_bias_init=mean_bias_init,
            layer_norm=policy_network_layer_norm,
            input_skip=policy_input_skip,
        )

    # --- Q module ----------------------------------------------------------
    # shared_q_backbone=True: single recurrent backbone + trunk, 2-head output.
    # shared_q_backbone=False: one 1-head module, instantiated twice via
    # stacked params and vmap (fully independent twin critics, stock-SAC style).
    q_module = rnn_shared.RecurrentMLP(
        rnn_hidden_size=rnn_hidden_size,
        output_layer_sizes=list(q_hidden_layer_sizes)
        + [2 if shared_q_backbone else 1],
        cell_type=cell_type,
        activation=activation,
        kernel_init=q_kernel_init,
        activate_final=False,
        layer_norm=q_network_layer_norm,
        input_skip=q_input_skip,
    )

    # --- Policy network wiring --------------------------------------------
    def policy_apply(processor_params, policy_params, obs, hidden, *, rng=None):
        if isinstance(obs, Mapping):
            obs = preprocess_observations_fn(
                obs[policy_obs_key],
                networks.normalizer_select(processor_params, policy_obs_key),
            )
        else:
            obs = preprocess_observations_fn(obs, processor_params)
        rngs = rnn_shared._policy_rngs(rng)
        if rngs is None:
            return policy_module.apply(policy_params, obs, hidden)
        return policy_module.apply(policy_params, obs, hidden, rngs=rngs)

    def policy_apply_sequence(
        processor_params, policy_params, obs_seq, hidden, *, rng=None
    ):
        if isinstance(obs_seq, Mapping):
            obs_seq = preprocess_observations_fn(
                obs_seq[policy_obs_key],
                networks.normalizer_select(processor_params, policy_obs_key),
            )
        else:
            obs_seq = preprocess_observations_fn(obs_seq, processor_params)
        rngs = rnn_shared._policy_rngs(rng)
        if rngs is None:
            return policy_module.apply(
                policy_params, obs_seq, hidden, method=policy_module.scan_forward
            )
        return policy_module.apply(
            policy_params,
            obs_seq,
            hidden,
            method=policy_module.scan_forward,
            rngs=rngs,
        )

    dummy_policy_obs = jnp.zeros((1, policy_obs_size))
    dummy_policy_hidden = rnn_shared.init_hidden_state(
        cell_type, rnn_hidden_size, 1
    )

    def policy_init(key):
        return policy_module.init(key, dummy_policy_obs, dummy_policy_hidden)

    def policy_init_hidden(batch_size):
        return rnn_shared.init_hidden_state(cell_type, rnn_hidden_size, batch_size)

    policy_network = rnn_shared.RecurrentNetwork(
        init=policy_init,
        apply=policy_apply,
        apply_sequence=policy_apply_sequence,
        init_hidden=policy_init_hidden,
    )

    # --- Q network wiring --------------------------------------------------
    def _preprocess_q_obs(processor_params, obs):
        if isinstance(obs, Mapping):
            return preprocess_observations_fn(
                obs[q_obs_key],
                networks.normalizer_select(processor_params, q_obs_key),
            )
        return preprocess_observations_fn(obs, processor_params)

    dummy_q_obs = jnp.zeros((1, q_obs_size))
    dummy_q_action = jnp.zeros((1, action_size))
    dummy_q_input = jnp.concatenate([dummy_q_obs, dummy_q_action], axis=-1)
    dummy_q_hidden = rnn_shared.init_hidden_state(cell_type, rnn_hidden_size, 1)

    if shared_q_backbone:

        def q_apply(processor_params, q_params, obs, action, hidden):
            obs = _preprocess_q_obs(processor_params, obs)
            inputs = jnp.concatenate([obs, action], axis=-1)
            return q_module.apply(q_params, inputs, hidden)

        def q_apply_sequence(
            processor_params, q_params, obs_seq, action_seq, hidden
        ):
            obs_seq = _preprocess_q_obs(processor_params, obs_seq)
            inputs_seq = jnp.concatenate([obs_seq, action_seq], axis=-1)
            return q_module.apply(
                q_params, inputs_seq, hidden, method=q_module.scan_forward
            )

        def q_init(key):
            return q_module.init(key, dummy_q_input, dummy_q_hidden)

        def q_init_hidden(batch_size):
            return rnn_shared.init_hidden_state(
                cell_type, rnn_hidden_size, batch_size
            )

    else:
        # Two independent critics: params/hiddens carry a leading [2] axis and
        # each apply vmaps the single-head module over that axis. The [B, 2]
        # output and opaque hidden keep the loss code unchanged.

        def q_apply(processor_params, q_params, obs, action, hidden):
            obs = _preprocess_q_obs(processor_params, obs)
            inputs = jnp.concatenate([obs, action], axis=-1)

            def single(p, h):
                return q_module.apply(p, inputs, h)

            q, new_hidden = jax.vmap(single)(q_params, hidden)  # q: [2, B, 1]
            return jnp.moveaxis(jnp.squeeze(q, -1), 0, -1), new_hidden

        def q_apply_sequence(
            processor_params, q_params, obs_seq, action_seq, hidden
        ):
            obs_seq = _preprocess_q_obs(processor_params, obs_seq)
            inputs_seq = jnp.concatenate([obs_seq, action_seq], axis=-1)

            def single(p, h):
                return q_module.apply(
                    p, inputs_seq, h, method=q_module.scan_forward
                )

            q, new_hidden = jax.vmap(single)(q_params, hidden)  # [2, T, B, 1]
            return jnp.moveaxis(jnp.squeeze(q, -1), 0, -1), new_hidden

        def q_init(key):
            keys = jax.random.split(key)
            params = [
                q_module.init(k, dummy_q_input, dummy_q_hidden) for k in keys
            ]
            return jax.tree_util.tree_map(
                lambda *xs: jnp.stack(xs, axis=0), *params
            )

        def q_init_hidden(batch_size):
            hidden = rnn_shared.init_hidden_state(
                cell_type, rnn_hidden_size, batch_size
            )
            return jax.tree_util.tree_map(
                lambda x: jnp.stack([x, x], axis=0), hidden
            )

    q_network = rnn_shared.RecurrentNetwork(
        init=q_init,
        apply=q_apply,
        apply_sequence=q_apply_sequence,
        init_hidden=q_init_hidden,
    )

    return RecurrentSACNetworks(
        policy_network=policy_network,
        q_network=q_network,
        parametric_action_distribution=parametric_action_distribution,
        rnn_hidden_size=rnn_hidden_size,
        cell_type=cell_type,
    )
