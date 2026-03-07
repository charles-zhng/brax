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

"""RNN-PPO networks with support for Vanilla RNN, GRU, and LSTM cells."""

import dataclasses
from typing import Any, Callable, Literal, Mapping, Sequence, Tuple, Union

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.types import PRNGKey
import flax
from flax import linen
import jax
import jax.numpy as jnp


# Type alias for RNN cell types
RNNCellType = Literal["simple", "gru", "lstm"]

# Type alias for hidden state - can be a single array (SimpleCell/GRU) or tuple (LSTM)
HiddenState = Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


def _policy_rngs(rng: PRNGKey | None):
    if rng is None:
        return None
    return {
        "dropout": rng,
        "policy": jax.random.fold_in(rng, 1),
    }


def get_rnn_cell(
    cell_type: RNNCellType,
    hidden_size: int,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
) -> linen.RNNCellBase:
    """Returns the appropriate RNN cell based on cell_type."""
    if cell_type == "simple":
        return linen.SimpleCell(features=hidden_size, kernel_init=kernel_init)
    elif cell_type == "gru":
        return linen.GRUCell(features=hidden_size, kernel_init=kernel_init)
    elif cell_type == "lstm":
        return linen.LSTMCell(features=hidden_size, kernel_init=kernel_init)
    else:
        raise ValueError(
            f"Unsupported RNN cell type: {cell_type}. Must be one of"
            ' "simple", "gru", or "lstm".'
        )


def init_hidden_state(
    cell_type: RNNCellType,
    hidden_size: int,
    batch_size: int,
) -> HiddenState:
    """Initialize hidden state for the given RNN cell type."""
    if cell_type == "lstm":
        # LSTM has (carry, hidden) state
        return (
            jnp.zeros((batch_size, hidden_size)),
            jnp.zeros((batch_size, hidden_size)),
        )
    else:
        # SimpleCell and GRU have single hidden state
        return jnp.zeros((batch_size, hidden_size))


class RecurrentMLP(linen.Module):
    """Recurrent module: single RNN cell followed by MLP output layers.

    Processes input through:
    1. RNN cell (obs -> hidden)
    2. Output MLP layers
    3. Optional separate mean output layer with configurable init and clipping
    """

    rnn_hidden_size: int
    output_layer_sizes: Sequence[int]
    cell_type: RNNCellType = "gru"
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    mean_clip_scale: float | None = None
    mean_kernel_init: Initializer | None = None

    def setup(self):
        self.rnn_cell = get_rnn_cell(
            self.cell_type, self.rnn_hidden_size, self.kernel_init
        )
        # If mean_kernel_init or mean_clip_scale is set, split off the final
        # layer so it can use a different initializer / clipping.
        if self.mean_kernel_init is not None or self.mean_clip_scale is not None:
            self.output_mlp = networks.MLP(
                layer_sizes=list(self.output_layer_sizes[:-1]),
                activation=self.activation,
                kernel_init=self.kernel_init,
                activate_final=True,
            )
            mean_kernel_init = (
                self.mean_kernel_init if self.mean_kernel_init is not None
                else self.kernel_init
            )
            self.mean_layer = linen.Dense(
                self.output_layer_sizes[-1], kernel_init=mean_kernel_init
            )
        else:
            self.output_mlp = networks.MLP(
                layer_sizes=list(self.output_layer_sizes),
                activation=self.activation,
                kernel_init=self.kernel_init,
                activate_final=False,
            )
            self.mean_layer = None

    def __call__(
        self, obs: jnp.ndarray, hidden: HiddenState
    ) -> Tuple[jnp.ndarray, HiddenState]:
        """Forward pass for single timestep.

        Args:
          obs: Observation of shape [batch, obs_dim]
          hidden: Hidden state from previous timestep

        Returns:
          Tuple of (output, new_hidden_state)
        """
        new_hidden, _ = self.rnn_cell(hidden, obs)
        # For LSTM, the output is the second element (hidden state h)
        if self.cell_type == "lstm":
            rnn_output = new_hidden[1]
        else:
            rnn_output = new_hidden
        output = self.output_mlp(rnn_output)
        if self.mean_layer is not None:
            output = self.mean_layer(output)
        if self.mean_clip_scale is not None:
            output = self.mean_clip_scale * (output / (1.0 + jnp.abs(output)))
        return output, new_hidden

    def scan_forward(
        self, obs_seq: jnp.ndarray, initial_hidden: HiddenState
    ) -> Tuple[jnp.ndarray, HiddenState]:
        """Forward pass over a sequence using scan.

        Args:
          obs_seq: Observations of shape [time, batch, obs_dim]
          initial_hidden: Initial hidden state

        Returns:
          Tuple of (outputs [time, batch, output_dim], final_hidden_state)
        """

        def step(hidden, obs):
            output, new_hidden = self(obs, hidden)
            return new_hidden, output

        final_hidden, outputs = jax.lax.scan(step, initial_hidden, obs_seq)
        return outputs, final_hidden


class RecurrentPolicyModule(linen.Module):
    """Recurrent policy module with learnable mean and (optionally) std."""

    param_size: int
    rnn_hidden_size: int
    output_layer_sizes: Sequence[int]
    cell_type: RNNCellType = "gru"
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    noise_std_type: Literal["scalar", "log"] = "scalar"
    init_noise_std: float = 1.0
    state_dependent_std: bool = False
    mean_clip_scale: float | None = None
    mean_kernel_init: Initializer | None = None

    def setup(self):
        self.recurrent_mlp = RecurrentMLP(
            rnn_hidden_size=self.rnn_hidden_size,
            output_layer_sizes=self.output_layer_sizes,
            cell_type=self.cell_type,
            activation=self.activation,
            kernel_init=self.kernel_init,
        )
        mean_kernel_init = (
            self.mean_kernel_init if self.mean_kernel_init is not None
            else self.kernel_init
        )
        self.mean_layer = linen.Dense(self.param_size, kernel_init=mean_kernel_init)

        if self.state_dependent_std:
            self.std_layer = linen.Dense(self.param_size, kernel_init=self.kernel_init)
        else:
            if self.noise_std_type == "scalar":
                self.std_param = networks.Param(
                    self.init_noise_std, size=self.param_size, name="std_param"
                )
            else:
                self.std_param = networks.LogParam(
                    self.init_noise_std, size=self.param_size, name="std_logparam"
                )

    def __call__(
        self, obs: jnp.ndarray, hidden: HiddenState
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], HiddenState]:
        """Single step forward pass.

        Args:
          obs: Observation [batch, obs_dim]
          hidden: Hidden state

        Returns:
          Tuple of ((mean, std), new_hidden)
        """
        features, new_hidden = self.recurrent_mlp(obs, hidden)
        mean = self.mean_layer(features)
        if self.mean_clip_scale is not None:
            mean = self.mean_clip_scale * (mean / (1.0 + jnp.abs(mean)))

        if self.state_dependent_std:
            log_std = self.std_layer(features)
            if self.noise_std_type == "log":
                std = jnp.exp(log_std)
            else:
                std = log_std
        else:
            std = self.std_param()
            std = jnp.broadcast_to(std, mean.shape)

        return (mean, std), new_hidden

    def scan_forward(
        self, obs_seq: jnp.ndarray, initial_hidden: HiddenState
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], HiddenState]:
        """Forward pass over sequence.

        Args:
          obs_seq: Observations [time, batch, obs_dim]
          initial_hidden: Initial hidden state

        Returns:
          Tuple of ((means [time, batch, param_size], stds), final_hidden)
        """

        def step(hidden, obs):
            (mean, std), new_hidden = self(obs, hidden)
            return new_hidden, (mean, std)

        final_hidden, (means, stds) = jax.lax.scan(step, initial_hidden, obs_seq)
        return (means, stds), final_hidden


class RecurrentValueModule(linen.Module):
    """Recurrent value function module."""

    rnn_hidden_size: int
    output_layer_sizes: Sequence[int]
    cell_type: RNNCellType = "gru"
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()

    def setup(self):
        self.recurrent_mlp = RecurrentMLP(
            rnn_hidden_size=self.rnn_hidden_size,
            output_layer_sizes=self.output_layer_sizes,
            cell_type=self.cell_type,
            activation=self.activation,
            kernel_init=self.kernel_init,
        )
        self.value_layer = linen.Dense(1, kernel_init=self.kernel_init)

    def __call__(
        self, obs: jnp.ndarray, hidden: HiddenState
    ) -> Tuple[jnp.ndarray, HiddenState]:
        """Single step forward pass.

        Args:
          obs: Observation [batch, obs_dim]
          hidden: Hidden state

        Returns:
          Tuple of (value [batch], new_hidden)
        """
        features, new_hidden = self.recurrent_mlp(obs, hidden)
        value = jnp.squeeze(self.value_layer(features), axis=-1)
        return value, new_hidden

    def scan_forward(
        self, obs_seq: jnp.ndarray, initial_hidden: HiddenState
    ) -> Tuple[jnp.ndarray, HiddenState]:
        """Forward pass over sequence.

        Args:
          obs_seq: Observations [time, batch, obs_dim]
          initial_hidden: Initial hidden state

        Returns:
          Tuple of (values [time, batch], final_hidden)
        """

        def step(hidden, obs):
            value, new_hidden = self(obs, hidden)
            return new_hidden, value

        final_hidden, values = jax.lax.scan(step, initial_hidden, obs_seq)
        return values, final_hidden


@dataclasses.dataclass
class RecurrentNetwork:
    """Recurrent network with init, apply, and hidden state initialization."""

    init: Callable[..., Any]
    apply: Callable[..., Any]
    apply_sequence: Callable[..., Any]
    init_hidden: Callable[[int], HiddenState]


@flax.struct.dataclass
class RNNPPONetworks:
    """RNN-PPO networks container.

    The policy network is recurrent (RNN/GRU/LSTM) while the value network
    is a feedforward MLP. This design allows the policy to maintain memory
    for partial observability while the value function processes observations
    independently.
    """

    policy_network: RecurrentNetwork
    value_network: RecurrentNetwork  # Feedforward MLP (init_hidden returns None)
    parametric_action_distribution: distribution.ParametricDistribution
    rnn_hidden_size: int  # Policy RNN hidden size
    cell_type: RNNCellType  # Policy RNN cell type


def _get_obs_state_size(obs_size: types.ObservationSize, obs_key: str) -> int:
    obs_size = obs_size[obs_key] if isinstance(obs_size, Mapping) else obs_size
    return jax.tree_util.tree_flatten(obs_size)[0][-1]


def make_inference_fn(rnn_ppo_networks: RNNPPONetworks, compute_value: bool = False):
    """Creates params and inference function for the RNN-PPO agent.

    Args:
      rnn_ppo_networks: The RNN-PPO networks.
      compute_value: If True, compute value during rollouts.
    """

    def make_policy(params: types.Params, deterministic: bool = False) -> Callable:
        """Creates a recurrent policy function.

        Returns a function that takes (observations, policy_hidden, key) and returns
        (action, extras, new_policy_hidden).
        """
        policy_network = rnn_ppo_networks.policy_network
        parametric_action_distribution = rnn_ppo_networks.parametric_action_distribution

        def policy(
            observations: types.Observation,
            policy_hidden: HiddenState,
            key_sample: PRNGKey,
        ) -> Tuple[types.Action, types.Extra, HiddenState]:
            """Recurrent policy function.

            Args:
              observations: Current observations
              policy_hidden: Policy network hidden state
              key_sample: Random key for sampling

            Returns:
              Tuple of (actions, extras, new_policy_hidden)
            """
            param_subset = (params[0], params[1])  # normalizer and policy params
            # Keep action sampling identical to previous behavior while deriving
            # a deterministic policy RNG for replaying stochastic layers.
            key_policy = jax.random.fold_in(key_sample, 1)
            logits, new_policy_hidden = policy_network.apply(
                *param_subset, observations, policy_hidden, rng=key_policy
            )

            if deterministic:
                actions = rnn_ppo_networks.parametric_action_distribution.mode(logits)
                extras = {"policy_hidden": new_policy_hidden}
                return actions, extras, new_policy_hidden

            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = parametric_action_distribution.postprocess(
                raw_actions
            )
            if log_prob.ndim == 0:
                batch_shape = (1,)
            else:
                batch_shape = log_prob.shape
            policy_rng = jnp.broadcast_to(
                key_policy, batch_shape + key_policy.shape
            )

            extras = {
                "log_prob": log_prob,
                "raw_action": raw_actions,
                "distribution_params": logits,
                "policy_rng": policy_rng,
                "policy_hidden": new_policy_hidden,
            }

            if compute_value:
                # Value network is feedforward MLP - hidden state is unused
                value, _ = rnn_ppo_networks.value_network.apply(
                    params[0], params[2], observations, None
                )
                extras["value"] = value

            return postprocessed_actions, extras, new_policy_hidden

        return policy

    return make_policy


def make_rnn_ppo_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_rnn_hidden_size: int = 64,
    policy_output_layer_sizes: Sequence[int] = (64,),
    value_hidden_layer_sizes: Sequence[int] = (128, 128),
    cell_type: RNNCellType = "gru",
    activation: ActivationFn = linen.swish,
    policy_obs_key: str = "state",
    value_obs_key: str = "state",
    distribution_type: Literal["normal", "tanh_normal"] = "tanh_normal",
    noise_std_type: Literal["scalar", "log"] = "scalar",
    init_noise_std: float = 1.0,
    state_dependent_std: bool = False,
    policy_network_kernel_init_fn: Initializer = jax.nn.initializers.lecun_uniform,
    policy_network_kernel_init_kwargs: Mapping[str, Any] | None = None,
    value_network_kernel_init_fn: Initializer = jax.nn.initializers.lecun_uniform,
    value_network_kernel_init_kwargs: Mapping[str, Any] | None = None,
    mean_clip_scale: float | None = None,
    mean_kernel_init_fn: Initializer | None = None,
    mean_kernel_init_kwargs: Mapping[str, Any] | None = None,
) -> RNNPPONetworks:
    """Make RNN-PPO networks with preprocessor.

    Policy architecture: obs -> RNN cell -> MLP output layers
    Value architecture: obs -> MLP (non-recurrent)

    Args:
      observation_size: Size of observations
      action_size: Size of action space
      preprocess_observations_fn: Function to preprocess observations
      policy_rnn_hidden_size: Hidden size of policy RNN cell
      policy_output_layer_sizes: MLP layer sizes after RNN in policy network
      value_hidden_layer_sizes: Hidden layer sizes for value MLP
      cell_type: Type of RNN cell ('simple', 'gru', 'lstm')
      activation: Activation function
      policy_obs_key: Key for policy observations
      value_obs_key: Key for value observations
      distribution_type: Type of action distribution
      noise_std_type: Type of noise std ('scalar' or 'log')
      init_noise_std: Initial noise standard deviation
      state_dependent_std: Whether std depends on state
      policy_network_kernel_init_fn: Kernel initializer function for policy
      policy_network_kernel_init_kwargs: Kwargs for policy kernel initializer
      value_network_kernel_init_fn: Kernel initializer function for value
      value_network_kernel_init_kwargs: Kwargs for value kernel initializer

    Returns:
      RNNPPONetworks containing policy and value networks
    """
    policy_kernel_init_kwargs = policy_network_kernel_init_kwargs or {}
    value_kernel_init_kwargs = value_network_kernel_init_kwargs or {}
    mean_kernel_init_kwargs_ = mean_kernel_init_kwargs or {}
    policy_kernel_init = policy_network_kernel_init_fn(**policy_kernel_init_kwargs)
    value_kernel_init = value_network_kernel_init_fn(**value_kernel_init_kwargs)
    mean_kernel_init = (
        mean_kernel_init_fn(**mean_kernel_init_kwargs_)
        if mean_kernel_init_fn is not None else None
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
    else:
        raise ValueError(
            f"Unsupported distribution type: {distribution_type}. Must be one"
            ' of "normal" or "tanh_normal".'
        )

    obs_size = _get_obs_state_size(observation_size, policy_obs_key)
    value_obs_size = _get_obs_state_size(observation_size, value_obs_key)

    # Create policy module
    if distribution_type == "tanh_normal":
        # For tanh_normal, we output the full param_size (means only, std learned separately)
        policy_module = RecurrentMLP(
            rnn_hidden_size=policy_rnn_hidden_size,
            output_layer_sizes=list(policy_output_layer_sizes)
            + [parametric_action_distribution.param_size],
            cell_type=cell_type,
            activation=activation,
            kernel_init=policy_kernel_init,
            mean_clip_scale=mean_clip_scale,
            mean_kernel_init=mean_kernel_init,
        )
    else:
        # For normal distribution, use the policy module with std
        policy_module = RecurrentPolicyModule(
            param_size=parametric_action_distribution.param_size,
            rnn_hidden_size=policy_rnn_hidden_size,
            output_layer_sizes=policy_output_layer_sizes,
            cell_type=cell_type,
            activation=activation,
            kernel_init=policy_kernel_init,
            noise_std_type=noise_std_type,
            init_noise_std=init_noise_std,
            state_dependent_std=state_dependent_std,
            mean_clip_scale=mean_clip_scale,
            mean_kernel_init=mean_kernel_init,
        )

    # Create value module (standard MLP, non-recurrent)
    value_module = networks.MLP(
        layer_sizes=list(value_hidden_layer_sizes) + [1],
        activation=activation,
        kernel_init=value_kernel_init,
    )

    # Policy network functions
    def policy_apply(processor_params, policy_params, obs, hidden, *, rng=None):
        if isinstance(obs, Mapping):
            obs = preprocess_observations_fn(
                obs[policy_obs_key],
                networks.normalizer_select(processor_params, policy_obs_key),
            )
        else:
            obs = preprocess_observations_fn(obs, processor_params)
        rngs = _policy_rngs(rng)
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
        rngs = _policy_rngs(rng)
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

    dummy_obs = jnp.zeros((1, obs_size))
    dummy_hidden = init_hidden_state(cell_type, policy_rnn_hidden_size, 1)

    def policy_init(key):
        return policy_module.init(key, dummy_obs, dummy_hidden)

    def policy_init_hidden(batch_size):
        return init_hidden_state(cell_type, policy_rnn_hidden_size, batch_size)

    policy_network = RecurrentNetwork(
        init=policy_init,
        apply=policy_apply,
        apply_sequence=policy_apply_sequence,
        init_hidden=policy_init_hidden,
    )

    # Value network functions (non-recurrent MLP)
    def value_apply(processor_params, value_params, obs, hidden):
        """Apply value network. Hidden state is ignored (non-recurrent)."""
        if isinstance(obs, Mapping):
            obs = preprocess_observations_fn(
                obs[value_obs_key],
                networks.normalizer_select(processor_params, value_obs_key),
            )
        else:
            obs = preprocess_observations_fn(obs, processor_params)
        value = jnp.squeeze(value_module.apply(value_params, obs), axis=-1)
        return value, hidden  # Return same hidden (unused)

    def value_apply_sequence(processor_params, value_params, obs_seq, hidden):
        """Apply value network over sequence. Hidden state is ignored."""
        if isinstance(obs_seq, Mapping):
            obs_seq = preprocess_observations_fn(
                obs_seq[value_obs_key],
                networks.normalizer_select(processor_params, value_obs_key),
            )
        else:
            obs_seq = preprocess_observations_fn(obs_seq, processor_params)
        # obs_seq is [time, batch, obs_dim], apply MLP to each timestep
        values = jnp.squeeze(value_module.apply(value_params, obs_seq), axis=-1)
        return values, hidden  # Return same hidden (unused)

    dummy_value_obs = jnp.zeros((1, value_obs_size))

    def value_init(key):
        return value_module.init(key, dummy_value_obs)

    def value_init_hidden(_batch_size):
        # Value network is non-recurrent, return None as placeholder
        return None

    value_network = RecurrentNetwork(
        init=value_init,
        apply=value_apply,
        apply_sequence=value_apply_sequence,
        init_hidden=value_init_hidden,
    )

    return RNNPPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
        rnn_hidden_size=policy_rnn_hidden_size,
        cell_type=cell_type,
    )
