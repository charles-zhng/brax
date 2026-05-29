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

"""Recurrent PPO losses.

See: https://arxiv.org/pdf/1707.06347.pdf

Note: This implementation assumes a recurrent policy network and a feedforward
(non-recurrent) value network. The value network is a standard MLP that processes
each observation independently.
"""

from typing import Any, Tuple

from brax.training import types
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.recurrent_ppo import networks as rnn_ppo_networks
from brax.training.types import Params
import flax
import jax
import jax.numpy as jnp


@flax.struct.dataclass
class RNNPPONetworkParams:
    """Contains training state for the learner."""

    policy: Params
    value: Params


def compute_rnn_ppo_loss(
    params: RNNPPONetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    progress: jnp.ndarray,
    rnn_ppo_network: rnn_ppo_networks.RNNPPONetworks,
    entropy_cost=1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
    vf_coefficient: float = 0.5,
    clipping_epsilon_value: float | None = None,
    activity_cost: float = 0.0,
    activity_derivative_cost: float = 0.0,
) -> Tuple[jnp.ndarray, types.Metrics]:
    """Computes RNN-PPO loss.

    The policy network is recurrent (RNN/GRU/LSTM) and processes sequences with
    initial hidden states stored in the data extras. The value network is a
    feedforward MLP that processes each observation independently.

    Args:
      params: Network parameters
      normalizer_params: Parameters of the normalizer
      data: Transition data with leading dimension [B, T]. Extra fields required:
        ['state_extras']['truncation']
        ['policy_extras']['raw_action']
        ['policy_extras']['log_prob']
        ['policy_extras']['initial_policy_hidden']
        If present, ['policy_extras']['policy_rng'] is used to replay stochastic
        policy layers deterministically.
      rng: Random key
      progress: Training progress in [0, 1] (env_steps / num_timesteps).
        Passed to `entropy_cost` if it is callable, enabling exploration
        annealing schedules.
      rnn_ppo_network: RNN-PPO networks
      entropy_cost: Entropy cost coefficient. Either a float, or a
        Callable[[progress], float] that maps a 0-d progress scalar to the
        current entropy weight (e.g. cosine decay).
      discounting: Discount factor
      reward_scaling: Reward multiplier
      gae_lambda: GAE lambda
      clipping_epsilon: Policy loss clipping epsilon
      normalize_advantage: Whether to normalize advantage estimate
      vf_coefficient: Coefficient for value function loss
      clipping_epsilon_value: Value function loss clipping epsilon
      activity_cost: L2 penalty on policy RNN hidden activity h_t (Codol et al.
        2024 use 0.01) to promote parsimonious/sparse activation.
      activity_derivative_cost: L2 penalty on the temporal derivative
        (h_t - h_{t-1}) of the policy RNN hidden activity (Codol et al. 2024 use
        0.1), masked across episode boundaries.

    Returns:
      A tuple (loss, metrics)
    """
    parametric_action_distribution = rnn_ppo_network.parametric_action_distribution
    policy_network = rnn_ppo_network.policy_network
    value_network = rnn_ppo_network.value_network

    def _extract_initial_hidden(hidden):
        if hidden is None:
            return None
        # Hidden states are stored at each timestep with shape [B, T, hidden_size].
        # We need only the first timestep's hidden state: [B, hidden_size].
        return jax.tree_util.tree_map(lambda x: x[:, 0] if x.ndim > 2 else x, hidden)

    def _reset_hidden(hidden, done):
        if hidden is None:
            return None
        done_expanded = done[..., None]
        if rnn_ppo_network.cell_type == "lstm":
            c, h = hidden
            c = jnp.where(done_expanded, 0.0, c)
            h = jnp.where(done_expanded, 0.0, h)
            return (c, h)
        return jnp.where(done_expanded, 0.0, hidden)

    def _hidden_activity(hidden):
        """Extract the recurrent activity h_t to regularize (LSTM: the h gate)."""
        if rnn_ppo_network.cell_type == "lstm":
            return hidden[1]
        return hidden

    def _masked_scan_policy(obs_seq, initial_hidden, done_seq, policy_rng_seq=None):
        """Process policy network over sequence with hidden state resets on done.

        Returns (policy_outputs, hidden_activity_seq), where hidden_activity_seq
        is the pre-reset recurrent activity h_t at each step [T, B, hidden_size],
        used for the activity / activity-derivative regularizers.
        """

        def step(hidden, inputs):
            if policy_rng_seq is None:
                obs_t, done_t = inputs
                output, new_hidden = policy_network.apply(
                    normalizer_params, params.policy, obs_t, hidden
                )
            else:
                obs_t, done_t, rng_t = inputs
                if rng_t.ndim == 1:
                    output, new_hidden = policy_network.apply(
                        normalizer_params,
                        params.policy,
                        obs_t,
                        hidden,
                        rng=rng_t,
                    )
                else:

                    def _apply_policy(obs_b, hidden_b, rng_b):
                        return policy_network.apply(
                            normalizer_params,
                            params.policy,
                            obs_b,
                            hidden_b,
                            rng=rng_b,
                        )

                    output, new_hidden = jax.vmap(_apply_policy, in_axes=(0, 0, 0))(
                        obs_t, hidden, rng_t
                    )
            # Capture genuine activity BEFORE the done-reset (the reset only
            # zeroes the state carried into the next episode).
            activity = _hidden_activity(new_hidden)
            new_hidden = _reset_hidden(new_hidden, done_t)
            return new_hidden, (output, activity)

        if policy_rng_seq is None:
            _, (outputs, hidden_activity) = jax.lax.scan(
                step, initial_hidden, (obs_seq, done_seq)
            )
        else:
            _, (outputs, hidden_activity) = jax.lax.scan(
                step, initial_hidden, (obs_seq, done_seq, policy_rng_seq)
            )
        return outputs, hidden_activity

    initial_policy_hidden = _extract_initial_hidden(
        data.extras["policy_extras"]["initial_policy_hidden"]
    )
    initial_value_hidden = _extract_initial_hidden(
        data.extras["policy_extras"].get("initial_value_hidden")
    )

    # Create a version of extras without hidden states for safe swapping
    policy_extras_for_swap = {
        k: v
        for k, v in data.extras["policy_extras"].items()
        if k
        not in (
            "initial_policy_hidden",
            "initial_value_hidden",
            "policy_hidden",
            "value_hidden",
        )
    }
    extras_for_swap = {
        "policy_extras": policy_extras_for_swap,
        "state_extras": data.extras["state_extras"],
    }
    data_for_swap = types.Transition(
        observation=data.observation,
        action=data.action,
        reward=data.reward,
        discount=data.discount,
        next_observation=data.next_observation,
        extras=extras_for_swap,
    )

    # Put the time dimension first: [B, T, ...] -> [T, B, ...]
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data_for_swap)

    done = data.discount < 0.5

    policy_rng = data.extras["policy_extras"].get("policy_rng")
    if policy_rng is not None:
        # policy_rng is stored with a batch dimension for shape consistency,
        # but rollout used a single key for the entire batch. Collapse back
        # to one key per timestep so replay matches the original RNG usage.
        if policy_rng.ndim == 3:
            policy_rng = policy_rng[:, 0]
        elif policy_rng.ndim == 2:
            policy_rng = policy_rng[0]

    # Process policy network over sequence with done-masked hidden resets
    policy_logits, policy_hidden_activity = _masked_scan_policy(
        data.observation,
        initial_policy_hidden,
        done,
        policy_rng,
    )

    def _masked_scan_value(obs_seq, initial_hidden, done_seq):
        """Process value network over sequence with hidden state resets on done."""

        def step(hidden, inputs):
            obs_t, done_t = inputs
            output, new_hidden = value_network.apply(
                normalizer_params, params.value, obs_t, hidden
            )
            new_hidden = _reset_hidden(new_hidden, done_t)
            return new_hidden, output

        final_hidden, outputs = jax.lax.scan(step, initial_hidden, (obs_seq, done_seq))
        return outputs, final_hidden

    if initial_value_hidden is None:
        # Value network is feedforward - apply directly to all observations.
        # value_network.apply returns (value, hidden) but hidden is unused for MLP.
        baseline, _ = value_network.apply(
            normalizer_params, params.value, data.observation, None
        )
        final_value_hidden = None
    else:
        baseline, final_value_hidden = _masked_scan_value(
            data.observation, initial_value_hidden, done
        )

    # Compute bootstrap value from terminal observation
    terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
    bootstrap_value, _ = value_network.apply(
        normalizer_params, params.value, terminal_obs, final_value_hidden
    )

    rewards = data.reward * reward_scaling
    truncation = data.extras["state_extras"]["truncation"]
    termination = (1 - data.discount) * (1 - truncation)

    target_action_log_probs = parametric_action_distribution.log_prob(
        policy_logits, data.extras["policy_extras"]["raw_action"]
    )
    behaviour_action_log_probs = data.extras["policy_extras"]["log_prob"]

    # Use the same GAE computation as regular PPO
    vs, advantages = ppo_losses.compute_gae(
        truncation=truncation,
        termination=termination,
        rewards=rewards,
        values=baseline,
        bootstrap_value=bootstrap_value,
        lambda_=gae_lambda,
        discount=discounting,
    )

    if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)

    surrogate_loss1 = rho_s * advantages
    surrogate_loss2 = (
        jnp.clip(rho_s, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages
    )

    policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

    # Value function loss
    v_error = vs - baseline
    v_loss = v_error * v_error
    if clipping_epsilon_value is not None:
        old_values = data.extras["policy_extras"]["value"]
        v_clipped = old_values + jnp.clip(
            baseline - old_values, -clipping_epsilon_value, clipping_epsilon_value
        )
        v_loss_clipped = (vs - v_clipped) ** 2
        v_loss = jnp.maximum(v_loss, v_loss_clipped)
    v_loss = jnp.mean(v_loss) * 0.5 * vf_coefficient

    # Entropy reward — entropy_cost can be a scalar or a callable schedule
    # of training progress. The branch is decided at trace time.
    if callable(entropy_cost):
        entropy_weight = entropy_cost(progress)
    else:
        entropy_weight = entropy_cost
    entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, rng))
    entropy_loss = entropy_weight * -entropy

    # Activity regularization on the policy RNN hidden states (Codol et al. 2024):
    # an L2 penalty on activity h_t and on its temporal derivative h_t - h_{t-1},
    # both averaged over (time, batch, units) and scale-invariant to hidden width.
    # hidden_activity has shape [T, B, hidden_size] (time-major, pre-reset).
    activity_l2 = jnp.mean(jnp.square(policy_hidden_activity))
    # Temporal derivative, masked across episode boundaries: a diff spanning a
    # done step (h_{t+1} starts a fresh episode) must not be penalized.
    activity_diff = policy_hidden_activity[1:] - policy_hidden_activity[:-1]
    not_done = (1.0 - done[:-1].astype(jnp.float32))[..., None]
    diff_sq = jnp.square(activity_diff) * not_done
    derivative_l2 = jnp.sum(diff_sq) / (
        jnp.sum(not_done) * policy_hidden_activity.shape[-1] + 1e-8
    )
    activity_loss = activity_cost * activity_l2
    activity_derivative_loss = activity_derivative_cost * derivative_l2

    total_loss = (
        policy_loss + v_loss + entropy_loss + activity_loss + activity_derivative_loss
    )

    new_dist = parametric_action_distribution.create_dist(policy_logits)
    if hasattr(new_dist, "kl_divergence"):
        old_dist_params = data.extras["policy_extras"]["distribution_params"]
        old_dist = parametric_action_distribution.create_dist(old_dist_params)
        kl = jnp.mean(new_dist.kl_divergence(old_dist))
        policy_dist_mean_std = jnp.mean(new_dist.scale)
    else:
        kl, policy_dist_mean_std = jnp.array(0.0), jnp.array(0.0)

    return total_loss, {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "v_loss": v_loss,
        "entropy_loss": entropy_loss,
        "entropy_cost": jnp.asarray(entropy_weight),
        "activity_loss": activity_loss,
        "activity_derivative_loss": activity_derivative_loss,
        "kl_mean": kl,
        "policy_dist_mean_std": policy_dist_mean_std,
    }
