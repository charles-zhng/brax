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

"""RNN-PPO losses.

See: https://arxiv.org/pdf/1707.06347.pdf
"""

from typing import Any, Tuple

from brax.training import types
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.rnn_ppo import networks as rnn_ppo_networks
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
    rnn_ppo_network: rnn_ppo_networks.RNNPPONetworks,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
    vf_coefficient: float = 0.5,
    clipping_epsilon_value: float | None = None,
) -> Tuple[jnp.ndarray, types.Metrics]:
  """Computes RNN-PPO loss.

  The key difference from standard PPO is that we process sequences with
  initial hidden states stored in the data extras.

  Args:
    params: Network parameters
    normalizer_params: Parameters of the normalizer
    data: Transition data with leading dimension [B, T]. Extra fields required:
      ['state_extras']['truncation']
      ['policy_extras']['raw_action']
      ['policy_extras']['log_prob']
      ['policy_extras']['initial_policy_hidden']
      ['policy_extras']['initial_value_hidden']
    rng: Random key
    rnn_ppo_network: RNN-PPO networks
    entropy_cost: Entropy cost coefficient
    discounting: Discount factor
    reward_scaling: Reward multiplier
    gae_lambda: GAE lambda
    clipping_epsilon: Policy loss clipping epsilon
    normalize_advantage: Whether to normalize advantage estimate
    vf_coefficient: Coefficient for value function loss
    clipping_epsilon_value: Value function loss clipping epsilon

  Returns:
    A tuple (loss, metrics)
  """
  parametric_action_distribution = rnn_ppo_network.parametric_action_distribution
  policy_network = rnn_ppo_network.policy_network
  value_network = rnn_ppo_network.value_network

  # Extract initial hidden states BEFORE swapping dimensions
  # Hidden states are stored at each timestep with shape [B, T, hidden_size]
  # We need only the first timestep's hidden state: [B, hidden_size]
  initial_policy_hidden = jax.tree_util.tree_map(
      lambda x: x[:, 0] if x.ndim > 2 else x,
      data.extras['policy_extras']['initial_policy_hidden']
  )
  initial_value_hidden = jax.tree_util.tree_map(
      lambda x: x[:, 0] if x.ndim > 2 else x,
      data.extras['policy_extras']['initial_value_hidden']
  )

  # Create a version of extras without hidden states for safe swapping
  policy_extras_for_swap = {
      k: v for k, v in data.extras['policy_extras'].items()
      if k not in ('initial_policy_hidden', 'initial_value_hidden',
                   'policy_hidden', 'value_hidden')
  }
  extras_for_swap = {
      'policy_extras': policy_extras_for_swap,
      'state_extras': data.extras['state_extras'],
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

  # Process entire sequence through policy network
  # obs shape: [T, B, obs_dim]
  policy_logits, _ = policy_network.apply_sequence(
      normalizer_params, params.policy, data.observation, initial_policy_hidden
  )

  # Process entire sequence through value network
  baseline, _ = value_network.apply_sequence(
      normalizer_params, params.value, data.observation, initial_value_hidden
  )

  # Compute bootstrap value from terminal observation
  terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
  # Get the final hidden state for value network to use for bootstrap
  _, final_value_hidden = value_network.apply_sequence(
      normalizer_params, params.value, data.observation, initial_value_hidden
  )
  bootstrap_value, _ = value_network.apply(
      normalizer_params, params.value, terminal_obs, final_value_hidden
  )

  rewards = data.reward * reward_scaling
  truncation = data.extras['state_extras']['truncation']
  termination = (1 - data.discount) * (1 - truncation)

  target_action_log_probs = parametric_action_distribution.log_prob(
      policy_logits, data.extras['policy_extras']['raw_action']
  )
  behaviour_action_log_probs = data.extras['policy_extras']['log_prob']

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
    old_values = data.extras['policy_extras']['value']
    v_clipped = old_values + jnp.clip(
        baseline - old_values, -clipping_epsilon_value, clipping_epsilon_value
    )
    v_loss_clipped = (vs - v_clipped) ** 2
    v_loss = jnp.maximum(v_loss, v_loss_clipped)
  v_loss = jnp.mean(v_loss) * 0.5 * vf_coefficient

  # Entropy reward
  entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, rng))
  entropy_loss = entropy_cost * -entropy

  total_loss = policy_loss + v_loss + entropy_loss

  new_dist = parametric_action_distribution.create_dist(policy_logits)
  if hasattr(new_dist, 'kl_divergence'):
    old_dist_params = data.extras['policy_extras']['distribution_params']
    old_dist = parametric_action_distribution.create_dist(old_dist_params)
    kl = jnp.mean(new_dist.kl_divergence(old_dist))
    policy_dist_mean_std = jnp.mean(new_dist.scale)
  else:
    kl, policy_dist_mean_std = jnp.array(0.0), jnp.array(0.0)

  return total_loss, {
      'total_loss': total_loss,
      'policy_loss': policy_loss,
      'v_loss': v_loss,
      'entropy_loss': entropy_loss,
      'kl_mean': kl,
      'policy_dist_mean_std': policy_dist_mean_std,
  }
