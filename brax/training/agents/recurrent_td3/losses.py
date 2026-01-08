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

"""Recurrent TD3 losses.

TD3 (Twin Delayed DDPG) with recurrent actor for partially observable
environments. Key differences from SAC:
- Deterministic policy (no entropy regularization)
- Target policy smoothing (clipped noise on target actions)
- Delayed policy updates (handled in train.py)
"""

from typing import Any, Tuple

from brax.training import types
from brax.training.agents.recurrent_td3 import networks as recurrent_td3_networks
from brax.training.agents.recurrent_ppo.networks import HiddenState
from brax.training.types import Params
from brax.training.types import PRNGKey
import jax
import jax.numpy as jnp


Transition = types.Transition


def make_losses(
    td3_network: recurrent_td3_networks.RecurrentTD3Networks,
    reward_scaling: float,
    discounting: float,
    target_noise: float = 0.2,
    noise_clip: float = 0.5,
):
  """Creates the TD3 losses.

  Args:
    td3_network: The recurrent TD3 networks.
    reward_scaling: Scaling factor for rewards.
    discounting: Discount factor (gamma).
    target_noise: Standard deviation of Gaussian noise added to target actions.
    noise_clip: Clipping range for target action noise.

  Returns:
    Tuple of (critic_loss_fn, actor_loss_fn).
  """
  actor_network = td3_network.actor_network
  q_network = td3_network.q_network

  def critic_loss(
      q_params: Params,
      target_q_params: Params,
      target_actor_params: Params,
      normalizer_params: Any,
      transitions: Transition,
      hidden_states: HiddenState,
      key: PRNGKey,
  ) -> Tuple[jnp.ndarray, types.Metrics]:
    """TD3 critic loss with target policy smoothing.

    Args:
      q_params: Current Q-network parameters.
      target_q_params: Target Q-network parameters.
      target_actor_params: Target actor parameters.
      normalizer_params: Observation normalizer parameters.
      transitions: Batch of transitions.
      hidden_states: Initial hidden states for the actor RNN.
      key: Random key for target noise.

    Returns:
      Tuple of (loss, metrics dict).
    """
    # Current Q-values for the actions taken
    q_old_action = q_network.apply(
        normalizer_params,
        q_params,
        transitions.observation,
        transitions.action,
    )

    # Target action with smoothing noise
    target_action, _ = actor_network.apply(
        normalizer_params,
        target_actor_params,
        transitions.next_observation,
        hidden_states,
    )

    # Add clipped noise for target policy smoothing
    noise = jax.random.normal(key, target_action.shape) * target_noise
    noise = jnp.clip(noise, -noise_clip, noise_clip)
    target_action = jnp.clip(target_action + noise, -1.0, 1.0)

    # Target Q-value (min of twin Q for conservative estimate)
    next_q = q_network.apply(
        normalizer_params,
        target_q_params,
        transitions.next_observation,
        target_action,
    )
    next_q = jnp.min(next_q, axis=-1)

    # Bellman target
    target_q = jax.lax.stop_gradient(
        transitions.reward * reward_scaling
        + transitions.discount * discounting * next_q
    )

    # MSE loss for both Q-networks
    q_error = q_old_action - jnp.expand_dims(target_q, -1)

    # Better bootstrapping for truncated episodes
    truncation = transitions.extras['state_extras']['truncation']
    q_error *= jnp.expand_dims(1 - truncation, -1)

    q_loss = 0.5 * jnp.mean(jnp.square(q_error))

    metrics = {
        'critic_loss': q_loss,
        'q1_mean': jnp.mean(q_old_action[..., 0]),
        'q2_mean': jnp.mean(q_old_action[..., 1]),
        'target_q_mean': jnp.mean(target_q),
    }

    return q_loss, metrics

  def actor_loss(
      actor_params: Params,
      normalizer_params: Any,
      q_params: Params,
      transitions: Transition,
      hidden_states: HiddenState,
  ) -> Tuple[jnp.ndarray, types.Metrics]:
    """TD3 actor loss (deterministic policy gradient).

    Args:
      actor_params: Current actor parameters.
      normalizer_params: Observation normalizer parameters.
      q_params: Current Q-network parameters (not target).
      transitions: Batch of transitions.
      hidden_states: Initial hidden states for the actor RNN.

    Returns:
      Tuple of (loss, metrics dict).
    """
    # Get actions from current actor
    action, _ = actor_network.apply(
        normalizer_params,
        actor_params,
        transitions.observation,
        hidden_states,
    )

    # Q-value for actor actions (use only first Q-network)
    q_action = q_network.apply(
        normalizer_params, q_params, transitions.observation, action
    )
    q1 = q_action[..., 0]

    # Maximize Q-value (minimize negative Q)
    loss = -jnp.mean(q1)

    metrics = {
        'actor_loss': loss,
        'actor_q1_mean': jnp.mean(q1),
    }

    return loss, metrics

  return critic_loss, actor_loss
