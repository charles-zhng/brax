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

"""Recurrent PPO losses."""

from typing import Any, Tuple

from brax.training import types
from brax.training.agents.recurrent_ppo import networks as recurrent_networks
from brax.training.types import Params
import flax
import jax
import jax.numpy as jnp


@flax.struct.dataclass
class RecurrentPPONetworkParams:
    """Contains training state for the learner."""

    model: Params


def compute_recurrent_ppo_loss(
    params: RecurrentPPONetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    ppo_network: recurrent_networks.RecurrentPPONetworks,
    entropy_cost: float = 1e-4,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
    vf_coefficient: float = 0.5,
    clipping_epsilon_value: float | None = None,
) -> Tuple[jnp.ndarray, types.Metrics]:
    """Computes PPO loss for recurrent agents."""
    parametric_action_distribution = ppo_network.parametric_action_distribution

    initial_core_state = data.extras["initial_core_state"]

    def swap_time(x):
        if isinstance(x, jnp.ndarray) and x.ndim >= 2:
            return jnp.swapaxes(x, 0, 1)
        return x

    data = types.Transition(
        observation=jax.tree_util.tree_map(swap_time, data.observation),
        action=jax.tree_util.tree_map(swap_time, data.action),
        reward=jax.tree_util.tree_map(swap_time, data.reward),
        discount=jax.tree_util.tree_map(swap_time, data.discount),
        next_observation=jax.tree_util.tree_map(swap_time, data.next_observation),
        extras={
            k: jax.tree_util.tree_map(swap_time, v)
            for k, v in data.extras.items()
            if k != "initial_core_state"
        },
    )
    truncation = data.extras["state_extras"]["truncation"]
    mask = (1.0 - truncation) * data.discount

    def rollout_fn(carry, inputs):
        core_state = carry
        obs, mask_t = inputs
        logits, value, new_core_state = ppo_network.apply_fn(
            normalizer_params, params.model, obs, core_state
        )
        new_core_state = ppo_network.mask_state_fn(new_core_state, mask_t)
        return new_core_state, (logits, value)

    _, (policy_logits, baseline) = jax.lax.scan(
        rollout_fn, initial_core_state, (data.observation, mask)
    )

    if "advantages" not in data.extras or "target_values" not in data.extras:
        raise ValueError("Advantages and target values must be precomputed.")
    advantages = data.extras["advantages"]
    vs = data.extras["target_values"]
    # Note: advantages and vs are already swapped via swap_time in the Transition reconstruction above
    if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    target_action_log_probs = parametric_action_distribution.log_prob(
        policy_logits, data.extras["policy_extras"]["raw_action"]
    )
    behaviour_action_log_probs = data.extras["policy_extras"]["log_prob"]

    rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)
    surrogate_loss1 = rho_s * advantages
    surrogate_loss2 = (
        jnp.clip(rho_s, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages
    )
    policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

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

    entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, rng))
    entropy_loss = entropy_cost * -entropy

    total_loss = policy_loss + v_loss + entropy_loss

    new_dist = parametric_action_distribution.create_dist(policy_logits)
    if hasattr(new_dist, "kl_divergence"):
        old_dist_params = data.extras["policy_extras"]["distribution_params"]
        old_dist = parametric_action_distribution.create_dist(old_dist_params)
        kl = jnp.mean(
            new_dist.kl_divergence(old_dist)
        )  # pytype: disable=attribute-error
        policy_dist_mean_std = jnp.mean(
            new_dist.scale
        )  # pytype: disable=attribute-error
    else:
        kl, policy_dist_mean_std = jnp.array(0.0), jnp.array(0.0)
    return total_loss, {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "v_loss": v_loss,
        "entropy_loss": entropy_loss,
        "kl_mean": kl,
        "policy_dist_mean_std": policy_dist_mean_std,
    }
