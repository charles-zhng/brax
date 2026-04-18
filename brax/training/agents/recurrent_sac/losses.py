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

"""Recurrent SAC losses.

Operates on batched sequences with leading shape ``[B, T, ...]``. Each loss:
  - extracts the initial policy hidden state stored at the first timestep of
    the stored sequence (``extras["policy_extras"]["initial_policy_hidden"][:, 0]``),
  - scans the policy network forward over the sequence with done-reset hidden,
  - zero-initializes the Q network hidden state and scans it forward over the
    sequence with done-reset hidden,
  - uses the first ``burn_in`` steps of each sequence to warm the Q hidden
    (stop-gradient) and computes the actual loss over the remaining
    ``unroll_length = T - burn_in`` steps.
"""

from typing import Any, Tuple

from brax.training import types
from brax.training.agents.recurrent_sac import networks as recurrent_sac_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
import jax
import jax.numpy as jnp


Transition = types.Transition


def _reset_hidden_on_done(hidden, done, cell_type):
    """Zero out hidden state entries where ``done`` is True.

    ``hidden`` may be a single array (SimpleCell/GRU) or a ``(c, h)`` tuple
    (LSTM). ``done`` broadcasts along the trailing feature dim.
    """
    done_expanded = done[..., None]
    if cell_type == "lstm":
        c, h = hidden
        return (
            jnp.where(done_expanded, 0.0, c),
            jnp.where(done_expanded, 0.0, h),
        )
    return jnp.where(done_expanded, 0.0, hidden)


def _extract_initial_hidden(hidden):
    """Pull the first-timestep hidden out of a [B, T, H] stored sequence."""
    if hidden is None:
        return None
    return jax.tree_util.tree_map(
        lambda x: x[:, 0] if x.ndim > 2 else x, hidden
    )


def _time_first(x):
    return jax.tree_util.tree_map(lambda y: jnp.swapaxes(y, 0, 1), x)


def _batch_first(x):
    return jax.tree_util.tree_map(lambda y: jnp.swapaxes(y, 0, 1), x)


def make_losses(
    recurrent_sac_network: recurrent_sac_networks.RecurrentSACNetworks,
    reward_scaling: float,
    discounting: float,
    action_size: int,
    burn_in: int = 0,
):
    """Build the alpha, critic, and actor loss functions.

    Args:
      recurrent_sac_network: Recurrent SAC network container.
      reward_scaling: Reward scaling factor.
      discounting: Discount factor (``gamma``).
      action_size: Action dimensionality (used for the entropy target).
      burn_in: Number of leading timesteps within each stored sequence to use
        for warming up RNN hidden states under ``stop_gradient`` before the
        loss is computed. Default ``0`` is equivalent to pure zero-init Q
        hidden (no burn-in).

    Returns:
      ``(alpha_loss, critic_loss, actor_loss)``.
    """
    target_entropy = -0.5 * action_size
    policy_network = recurrent_sac_network.policy_network
    q_network = recurrent_sac_network.q_network
    param_dist = recurrent_sac_network.parametric_action_distribution
    cell_type = recurrent_sac_network.cell_type

    def _policy_forward_scan(
        policy_params: Params,
        normalizer_params: Any,
        obs_seq: jnp.ndarray,
        done_seq: jnp.ndarray,
        initial_hidden,
        key: PRNGKey,
    ):
        """Scan policy over a [B, T, ...] sequence with done-reset hidden.

        Returns:
          logits_seq:       [B, T, param_size]
          raw_action_seq:   [B, T, action_dim]   (sample_no_postprocessing)
          log_prob_seq:     [B, T]
          hidden_after_seq: [B, T, H]            (hidden after consuming obs_t,
                                                  done-reset applied at t)
        """
        T = done_seq.shape[1]
        keys_t = jax.random.split(key, T)  # [T, 2]
        obs_seq_t = _time_first(obs_seq)
        done_seq_t = jnp.swapaxes(done_seq, 0, 1)

        def step(hidden, inputs):
            obs_t, done_t, key_t = inputs
            logits, new_hidden = policy_network.apply(
                normalizer_params, policy_params, obs_t, hidden
            )
            raw_action = param_dist.sample_no_postprocessing(logits, key_t)
            log_prob = param_dist.log_prob(logits, raw_action)
            new_hidden = _reset_hidden_on_done(new_hidden, done_t, cell_type)
            return new_hidden, (logits, raw_action, log_prob, new_hidden)

        _, (logits_seq_t, raw_action_seq_t, log_prob_seq_t, hidden_after_t) = (
            jax.lax.scan(step, initial_hidden, (obs_seq_t, done_seq_t, keys_t))
        )
        return (
            jnp.swapaxes(logits_seq_t, 0, 1),
            jnp.swapaxes(raw_action_seq_t, 0, 1),
            jnp.swapaxes(log_prob_seq_t, 0, 1),
            _batch_first(hidden_after_t),
        )

    def _q_forward_scan(
        q_params: Params,
        normalizer_params: Any,
        obs_seq,
        action_seq: jnp.ndarray,
        done_seq: jnp.ndarray,
        initial_hidden,
    ) -> jnp.ndarray:
        """Scan twin Q over a [B, T, ...] sequence with done-reset hidden.

        Returns ``q_seq`` with shape ``[B, T, 2]``.
        """
        obs_seq_t = _time_first(obs_seq)
        action_seq_t = jnp.swapaxes(action_seq, 0, 1)
        done_seq_t = jnp.swapaxes(done_seq, 0, 1)

        def step(hidden, inputs):
            obs_t, action_t, done_t = inputs
            q_t, new_hidden = q_network.apply(
                normalizer_params, q_params, obs_t, action_t, hidden
            )
            new_hidden = _reset_hidden_on_done(new_hidden, done_t, cell_type)
            return new_hidden, q_t

        _, q_seq_t = jax.lax.scan(
            step, initial_hidden, (obs_seq_t, action_seq_t, done_seq_t)
        )
        return jnp.swapaxes(q_seq_t, 0, 1)

    def _apply_policy_on_next_obs(
        policy_params: Params,
        normalizer_params: Any,
        next_obs_seq,
        hidden_after_seq,
    ) -> jnp.ndarray:
        """Apply policy to next_obs[:, t] using hidden_after[:, t].

        All (b, t) positions are independent given their respective hiddens,
        so we flatten to a single batched call. Returns logits of shape
        ``[B, T, param_size]``.
        """
        leaf = (
            next_obs_seq
            if isinstance(next_obs_seq, jnp.ndarray)
            else next(iter(jax.tree_util.tree_leaves(next_obs_seq)))
        )
        B, T = leaf.shape[0], leaf.shape[1]

        def _flatten(x):
            return x.reshape(B * T, *x.shape[2:])

        next_obs_flat = jax.tree_util.tree_map(_flatten, next_obs_seq)
        hidden_flat = jax.tree_util.tree_map(_flatten, hidden_after_seq)
        logits_flat, _ = policy_network.apply(
            normalizer_params, policy_params, next_obs_flat, hidden_flat
        )
        return logits_flat.reshape(B, T, -1)

    # ---------------------------------------------------------------- losses

    def alpha_loss(
        log_alpha: jnp.ndarray,
        policy_params: Params,
        normalizer_params: Any,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        """Automatic entropy-temperature loss (Eq 18 of SAC paper)."""
        done_seq = transitions.discount < 0.5
        initial_hidden = _extract_initial_hidden(
            transitions.extras["policy_extras"]["initial_policy_hidden"]
        )
        _, _, log_prob_seq, _ = _policy_forward_scan(
            policy_params,
            normalizer_params,
            transitions.observation,
            done_seq,
            initial_hidden,
            key,
        )
        log_prob_bp = log_prob_seq[:, burn_in:]
        alpha = jnp.exp(log_alpha)
        return jnp.mean(
            alpha * jax.lax.stop_gradient(-log_prob_bp - target_entropy)
        )

    def critic_loss(
        q_params: Params,
        policy_params: Params,
        normalizer_params: Any,
        target_q_params: Params,
        alpha: jnp.ndarray,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        """Twin-Q Bellman loss with entropy-regularized targets."""
        done_seq = transitions.discount < 0.5

        # Extract initial hidden state from stored extras.
        initial_policy_hidden = _extract_initial_hidden(
            transitions.extras["policy_extras"]["initial_policy_hidden"]
        )

        # Forward scan of the policy over obs_seq produces hidden_after_seq,
        # which is the correct initial hidden for processing next_obs_t.
        policy_key, next_sample_key = jax.random.split(key)
        _, _, _, hidden_after_seq = _policy_forward_scan(
            policy_params,
            normalizer_params,
            transitions.observation,
            done_seq,
            initial_policy_hidden,
            policy_key,
        )

        # Compute policy distribution at next_obs.
        next_logits_seq = _apply_policy_on_next_obs(
            policy_params,
            normalizer_params,
            transitions.next_observation,
            hidden_after_seq,
        )
        next_raw_action_seq = param_dist.sample_no_postprocessing(
            next_logits_seq, next_sample_key
        )
        next_log_prob_seq = param_dist.log_prob(
            next_logits_seq, next_raw_action_seq
        )
        next_action_seq = param_dist.postprocess(next_raw_action_seq)

        # Target Q scan over (next_obs, next_action) with zero hidden.
        leaf = (
            transitions.observation
            if isinstance(transitions.observation, jnp.ndarray)
            else next(iter(jax.tree_util.tree_leaves(transitions.observation)))
        )
        B = leaf.shape[0]
        zero_q_hidden = q_network.init_hidden(B)
        next_q_seq = _q_forward_scan(
            target_q_params,
            normalizer_params,
            transitions.next_observation,
            next_action_seq,
            done_seq,
            zero_q_hidden,
        )
        next_v_seq = jnp.min(next_q_seq, axis=-1) - alpha * next_log_prob_seq
        target_q_seq = jax.lax.stop_gradient(
            transitions.reward * reward_scaling
            + transitions.discount * discounting * next_v_seq
        )  # [B, T]

        # Current Q scan over (obs, action) with zero hidden.
        current_q_seq = _q_forward_scan(
            q_params,
            normalizer_params,
            transitions.observation,
            transitions.action,
            done_seq,
            zero_q_hidden,
        )  # [B, T, 2]

        q_error = current_q_seq - target_q_seq[..., None]  # [B, T, 2]

        # Better bootstrapping on truncated transitions (matches stock SAC).
        truncation = transitions.extras["state_extras"]["truncation"]  # [B, T]
        q_error = q_error * jnp.expand_dims(1.0 - truncation, -1)

        # Burn-in slice: drop first burn_in steps from the loss.
        q_error_bp = q_error[:, burn_in:]
        return 0.5 * jnp.mean(jnp.square(q_error_bp))

    def actor_loss(
        policy_params: Params,
        normalizer_params: Any,
        q_params: Params,
        alpha: jnp.ndarray,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        """Policy loss: ``E[alpha * log pi - min_i Q_i(s, pi(s))]``."""
        done_seq = transitions.discount < 0.5
        initial_policy_hidden = _extract_initial_hidden(
            transitions.extras["policy_extras"]["initial_policy_hidden"]
        )
        _, raw_action_seq, log_prob_seq, _ = _policy_forward_scan(
            policy_params,
            normalizer_params,
            transitions.observation,
            done_seq,
            initial_policy_hidden,
            key,
        )
        action_seq = param_dist.postprocess(raw_action_seq)

        leaf = (
            transitions.observation
            if isinstance(transitions.observation, jnp.ndarray)
            else next(iter(jax.tree_util.tree_leaves(transitions.observation)))
        )
        B = leaf.shape[0]
        zero_q_hidden = q_network.init_hidden(B)
        q_seq = _q_forward_scan(
            q_params,
            normalizer_params,
            transitions.observation,
            action_seq,
            done_seq,
            zero_q_hidden,
        )  # [B, T, 2]

        min_q_seq = jnp.min(q_seq, axis=-1)  # [B, T]
        actor_loss_seq = alpha * log_prob_seq - min_q_seq
        return jnp.mean(actor_loss_seq[:, burn_in:])

    return alpha_loss, critic_loss, actor_loss
