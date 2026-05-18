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

"""Recurrent Soft Actor-Critic training.

Off-policy SAC for RNN policies and twin recurrent Q networks. Sequences are
collected via ``generate_unroll_rnn`` (reused from ``recurrent_ppo``) and
stored as atomic items in a ``UniformSamplingQueue`` — each buffer element is
a sequence of length ``collect_len = unroll_length + burn_in``. At loss time,
the policy's initial hidden is recovered from the stored sequence; the Q
network's hidden is zero-initialized and warmed up over the ``burn_in``
leading steps before backprop.
"""

import functools
import time
from typing import Any, Callable, Mapping, Optional, Tuple

from absl import logging
from brax import base
from brax import envs
from brax.training import gradients
from brax.training import pmap
from brax.training import replay_buffers
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.recurrent_ppo import train as rnn_ppo_train
from brax.training.agents.recurrent_sac import losses as recurrent_sac_losses
from brax.training.agents.recurrent_sac import networks as recurrent_sac_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax


Metrics = types.Metrics
Transition = types.Transition
HiddenState = recurrent_sac_networks.HiddenState
ReplayBufferState = Any
InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]

_PMAP_AXIS_NAME = "i"

# Keys in policy_extras that should be stripped before inserting into the
# replay buffer — we only need initial_policy_hidden at loss time.
_EXTRAS_TO_KEEP = ("initial_policy_hidden",)


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the recurrent SAC learner."""

    policy_optimizer_state: optax.OptState
    policy_params: Params
    q_optimizer_state: optax.OptState
    q_params: Params
    target_q_params: Params
    gradient_steps: types.UInt64
    env_steps: types.UInt64
    alpha_optimizer_state: optax.OptState
    alpha_params: Params
    normalizer_params: running_statistics.RunningStatisticsState


def _unpmap(v):
    return jax.tree_util.tree_map(
        lambda x: x.addressable_shards[0].data.squeeze(0), v
    )


def _prune_policy_extras(transitions: Transition) -> Transition:
    """Keep only the extras fields that the loss actually consumes.

    ``generate_unroll_rnn`` stores all outputs of the inference function in
    ``policy_extras``. For recurrent SAC the loss only needs
    ``initial_policy_hidden``.
    """
    policy_extras = transitions.extras.get("policy_extras", {})
    pruned = {k: v for k, v in policy_extras.items() if k in _EXTRAS_TO_KEEP}
    new_extras = dict(transitions.extras)
    new_extras["policy_extras"] = pruned
    return transitions._replace(extras=new_extras)


def _init_training_state(
    key: PRNGKey,
    obs_shape,
    local_devices_to_use: int,
    recurrent_sac_network: recurrent_sac_networks.RecurrentSACNetworks,
    alpha_optimizer: optax.GradientTransformation,
    policy_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation,
    normalize_observations_std_eps: float,
    normalize_observations_mode: str,
    init_log_alpha: float = 0.0,
) -> TrainingState:
    """Initialize training state and replicate it across devices."""
    key_policy, key_q = jax.random.split(key)

    log_alpha = jnp.asarray(init_log_alpha, dtype=jnp.float32)
    alpha_optimizer_state = alpha_optimizer.init(log_alpha)

    policy_params = recurrent_sac_network.policy_network.init(key_policy)
    policy_optimizer_state = policy_optimizer.init(policy_params)

    q_params = recurrent_sac_network.q_network.init(key_q)
    q_optimizer_state = q_optimizer.init(q_params)

    normalizer_params = running_statistics.init_state(
        obs_shape,
        std_eps=normalize_observations_std_eps,
        mode=normalize_observations_mode,
    )

    training_state = TrainingState(
        policy_optimizer_state=policy_optimizer_state,
        policy_params=policy_params,
        q_optimizer_state=q_optimizer_state,
        q_params=q_params,
        target_q_params=q_params,
        gradient_steps=types.UInt64(hi=0, lo=0),
        env_steps=types.UInt64(hi=0, lo=0),
        alpha_optimizer_state=alpha_optimizer_state,
        alpha_params=log_alpha,
        normalizer_params=normalizer_params,
    )
    return jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
    )


class RecurrentSACEvaluator:
    """Runs evaluation episodes with a recurrent SAC policy."""

    def __init__(
        self,
        eval_env: envs.Env,
        eval_policy_fn: Callable[[Params], Callable],
        recurrent_sac_network: recurrent_sac_networks.RecurrentSACNetworks,
        num_eval_envs: int,
        episode_length: int,
        action_repeat: int,
        key: PRNGKey,
    ):
        self._key = key
        self._eval_walltime = 0.0
        self._num_eval_envs = num_eval_envs

        eval_env = envs.training.EvalWrapper(eval_env)
        self._eval_state_to_donate = jax.jit(eval_env.reset)(
            jax.random.split(key, num_eval_envs)
        )

        def generate_eval_unroll(
            eval_env_state_donated: envs.State,
            policy_params,
            key: PRNGKey,
        ) -> envs.State:
            reset_keys = jax.random.split(key, num_eval_envs)
            eval_first_state = eval_env.reset(reset_keys)
            policy_hidden = recurrent_sac_network.policy_network.init_hidden(
                num_eval_envs
            )
            final_state, _, _ = rnn_ppo_train.generate_unroll_rnn(
                eval_env,
                eval_first_state,
                eval_policy_fn(policy_params),
                policy_hidden,
                key,
                unroll_length=episode_length // action_repeat,
                store_initial_hidden=False,
                rnn_ppo_network=recurrent_sac_network,
            )
            return final_state

        self._generate_eval_unroll = jax.jit(
            generate_eval_unroll, donate_argnums=(0,), keep_unused=True
        )
        self._steps_per_unroll = episode_length * num_eval_envs

    def run_evaluation(
        self,
        policy_params,
        training_metrics: Metrics,
        aggregate_episodes: bool = True,
    ) -> Metrics:
        self._key, unroll_key = jax.random.split(self._key)
        t = time.time()
        eval_state = self._generate_eval_unroll(
            self._eval_state_to_donate, policy_params, unroll_key
        )
        self._eval_state_to_donate = eval_state

        eval_metrics = eval_state.info["eval_metrics"]
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        episode_lengths = np.maximum(eval_metrics.episode_steps, 1.0).astype(float)

        def _agg(metric, fn, to_aggregate, to_normalize, episode_lengths):
            if not to_aggregate:
                return metric
            if to_normalize:
                return fn(metric / episode_lengths)
            return fn(metric)

        metrics: Metrics = {}
        for fn in [np.mean, np.std]:
            suffix = "_std" if fn == np.std else ""
            for name, value in eval_metrics.episode_metrics.items():
                metrics[f"eval/episode_{name}{suffix}"] = _agg(
                    value,
                    fn,
                    aggregate_episodes,
                    name.endswith("per_step"),
                    episode_lengths,
                )
        metrics["eval/avg_episode_length"] = np.mean(eval_metrics.episode_steps)
        metrics["eval/std_episode_length"] = np.std(eval_metrics.episode_steps)
        metrics["eval/epoch_eval_time"] = epoch_eval_time
        metrics["eval/sps"] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime = self._eval_walltime + epoch_eval_time
        return {
            "eval/walltime": self._eval_walltime,
            **training_metrics,
            **metrics,
        }


def train(
    environment: envs.Env,
    num_timesteps: int,
    episode_length: int,
    wrap_env: bool = True,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_eval_envs: int = 128,
    learning_rate: float = 3e-4,
    discounting: float = 0.99,
    seed: int = 0,
    batch_size: int = 64,
    unroll_length: int = 32,
    burn_in: int = 0,
    num_evals: int = 1,
    normalize_observations: bool = False,
    normalize_observations_std_eps: float = 0.0,
    normalize_observations_mode: str = "welford",
    max_devices_per_host: Optional[int] = None,
    reward_scaling: float = 1.0,
    tau: float = 0.005,
    min_replay_size: int = 0,
    max_replay_size: Optional[int] = None,
    grad_updates_per_step: int = 1,
    deterministic_eval: bool = False,
    network_factory: types.NetworkFactory[
        recurrent_sac_networks.RecurrentSACNetworks
    ] = recurrent_sac_networks.make_recurrent_sac_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    eval_env: Optional[envs.Env] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    alpha_learning_rate: float = 3e-4,
    target_entropy: Optional[float] = None,
    init_log_alpha: float = 0.0,
):
    """Recurrent SAC training.

    Args:
      environment: Environment to train on.
      num_timesteps: Total environment steps for the full run.
      episode_length: Maximum episode length (passed to the wrapper).
      wrap_env: Whether to wrap the env for training / auto-reset.
      wrap_env_fn: Optional custom wrapper.
      action_repeat: Number of env steps per action.
      num_envs: Number of parallel training envs (total across devices).
      num_eval_envs: Number of parallel eval envs.
      learning_rate: Adam learning rate for the policy and Q optimizers.
      discounting: Discount factor (``gamma``).
      seed: Random seed.
      batch_size: Number of sequences per gradient step.
      unroll_length: Number of gradient-carrying timesteps per sampled sequence.
      burn_in: Leading timesteps used to warm RNN hidden states without
        gradient. Stored sequence length is ``unroll_length + burn_in``.
      num_evals: Number of evaluation rounds across training.
      normalize_observations: Whether to normalize observations.
      normalize_observations_std_eps: Stability epsilon for normalizer.
      normalize_observations_mode: ``"welford"`` or ``"ema"``.
      max_devices_per_host: Cap on devices to use per host.
      reward_scaling: Reward scaling factor passed to the critic loss.
      tau: Polyak averaging coefficient for the target Q network.
      min_replay_size: Env-step count that must be in the buffer before
        training gradient steps begin.
      max_replay_size: Maximum env-step capacity of the buffer (internally
        divided by ``collect_len`` to get sequence capacity). Defaults to
        ``num_timesteps``.
      grad_updates_per_step: Number of SGD steps per collected sequence batch.
      deterministic_eval: Whether eval policy takes the distribution mode.
      network_factory: Factory producing ``RecurrentSACNetworks``.
      progress_fn: Called with ``(current_step, metrics)`` after each eval.
      policy_params_fn: Called with ``(current_step, make_policy, params)``
        after each eval.
      eval_env: Optional separate environment for eval.
      randomization_fn: Optional domain-randomization function.

    Returns:
      ``(make_policy, (normalizer_params, policy_params, q_params), metrics)``.
    """
    process_id = jax.process_index()
    local_devices_to_use = jax.local_device_count()
    if max_devices_per_host is not None:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    device_count = local_devices_to_use * jax.process_count()
    logging.info(
        "local_device_count: %s; total_device_count: %s",
        local_devices_to_use,
        device_count,
    )

    if burn_in < 0:
        raise ValueError(f"burn_in must be >= 0, got {burn_in}")
    collect_len = unroll_length + burn_in

    if min_replay_size >= num_timesteps:
        raise ValueError(
            "No training will happen because min_replay_size >= num_timesteps"
        )

    if max_replay_size is None:
        max_replay_size = num_timesteps

    assert num_envs % device_count == 0, (
        f"num_envs ({num_envs}) must be divisible by device_count ({device_count})"
    )

    # Per actor_step, each of num_envs envs produces collect_len env steps.
    env_steps_per_actor_step = action_repeat * num_envs * collect_len
    num_prefill_actor_steps = -(-min_replay_size // (num_envs * collect_len))
    num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
    if num_timesteps - num_prefill_env_steps < 0:
        raise ValueError(
            f"num_timesteps {num_timesteps} < num_prefill_env_steps "
            f"{num_prefill_env_steps}. Increase num_timesteps or decrease "
            "min_replay_size / unroll_length / num_envs."
        )

    num_evals_after_init = max(num_evals - 1, 1)
    num_training_steps_per_epoch = -(
        -(num_timesteps - num_prefill_env_steps)
        // (num_evals_after_init * env_steps_per_actor_step)
    )

    # -- Env wrapping ------------------------------------------------------
    env = environment
    rng = jax.random.PRNGKey(seed)
    rng, key_env = jax.random.split(rng)
    if wrap_env:
        v_randomization_fn = None
        if randomization_fn is not None:
            v_randomization_fn = functools.partial(
                randomization_fn,
                rng=jax.random.split(
                    key_env, num_envs // jax.process_count() // local_devices_to_use
                ),
            )
        if wrap_env_fn is not None:
            wrap_for_training = wrap_env_fn
        else:
            wrap_for_training = envs.training.wrap
        env = wrap_for_training(
            env,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=v_randomization_fn,
        )

    # -- Network + optimizers ---------------------------------------------
    normalize_fn = lambda x, y: x
    if normalize_observations:
        normalize_fn = running_statistics.normalize
    recurrent_sac_network = network_factory(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=normalize_fn,
    )
    make_policy = recurrent_sac_networks.make_inference_fn(recurrent_sac_network)

    alpha_optimizer = optax.adam(learning_rate=alpha_learning_rate)
    policy_optimizer = optax.adam(learning_rate=learning_rate)
    q_optimizer = optax.adam(learning_rate=learning_rate)

    alpha_loss_fn, critic_loss_fn, actor_loss_fn = recurrent_sac_losses.make_losses(
        recurrent_sac_network=recurrent_sac_network,
        reward_scaling=reward_scaling,
        discounting=discounting,
        action_size=env.action_size,
        burn_in=burn_in,
        target_entropy=target_entropy,
    )
    alpha_update = gradients.gradient_update_fn(
        alpha_loss_fn, alpha_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
    )
    critic_update = gradients.gradient_update_fn(
        critic_loss_fn, q_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
    )
    actor_update = gradients.gradient_update_fn(
        actor_loss_fn, policy_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
    )

    # -- Replay buffer -----------------------------------------------------
    # Build a dummy transition whose leading dim is `collect_len` — each
    # stored buffer element is a full sequence.
    def _per_env_dummy_obs_leaf(sz):
        return jnp.zeros((sz,), jnp.float32)

    def _make_dummy_obs(obs_size):
        if isinstance(obs_size, Mapping):
            return {k: _per_env_dummy_obs_leaf(v) for k, v in obs_size.items()}
        return _per_env_dummy_obs_leaf(obs_size)

    dummy_obs = _make_dummy_obs(env.observation_size)
    dummy_action = jnp.zeros((env.action_size,), jnp.float32)

    # Hidden state example for storage: per-timestep initial policy hidden.
    dummy_hidden_single = recurrent_sac_network.policy_network.init_hidden(1)
    # Strip the leading batch dim (=1) to get per-env shape.
    dummy_hidden_per_env = jax.tree_util.tree_map(
        lambda x: x[0], dummy_hidden_single
    )

    def _stack_over_time(x):
        return jnp.stack([x] * collect_len, axis=0)

    dummy_initial_hidden_seq = jax.tree_util.tree_map(
        _stack_over_time, dummy_hidden_per_env
    )
    dummy_obs_seq = jax.tree_util.tree_map(_stack_over_time, dummy_obs)
    dummy_action_seq = _stack_over_time(dummy_action)

    dummy_transition_seq = Transition(
        observation=dummy_obs_seq,
        action=dummy_action_seq,
        reward=jnp.zeros((collect_len,), jnp.float32),
        discount=jnp.zeros((collect_len,), jnp.float32),
        next_observation=dummy_obs_seq,
        extras={
            "state_extras": {
                "truncation": jnp.zeros((collect_len,), jnp.float32),
            },
            "policy_extras": {
                "initial_policy_hidden": dummy_initial_hidden_seq,
            },
        },
    )

    max_sequences_per_device = max(
        1, max_replay_size // device_count // collect_len
    )
    sample_batch_size = max(1, batch_size * grad_updates_per_step // device_count)
    replay_buffer = replay_buffers.UniformSamplingQueue(
        max_replay_size=max_sequences_per_device,
        dummy_data_sample=dummy_transition_seq,
        sample_batch_size=sample_batch_size,
    )

    # -- sgd_step ----------------------------------------------------------
    def sgd_step(carry, sequences: Transition):
        training_state, key = carry
        key, key_alpha, key_critic, key_actor = jax.random.split(key, 4)

        a_loss, alpha_params, alpha_optimizer_state = alpha_update(
            training_state.alpha_params,
            training_state.policy_params,
            training_state.normalizer_params,
            sequences,
            key_alpha,
            optimizer_state=training_state.alpha_optimizer_state,
        )
        alpha = jnp.exp(training_state.alpha_params)

        c_loss, q_params, q_optimizer_state = critic_update(
            training_state.q_params,
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.target_q_params,
            alpha,
            sequences,
            key_critic,
            optimizer_state=training_state.q_optimizer_state,
        )

        p_loss, policy_params, policy_optimizer_state = actor_update(
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.q_params,
            alpha,
            sequences,
            key_actor,
            optimizer_state=training_state.policy_optimizer_state,
        )

        new_target_q_params = jax.tree_util.tree_map(
            lambda t, q: t * (1 - tau) + q * tau,
            training_state.target_q_params,
            q_params,
        )

        metrics = {
            "critic_loss": c_loss,
            "actor_loss": p_loss,
            "alpha_loss": a_loss,
            "alpha": jnp.exp(alpha_params),
        }

        new_state = TrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            q_optimizer_state=q_optimizer_state,
            q_params=q_params,
            target_q_params=new_target_q_params,
            gradient_steps=training_state.gradient_steps + 1,
            env_steps=training_state.env_steps,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params,
            normalizer_params=training_state.normalizer_params,
        )
        return (new_state, key), metrics

    # -- get_experience ----------------------------------------------------
    def get_experience(
        training_state: TrainingState,
        env_state: envs.State,
        policy_hidden: HiddenState,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[
        running_statistics.RunningStatisticsState,
        envs.State,
        HiddenState,
        ReplayBufferState,
    ]:
        policy = make_policy(
            (training_state.normalizer_params, training_state.policy_params)
        )
        next_env_state, transitions, new_policy_hidden = (
            rnn_ppo_train.generate_unroll_rnn(
                env,
                env_state,
                policy,
                policy_hidden,
                key,
                unroll_length=collect_len,
                extra_fields=("truncation",),
                store_initial_hidden=True,
                rnn_ppo_network=recurrent_sac_network,
            )
        )
        # transitions: [collect_len, envs_per_device, ...]  ->  [envs_per_device, collect_len, ...]
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1), transitions
        )
        transitions = _prune_policy_extras(transitions)

        # Update normalizer with observations flattened over (env, time).
        def _flatten_for_normalize(x):
            return x.reshape((-1,) + x.shape[2:])

        flat_obs = jax.tree_util.tree_map(
            _flatten_for_normalize, transitions.observation
        )
        normalizer_params = running_statistics.update(
            training_state.normalizer_params,
            flat_obs,
            pmap_axis_name=_PMAP_AXIS_NAME,
        )

        buffer_state = replay_buffer.insert(buffer_state, transitions)
        return normalizer_params, next_env_state, new_policy_hidden, buffer_state

    # -- training_step -----------------------------------------------------
    def training_step(
        training_state: TrainingState,
        env_state: envs.State,
        policy_hidden: HiddenState,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ):
        experience_key, training_key = jax.random.split(key)
        normalizer_params, env_state, policy_hidden, buffer_state = get_experience(
            training_state,
            env_state,
            policy_hidden,
            buffer_state,
            experience_key,
        )
        training_state = training_state.replace(
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + env_steps_per_actor_step,
        )

        buffer_state, sequences = replay_buffer.sample(buffer_state)
        # sequences: [batch_size * grad_updates_per_step, collect_len, ...]
        # -> [grad_updates_per_step, batch_per_grad, collect_len, ...]
        sequences = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (grad_updates_per_step, -1) + x.shape[1:]),
            sequences,
        )
        (training_state, _), metrics = jax.lax.scan(
            sgd_step, (training_state, training_key), sequences
        )
        metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        return training_state, env_state, policy_hidden, buffer_state, metrics

    # -- prefill ----------------------------------------------------------
    def prefill_replay_buffer(
        training_state: TrainingState,
        env_state: envs.State,
        policy_hidden: HiddenState,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ):
        def f(carry, _):
            ts, es, ph, bs, k = carry
            k, k_new = jax.random.split(k)
            new_norm, es, ph, bs = get_experience(ts, es, ph, bs, k)
            ts = ts.replace(
                normalizer_params=new_norm,
                env_steps=ts.env_steps + env_steps_per_actor_step,
            )
            return (ts, es, ph, bs, k_new), ()

        (ts, es, ph, bs, _), _ = jax.lax.scan(
            f,
            (training_state, env_state, policy_hidden, buffer_state, key),
            (),
            length=num_prefill_actor_steps,
        )
        return ts, es, ph, bs

    prefill_replay_buffer = jax.pmap(
        prefill_replay_buffer,
        axis_name=_PMAP_AXIS_NAME,
        donate_argnums=(0, 1, 2, 3),
    )

    # -- training_epoch ---------------------------------------------------
    def training_epoch(
        training_state: TrainingState,
        env_state: envs.State,
        policy_hidden: HiddenState,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ):
        def f(carry, _):
            ts, es, ph, bs, k = carry
            k, k_new = jax.random.split(k)
            ts, es, ph, bs, m = training_step(ts, es, ph, bs, k)
            return (ts, es, ph, bs, k_new), m

        (ts, es, ph, bs, _), metrics = jax.lax.scan(
            f,
            (training_state, env_state, policy_hidden, buffer_state, key),
            (),
            length=num_training_steps_per_epoch,
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return ts, es, ph, bs, metrics

    training_epoch = jax.pmap(
        training_epoch,
        axis_name=_PMAP_AXIS_NAME,
        donate_argnums=(0, 1, 2, 3),
    )

    training_walltime = 0.0

    def training_epoch_with_timing(
        training_state: TrainingState,
        env_state: envs.State,
        policy_hidden: HiddenState,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ):
        nonlocal training_walltime
        t = time.time()
        ts, es, ph, bs, metrics = training_epoch(
            training_state, env_state, policy_hidden, buffer_state, key
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
        epoch_time = time.time() - t
        training_walltime += epoch_time
        sps = (
            env_steps_per_actor_step * num_training_steps_per_epoch
        ) / epoch_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        return ts, es, ph, bs, metrics

    # -- Init training state + buffer + env + eval ------------------------
    global_key, local_key = jax.random.split(rng)
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)

    # Env init — per-device reset. Do this first so we can derive the
    # per-sample observation shape from the resulting env_state.obs.
    env_keys = jax.random.split(env_key, num_envs // jax.process_count())
    env_keys = jnp.reshape(
        env_keys, (local_devices_to_use, -1) + env_keys.shape[1:]
    )
    env_state = jax.pmap(env.reset)(env_keys)

    # env_state.obs has shape [local_devices, envs_per_device, *obs_shape].
    # Drop the two leading dims to get the per-sample obs shape for the
    # running-statistics normalizer.
    per_sample_obs_shape = jax.tree_util.tree_map(
        lambda x: specs.Array(x.shape[2:], jnp.dtype("float32")),
        env_state.obs,
    )

    training_state = _init_training_state(
        key=global_key,
        obs_shape=per_sample_obs_shape,
        local_devices_to_use=local_devices_to_use,
        recurrent_sac_network=recurrent_sac_network,
        alpha_optimizer=alpha_optimizer,
        policy_optimizer=policy_optimizer,
        q_optimizer=q_optimizer,
        normalize_observations_std_eps=normalize_observations_std_eps,
        normalize_observations_mode=normalize_observations_mode,
        init_log_alpha=init_log_alpha,
    )
    del global_key

    # Replay buffer init per device.
    buffer_state = jax.pmap(replay_buffer.init)(
        jax.random.split(rb_key, local_devices_to_use)
    )

    # Policy hidden init — [local_devices, envs_per_device, hidden].
    envs_per_device = num_envs // device_count

    def _init_policy_hidden():
        hidden = recurrent_sac_network.policy_network.init_hidden(envs_per_device)
        return jax.device_put_replicated(
            hidden, jax.local_devices()[:local_devices_to_use]
        )

    policy_hidden = _init_policy_hidden()

    # Eval setup.
    if not eval_env:
        eval_env = environment
    if wrap_env:
        if randomization_fn is not None:
            v_randomization_fn = functools.partial(
                randomization_fn, rng=jax.random.split(eval_key, num_eval_envs)
            )
        else:
            v_randomization_fn = None
        eval_env = (wrap_env_fn or envs.training.wrap)(
            eval_env,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=v_randomization_fn,
        )

    evaluator = RecurrentSACEvaluator(
        eval_env,
        functools.partial(make_policy, deterministic=deterministic_eval),
        recurrent_sac_network,
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )

    # Initial eval.
    metrics: Metrics = {}
    if process_id == 0 and num_evals > 1:
        metrics = evaluator.run_evaluation(
            _unpmap(
                (training_state.normalizer_params, training_state.policy_params)
            ),
            training_metrics={},
        )
        logging.info(metrics)
        progress_fn(0, metrics)

    # Prefill replay buffer.
    t = time.time()
    prefill_key, local_key = jax.random.split(local_key)
    prefill_keys = jax.random.split(prefill_key, local_devices_to_use)
    training_state, env_state, policy_hidden, buffer_state = prefill_replay_buffer(
        training_state, env_state, policy_hidden, buffer_state, prefill_keys
    )
    replay_size = jnp.sum(jax.vmap(replay_buffer.size)(buffer_state)) * jax.process_count()
    logging.info("replay size after prefill: %s sequences", replay_size)
    training_walltime = time.time() - t

    current_step = 0
    for _ in range(num_evals_after_init):
        logging.info("step %s", current_step)

        epoch_key, local_key = jax.random.split(local_key)
        epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
        training_state, env_state, policy_hidden, buffer_state, training_metrics = (
            training_epoch_with_timing(
                training_state, env_state, policy_hidden, buffer_state, epoch_keys
            )
        )
        current_step = int(_unpmap(training_state.env_steps))

        if process_id == 0:
            params = _unpmap(
                (
                    training_state.normalizer_params,
                    training_state.policy_params,
                    training_state.q_params,
                )
            )
            policy_params_fn(current_step, make_policy, params)

            metrics = evaluator.run_evaluation(
                _unpmap(
                    (
                        training_state.normalizer_params,
                        training_state.policy_params,
                    )
                ),
                training_metrics,
            )
            logging.info(metrics)
            progress_fn(current_step, metrics)

    total_steps = current_step
    if total_steps < num_timesteps:
        logging.warning(
            "Total env steps %d below requested num_timesteps %d",
            total_steps,
            num_timesteps,
        )

    params = _unpmap(
        (
            training_state.normalizer_params,
            training_state.policy_params,
            training_state.q_params,
        )
    )

    pmap.assert_is_replicated(training_state)
    logging.info("total steps: %s", total_steps)
    pmap.synchronize_hosts()
    return make_policy, params, metrics
