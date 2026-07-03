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

"""Recurrent PPO training.

This module implements PPO with recurrent neural networks (RNN, GRU, LSTM).
See: https://arxiv.org/pdf/1707.06347.pdf
"""

import functools
import time
from typing import Any, Callable, Mapping, Optional, Tuple, Union

from absl import logging
from brax import base
from brax import envs
from brax.training import gradients
from brax.training import logger as metric_logger
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.ppo import optimizer as ppo_optimizer
from brax.training.agents.recurrent_ppo import checkpoint
from brax.training.agents.recurrent_ppo import losses as rnn_ppo_losses
from brax.training.agents.recurrent_ppo import networks as rnn_ppo_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics
HiddenState = rnn_ppo_networks.HiddenState

_PMAP_AXIS_NAME = "i"


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    optimizer_state: optax.OptState
    params: rnn_ppo_losses.RNNPPONetworkParams
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: types.UInt64


def _unpmap(v):
    # Avoid degraded performance under the new jax.pmap.
    return jax.tree_util.tree_map(
        lambda x: x.addressable_shards[0].data.squeeze(0), v
    )


def _strip_weak_type(tree):
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return jnp.astype(leaf, leaf.dtype)

    return jax.tree_util.tree_map(f, tree)


def _maybe_wrap_env(
    env: envs.Env,
    wrap_env: bool,
    num_envs: int,
    episode_length: Optional[int],
    action_repeat: int,
    device_count: int,
    key_env: PRNGKey,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
):
    """Wraps the environment for training/eval if wrap_env is True."""
    if not wrap_env:
        return env
    if episode_length is None:
        raise ValueError("episode_length must be specified in rnn_ppo.train")
    v_randomization_fn = None
    if randomization_fn is not None:
        randomization_batch_size = num_envs // device_count
        randomization_rng = jax.random.split(key_env, randomization_batch_size)
        v_randomization_fn = functools.partial(randomization_fn, rng=randomization_rng)
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
    return env


def _remove_pixels(
    obs: Union[jnp.ndarray, Mapping[str, jax.Array]],
) -> Union[jnp.ndarray, Mapping[str, jax.Array]]:
    """Removes pixel observations from the observation dict."""
    if not isinstance(obs, Mapping):
        return obs
    return {k: v for k, v in obs.items() if not k.startswith("pixels/")}


def _random_translate_pixels(
    obs: Mapping[str, jax.Array], key: PRNGKey
) -> Mapping[str, jax.Array]:
    """Apply random translations to B x T x ... pixel observations.

    The same shift is applied across the unroll_length (T) dimension.

    Args:
      obs: a dictionary of observations
      key: a PRNGKey

    Returns:
      A dictionary of observations with translated pixels
    """

    @jax.vmap
    def rt_all_views(
        ub_obs: Mapping[str, jax.Array], key: PRNGKey
    ) -> Mapping[str, jax.Array]:
        # Expects dictionary of unbatched observations.
        def rt_view(img: jax.Array, padding: int, key: PRNGKey) -> jax.Array:  # TxHxWxC
            # Randomly translates a set of pixel inputs.
            # Adapted from
            # https://github.com/ikostrikov/jaxrl/blob/main/jaxrl/agents/drq/augmentations.py
            crop_from = jax.random.randint(key, (2,), 0, 2 * padding + 1)
            zero = jnp.zeros((1,), dtype=jnp.int32)
            crop_from = jnp.concatenate([zero, crop_from, zero])
            padded_img = jnp.pad(
                img,
                ((0, 0), (padding, padding), (padding, padding), (0, 0)),
                mode="edge",
            )
            return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)

        out = {}
        for k_view, v_view in ub_obs.items():
            if k_view.startswith("pixels/"):
                key, key_shift = jax.random.split(key)
                out[k_view] = rt_view(v_view, 4, key_shift)
        return {**ub_obs, **out}

    bdim = next(iter(obs.items()), None)[1].shape[0]
    keys = jax.random.split(key, bdim)
    obs = rt_all_views(obs, keys)
    return obs


def _reset_hidden_on_done(
    hidden: HiddenState,
    done: jnp.ndarray,
    rnn_ppo_network: rnn_ppo_networks.RNNPPONetworks,
) -> HiddenState:
    """Reset policy hidden state where episodes are done.

    Args:
      hidden: Current policy hidden state
      done: Done flags [batch_size]
      rnn_ppo_network: The RNN-PPO networks (for cell type info)

    Returns:
      Hidden state with zeros where done=True
    """
    # Expand done to match hidden state shape
    done_expanded = done[..., None]  # [batch, 1]

    if rnn_ppo_network.cell_type == "lstm":
        # LSTM has tuple state (carry, hidden)
        c, h = hidden
        c = jnp.where(done_expanded, 0.0, c)
        h = jnp.where(done_expanded, 0.0, h)
        return (c, h)
    else:
        # SimpleCell and GRU have single state
        return jnp.where(done_expanded, 0.0, hidden)


def actor_step_rnn(
    env: envs.Env,
    env_state: envs.State,
    policy: Callable,
    policy_hidden: HiddenState,
    key: PRNGKey,
    extra_fields: tuple = (),
    store_initial_hidden: bool = True,
    rnn_ppo_network: Optional[rnn_ppo_networks.RNNPPONetworks] = None,
) -> Tuple[envs.State, types.Transition, HiddenState]:
    """Collect data with recurrent policy.

    Args:
      env: Environment
      env_state: Current environment state
      policy: Recurrent policy function
      policy_hidden: Policy network hidden state
      key: Random key
      extra_fields: Extra fields to collect from env state
      store_initial_hidden: Whether to store initial hidden state in extras
      rnn_ppo_network: RNN-PPO networks for hidden state reset

    Returns:
      Tuple of (next_state, transition, new_policy_hidden)
    """
    actions, policy_extras, new_policy_hidden = policy(
        env_state.obs, policy_hidden, key
    )
    nstate = env.step(env_state, actions)
    state_extras = {x: nstate.info[x] for x in extra_fields}

    if store_initial_hidden:
        # Store the initial policy hidden state for this step (before the step was taken)
        policy_extras["initial_policy_hidden"] = policy_hidden

    transition = types.Transition(
        observation=env_state.obs,
        action=actions,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation=nstate.obs,
        extras={"policy_extras": policy_extras, "state_extras": state_extras},
    )

    # Reset policy hidden state where episodes are done
    if rnn_ppo_network is not None:
        new_policy_hidden = _reset_hidden_on_done(
            new_policy_hidden, nstate.done, rnn_ppo_network
        )

    return nstate, transition, new_policy_hidden


def generate_unroll_rnn(
    env: envs.Env,
    env_state: envs.State,
    policy: Callable,
    policy_hidden: HiddenState,
    key: PRNGKey,
    unroll_length: int,
    extra_fields: tuple = (),
    store_initial_hidden: bool = True,
    rnn_ppo_network: Optional[rnn_ppo_networks.RNNPPONetworks] = None,
) -> Tuple[envs.State, types.Transition, HiddenState]:
    """Collect trajectories with recurrent policy.

    Args:
      env: Environment
      env_state: Current environment state
      policy: Recurrent policy function
      policy_hidden: Policy network hidden state
      key: Random key
      unroll_length: Number of steps to unroll
      extra_fields: Extra fields to collect
      store_initial_hidden: Whether to store initial hidden state in extras
      rnn_ppo_network: RNN-PPO networks for hidden state reset

    Returns:
      Tuple of (final_state, transitions, final_policy_hidden)
    """

    def f(carry, unused_t):
        state, hidden, current_key = carry
        current_key, next_key = jax.random.split(current_key)
        nstate, transition, new_hidden = actor_step_rnn(
            env,
            state,
            policy,
            hidden,
            current_key,
            extra_fields=extra_fields,
            store_initial_hidden=store_initial_hidden,
            rnn_ppo_network=rnn_ppo_network,
        )
        return (nstate, new_hidden, next_key), transition

    (final_state, final_hidden, _), data = jax.lax.scan(
        f, (env_state, policy_hidden, key), (), length=unroll_length
    )
    return final_state, data, final_hidden


class RecurrentEvaluator:
    """Class to run evaluations with recurrent policies."""

    def __init__(
        self,
        eval_env: envs.Env,
        eval_policy_fn: Callable[[Params], Callable],
        rnn_ppo_network: rnn_ppo_networks.RNNPPONetworks,
        num_eval_envs: int,
        episode_length: int,
        action_repeat: int,
        key: PRNGKey,
    ):
        """Init."""
        self._key = key
        self._eval_walltime = 0.0
        self._rnn_ppo_network = rnn_ppo_network
        self._num_eval_envs = num_eval_envs

        eval_env = envs.training.EvalWrapper(eval_env)
        self._eval_state_to_donate = jax.jit(eval_env.reset)(
            jax.random.split(key, num_eval_envs)
        )

        def generate_eval_unroll(
            eval_env_state_donated: envs.State,
            policy_params: Params,
            key: PRNGKey,
        ) -> envs.State:
            reset_keys = jax.random.split(key, num_eval_envs)
            eval_first_state = eval_env.reset(reset_keys)

            # Initialize policy hidden state
            policy_hidden = rnn_ppo_network.policy_network.init_hidden(num_eval_envs)

            final_state, _, _ = generate_unroll_rnn(
                eval_env,
                eval_first_state,
                eval_policy_fn(policy_params),
                policy_hidden,
                key,
                unroll_length=episode_length // action_repeat,
                store_initial_hidden=False,
                rnn_ppo_network=rnn_ppo_network,
            )
            return final_state

        self._generate_eval_unroll = jax.jit(
            generate_eval_unroll, donate_argnums=(0,), keep_unused=True
        )
        self._steps_per_unroll = episode_length * num_eval_envs

    def run_evaluation(
        self,
        policy_params: Params,
        training_metrics: Metrics,
        aggregate_episodes: bool = True,
    ) -> Metrics:
        """Run one epoch of evaluation."""
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

        def _agg_fn(metric, fn, to_aggregate, to_normalize, episode_lengths):
            if not to_aggregate:
                return metric
            if to_normalize:
                return fn(metric / episode_lengths)
            return fn(metric)

        metrics = {}
        for fn in [np.mean, np.std]:
            suffix = "_std" if fn == np.std else ""
            for name, value in eval_metrics.episode_metrics.items():
                metrics[f"eval/episode_{name}{suffix}"] = _agg_fn(
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
        metrics = {
            "eval/walltime": self._eval_walltime,
            **training_metrics,
            **metrics,
        }

        return metrics


def _force_std_leaf(policy_params, std):
    """Overwrite the policy's action-noise std leaves with `std`.

    Used to pin a non-learned, scheduled action-noise std (annealed
    exploration). Handles both the 'scalar' ('std_param') and 'log'
    ('std_logparam') parameterizations; raises if the policy has neither
    (e.g. state-dependent std), where forcing a scheduled std is undefined.
    """
    matched = 0

    def f(path, leaf):
        nonlocal matched
        keys = {str(getattr(k, "key", k)) for k in path}
        if "std_param" in keys:
            matched += 1
            return jnp.full_like(leaf, std)
        if "std_logparam" in keys:
            matched += 1
            return jnp.full_like(leaf, jnp.log(std))
        return leaf

    new_params = jax.tree_util.tree_map_with_path(f, policy_params)
    if matched == 0:
        raise ValueError(
            "noise_std_schedule requires a policy with a non-state-dependent"
            " std leaf ('std_param' or 'std_logparam'); none found."
        )
    return new_params


def train(
    environment: envs.Env,
    num_timesteps: int,
    max_devices_per_host: Optional[int] = None,
    # high-level control flow
    wrap_env: bool = True,
    augment_pixels: bool = False,
    # environment wrapper
    num_envs: int = 1,
    episode_length: Optional[int] = None,
    action_repeat: int = 1,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    # ppo params
    learning_rate: float = 1e-4,
    entropy_cost=1e-4,  # float | Callable[[jnp.ndarray], jnp.ndarray]
    # Annealed action-noise exploration (Codol et al. 2024): when set, the policy's
    # non-state-dependent std is FORCED to noise_std_schedule(progress) after every
    # minibatch update instead of being learned (which collapses it). Works with
    # 'scalar' and 'log' std parameterizations; incompatible with state-dependent
    # std. Progress-gated proxy for the paper's return-gated sigma. None => off
    # (learned std).
    noise_std_schedule: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    discounting: float = 0.9,
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    num_resets_per_eval: int = 0,
    normalize_observations: bool = False,
    normalize_observations_std_eps: float = 0.0,
    normalize_observations_mode: str = "welford",
    reward_scaling: float = 1.0,
    clipping_epsilon: float = 0.3,
    clipping_epsilon_value: float | None = None,
    gae_lambda: float = 0.95,
    max_grad_norm: Optional[float] = None,
    normalize_advantage: bool = True,
    vf_loss_coefficient: float = 0.5,
    activity_cost: float = 0.0,
    activity_derivative_cost: float = 0.0,
    bootstrap_on_timeout: bool = False,
    desired_kl: float = 0.01,
    learning_rate_schedule: Optional[Union[str, ppo_optimizer.LRSchedule]] = None,
    network_factory: types.NetworkFactory[
        rnn_ppo_networks.RNNPPONetworks
    ] = rnn_ppo_networks.make_rnn_ppo_networks,
    seed: int = 0,
    use_pmap_on_reset: bool = True,
    # eval
    num_evals: int = 1,
    eval_env: Optional[envs.Env] = None,
    num_eval_envs: int = 128,
    deterministic_eval: bool = False,
    # training metrics
    log_training_metrics: bool = False,
    training_metrics_steps: Optional[int] = None,
    # callbacks
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    # checkpointing
    save_checkpoint_path: Optional[str] = None,
    restore_checkpoint_path: Optional[str] = None,
    restore_params: Optional[Any] = None,
    restore_value_fn: bool = True,
    run_evals: bool = True,
):
    """RNN-PPO training.

    Args:
      environment: the environment to train
      num_timesteps: the total number of environment steps to use during training
      max_devices_per_host: maximum number of chips to use per host process
      wrap_env: If True, wrap the environment for training.
      augment_pixels: whether to add image augmentation to pixel inputs
      num_envs: the number of parallel environments to use for rollouts
      episode_length: the length of an environment episode
      action_repeat: the number of timesteps to repeat an action
      wrap_env_fn: a custom function that wraps the environment for training
      randomization_fn: a user-defined callback function for domain randomization
      learning_rate: learning rate for ppo loss
      entropy_cost: entropy reward for ppo loss
      discounting: discounting rate
      unroll_length: the number of timesteps to unroll in each environment
      batch_size: the batch size for each minibatch SGD step
      num_minibatches: the number of times to run the SGD step
      num_updates_per_batch: the number of times to run the gradient update
      num_resets_per_eval: the number of environment resets between each eval
      normalize_observations: whether to normalize observations
      normalize_observations_std_eps: small value for obs normalization stability
      normalize_observations_mode: method to use for running statistics
      reward_scaling: float scaling for reward
      clipping_epsilon: clipping epsilon for PPO loss
      clipping_epsilon_value: Value function loss clipping epsilon
      gae_lambda: General advantage estimation lambda
      max_grad_norm: gradient clipping norm value
      normalize_advantage: whether to normalize advantage estimate
      vf_loss_coefficient: Coefficient for value function loss
      activity_cost: L2 penalty on policy RNN hidden activity (Codol et al. 2024
        use 0.01); 0.0 disables.
      activity_derivative_cost: L2 penalty on the temporal derivative of the
        policy RNN hidden activity (Codol et al. 2024 use 0.1); 0.0 disables.
      bootstrap_on_timeout: if True, bootstrap value on time_out steps
      desired_kl: Desired KL divergence for adaptive KL learning rate
      learning_rate_schedule: Learning rate schedule for the optimizer
      network_factory: function that generates RNN networks
      seed: random seed
      use_pmap_on_reset: if True, use pmap for env.reset across devices
      num_evals: the number of evals to run during training
      eval_env: an optional environment for eval only
      num_eval_envs: the number of envs to use for evaluation
      deterministic_eval: whether to run the eval with a deterministic policy
      log_training_metrics: whether to log training metrics
      training_metrics_steps: steps between logging training metrics
      progress_fn: a user-defined callback function for reporting metrics
      policy_params_fn: a user-defined callback function for saving checkpoints
      save_checkpoint_path: the path used to save checkpoints
      restore_checkpoint_path: the path used to restore previous model params
      restore_params: raw network parameters to restore
      restore_value_fn: whether to restore the value function
      run_evals: if True, run evaluator during training

    Returns:
      Tuple of (make_policy function, network params, metrics)
    """
    assert batch_size * num_minibatches % num_envs == 0

    xt = time.time()

    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    logging.info(
        "Device count: %d, process count: %d (id %d), local device count: %d, "
        "devices to be used count: %d",
        jax.device_count(),
        process_count,
        process_id,
        local_device_count,
        local_devices_to_use,
    )
    device_count = local_devices_to_use * process_count

    env_step_per_training_step = (
        batch_size * unroll_length * num_minibatches * action_repeat
    )
    num_evals_after_init = max(num_evals - 1, 1)
    num_training_steps_per_epoch = np.ceil(
        num_timesteps
        / (
            num_evals_after_init
            * env_step_per_training_step
            * max(num_resets_per_eval, 1)
        )
    ).astype(int)

    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_env, eval_key = jax.random.split(local_key, 3)
    key_policy, key_value = jax.random.split(global_key)
    del global_key

    assert num_envs % device_count == 0

    env = _maybe_wrap_env(
        environment,
        wrap_env,
        num_envs,
        episode_length,
        action_repeat,
        device_count,
        key_env,
        wrap_env_fn,
        randomization_fn,
    )

    def reset_fn_donated_env_state(env_state_donated, key_envs):
        return env.reset(key_envs)

    key_envs = jax.random.split(key_env, num_envs // process_count)
    key_envs = jnp.reshape(key_envs, (local_devices_to_use, -1) + key_envs.shape[1:])
    if local_devices_to_use > 1 or use_pmap_on_reset:
        reset_fn_ = jax.pmap(env.reset, axis_name=_PMAP_AXIS_NAME)
        env_state = reset_fn_(key_envs)
        reset_fn = jax.pmap(
            reset_fn_donated_env_state,
            axis_name=_PMAP_AXIS_NAME,
            donate_argnums=(0,),
        )
    else:
        reset_fn_ = jax.jit(jax.vmap(env.reset))
        env_state = reset_fn_(key_envs)
        reset_fn = jax.jit(
            reset_fn_donated_env_state, donate_argnums=(0,), keep_unused=True
        )

    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)

    normalize = lambda x, y: x
    if normalize_observations:
        normalize = running_statistics.normalize
    rnn_ppo_network = network_factory(
        obs_shape, env.action_size, preprocess_observations_fn=normalize
    )
    make_policy = rnn_ppo_networks.make_inference_fn(
        rnn_ppo_network,
        compute_value=bootstrap_on_timeout or clipping_epsilon_value is not None,
    )

    # Optimizer
    base_optimizer = optax.adam(learning_rate=learning_rate)
    lr_schedule = learning_rate_schedule or ppo_optimizer.LRSchedule.NONE
    lr_schedule = ppo_optimizer.LRSchedule(lr_schedule)
    lr_is_adaptive_kl = lr_schedule == ppo_optimizer.LRSchedule.ADAPTIVE_KL
    if lr_is_adaptive_kl:
        base_optimizer = optax.inject_hyperparams(optax.adam)(
            learning_rate=learning_rate
        )
    if max_grad_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            base_optimizer,
        )
    else:
        optimizer = base_optimizer

    loss_fn = functools.partial(
        rnn_ppo_losses.compute_rnn_ppo_loss,
        rnn_ppo_network=rnn_ppo_network,
        entropy_cost=entropy_cost,
        discounting=discounting,
        reward_scaling=reward_scaling,
        gae_lambda=gae_lambda,
        clipping_epsilon=clipping_epsilon,
        normalize_advantage=normalize_advantage,
        vf_coefficient=vf_loss_coefficient,
        clipping_epsilon_value=clipping_epsilon_value,
        activity_cost=activity_cost,
        activity_derivative_cost=activity_derivative_cost,
    )

    loss_and_pgrad_fn = gradients.loss_and_pgrad(
        loss_fn, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
    )

    steps_between_logging = training_metrics_steps or env_step_per_training_step
    metrics_aggregator = metric_logger.EpisodeMetricsLogger(
        steps_between_logging=steps_between_logging,
        progress_fn=progress_fn,
    )

    # Number of envs per device
    envs_per_device = num_envs // device_count

    def minibatch_step(
        carry,
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
        progress: jnp.ndarray,
    ):
        optimizer_state, params, key = carry
        key, key_loss = jax.random.split(key)
        (_, metrics), grads = loss_and_pgrad_fn(
            params, normalizer_params, data, key_loss, progress
        )

        if lr_is_adaptive_kl:
            kl_mean = metrics["kl_mean"]
            kl_mean = jax.lax.pmean(kl_mean, axis_name=_PMAP_AXIS_NAME)
            optimizer_state, lr = ppo_optimizer.adaptive_kl_learning_rate(
                optimizer_state, kl_mean, desired_kl
            )
        else:
            lr = jnp.array(learning_rate)
        metrics["learning_rate"] = lr

        params_update, optimizer_state = optimizer.update(grads, optimizer_state)
        params = optax.apply_updates(params, params_update)

        # Annealed action-noise exploration: pin the policy std to the schedule
        # (progress-gated) instead of the collapsing learned value.
        if noise_std_schedule is not None:
            params = params.replace(
                policy=_force_std_leaf(params.policy, noise_std_schedule(progress))
            )

        return (optimizer_state, params, key), metrics

    def sgd_step(
        carry,
        unused_t,
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
        progress: jnp.ndarray,
    ):
        optimizer_state, params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        if augment_pixels:
            key, key_rt = jax.random.split(key)
            r_translate = functools.partial(_random_translate_pixels, key=key_rt)
            data = types.Transition(
                observation=r_translate(data.observation),
                action=data.action,
                reward=data.reward,
                discount=data.discount,
                next_observation=r_translate(data.next_observation),
                extras=data.extras,
            )

        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x

        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(
                minibatch_step,
                normalizer_params=normalizer_params,
                progress=progress,
            ),
            (optimizer_state, params, key_grad),
            shuffled_data,
            length=num_minibatches,
        )

        return (optimizer_state, params, key), metrics

    def training_step(
        carry: Tuple[TrainingState, envs.State, HiddenState, PRNGKey],
        unused_t,
    ) -> Tuple[Tuple[TrainingState, envs.State, HiddenState, PRNGKey], Metrics]:
        training_state, state, policy_hidden, key = carry
        key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

        policy = make_policy(
            (
                training_state.normalizer_params,
                training_state.params.policy,
                training_state.params.value,
            )
        )

        def f(carry, unused_t):
            current_state, current_hidden, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            initial_hidden = current_hidden
            extra_fields = ["truncation", "episode_metrics", "episode_done"]
            if bootstrap_on_timeout:
                extra_fields.append("time_out")
            next_state, data, new_hidden = generate_unroll_rnn(
                env,
                current_state,
                policy,
                current_hidden,
                current_key,
                unroll_length,
                extra_fields=tuple(extra_fields),
                store_initial_hidden=False,
                rnn_ppo_network=rnn_ppo_network,
            )
            return (next_state, new_hidden, next_key), (data, initial_hidden)

        (state, policy_hidden, _), (data, initial_policy_hidden) = jax.lax.scan(
            f,
            (state, policy_hidden, key_generate_unroll),
            (),
            length=batch_size * num_minibatches // num_envs,
        )
        # Have leading dimensions (batch_size * num_minibatches, unroll_length)
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
        )
        initial_policy_hidden = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]),
            initial_policy_hidden,
        )
        assert data.discount.shape[1:] == (unroll_length,)

        if bootstrap_on_timeout:
            time_out = data.extras["state_extras"]["time_out"]
            value = data.extras["policy_extras"]["value"]
            data = types.Transition(
                observation=data.observation,
                action=data.action,
                reward=data.reward + discounting * time_out * value,
                discount=data.discount,
                next_observation=data.next_observation,
                extras=data.extras,
            )

        policy_extras = dict(data.extras["policy_extras"])
        policy_extras["initial_policy_hidden"] = initial_policy_hidden
        policy_extras.setdefault("initial_value_hidden", None)
        data = types.Transition(
            observation=data.observation,
            action=data.action,
            reward=data.reward,
            discount=data.discount,
            next_observation=data.next_observation,
            extras={
                "policy_extras": policy_extras,
                "state_extras": data.extras["state_extras"],
            },
        )

        normalizer_params = training_state.normalizer_params
        if not lr_is_adaptive_kl:
            normalizer_params = running_statistics.update(
                normalizer_params,
                _remove_pixels(data.observation),
                pmap_axis_name=_PMAP_AXIS_NAME,
            )

        # training_state.env_steps is a UInt64(hi, lo) flax dataclass — recombine
        # via float arithmetic for the entropy-schedule progress fraction.
        env_steps_f32 = (
            jnp.asarray(training_state.env_steps.hi, jnp.float32)
            * jnp.float32(2.0**32)
            + jnp.asarray(training_state.env_steps.lo, jnp.float32)
        )
        progress = jnp.minimum(
            env_steps_f32 / jnp.float32(max(num_timesteps, 1)),
            jnp.float32(1.0),
        )
        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(
                sgd_step,
                data=data,
                normalizer_params=normalizer_params,
                progress=progress,
            ),
            (training_state.optimizer_state, training_state.params, key_sgd),
            (),
            length=num_updates_per_batch,
        )

        if lr_is_adaptive_kl:
            normalizer_params = running_statistics.update(
                normalizer_params,
                _remove_pixels(data.observation),
                pmap_axis_name=_PMAP_AXIS_NAME,
            )

        new_training_state = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + env_step_per_training_step,
        )

        if log_training_metrics:
            jax.debug.callback(
                metrics_aggregator.update_episode_metrics,
                data.extras["state_extras"]["episode_metrics"],
                data.extras["state_extras"]["episode_done"],
                metrics,
            )

        return (new_training_state, state, policy_hidden, new_key), metrics

    def training_epoch(
        training_state: TrainingState,
        state: envs.State,
        policy_hidden: HiddenState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, HiddenState, Metrics]:
        (training_state, state, policy_hidden, _), loss_metrics = jax.lax.scan(
            training_step,
            (training_state, state, policy_hidden, key),
            (),
            length=num_training_steps_per_epoch,
        )
        loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
        return training_state, state, policy_hidden, loss_metrics

    training_epoch = jax.pmap(
        training_epoch,
        axis_name=_PMAP_AXIS_NAME,
        donate_argnums=(
            0,
            1,
            2,
        ),
    )

    training_walltime = 0

    def training_epoch_with_timing(
        training_state: TrainingState,
        env_state: envs.State,
        policy_hidden: HiddenState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, HiddenState, Metrics]:
        nonlocal training_walltime
        t = time.time()
        training_state, env_state, policy_hidden = _strip_weak_type(
            (training_state, env_state, policy_hidden)
        )
        result = training_epoch(training_state, env_state, policy_hidden, key)
        training_state, env_state, policy_hidden, metrics = _strip_weak_type(result)

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (
            num_training_steps_per_epoch
            * env_step_per_training_step
            * max(num_resets_per_eval, 1)
        ) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        return training_state, env_state, policy_hidden, metrics

    # Initialize model params and training state
    init_params = rnn_ppo_losses.RNNPPONetworkParams(
        policy=rnn_ppo_network.policy_network.init(key_policy),
        value=rnn_ppo_network.value_network.init(key_value),
    )

    obs_shape = jax.tree_util.tree_map(
        lambda x: specs.Array(x.shape[-1:], jnp.dtype("float32")), env_state.obs
    )
    training_state = TrainingState(
        optimizer_state=optimizer.init(init_params),
        params=init_params,
        normalizer_params=running_statistics.init_state(
            _remove_pixels(obs_shape),
            std_eps=normalize_observations_std_eps,
            mode=normalize_observations_mode,
        ),
        env_steps=types.UInt64(hi=0, lo=0),
    )

    if restore_checkpoint_path is not None:
        params = checkpoint.load(restore_checkpoint_path)
        value_params = params[2] if restore_value_fn else init_params.value
        training_state = training_state.replace(
            normalizer_params=params[0],
            params=training_state.params.replace(policy=params[1], value=value_params),
        )

    if restore_params is not None:
        logging.info("Restoring TrainingState from `restore_params`.")
        value_params = restore_params[2] if restore_value_fn else init_params.value
        training_state = training_state.replace(
            normalizer_params=restore_params[0],
            params=training_state.params.replace(
                policy=restore_params[1], value=value_params
            ),
        )

    if num_timesteps == 0:
        return (
            make_policy,
            (
                training_state.normalizer_params,
                training_state.params.policy,
                training_state.params.value,
            ),
            {},
        )

    # jax.device_put_replicated was removed in jax 0.10; replicate across
    # devices via an explicit sharding (mirrors brax upstream ppo/train.py).
    devices = jax.local_devices()[:local_devices_to_use]
    mesh = jax.sharding.Mesh(np.array(devices), ('_device_put_sharded',))
    sharding = jax.NamedSharding(mesh, jax.P('_device_put_sharded'))

    def _replicate(x):
        if isinstance(x, jax.Array):
            return jax.device_put(jnp.stack([x] * len(devices)), sharding)
        return jax.device_put(np.stack([x] * len(devices)), sharding)

    training_state = jax.tree_util.tree_map(_replicate, training_state)

    # Initialize policy hidden state for training
    # Shape: [num_devices, envs_per_device, hidden_size]
    def init_policy_hidden_for_training():
        hidden = rnn_ppo_network.policy_network.init_hidden(envs_per_device)
        # Replicate across devices (reuses the sharding defined above).
        return jax.tree_util.tree_map(_replicate, hidden)

    policy_hidden = init_policy_hidden_for_training()

    eval_env = _maybe_wrap_env(
        eval_env or environment,
        wrap_env,
        num_eval_envs,
        episode_length,
        action_repeat,
        device_count=1,
        key_env=eval_key,
        wrap_env_fn=wrap_env_fn,
        randomization_fn=randomization_fn,
    )
    evaluator = RecurrentEvaluator(
        eval_env,
        functools.partial(make_policy, deterministic=deterministic_eval),
        rnn_ppo_network,
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )

    training_metrics = {}
    current_step = 0

    # Run initial eval
    metrics = {}
    if process_id == 0 and num_evals > 1 and run_evals:
        metrics = evaluator.run_evaluation(
            _unpmap(
                (
                    training_state.normalizer_params,
                    training_state.params.policy,
                    training_state.params.value,
                )
            ),
            training_metrics={},
        )
        logging.info(metrics)
        progress_fn(0, metrics)

    # Run initial policy_params_fn
    params = _unpmap(
        (
            training_state.normalizer_params,
            training_state.params.policy,
            training_state.params.value,
        )
    )
    policy_params_fn(current_step, make_policy, params)

    for it in range(num_evals_after_init):
        logging.info("starting iteration %s %s", it, time.time() - xt)

        for _ in range(max(num_resets_per_eval, 1)):
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            (training_state, env_state, policy_hidden, training_metrics) = (
                training_epoch_with_timing(
                    training_state, env_state, policy_hidden, epoch_keys
                )
            )
            current_step = int(_unpmap(training_state.env_steps))

            key_envs = jax.vmap(
                lambda x, s: jax.random.split(x[0], s), in_axes=(0, None)
            )(key_envs, key_envs.shape[1])
            if num_resets_per_eval > 0:
                env_state = reset_fn(env_state, key_envs)
                # Reset policy hidden state on env reset
                policy_hidden = init_policy_hidden_for_training()

        if process_id != 0:
            continue

        params = _unpmap(
            (
                training_state.normalizer_params,
                training_state.params.policy,
                training_state.params.value,
            )
        )

        policy_params_fn(current_step, make_policy, params)

        if save_checkpoint_path is not None:
            ckpt_config = checkpoint.network_config(
                observation_size=obs_shape,
                action_size=env.action_size,
                normalize_observations=normalize_observations,
                network_factory=network_factory,
            )
            checkpoint.save(save_checkpoint_path, current_step, params, ckpt_config)

        if num_evals > 0:
            metrics = training_metrics
            if run_evals:
                metrics = evaluator.run_evaluation(
                    params,
                    training_metrics,
                )
            logging.info(metrics)
            progress_fn(current_step, metrics)

    total_steps = current_step
    if not total_steps >= num_timesteps:
        raise AssertionError(
            f"Total steps {total_steps} is less than `num_timesteps`="
            f" {num_timesteps}."
        )

    pmap.assert_is_replicated(training_state)
    params = _unpmap(
        (
            training_state.normalizer_params,
            training_state.params.policy,
            training_state.params.value,
        )
    )
    logging.info("total steps: %s", total_steps)
    pmap.synchronize_hosts()
    return (make_policy, params, metrics)
