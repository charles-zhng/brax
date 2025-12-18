# Copyright 2025 DeepMind Technologies Limited
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
# ==============================================================================
"""Train a Recurrent PPO agent using JAX on the specified environment."""

import datetime
import functools
import json
import os
import time
import warnings

from absl import app
from absl import flags
from absl import logging
from brax.training.agents.recurrent_ppo import networks as recurrent_ppo_networks
from brax.training.agents.recurrent_ppo import train as recurrent_ppo
from etils import epath
import jax
import jax.numpy as jp
import imageio
import matplotlib.pyplot as plt
from ml_collections import config_dict
import mujoco
import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import dm_control_suite_params
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params
import tensorboardX
import wandb


xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)

# Suppress warnings

# Suppress RuntimeWarnings from JAX
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# Suppress DeprecationWarnings from JAX
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# Suppress UserWarnings from absl (used by JAX and TensorFlow)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")


_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "PendulumSwingup",
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
)
_IMPL = flags.DEFINE_enum("impl", "jax", ["jax", "warp"], "MJX implementation")
_LOAD_CHECKPOINT_PATH = flags.DEFINE_string(
    "load_checkpoint_path", None, "Path to load checkpoint from"
)
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name")
_PLAY_ONLY = flags.DEFINE_boolean(
    "play_only", False, "If true, only play with the model and do not train"
)
_USE_WANDB = flags.DEFINE_boolean(
    "use_wandb",
    False,
    "Use Weights & Biases for logging (ignored in play-only mode)",
)
_USE_TB = flags.DEFINE_boolean(
    "use_tb", False, "Use TensorBoard for logging (ignored in play-only mode)"
)
_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "domain_randomization", False, "Use domain randomization"
)
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_NUM_TIMESTEPS = flags.DEFINE_integer("num_timesteps", 1_000_000, "Number of timesteps")
_NUM_VIDEOS = flags.DEFINE_integer(
    "num_videos", 1, "Number of videos to record after training."
)
_NUM_EVALS = flags.DEFINE_integer("num_evals", 5, "Number of evaluations")
_REWARD_SCALING = flags.DEFINE_float("reward_scaling", 10.0, "Reward scaling")
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", 1000, "Episode length")
_NORMALIZE_OBSERVATIONS = flags.DEFINE_boolean(
    "normalize_observations", True, "Normalize observations"
)
_ACTION_REPEAT = flags.DEFINE_integer("action_repeat", 1, "Action repeat")
_UNROLL_LENGTH = flags.DEFINE_integer("unroll_length", 16, "Unroll length")
_BPTT_LENGTH = flags.DEFINE_integer(
    "bptt_length",
    None,
    "Backpropagation through time length (must divide unroll_length). Defaults to unroll_length if not specified.",
)
_NUM_MINIBATCHES = flags.DEFINE_integer("num_minibatches", 8, "Number of minibatches")
_NUM_UPDATES_PER_BATCH = flags.DEFINE_integer(
    "num_updates_per_batch", 2, "Number of updates per batch"
)
_DISCOUNTING = flags.DEFINE_float("discounting", 0.97, "Discounting")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 3e-4, "Learning rate")
_ENTROPY_COST = flags.DEFINE_float("entropy_cost", 1e-2, "Entropy cost")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 512, "Number of environments")
_NUM_EVAL_ENVS = flags.DEFINE_integer(
    "num_eval_envs", 128, "Number of evaluation environments"
)
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 256, "Batch size")
_MAX_GRAD_NORM = flags.DEFINE_float("max_grad_norm", 1.0, "Max grad norm")
_CLIPPING_EPSILON = flags.DEFINE_float(
    "clipping_epsilon", 0.3, "Clipping epsilon for PPO"
)
_GAE_LAMBDA = flags.DEFINE_float("gae_lambda", 0.95, "GAE lambda")

# Recurrent network configuration
_CORE_TYPE = flags.DEFINE_enum(
    "core_type", "rnn", ["gru", "lstm", "rnn"], "Type of recurrent cell"
)
_HIDDEN_SIZE = flags.DEFINE_integer("hidden_size", 64, "Size of recurrent hidden state")
_POLICY_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "policy_hidden_layer_sizes",
    [64],
    "Policy hidden layer sizes (before recurrent core)",
)
_VALUE_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "value_hidden_layer_sizes",
    [64],
    "Value hidden layer sizes (after recurrent core)",
)
_POLICY_OBS_KEY = flags.DEFINE_string("policy_obs_key", "state", "Policy obs key")

_DETERMINISTIC_EVAL = flags.DEFINE_boolean(
    "deterministic_eval", True, "Use deterministic policy during evaluation"
)
_RUN_EVALS = flags.DEFINE_boolean(
    "run_evals",
    True,
    "Run evaluation rollouts between policy updates.",
)
_LOG_TRAINING_METRICS = flags.DEFINE_boolean(
    "log_training_metrics",
    False,
    "Whether to log training metrics and callback to progress_fn. Significantly"
    " slows down training if too frequent.",
)
_TRAINING_METRICS_STEPS = flags.DEFINE_integer(
    "training_metrics_steps",
    1_000_000,
    "Number of steps between logging training metrics. Increase if training"
    " experiences slowdown.",
)


def get_rl_config(env_name: str) -> config_dict.ConfigDict:
    """Get default RL config for the environment and modify for recurrent PPO."""
    if env_name in mujoco_playground.manipulation._envs:
        return manipulation_params.brax_ppo_config(env_name, _IMPL.value)
    elif env_name in mujoco_playground.locomotion._envs:
        return locomotion_params.brax_ppo_config(env_name, _IMPL.value)
    elif env_name in mujoco_playground.dm_control_suite._envs:
        return dm_control_suite_params.brax_ppo_config(env_name, _IMPL.value)

    raise ValueError(f"Env {env_name} not found in {registry.ALL_ENVS}.")


def main(argv):
    """Run training and evaluation for the specified environment."""

    del argv

    # Load environment configuration
    env_cfg = registry.get_default_config(_ENV_NAME.value)
    env_cfg["impl"] = _IMPL.value

    ppo_params = get_rl_config(_ENV_NAME.value)

    # Override with command line flags
    if _NUM_TIMESTEPS.present:
        ppo_params.num_timesteps = _NUM_TIMESTEPS.value
    if _PLAY_ONLY.present:
        ppo_params.num_timesteps = 0
    if _NUM_EVALS.present:
        ppo_params.num_evals = _NUM_EVALS.value
    if _REWARD_SCALING.present:
        ppo_params.reward_scaling = _REWARD_SCALING.value
    if _EPISODE_LENGTH.present:
        ppo_params.episode_length = _EPISODE_LENGTH.value
    if _NORMALIZE_OBSERVATIONS.present:
        ppo_params.normalize_observations = _NORMALIZE_OBSERVATIONS.value
    if _ACTION_REPEAT.present:
        ppo_params.action_repeat = _ACTION_REPEAT.value
    if _UNROLL_LENGTH.present:
        ppo_params.unroll_length = _UNROLL_LENGTH.value
    if _NUM_MINIBATCHES.present:
        ppo_params.num_minibatches = _NUM_MINIBATCHES.value
    if _NUM_UPDATES_PER_BATCH.present:
        ppo_params.num_updates_per_batch = _NUM_UPDATES_PER_BATCH.value
    if _DISCOUNTING.present:
        ppo_params.discounting = _DISCOUNTING.value
    if _LEARNING_RATE.present:
        ppo_params.learning_rate = _LEARNING_RATE.value
    if _ENTROPY_COST.present:
        ppo_params.entropy_cost = _ENTROPY_COST.value
    if _NUM_ENVS.present:
        ppo_params.num_envs = _NUM_ENVS.value
    if _NUM_EVAL_ENVS.present:
        ppo_params.num_eval_envs = _NUM_EVAL_ENVS.value
    if _BATCH_SIZE.present:
        ppo_params.batch_size = _BATCH_SIZE.value
    if _MAX_GRAD_NORM.present:
        ppo_params.max_grad_norm = _MAX_GRAD_NORM.value
    if _CLIPPING_EPSILON.present:
        ppo_params.clipping_epsilon = _CLIPPING_EPSILON.value

    env = registry.load(_ENV_NAME.value, config=env_cfg)

    if _RUN_EVALS.present:
        ppo_params.run_evals = _RUN_EVALS.value
    if _LOG_TRAINING_METRICS.present:
        ppo_params.log_training_metrics = _LOG_TRAINING_METRICS.value
    if _TRAINING_METRICS_STEPS.present:
        ppo_params.training_metrics_steps = _TRAINING_METRICS_STEPS.value

    print(f"Environment Config:\n{env_cfg}")
    print(f"Base PPO Training Parameters:\n{ppo_params}")

    # Generate unique experiment name
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    exp_name = f"{_ENV_NAME.value}-recurrent-{_CORE_TYPE.value}-{timestamp}"
    if _SUFFIX.value is not None:
        exp_name += f"-{_SUFFIX.value}"
    print(f"Experiment name: {exp_name}")

    # Set up logging directory
    logdir = epath.Path("logs").resolve() / exp_name
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"Logs are being stored in: {logdir}")

    # Initialize Weights & Biases if required
    if _USE_WANDB.value and not _PLAY_ONLY.value:
        wandb.init(project="mjxrl-recurrent", name=exp_name)
        wandb.config.update(env_cfg.to_dict())
        wandb.config.update(
            {
                "env_name": _ENV_NAME.value,
                "core_type": _CORE_TYPE.value,
                "hidden_size": _HIDDEN_SIZE.value,
            }
        )

    # Initialize TensorBoard if required
    if _USE_TB.value and not _PLAY_ONLY.value:
        writer = tensorboardX.SummaryWriter(logdir)

    # Handle checkpoint loading
    if _LOAD_CHECKPOINT_PATH.value is not None:
        # Convert to absolute path
        ckpt_path = epath.Path(_LOAD_CHECKPOINT_PATH.value).resolve()
        if ckpt_path.is_dir():
            latest_ckpts = list(ckpt_path.glob("*"))
            latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
            latest_ckpts.sort(key=lambda x: int(x.name))
            latest_ckpt = latest_ckpts[-1]
            restore_checkpoint_path = latest_ckpt
            print(f"Restoring from: {restore_checkpoint_path}")
        else:
            restore_checkpoint_path = ckpt_path
            print(f"Restoring from checkpoint: {restore_checkpoint_path}")
    else:
        print("No checkpoint path provided, not restoring from checkpoint")
        restore_checkpoint_path = None

    # Set up checkpoint directory
    ckpt_path = logdir / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint path: {ckpt_path}")

    # Save environment configuration
    with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
        json.dump(env_cfg.to_dict(), fp, indent=4)

    # Create recurrent network factory
    network_factory = functools.partial(
        recurrent_ppo_networks.make_recurrent_ppo_networks,
        core_type=_CORE_TYPE.value,
        hidden_size=_HIDDEN_SIZE.value,
        policy_hidden_layer_sizes=tuple(map(int, _POLICY_HIDDEN_LAYER_SIZES.value)),
        value_hidden_layer_sizes=tuple(map(int, _VALUE_HIDDEN_LAYER_SIZES.value)),
        policy_obs_key=_POLICY_OBS_KEY.value,
    )

    # Build training parameters dict
    unroll_length = ppo_params.get("unroll_length", _UNROLL_LENGTH.value)
    bptt_length = (
        _BPTT_LENGTH.value
        if _BPTT_LENGTH.present and _BPTT_LENGTH.value is not None
        else unroll_length
    )
    training_params = dict(
        num_timesteps=ppo_params.get("num_timesteps", _NUM_TIMESTEPS.value),
        num_evals=ppo_params.get("num_evals", _NUM_EVALS.value),
        reward_scaling=ppo_params.get("reward_scaling", _REWARD_SCALING.value),
        episode_length=ppo_params.get("episode_length", _EPISODE_LENGTH.value),
        normalize_observations=ppo_params.get(
            "normalize_observations", _NORMALIZE_OBSERVATIONS.value
        ),
        action_repeat=ppo_params.get("action_repeat", _ACTION_REPEAT.value),
        unroll_length=unroll_length,
        bptt_length=bptt_length,
        num_minibatches=ppo_params.get("num_minibatches", _NUM_MINIBATCHES.value),
        num_updates_per_batch=ppo_params.get(
            "num_updates_per_batch", _NUM_UPDATES_PER_BATCH.value
        ),
        discounting=ppo_params.get("discounting", _DISCOUNTING.value),
        learning_rate=ppo_params.get("learning_rate", _LEARNING_RATE.value),
        entropy_cost=ppo_params.get("entropy_cost", _ENTROPY_COST.value),
        num_envs=ppo_params.get("num_envs", _NUM_ENVS.value),
        num_eval_envs=ppo_params.get("num_eval_envs", _NUM_EVAL_ENVS.value),
        batch_size=ppo_params.get("batch_size", _BATCH_SIZE.value),
        max_grad_norm=ppo_params.get("max_grad_norm", _MAX_GRAD_NORM.value),
        clipping_epsilon=ppo_params.get("clipping_epsilon", _CLIPPING_EPSILON.value),
        gae_lambda=_GAE_LAMBDA.value,
        deterministic_eval=_DETERMINISTIC_EVAL.value,
        run_evals=ppo_params.get("run_evals", _RUN_EVALS.value),
        log_training_metrics=ppo_params.get(
            "log_training_metrics", _LOG_TRAINING_METRICS.value
        ),
    )

    if _TRAINING_METRICS_STEPS.present:
        training_params["training_metrics_steps"] = _TRAINING_METRICS_STEPS.value

    randomization_fn = None
    if _DOMAIN_RANDOMIZATION.value:
        randomization_fn = registry.get_domain_randomizer(_ENV_NAME.value)

    print(f"Recurrent PPO Training Parameters:")
    for k, v in training_params.items():
        print(f"  {k}: {v}")
    print(f"  core_type: {_CORE_TYPE.value}")
    print(f"  hidden_size: {_HIDDEN_SIZE.value}")

    train_fn = functools.partial(
        recurrent_ppo,
        **training_params,
        network_factory=network_factory,
        seed=_SEED.value,
        restore_checkpoint_path=restore_checkpoint_path,
        save_checkpoint_path=ckpt_path,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        randomization_fn=randomization_fn,
    )

    times = [time.monotonic()]
    reward_history = {"steps": [], "rewards": []}
    v_loss_history = {"steps": [], "v_losses": []}

    # Progress function for logging
    def progress(num_steps, metrics):
        times.append(time.monotonic())
        # Track reward for plotting
        if _RUN_EVALS.value:
            if "eval/episode_reward" in metrics:
                reward_history["steps"].append(num_steps)
                reward_history["rewards"].append(metrics["eval/episode_reward"])

            if "training/v_loss" in metrics:
                v_loss_history["steps"].append(num_steps)
                v_loss_history["v_losses"].append(metrics["training/v_loss"])

        # Log to Weights & Biases
        if _USE_WANDB.value and not _PLAY_ONLY.value:
            wandb.log(metrics, step=num_steps)

        # Log to TensorBoard
        if _USE_TB.value and not _PLAY_ONLY.value:
            for key, value in metrics.items():
                writer.add_scalar(key, value, num_steps)
            writer.flush()
        if _RUN_EVALS.value:
            print(f"{num_steps}: reward={metrics['eval/episode_reward']:.3f}")
        if _LOG_TRAINING_METRICS.value:
            if "episode/sum_reward" in metrics:
                print(
                    f"{num_steps}: mean episode"
                    f" reward={metrics['episode/sum_reward']:.3f}"
                )

    # Load evaluation environment.
    eval_env = registry.load(_ENV_NAME.value, config=env_cfg)

    # Train or load the model
    make_inference_fn, params, _ = train_fn(  # pylint: disable=no-value-for-parameter
        environment=env,
        progress_fn=progress,
        eval_env=eval_env,
    )

    print("Done training.")
    if len(times) > 1:
        print(f"Time to JIT compile: {times[1] - times[0]}")
        print(f"Time to train: {times[-1] - times[1]}")

    # Save reward curve plot
    if reward_history["steps"]:
        plt.figure(figsize=(10, 6))
        plt.plot(
            reward_history["steps"], reward_history["rewards"], marker="o", markersize=3
        )
        plt.xlabel("Timesteps")
        plt.ylabel("Episode Reward")
        plt.title(
            f"Reward Curve - {_ENV_NAME.value} (Recurrent PPO - {_CORE_TYPE.value})"
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        reward_plot_path = str(ckpt_path / "reward_curve.png")
        plt.savefig(reward_plot_path, dpi=150)
        plt.close()
        print(f"Reward curve saved to: {reward_plot_path}")

    if v_loss_history["steps"]:
        plt.figure(figsize=(10, 6))
        plt.plot(
            v_loss_history["steps"],
            v_loss_history["v_losses"],
            marker="o",
            markersize=3,
        )
        plt.xlabel("Timesteps")
        plt.ylabel("V Loss")
        plt.title(
            f"V Loss Curve - {_ENV_NAME.value} (Recurrent PPO - {_CORE_TYPE.value})"
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        v_loss_plot_path = str(ckpt_path / "v_loss_curve.png")
        plt.savefig(v_loss_plot_path, dpi=150)
        plt.close()
        print(f"V loss curve saved to: {v_loss_plot_path}")

    print("Starting inference...")

    # Rebuild network for inference to get initial state function
    ppo_network = network_factory(
        observation_size=env.observation_size,
        action_size=env.action_size,
    )

    # Create inference function.
    inference_fn = make_inference_fn(params, deterministic=True)
    jit_inference_fn = jax.jit(inference_fn)
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)

    # Run evaluation rollouts.
    def do_rollout(rng, state, core_state):
        empty_data = state.data.__class__(
            **{k: None for k in state.data.__annotations__}
        )  # pytype: disable=attribute-error
        empty_traj = state.__class__(
            **{k: None for k in state.__annotations__}
        )  # pytype: disable=attribute-error
        empty_traj = empty_traj.replace(data=empty_data)

        def step(carry, _):
            state, rnn_state, rng = carry
            rng, act_key = jax.random.split(rng)
            act, _, new_rnn_state = jit_inference_fn(state.obs, rnn_state, act_key)
            state = eval_env.step(state, act)
            # Mask recurrent state on episode termination
            mask = 1.0 - state.done
            masked_rnn_state = ppo_network.mask_state_fn(new_rnn_state, mask)
            traj_data = empty_traj.tree_replace(
                {
                    "data.qpos": state.data.qpos,
                    "data.qvel": state.data.qvel,
                    "data.time": state.data.time,
                    "data.ctrl": state.data.ctrl,
                    "data.mocap_pos": state.data.mocap_pos,
                    "data.mocap_quat": state.data.mocap_quat,
                    "data.xfrc_applied": state.data.xfrc_applied,
                }
            )
            return (state, masked_rnn_state, rng), traj_data

        _, traj = jax.lax.scan(
            step, (state, core_state, rng), None, length=_EPISODE_LENGTH.value
        )
        return traj

    rng = jax.random.split(jax.random.PRNGKey(_SEED.value), _NUM_VIDEOS.value)
    reset_states = jax.jit(jax.vmap(jit_reset))(rng)
    initial_core_states = ppo_network.initial_state_fn(_NUM_VIDEOS.value)
    traj_stacked = jax.jit(jax.vmap(do_rollout))(rng, reset_states, initial_core_states)
    trajectories = [None] * _NUM_VIDEOS.value
    for i in range(_NUM_VIDEOS.value):
        t = jax.tree.map(lambda x, i=i: x[i], traj_stacked)
        trajectories[i] = [
            jax.tree.map(lambda x, j=j: x[j], t) for j in range(_EPISODE_LENGTH.value)
        ]

    # Render and save the rollout.
    render_every = 2
    fps = 1.0 / eval_env.dt / render_every
    print(f"FPS for rendering: {fps}")
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    for i, rollout in enumerate(trajectories):
        traj = rollout[::render_every]
        frames = eval_env.render(traj, height=480, width=640, scene_option=scene_option)
        video_path = str(ckpt_path / f"rollout{i}.mp4")
        imageio.mimsave(video_path, frames, fps=fps)
        print(f"Rollout video saved to: {video_path}")


if __name__ == "__main__":
    app.run(main)
