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
"""Environment registry for the experiment harness.

Registered environments:
  pendulum          - mujoco_playground PendulumSwingup
  cartpole          - mujoco_playground CartpoleBalance
  pendulum_partial  - PendulumSwingupPartialObs (POMDP; dict obs with
                      'state' (2D, policy) and 'privileged_state' (3D, value))
  fast              - brax's trivial debug env (CPU smoke tests; no video)
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from brax import envs as brax_envs
from experiments.wrappers import wrap_mjx_env

from mujoco_playground import registry
from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.dm_control_suite import common

_XML_PATH = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "pendulum.xml"
_ANGLE_BOUND = 8
_COSINE_BOUND = np.cos(np.deg2rad(_ANGLE_BOUND))


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.01,
        episode_length=1000,
        action_repeat=1,
        vision=False,
        impl="jax",
        nconmax=0,
        njmax=0,
    )


class PendulumSwingupPartialObs(mjx_env.MjxEnv):
    """Swingup environment with partial observation (angular velocity hidden).

    This is identical to PendulumSwingup except the policy observation only
    includes pole orientation (2D) and excludes angular velocity. This creates
    a POMDP that requires memory to solve optimally.

    Observations are returned as a dict:
      - 'state': Partial observation (2D) - pole orientation only (for policy)
      - 'privileged_state': Full observation (3D) - orientation + velocity (for value)
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)
        if self._config.vision:
            raise NotImplementedError(
                f"Vision not implemented for {self.__class__.__name__}."
            )

        self._xml_path = _XML_PATH.as_posix()
        self._model_assets = common.get_assets()
        self._mj_model = mujoco.MjModel.from_xml_string(
            _XML_PATH.read_text(), self._model_assets
        )
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._post_init()

    def _post_init(self) -> None:
        self._pole_body_id = self.mj_model.body("pole").id
        hinge_joint_id = self.mj_model.joint("hinge").id
        self._hinge_qposadr = self.mj_model.jnt_qposadr[hinge_joint_id]
        self._hinge_qveladr = self.mj_model.jnt_dofadr[hinge_joint_id]

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, rng1 = jax.random.split(rng)

        qpos = jp.zeros(self.mjx_model.nq)
        qpos = qpos.at[self._hinge_qposadr].set(jax.random.uniform(rng1) * jp.pi)

        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            impl=self.mjx_model.impl.value,
            naconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        metrics = {}
        info = {"rng": rng}

        reward, done = jp.zeros(2)  # pylint: disable=redefined-outer-name
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)
        reward = self._get_reward(
            data, action, state.info, state.metrics
        )  # pylint: disable=redefined-outer-name
        obs = self._get_obs(data, state.info)
        done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = done.astype(float)
        return mjx_env.State(data, obs, reward, done, state.metrics, state.info)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> dict[str, jax.Array]:
        del info  # Unused.
        # Return dict with partial obs for policy and full obs for value network
        orientation = self._pole_orientation(data)
        velocity = self._angular_velocity(data)
        return {
            "state": orientation,  # Partial: policy sees only orientation (2D)
            "privileged_state": jp.concatenate(
                [orientation, velocity.reshape(1)]
            ),  # Full: value sees orientation + velocity (3D)
        }

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
    ) -> jax.Array:
        del action, info, metrics  # Unused.
        return reward.tolerance(
            self._pole_vertical(data),
            (_COSINE_BOUND, 1),
            margin=2.0,
            sigmoid="gaussian",
        )

    def _pole_vertical(self, data: mjx.Data) -> jax.Array:
        """Returns vertical (z) component of pole frame."""
        return data.xmat[self._pole_body_id, 2, 2]

    def _angular_velocity(self, data: mjx.Data) -> jax.Array:
        """Returns the angular velocity of the pole."""
        return data.qvel[self._hinge_qveladr]

    def _pole_orientation(self, data: mjx.Data) -> jax.Array:
        """Returns both horizontal and vertical components of pole frame."""
        xz = data.xmat[self._pole_body_id, 0, 2]
        zz = data.xmat[self._pole_body_id, 2, 2]
        return jp.array([xz, zz])

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def observation_size(self) -> dict[str, int]:
        return {
            "state": 2,  # Pole orientation (xz, zz)
            "privileged_state": 3,  # Pole orientation + angular velocity
        }

    @property
    def action_size(self) -> int:
        return self.mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model


# ============================================================================
# Registry
# ============================================================================

def _load_playground(env_name):
    def ctor(config_overrides=None):
        # Default to the JAX backend: with warp-lang installed, playground's
        # default impl resolves to warp, which cannot run on CPU-only nodes.
        overrides = {'impl': 'jax', **(config_overrides or {})}
        return registry.load(env_name, config_overrides=overrides)

    return ctor


def _load_partial_pendulum(config_overrides=None):
    return PendulumSwingupPartialObs(config_overrides=config_overrides)


def _load_fast(config_overrides=None):
    if config_overrides:
        raise ValueError("env_config is not supported for the 'fast' env")
    return brax_envs.get_environment('fast')


# name -> (constructor, wrap_env_fn or None). A None wrap_env_fn means the env
# comes from the brax registry and uses brax's default training wrappers.
_ENVS = {
    'pendulum': (_load_playground('PendulumSwingup'), wrap_mjx_env),
    'cartpole': (_load_playground('CartpoleBalance'), wrap_mjx_env),
    'pendulum_partial': (_load_partial_pendulum, wrap_mjx_env),
    'fast': (_load_fast, None),
}


def env_names():
    return sorted(_ENVS)


def make_env(name: str, config_overrides=None):
    """Returns (env, wrap_env_fn) for a registered environment name."""
    if name not in _ENVS:
        raise ValueError(f'Unknown env {name!r}. Available: {env_names()}')
    ctor, wrap_env_fn = _ENVS[name]
    return ctor(config_overrides), wrap_env_fn
