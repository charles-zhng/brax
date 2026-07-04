"""Algorithm registry for the experiment harness.

Each entry knows how to build its network factory from config, how to call its
``train()``, and how to build a single-env inference step for video rollouts.
"""

import functools
from dataclasses import dataclass
from typing import Any, Callable, Dict

import jax

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo_train
from brax.training.agents.recurrent_ppo import networks as rnn_ppo_networks
from brax.training.agents.recurrent_ppo import train as rnn_ppo_train
from brax.training.agents.recurrent_ppo_feature import networks as rnn_ppo_feature_networks
from brax.training.agents.recurrent_ppo_feature import train as rnn_ppo_feature_train
from brax.training.agents.recurrent_sac import networks as rnn_sac_networks
from brax.training.agents.recurrent_sac import train as rnn_sac_train
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac_train


@dataclass(frozen=True)
class Algo:
    """One trainable algorithm.

    Attributes:
      train_fn: the agent's train() function.
      make_networks: network factory taking (obs_size, action_size, **kwargs).
      recurrent: whether the inference policy threads an RNN hidden state.
      inference_params: maps train()'s returned params tuple to the tuple the
        inference policy expects.
    """

    train_fn: Callable[..., Any]
    make_networks: Callable[..., Any]
    recurrent: bool
    inference_params: Callable[[Any], Any] = lambda params: params


ALGOS: Dict[str, Algo] = {
    'ff_ppo': Algo(
        train_fn=ppo_train.train,
        make_networks=ppo_networks.make_ppo_networks,
        recurrent=False,
    ),
    'rnn_ppo': Algo(
        train_fn=rnn_ppo_train.train,
        make_networks=rnn_ppo_networks.make_rnn_ppo_networks,
        recurrent=True,
    ),
    'rnn_ppo_feature': Algo(
        train_fn=rnn_ppo_feature_train.train,
        make_networks=rnn_ppo_feature_networks.make_rnn_ppo_networks,
        recurrent=True,
    ),
    'rnn_sac': Algo(
        train_fn=rnn_sac_train.train,
        make_networks=rnn_sac_networks.make_recurrent_sac_networks,
        recurrent=True,
        # Inference consumes (normalizer, policy); train() returns
        # (normalizer, policy, q) — drop q_params.
        inference_params=lambda params: (params[0], params[1]),
    ),
    # Stock brax SAC (feedforward baseline). Flat observations only —
    # make_sac_networks has no dict-obs support; use the *_partial_flat envs.
    'ff_sac': Algo(
        train_fn=sac_train.train,
        make_networks=sac_networks.make_sac_networks,
        recurrent=False,
    ),
}


def algo_names():
    return sorted(ALGOS)


def get_algo(name: str) -> Algo:
    if name not in ALGOS:
        raise ValueError(f'Unknown algo {name!r}. Available: {algo_names()}')
    return ALGOS[name]


def make_network_factory(algo: Algo, network_kwargs: Dict[str, Any]):
    """Binds config network kwargs into the factory brax train() expects."""
    if not network_kwargs:
        return algo.make_networks
    return functools.partial(algo.make_networks, **network_kwargs)


def make_act_fn(algo: Algo, network, make_policy, params):
    """Returns (act_fn, initial_carry) for single-env deterministic rollout.

    act_fn(obs, carry, key) -> (action, new_carry). For feedforward algos the
    carry is None and passed through unchanged.
    """
    policy = make_policy(algo.inference_params(params), deterministic=True)

    if algo.recurrent:
        initial_carry = network.policy_network.init_hidden(1)

        @jax.jit
        def act(obs, carry, key):
            obs_b = jax.tree_util.tree_map(lambda x: x[None, ...], obs)
            action, _, new_carry = policy(obs_b, carry, key)
            return action[0], new_carry

        return act, initial_carry

    @jax.jit
    def act(obs, carry, key):
        obs_b = jax.tree_util.tree_map(lambda x: x[None, ...], obs)
        action, _ = policy(obs_b, key)
        return action[0], carry

    return act, None
