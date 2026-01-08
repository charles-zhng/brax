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

"""Optimizers for Recurrent TD3."""

import optax


def make_optimizer(
    learning_rate: float,
    max_grad_norm: float | None = None,
) -> optax.GradientTransformation:
  """Creates an Adam optimizer with optional gradient clipping.

  Args:
    learning_rate: Learning rate for Adam.
    max_grad_norm: If specified, clips gradients by global norm.

  Returns:
    An optax optimizer.
  """
  if max_grad_norm is not None:
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate=learning_rate),
    )
  return optax.adam(learning_rate=learning_rate)
