from typing import Tuple
import jax
import jax.numpy as jnp
from flax import struct
import abc
import chex


class BaseGridGoal(struct.PyTreeNode):
    """Base class for grid world goals using JAX patterns."""

    @abc.abstractmethod
    def __call__(self, state: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Returns (done, reward)."""
        pass


class AgentHasItemFromRulesetGoal(BaseGridGoal):
    color_sensitive: bool = struct.field(pytree_node=False, default=True)

    @jax.jit
    def __call__(self, state):
        # Use precomputed goal information (O(1) instead of O(R²))
        target_item = state.goal_item
        target_color = state.goal_color

        inv_has = state.inventories[target_item]
        if self.color_sensitive:
            col_has = state.inventory_colors[target_item]
            color_ok = col_has == target_color
        else:
            color_ok = jnp.ones_like(inv_has, dtype=jnp.bool_)

        goal_achieved = inv_has & color_ok

        reward = goal_achieved.astype(jnp.float32)
        return goal_achieved, reward
