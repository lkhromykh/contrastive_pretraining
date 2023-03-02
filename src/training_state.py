from typing import NamedTuple

import jax
import jax.numpy as jnp
import haiku as hk
import optax


@jax.tree_util.register_pytree_node_class
class TrainingState(NamedTuple):
    params: hk.Params
    target_params: hk.Params
    opt_state: optax.OptState
    rng: jax.random.PRNGKey
    target_update_var: float | int
    step: int

    tx: optax.TransformUpdateFn

    def update(self, grads: hk.Params):
        params = self.params
        target_params = self.target_params
        opt_state = self.opt_state
        step = self.step

        updates, opt_state = self.tx(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)

        if isinstance(tv := self.target_update_var, int):
            # Hard update.
            target_params = optax.periodic_update(
                params, target_params, step, tv)
        else:
            # Polyak update.
            target_params = optax.incremental_update(params, target_params, tv)

        return self._replace(
            params=params,
            target_params=target_params,
            opt_state=opt_state,
            step=step + 1
        )

    @classmethod
    def init(cls,
             rng: jax.random.PRNGKey,
             params: hk.Params,
             optim: optax.GradientTransformation,
             target_update_var: float | int
             ):
        if isinstance(tv := target_update_var, int):
            tv = jnp.int32(tv)
        else:
            tv = jnp.float32(tv)
        return cls(
            params=params,
            target_params=params,
            opt_state=optim.init(params),
            rng=rng,
            tx=optim.update,
            target_update_var=tv,
            step=jnp.int32(0)
        )

    def tree_flatten(self):
        children = (
            self.params,
            self.target_params,
            self.opt_state,
            self.rng,
            self.target_update_var,
            self.step
        )
        return children, (self.tx,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, *aux)
