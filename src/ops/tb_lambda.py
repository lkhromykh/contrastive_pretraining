import jax
import chex


def tree_backup(q_t: chex.Array,
                v_tp1: chex.Array,
                r_t: chex.Array,
                disc_t: chex.Array,
                pi_t: chex.Array,
                lambda_: float
                ) -> chex.Array:
    """Tree-backup, TB(λ)"""
    pi_t = lambda_ * pi_t
    xs = (q_t, v_tp1, r_t, disc_t, pi_t)
    chex.assert_rank(xs, 1)
    chex.assert_type(xs, float)
    chex.assert_scalar_non_negative(lambda_)

    def fn(acc, x):
        q, next_v, r, disc, c = x
        resid = r + disc * next_v - q
        acc = resid + disc * c * acc
        return acc, acc

    _, resid_t = jax.lax.scan(fn, 0., xs, reverse=True)
    return q_t + resid_t
