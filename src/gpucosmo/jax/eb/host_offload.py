"""Run scalar sequential stages (Recfast, thermo table) on the host CPU.

On the GPU these stages are long chains of single-thread fp64 kernels plus a
host round trip per ``while`` iteration: the matter-P(k) thermo build takes
1.3 s on an RTX 5070 against 11 ms for the same XLA program compiled for the
CPU. ``on_host`` keeps the algorithm (and the JAX code) identical and only
changes where it executes, via ``jax.pure_callback``.

Forward mode works to any order: the JVP rule evaluates ``jax.jvp(fn)`` on the host
through ``on_host`` itself, so a second derivative (forward over forward, e.g. for the
DALI expansion's Hessian) nests one more host call. Reverse mode is not defined
(``pure_callback`` has no transpose); callers that need ``jax.grad`` keep the device path.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np


def _cpu_device():
    return jax.devices("cpu")[0]


@functools.lru_cache(maxsize=None)
def _has_cpu() -> bool:
    try:
        jax.devices("cpu")
    except RuntimeError:
        return False
    return True


def on_host(fn):
    """Wrap ``fn(*scalars) -> pytree of arrays`` to run on the CPU backend.

    ``fn`` takes scalar arguments only. Under ``vmap`` the callback receives the
    whole batch at once and evaluates it with a CPU ``vmap`` (one host call per
    device batch).
    """
    cpu_single = jax.jit(fn, backend="cpu")
    cpu_batch = jax.jit(jax.vmap(fn), backend="cpu")

    def _out_struct(n_args):
        scalar = jax.ShapeDtypeStruct((), jnp.float64)
        out = jax.eval_shape(fn, *([scalar] * n_args))
        return jax.tree.map(lambda s: jax.ShapeDtypeStruct(s.shape, s.dtype), out)

    def _host(*args):
        args = [np.asarray(a) for a in args]
        with jax.default_device(_cpu_device()):
            if args[0].ndim == 0:
                out = cpu_single(*args)
            else:
                lead = args[0].shape
                flat = [a.reshape(-1) for a in args]
                out = cpu_batch(*flat)
                out = jax.tree.map(lambda x: x.reshape(lead + x.shape[1:]), out)
        return jax.tree.map(np.asarray, out)

    @jax.custom_jvp
    def via_host(*args):
        return jax.pure_callback(
            _host, _out_struct(len(args)), *args, vmap_method="broadcast_all",
        )

    @via_host.defjvp
    def _via_host_jvp(primals, tangents):
        n = len(primals)
        tangents = tuple(jnp.zeros_like(p) if type(t).__name__ == "SymbolicZero" else t
                         for p, t in zip(primals, tangents))
        out = via_host(*primals)
        t_out = _jvp_on_host(fn, n)(*primals, *tangents)
        return out, t_out

    @functools.wraps(fn)
    def wrapped(*args):
        args = tuple(jnp.asarray(a, dtype=jnp.float64) for a in args)
        if jax.default_backend() == "cpu" or not _has_cpu():
            # Already on the host, or no CPU backend (JAX_PLATFORMS=cuda): inline.
            return fn(*args)
        return via_host(*args)

    return wrapped


@functools.lru_cache(maxsize=None)
def _jvp_on_host_cached(fn, n):
    def jvp_fn(*a):
        return jax.jvp(fn, tuple(a[:n]), tuple(a[n:]))[1]

    return on_host(jvp_fn)


def _jvp_on_host(fn, n):
    """``(*primals, *tangents) -> tangent of fn``, itself evaluated on the host (and
    forward-differentiable again)."""
    return _jvp_on_host_cached(fn, n)


__all__ = ["on_host"]
