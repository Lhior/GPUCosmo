"""candl under recent JAX: ``jnp.atleast_1d`` / ``atleast_2d`` no longer accept Python lists
(JAX ≥ 0.10), which candl passes in ``GaussianPrior`` (its constructor and ``log_like``; the
SPT-3G 2018 TTTEEE and D1 BB data sets fail to load or evaluate). ``ensure_candl_compat()``
gives ``candl.likelihood`` a ``jnp`` whose ``atleast_*d`` convert lists first; everything else is
JAX's. In-process only; idempotent."""

from __future__ import annotations

_DONE = False


class _JnpShim:
    def __init__(self, jnp):
        self._jnp = jnp

    def __getattr__(self, name):
        return getattr(self._jnp, name)

    def atleast_1d(self, *arys):
        return self._jnp.atleast_1d(*(self._fix(a) for a in arys))

    def atleast_2d(self, *arys):
        return self._jnp.atleast_2d(*(self._fix(a) for a in arys))

    def _fix(self, a):
        if isinstance(a, (list, tuple)):
            return self._jnp.stack([self._jnp.asarray(x) for x in a]) if a else self._jnp.zeros(0)
        return a


def ensure_candl_compat() -> bool:
    """Patch ``candl.likelihood`` (if candl is installed); returns whether candl is available."""
    global _DONE
    try:
        import candl.likelihood as cl
    except Exception:  # noqa: BLE001 (candl absent / broken: nothing to patch)
        return False
    if not _DONE and not isinstance(cl.jnp, _JnpShim):
        cl.jnp = _JnpShim(cl.jnp)
    _DONE = True
    return True


__all__ = ["ensure_candl_compat"]
