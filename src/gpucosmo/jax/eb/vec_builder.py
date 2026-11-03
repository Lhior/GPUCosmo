"""Assemble a small state vector from static-index pieces in one concatenate.

Chained ``y.at[i].set(v)`` on a per-mode RHS vector lowers to one in-place
dynamic-update-slice per call, and XLA:GPU keeps each as a separate kernel:
~25 launches per ``derivs`` evaluation, the dominant cost of the (latency
bound) mode evolution. ``VecBuilder`` records the same updates with the same
"last write wins" semantics and emits a single ``concatenate`` that fuses with
the producers. The values are bitwise those of the ``.at[].set`` chain.
"""

from __future__ import annotations

import jax.numpy as jnp


class _At:
    __slots__ = ("_b",)

    def __init__(self, builder):
        self._b = builder

    def __getitem__(self, idx):
        return _AtIdx(self._b, idx)


class _AtIdx:
    __slots__ = ("_b", "_idx")

    def __init__(self, builder, idx):
        self._b, self._idx = builder, idx

    def set(self, val):
        return self._b._set(self._idx, val)


class VecBuilder:
    """Drop-in for the ``.at[static].set`` / static ``[]`` subset of a 1-D array."""

    def __init__(self, n: int, base=None, dtype=jnp.float64):
        self.n = int(n)
        self.dtype = dtype
        self.base = None if base is None else jnp.asarray(base, dtype=dtype)
        self._ups: list[tuple[int, int, object]] = []

    @property
    def at(self):
        return _At(self)

    def _range(self, idx) -> tuple[int, int, bool]:
        if isinstance(idx, slice):
            lo, hi, step = idx.indices(self.n)
            if step != 1:
                raise IndexError("VecBuilder supports unit-step slices only")
            return lo, hi, False
        i = int(idx)
        if i < 0:
            i += self.n
        return i, i + 1, True

    def _set(self, idx, val):
        lo, hi, scalar = self._range(idx)
        if hi <= lo:
            return self
        v = jnp.asarray(val, dtype=self.dtype)
        v = jnp.reshape(v, (1,)) if scalar else jnp.broadcast_to(v, (hi - lo,))
        self._ups.append((lo, hi, v))
        return self

    def _pieces(self, lo: int, hi: int):
        owner = [-1] * (hi - lo)
        for u, (a, b, _v) in enumerate(self._ups):
            for i in range(max(a, lo), min(b, hi)):
                owner[i - lo] = u
        out, i = [], lo
        while i < hi:
            u = owner[i - lo]
            j = i
            while j < hi and owner[j - lo] == u:
                j += 1
            if u < 0:
                out.append(jnp.zeros(j - i, self.dtype) if self.base is None else self.base[i:j])
            else:
                a, _b, v = self._ups[u]
                out.append(v[i - a : j - a])
            i = j
        return out

    def __getitem__(self, idx):
        lo, hi, scalar = self._range(idx)
        p = self._pieces(lo, hi)
        v = p[0] if len(p) == 1 else jnp.concatenate(p)
        return v[0] if scalar else v

    def build(self):
        p = self._pieces(0, self.n)
        return p[0] if len(p) == 1 else jnp.concatenate(p)


__all__ = ["VecBuilder"]
