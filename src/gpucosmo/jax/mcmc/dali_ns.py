#!/usr/bin/env python3
"""DALI posterior tools shared by the evidence sampler (``dali_smc``) and its CLI.

The evidence is that of the DALI likelihood: the likelihood codes evaluated on the order-p DALI
expansion of the exact Boltzmann theory about the MAP (``inference.dali_expansion``; default p = 3, the
triplet), with the prior π,

    Z_DALI = ∫ L_DALI π dx.

The tempered SMC (``dali_smc``) samples L_DALI π and estimates Z_DALI. Its accuracy as an
estimate of the exact evidence is that of the expansion; the DALI hierarchy measures it: the
same expansion truncated to order p − 1, re-evaluated at the order-p posterior draws, gives the
last order's effect on log Z and on the posterior (``dali_order_check``). No emulator is
trained and no exact likelihood is evaluated while sampling: the expansion costs one
finite-difference design of exact theory evaluations per model (1 + 2n² at second order,
1 + 4n + 2n(n − 1) + 8 C(n, 3) at third).

This module holds exact prior draws restricted to the hard cuts (``sample_prior``; Z_DALI is an
evidence under the normalized prior restricted to the physical region, directly comparable
between models), narrowed sampling boxes (``narrowed_prior_view``), the batched DALI
log-likelihood, the Laplace approximation at the MAP (``laplace_gaussian``, for
``--auto-boxes``) and the order check.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import truncnorm

jax.config.update("jax_enable_x64", True)


# ----------------------------------------------------------------------------
# prior
# ----------------------------------------------------------------------------
def separable_logprior(bundle, x: np.ndarray) -> np.ndarray:
    """Unnormalized log of the separable prior part (Gaussian terms only; uniform = 0)."""
    mu, sd = np.asarray(bundle.gauss_mu, dtype=float), np.asarray(bundle.gauss_sd, dtype=float)
    gauss = np.isfinite(sd)
    z = np.where(gauss, (np.asarray(x) - mu) / np.where(gauss, sd, 1.0), 0.0)
    return -0.5 * np.sum(z * z, axis=-1)


def sample_prior(bundle, n: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """``n`` exact draws from the normalized prior of ``bundle``.

    Separable part: uniform on [lo, hi] or Gaussian N(μ, σ) truncated to [lo, hi]
    (``bundle.gauss_mu/gauss_sd``, σ = inf for uniform). The remainder of
    ``bundle.log_prior_fn`` (hard cuts such as μ0 < 2Σ0 + 1 or w0 + wa < 0, joint
    Gaussian terms such as plik's SZ prior) is applied by rejection. Returns the
    points and the acceptance fraction (= prior mass of the physical region under
    the separable prior).
    """
    lo, hi = np.asarray(bundle.prior_lo, dtype=float), np.asarray(bundle.prior_hi, dtype=float)
    mu, sd = np.asarray(bundle.gauss_mu, dtype=float), np.asarray(bundle.gauss_sd, dtype=float)
    gauss = np.isfinite(sd)
    d = lo.size
    out, n_try, n_ok = [], 0, 0
    while n_ok < n:
        m = int(max(4 * (n - n_ok), 2048))
        x = np.empty((m, d))
        for i in range(d):
            if gauss[i]:
                a, b = (lo[i] - mu[i]) / sd[i], (hi[i] - mu[i]) / sd[i]
                x[:, i] = truncnorm.rvs(a, b, loc=mu[i], scale=sd[i], size=m, random_state=rng)
            else:
                x[:, i] = rng.uniform(lo[i], hi[i], m)
        lp = np.asarray(bundle.log_prior_fn(jnp.asarray(x)), dtype=float)
        lsep = separable_logprior(bundle, x)
        extra = lp - lsep  # 0 inside the cuts for separable priors; ≤ 0 joint Gaussian terms
        finite = np.isfinite(extra)
        if finite.any():
            c = max(0.0, float(np.max(extra[finite])))
            acc = finite & (np.log(rng.uniform(size=m)) < extra - c)
        else:
            acc = finite
        out.append(x[acc])
        n_try += m
        n_ok += int(acc.sum())
    x = np.concatenate(out)[:n]
    return x, {"accept_fraction": n_ok / n_try, "n_tried": n_try}


# ----------------------------------------------------------------------------
# DALI likelihood
# ----------------------------------------------------------------------------
def make_dali_loglike(bundle) -> Callable[[jnp.ndarray, Any], jnp.ndarray]:
    """``ll_s(x, tt)``: the bundle's likelihood code on the DALI-expanded theory ``tt``
    (a pytree: expansion coefficients + the likelihood's resident arrays), passed as an
    argument so jitted programs do not embed multi-GB constants. A bundle view with a
    ``dali_loglike`` attribute (e.g. ``CompressedDALI.loglike``, with ``tt`` its
    ``data``) supplies its own function."""
    own = getattr(bundle, "dali_loglike", None)
    if own is not None:
        return own
    S = bundle.dali

    def ll_s(x, tt):
        return bundle.log_like_from_theory_fn(x, S.theory(tt, x))

    return ll_s


def separable_logdensity(lo, hi, mu, sd, X: np.ndarray) -> np.ndarray:
    """Normalized log density of the separable prior: uniform on [lo, hi], or N(μ, σ)
    truncated to [lo, hi] (σ finite); −inf outside the box."""
    from scipy.special import ndtr

    X = np.atleast_2d(np.asarray(X, dtype=float))
    gauss = np.isfinite(sd)
    s = np.where(gauss, sd, 1.0)
    a, b = (lo - mu) / s, (hi - mu) / s
    log_norm = np.where(gauss, np.log(np.where(gauss, ndtr(b) - ndtr(a), 1.0)) + np.log(s)
                        + 0.5 * np.log(2 * np.pi), np.log(hi - lo))
    z = np.where(gauss, (X - mu) / s, 0.0)
    out = np.sum(-0.5 * z * z - log_norm, axis=-1)
    inside = np.all((X >= lo) & (X <= hi), axis=-1)
    return np.where(inside, out, -np.inf)


def laplace_gaussian(bundle, tt, x_map, *, rel_step: float = 1e-4):
    """(mean, cov, n_floored) of the DALI posterior's Laplace approximation at ``x_map``.

    The Hessian is central differences of the compiled gradient (steps ``rel_step``·(1+|x|)),
    as ``run_dali_hmc``'s MAP stage: forward-over-reverse (``jax.hessian``) through the
    full-shape model compiles for > 10 min and > 10 GB of host memory. ``tt`` (expansion tables +
    multi-GB likelihood arrays) is an argument, not a closure (it would be compiled in as a
    constant). Directions with less curvature than the prior box (e.g. a parameter pinned at a
    prior edge) are floored at the box variance."""
    ll_s = make_dali_loglike(bundle)
    vg = jax.jit(jax.value_and_grad(lambda x, t: ll_s(x, t) + bundle.log_prior_fn(x)))
    x0 = np.asarray(x_map, dtype=float)
    n = x0.size
    e = rel_step * (1.0 + np.abs(x0))
    rows = []
    for i in range(n):
        d = np.zeros(n)
        d[i] = e[i]
        gp = np.asarray(vg(jnp.asarray(x0 + d), tt)[1], dtype=float)
        gm = np.asarray(vg(jnp.asarray(x0 - d), tt)[1], dtype=float)
        rows.append((gp - gm) / (2.0 * e[i]))
    H = -np.stack(rows)
    H = 0.5 * (H + H.T)
    if not np.all(np.isfinite(H)):
        raise ValueError("non-finite Laplace Hessian (MAP at a prior edge? reduce rel_step)")
    width = np.asarray(bundle.prior_hi, dtype=float) - np.asarray(bundle.prior_lo, dtype=float)
    D = np.diag(width)  # box units: the prior variance is 1/12 per axis
    lam, V = np.linalg.eigh(D @ H @ D)
    n_bad = int(np.sum(lam <= 12.0))
    cov_u = (V / np.maximum(lam, 12.0)) @ V.T
    return x0, D @ cov_u @ D, n_bad


def narrowed_prior_view(bundle, box: dict | None):
    """A view of ``bundle`` whose uniform boxes are replaced by ``box`` {name: (lo, hi)},
    and log(V_narrow / V_box). Only uniform parameters may be narrowed (narrowing a
    truncated Gaussian changes its normalization differently), and only inward."""
    from types import SimpleNamespace

    names = list(bundle.param_names)
    lo = np.asarray(bundle.prior_lo, dtype=float).copy()
    hi = np.asarray(bundle.prior_hi, dtype=float).copy()
    sd = np.asarray(bundle.gauss_sd, dtype=float)
    log_ratio = 0.0
    for n, (a, b) in (box or {}).items():
        i = names.index(n)
        if np.isfinite(sd[i]):
            raise ValueError(f"sampling_box: {n} has a Gaussian prior; only uniform parameters can be narrowed")
        if not (lo[i] <= a < b <= hi[i]):
            raise ValueError(f"sampling_box: {n} [{a}, {b}] not inside the prior box [{lo[i]}, {hi[i]}]")
        log_ratio += float(np.log((b - a) / (hi[i] - lo[i])))
        lo[i], hi[i] = a, b
    view = SimpleNamespace(param_names=bundle.param_names, prior_lo=jnp.asarray(lo), prior_hi=jnp.asarray(hi),
                           gauss_mu=bundle.gauss_mu, gauss_sd=bundle.gauss_sd, log_prior_fn=bundle.log_prior_fn)
    return view, log_ratio


def make_batched_dali_loglike(bundle, tt, n_batch: int, ceiling: float | None = None,
                                   stats: dict | None = None):
    """``f(X) -> np.ndarray``: DALI log-likelihood of the rows of ``X`` in fixed-size
    GPU batches (padded), −inf where non-finite or above ``ceiling`` (counted in
    ``stats["n_above_ceiling"]``, with the largest such value and its point)."""
    ll_s = make_dali_loglike(bundle)
    f = jax.jit(lambda X, t: jax.vmap(lambda x: ll_s(x, t))(X))

    def batched(X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        m = X.shape[0]
        out = np.empty(m)
        for i0 in range(0, m, n_batch):
            xb = X[i0:i0 + n_batch]
            pad = n_batch - xb.shape[0]
            xb_p = np.concatenate([xb, np.repeat(xb[-1:], pad, axis=0)]) if pad else xb
            out[i0:i0 + xb.shape[0]] = np.asarray(f(jnp.asarray(xb_p), tt), dtype=float)[:xb.shape[0]]
        if ceiling is not None:
            bad = np.isfinite(out) & (out > ceiling)
            if bad.any() and stats is not None:
                stats["n_above_ceiling"] = stats.get("n_above_ceiling", 0) + int(bad.sum())
                j = int(np.argmax(np.where(bad, out, -np.inf)))
                if out[j] > stats.get("max_above_ceiling", -np.inf):
                    stats["max_above_ceiling"] = float(out[j])
                    stats["point_above_ceiling"] = X[j].tolist()
            out = np.where(bad, -np.inf, out)
        return np.where(np.isfinite(out), out, -np.inf)

    return batched


# ----------------------------------------------------------------------------
# convergence of the DALI hierarchy
# ----------------------------------------------------------------------------
def dali_order_check(X: np.ndarray, ll_p: np.ndarray, ll_q: np.ndarray, names, *, order: int,
                     lower_order: int | None = None) -> dict:
    """Change of the evidence and posterior from the order-q (default p − 1) to the order-p DALI.

    ``X``: draws of the order-p DALI posterior, ``ll_p`` / ``ll_q`` the order-p and the truncated
    order-q DALI log-likelihoods there. Importance weights w = L_q / L_p give

        log Z_q − log Z_p = log E_p[w],

    and the order-q posterior moments, so the last order's effect on log Z and on each
    parameter's mean (in σ of the order-p posterior) and width. The terms of orders q + 1 … p
    measure the truncation error of the order-q expansion; that of the order-p one is
    expected smaller when the hierarchy converges. Consecutive orders add terms of
    alternating parity in each parameter, so a step p − 1 → p cannot see an error dominated by
    the next term of the other parity (a missing quartic term leaves the 2 → 3 step small):
    compare p − 2 → p too, or run order p + 1. Costs two DALI evaluations per draw,
    no exact theory call. The weights' ESS fraction says whether the order-p draws cover
    the order-q posterior (if it is low the estimate is unreliable and the expansion far
    from converged)."""
    X = np.asarray(X, dtype=float)
    ok = np.isfinite(ll_p) & np.isfinite(ll_q)
    Xf = X[ok]
    lw = np.asarray(ll_q, dtype=float)[ok] - np.asarray(ll_p, dtype=float)[ok]
    if lw.size == 0:
        raise ValueError("DALI order check: no draw with finite log L at both orders")
    m = float(lw.max())
    w = np.exp(lw - m)
    dlogz = m + float(np.log(np.mean(w)))
    err = float(np.std(w) / (np.sqrt(w.size) * np.mean(w)))
    wn = w / w.sum()
    ess = float(1.0 / np.sum(wn ** 2))
    mean_p, sd_p = Xf.mean(axis=0), Xf.std(axis=0)
    mean_q = wn @ Xf
    sd_q = np.sqrt(wn @ (Xf - mean_q) ** 2)
    sd_safe = np.where(sd_p > 0, sd_p, np.inf)
    shift = (mean_q - mean_p) / sd_safe
    names = [str(n) for n in names]
    return dict(
        order=int(order), lower_order=int(order) - 1 if lower_order is None else int(lower_order),
        n_draws=int(X.shape[0]), n_nonfinite=int((~ok).sum()),
        delta_logz=dlogz, delta_logz_err=err, ess_fraction=ess / w.size,
        loglike_diff_mean=float(lw.mean()), loglike_diff_std=float(lw.std()),
        mean_shift_sigma=dict(zip(names, shift.tolist())), std_ratio=dict(zip(names, (sd_q / sd_safe).tolist())),
        max_abs_mean_shift_sigma=float(np.max(np.abs(shift))),
        max_abs_mean_shift_param=names[int(np.argmax(np.abs(shift)))],
    )


__all__ = ["dali_order_check", "laplace_gaussian", "make_batched_dali_loglike", "make_dali_loglike",
           "narrowed_prior_view", "sample_prior", "separable_logdensity", "separable_logprior"]
