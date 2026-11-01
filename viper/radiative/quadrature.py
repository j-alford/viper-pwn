"""Error-controlled particle-energy quadrature."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import special

__all__ = ["trapz_loglog", "QuadratureInfo", "QuadratureWarning",
           "integrate_population", "integrate_piecewise"]

_START_PER_DECADE = 8.0
_LN10 = np.log(10.0)


class QuadratureWarning(UserWarning):
    """Refinement cap reached."""


def _logmean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Logarithmic mean."""
    pos = (a > 0) & (b > 0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        lm = a * special.exprel(np.log(b / a))
    return np.where(pos, lm, 0.5 * (a + b))


def trapz_loglog(y: np.ndarray, x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Power-law-exact trapezoid."""
    y = np.moveaxis(np.asarray(y, dtype=float), axis, 0)
    x = np.asarray(x, dtype=float)
    shape = (-1,) + (1,) * (y.ndim - 1)
    g = y * x.reshape(shape)
    h = np.diff(np.log(x)).reshape(shape)
    seg = h * _logmean(g[:-1], g[1:])
    lin = 0.5 * (y[:-1] + y[1:]) * np.diff(x).reshape(shape)
    pos = (g[:-1] > 0) & (g[1:] > 0)
    return np.sum(np.where(pos, seg, lin), axis=0)


def _trapz_uniform(g: np.ndarray, h) -> np.ndarray:
    """Uniform-grid trapz_loglog."""
    return h * np.sum(_logmean(g[:-1], g[1:]), axis=0)


@dataclass
class QuadratureInfo:
    """Last quadrature outcome."""

    rel_error: float
    per_decade: float
    n_points: int
    converged: bool


def integrate_population(kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
                         density: Callable[[np.ndarray], np.ndarray],
                         E_min: float, E_max: float, eps: np.ndarray,
                         rtol: float = 1e-4, floor: float = 1e-6,
                         max_per_decade: float = 4096.0,
                         threshold: np.ndarray | None = None,
                         delta_min: np.ndarray | None = None,
                         ) -> tuple[np.ndarray, QuadratureInfo]:
    """Single-piece population integral."""
    return integrate_piecewise(lambda E, e, k: kernel(E, e), density,
                               [E_min, E_max], eps, threshold, delta_min,
                               rtol=rtol, floor=floor,
                               max_per_decade=max_per_decade)


def integrate_piecewise(kernel: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
                        density: Callable[[np.ndarray], np.ndarray],
                        edges: np.ndarray, eps: np.ndarray,
                        threshold: np.ndarray | None = None,
                        delta_min: np.ndarray | None = None,
                        rtol: float = 1e-4, floor: float = 1e-6,
                        max_per_decade: float = 4096.0,
                        ) -> tuple[np.ndarray, QuadratureInfo]:
    """Population integral over pieces."""
    eps = np.asarray(eps, dtype=float).ravel()
    n_eps = eps.size
    result = np.zeros(n_eps)
    errs = np.zeros(n_eps)
    edges = np.asarray(edges, dtype=float).ravel()
    n_pc = edges.size - 1
    if n_eps == 0 or n_pc < 1:
        return result, QuadratureInfo(0.0, 0.0, 0, True)
    shared = threshold is None
    if shared:
        thr = np.zeros(n_eps)
        a = np.repeat(edges[:-1, None], n_eps, axis=1)
        b = np.repeat(edges[1:, None], n_eps, axis=1)
        ok = (b > a) & (a > 0)
        dlo = np.where(ok, a, 1.0)
    else:
        thr = np.broadcast_to(np.asarray(threshold, dtype=float), (n_eps,))
        dmin = np.zeros(n_eps) if delta_min is None else \
            np.broadcast_to(np.asarray(delta_min, dtype=float), (n_eps,))
        a = edges[:-1, None] - thr[None, :]
        b = edges[1:, None] - thr[None, :]
        dlo = np.maximum(a, dmin[None, :])
        ok = (b > dlo) & (dlo > 0)
        dlo = np.where(ok, dlo, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lo = np.log(dlo)
        span = np.where(ok, np.log(np.where(ok, b, 1.0)) - lo, 0.0)
    active = np.nonzero(np.any(span > 0, axis=0))[0]
    pieces = [k for k in range(n_pc) if np.any(span[k] > 0)]
    if active.size == 0:
        return result, QuadratureInfo(0.0, 0.0, 0, True)
    nseg = {k: max(4, int(np.ceil(_START_PER_DECADE * span[k].max() / _LN10)))
            for k in pieces}

    def evaluate(k: int, frac: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Integrand on piece k."""
        if shared:
            E = np.exp(lo[k, 0] + span[k, 0] * frac)
            return (E * density(E))[:, None] * kernel(E, eps[cols], k)
        d = np.exp(lo[k, cols][None, :] + span[k, cols][None, :] * frac[:, None])
        E = thr[cols][None, :] + d
        N = density(E.ravel()).reshape(E.shape)
        out = (d * N) * kernel(E, eps[cols], k)
        return np.where(span[k, cols][None, :] > 0, out, 0.0)

    grid = {k: np.linspace(0.0, 1.0, nseg[k] + 1) for k in pieces}
    G = {k: evaluate(k, grid[k], active) for k in pieces}
    h = {k: span[k, active] / nseg[k] for k in pieces}
    T = sum(_trapz_uniform(G[k], h[k]) for k in pieces)
    R = T.copy()
    w2 = eps**2
    level = 0
    max_used = 0
    while active.size:
        Tn = 0.0
        for k in pieces:
            s01 = grid[k]
            mid = 0.5 * (s01[:-1] + s01[1:])
            Gm = evaluate(k, mid, active)
            Gn = np.empty((2 * nseg[k] + 1, active.size))
            Gn[0::2] = G[k]
            Gn[1::2] = Gm
            sn = np.empty(2 * nseg[k] + 1)
            sn[0::2] = s01
            sn[1::2] = mid
            nseg[k] *= 2
            h[k] = 0.5 * h[k]
            G[k], grid[k] = Gn, sn
            Tn = Tn + _trapz_uniform(Gn, h[k])
        level += 1
        Rn = Tn + (Tn - T) / 3.0
        err_abs = np.abs(Rn - R) if level > 1 else np.abs(Tn - T)
        result[active] = Rn
        with np.errstate(divide="ignore", invalid="ignore"):
            errs[active] = np.where(Rn != 0, err_abs / np.abs(Rn),
                                    np.where(err_abs == 0, 0.0, np.inf))
        max_used = sum(nseg[k] + 1 for k in pieces)
        sedv = np.abs(result) * w2
        insig = sedv[active] < floor * sedv.max()
        with np.errstate(divide="ignore"):
            dens = np.min([np.where(span[k, active] > 0,
                                    nseg[k] * _LN10 / span[k, active], np.inf)
                           for k in pieces], axis=0)
        done = (errs[active] <= rtol) | insig | (dens >= max_per_decade)
        keep = ~done
        active = active[keep]
        for k in pieces:
            G[k], h[k] = G[k][:, keep], h[k][keep]
        T, R = Tn[keep], Rn[keep]
    sedv = np.abs(result) * w2
    sig = (sedv >= floor * sedv.max()) & (sedv > 0)
    err = float(np.max(errs[sig])) if np.any(sig) else 0.0
    per_dec = float(min(nseg[k] * _LN10 / span[k].max() for k in pieces))
    return result, QuadratureInfo(err, per_dec, max_used, err <= rtol)
