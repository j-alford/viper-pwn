"""Error-controlled electron-energy quadrature."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import special

__all__ = ["trapz_loglog", "QuadratureInfo", "integrate_population"]


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
                         start_per_decade: float = 8.0,
                         max_per_decade: float = 4096.0,
                         richardson: bool = True,
                         threshold: np.ndarray | None = None,
                         delta_min: np.ndarray | None = None,
                         ) -> tuple[np.ndarray, QuadratureInfo]:
    """Error-controlled population integral."""
    eps = np.asarray(eps, dtype=float).ravel()
    n_eps = eps.size
    result = np.zeros(n_eps)
    errs = np.zeros(n_eps)
    if n_eps == 0:
        return result, QuadratureInfo(0.0, 0.0, 0, True)
    shared = threshold is None
    if shared:
        lo = np.full(n_eps, np.log(E_min))
        hi = np.full(n_eps, np.log(E_max))
        thr = np.zeros(n_eps)
    else:
        thr = np.broadcast_to(np.asarray(threshold, dtype=float), (n_eps,))
        dlo = np.maximum(E_min - thr, 0.0)
        if delta_min is not None:
            dlo = np.maximum(dlo, np.broadcast_to(delta_min, (n_eps,)))
        dhi = E_max - thr
        valid = (dhi > dlo) & (dlo > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            lo = np.where(valid, np.log(np.where(valid, dlo, 1.0)), 0.0)
            hi = np.where(valid, np.log(np.where(valid, dhi, 1.0)), 0.0)
    span = hi - lo
    active = np.nonzero(span > 0)[0]
    if active.size == 0:
        return result, QuadratureInfo(0.0, 0.0, 0, True)
    decades = span[active] / np.log(10.0)
    nseg = max(8, int(np.ceil(start_per_decade * decades.max())))
    s01 = np.linspace(0.0, 1.0, nseg + 1)

    def evaluate(frac: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Integrand on the grid."""
        if shared:
            lnE = lo[0] + span[0] * frac
            E = np.exp(lnE)
            return (E * density(E))[:, None] * kernel(E, eps[cols])
        d = np.exp(lo[cols][None, :] + span[cols][None, :] * frac[:, None])
        E = thr[cols][None, :] + d
        N = density(E.ravel()).reshape(E.shape)
        return (d * N) * kernel(E, eps[cols])

    G = evaluate(s01, active)
    h = span[active] / nseg
    T = _trapz_uniform(G, h)
    R = T.copy()
    w2 = eps**2
    level = 0
    max_used = 0
    while active.size:
        mid = 0.5 * (s01[:-1] + s01[1:])
        Gm = evaluate(mid, active)
        Gn = np.empty((2 * nseg + 1, active.size))
        Gn[0::2] = G
        Gn[1::2] = Gm
        sn = np.empty(2 * nseg + 1)
        sn[0::2] = s01
        sn[1::2] = mid
        nseg *= 2
        h = 0.5 * h
        level += 1
        Tn = _trapz_uniform(Gn, h)
        if richardson:
            Rn = Tn + (Tn - T) / 3.0
            err_abs = np.abs(Rn - R) if level > 1 else np.abs(Tn - T)
        else:
            Rn = Tn
            err_abs = np.abs(Tn - T) / 3.0
        result[active] = Rn
        with np.errstate(divide="ignore", invalid="ignore"):
            errs[active] = np.where(Rn != 0, err_abs / np.abs(Rn),
                                    np.where(err_abs == 0, 0.0, np.inf))
        max_used = nseg + 1
        sedv = np.abs(result) * w2
        smax = sedv.max()
        insig = sedv[active] < floor * smax
        dens = nseg / (span[active] / np.log(10.0))
        done = (errs[active] <= rtol) | insig | (dens >= max_per_decade)
        keep = ~done
        active = active[keep]
        G, T, R, h = Gn[:, keep], Tn[keep], Rn[keep], h[keep]
        s01 = sn
    sedv = np.abs(result) * w2
    sig = (sedv >= floor * sedv.max()) & (sedv > 0)
    err = float(np.max(errs[sig])) if np.any(sig) else 0.0
    per_dec = (max_used - 1) / max(np.max(span) / np.log(10.0), 1e-300)
    return result, QuadratureInfo(err, per_dec, max_used, err <= rtol)
