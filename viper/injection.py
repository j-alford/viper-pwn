"""Pluggable electron-injection spectral shapes (continuous component).

The Gelfand, Slane & Zhang (2009) model injects a (broken) power law
plus an optional relativistic Maxwellian. This module generalizes the
*continuous* component: the injected dN/dE shape is selected by a
string name from :data:`INJECTION_MODELS`, and users can register
arbitrary continuous shapes of their own.

A shape function is the *unnormalized* differential spectrum

    fn(e_gev, e_break, p1, p2) -> dN/dE   (arbitrary normalization)

vectorized over ``e_gev`` [GeV], non-negative and finite on
``[e_min, e_max]``. Custom shapes must be registered at module level
of the driver / source-config module (MCMC workers re-import it under
``spawn``). Parameter interpretation per built-in form:

============================  =========================  ==========  ==========
name                          e_break                    p1          p2
============================  =========================  ==========  ==========
``'broken_powerlaw'``         break energy (0 = none)    low index   high index
``'log_parabola'``            pivot energy               alpha       beta
``'powerlaw_cutoff'``         e-folding cutoff energy    index       (ignored)
============================  =========================  ==========  ==========
"""

from __future__ import annotations

import numpy as np

from . import constants as C

# Gauss-Legendre nodes per grid bin.
N_NODES = 8
# Panels per decade for the normalization integral.
NORM_PANELS_PER_DECADE = 20


def _shape_broken_powerlaw(e_gev, e_break, p1, p2):
    """Broken power law, continuous at the break (registry/testing form).

    The integrator never calls this; ``'broken_powerlaw'`` uses
    the analytic path in :mod:`viper.particles`.
    """
    if e_break <= 0:
        raise ValueError(
            "broken_powerlaw shape requires e_break > 0 (the analytic "
            "path handles the e_break = 0 single-power-law convention)")
    x = e_gev / e_break
    return np.where(x < 1.0, x ** -p1, x ** -p2)


def _shape_log_parabola(e_gev, e_break, p1, p2):
    """Log parabola: dN/dE = (E/E_piv)^-(alpha + beta*log10(E/E_piv)).

    ``e_break`` is the pivot energy E_piv [GeV], ``p1`` = alpha (index
    at the pivot), ``p2`` = beta (curvature; 0 reduces to a single
    power law of index alpha).
    """
    if e_break <= 0:
        raise ValueError(
            "log_parabola uses e_break as the pivot energy; "
            f"set e_break > 0 (got {e_break})")
    x = e_gev / e_break
    return x ** -(p1 + p2 * np.log10(x))


def _shape_powerlaw_cutoff(e_gev, e_break, p1, p2):
    """Exponentially cut-off power law: dN/dE = E^-p1 * exp(-E/E_cut).

    ``e_break`` is the e-folding cutoff energy E_cut [GeV]; ``p2`` is
    ignored (freeze it when fitting this form).
    """
    if e_break <= 0:
        raise ValueError(
            "powerlaw_cutoff uses e_break as the cutoff energy; "
            f"set e_break > 0 (got {e_break})")
    return e_gev ** -p1 * np.exp(-e_gev / e_break)


INJECTION_MODELS = {
    'broken_powerlaw': _shape_broken_powerlaw,
    'log_parabola': _shape_log_parabola,
    'powerlaw_cutoff': _shape_powerlaw_cutoff,
}

_BUILTIN_MODELS = frozenset(INJECTION_MODELS)


def register_injection_model(name: str, shape_fn) -> None:
    """Register a custom continuous injection shape under *name*.

    ``shape_fn(e_gev, e_break, p1, p2)`` must return the unnormalized
    dN/dE as an array shaped like ``e_gev``, non-negative and finite.
    Re-registering a custom name overwrites it; built-in names cannot
    be overridden.

    For MCMC fits, call at module level of the driver or source-config
    module, not inside ``if __name__ == '__main__'`` (``spawn`` workers
    re-import the driver).
    """
    if name in _BUILTIN_MODELS:
        raise ValueError(
            f"cannot override built-in injection model {name!r}")
    if not callable(shape_fn):
        raise TypeError(
            f"shape_fn for {name!r} must be callable, got "
            f"{type(shape_fn).__name__}")
    INJECTION_MODELS[name] = shape_fn


def get_injection_shape(name: str):
    """Look up a registered shape function, with a helpful error."""
    fn = INJECTION_MODELS.get(name)
    if fn is None:
        raise ValueError(
            f"unknown injection_model {name!r}; choose from "
            f"{sorted(INJECTION_MODELS)} or add one with "
            f"viper.injection.register_injection_model")
    return fn


def _gl_nodes_lnE(lo: np.ndarray, hi: np.ndarray, n_nodes: int):
    """Gauss-Legendre nodes/weights for integration in ln E.

    Returns ``E`` and ``w`` shaped ``(len(lo), n_nodes)`` such that
    ``sum(w * g(E) * E, axis=1)`` approximates ``int g(E) dE``; the
    extra factor E is the ``dE = E dlnE`` Jacobian.
    """
    x, w = np.polynomial.legendre.leggauss(n_nodes)
    ln_lo = np.log(lo)[:, None]
    ln_hi = np.log(hi)[:, None]
    half = 0.5 * (ln_hi - ln_lo)
    e_nodes = np.exp(0.5 * (ln_lo + ln_hi) + half * x[None, :])
    return e_nodes, half * w[None, :]


def injection_bin_counts(
    name: str,
    grid,
    e_min: float,
    e_max: float,
    e_break: float,
    p1: float,
    p2: float,
) -> np.ndarray:
    """Per-bin particle counts injected per unit energy_40, for shape *name*.

    Normalized so the total injected energy over ``[e_min, e_max]`` is
    exactly 1 x 10^40 erg. The result depends only on the grid and the
    spectral parameters, so callers cache it and scale by the per-step
    injected energy:

        grid.number += energy_40 * counts

    Parameters
    ----------
    name : str
        Key in :data:`INJECTION_MODELS`.
    grid : SpectralGrid
        Electron grid whose ``bin_min`` / ``bin_max`` define the bins.
    e_min, e_max : float
        Injection support [GeV] (hard truncation).
    e_break, p1, p2 : float
        Spectral parameters, interpreted by the shape.

    Returns
    -------
    ndarray
        Counts per bin (same units as ``grid.number``) for
        ``energy_40 = 1``; zero outside the support.
    """
    shape_fn = get_injection_shape(name)
    if not (e_min > 0):
        raise ValueError(f"e_min must be > 0, got {e_min}")
    if not (e_max > e_min):
        raise ValueError(
            f"e_max must be > e_min, got e_max={e_max} <= e_min={e_min}")

    def _integrate(lo, hi, weight_extra_E=False):
        """Per-interval integral of f dE (or E f dE), split at e_break.

        Piecewise split keeps kinked shapes at full quadrature accuracy.
        """
        pieces = [(lo, hi)]
        if e_break > 0:
            strad = (lo < e_break) & (hi > e_break)
            if np.any(strad):
                lo1, hi1 = lo.copy(), hi.copy()
                hi1[strad] = e_break
                lo2 = np.full_like(lo, np.nan)
                hi2 = np.full_like(hi, np.nan)
                lo2[strad], hi2[strad] = e_break, hi[strad]
                pieces = [(lo1, hi1), (lo2, hi2)]
        total = np.zeros(len(lo))
        for plo, phi in pieces:
            valid = np.isfinite(plo) & (phi > plo)
            if not np.any(valid):
                continue
            e_nodes, w = _gl_nodes_lnE(plo[valid], phi[valid], N_NODES)
            f_nodes = np.asarray(
                shape_fn(e_nodes, e_break, p1, p2), dtype=float)
            if not np.all(np.isfinite(f_nodes)) or np.any(f_nodes < 0):
                raise ValueError(
                    f"injection model {name!r} returned negative or "
                    f"non-finite dN/dE on [{e_min:g}, {e_max:g}] GeV")
            if weight_extra_E:
                f_nodes = f_nodes * e_nodes
            total[valid] += np.sum(w * f_nodes * e_nodes, axis=1)
        return total

    n_panels = max(1, int(np.ceil(
        np.log10(e_max / e_min) * NORM_PANELS_PER_DECADE)))
    edges = np.exp(np.linspace(np.log(e_min), np.log(e_max), n_panels + 1))
    energy_int = float(np.sum(
        _integrate(edges[:-1], edges[1:], weight_extra_E=True)))
    if not (energy_int > 0 and np.isfinite(energy_int)):
        raise ValueError(
            f"injection model {name!r} integrates to zero or non-finite "
            f"energy on [{e_min:g}, {e_max:g}] GeV")

    lo = np.maximum(grid.bin_min, e_min)
    hi = np.minimum(grid.bin_max, e_max)
    counts = np.zeros(grid.n_bins)
    overlap = hi > lo
    if np.any(overlap):
        counts[overlap] = _integrate(lo[overlap], hi[overlap])

    # grid.number in 10^40-particle units (analytic injectors' convention)
    return counts * (C.GeV_per_erg / energy_int)
