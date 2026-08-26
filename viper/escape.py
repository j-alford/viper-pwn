"""The Gelfand, Slane & Zhang (2009) model evolves the electron spectrum with
adiabatic + synchrotron + IC losses only; particles are otherwise confined.
This module supplies an *energy-dependent escape timescale* ``tau_esc(E)`` for
an optional one-zone escape sink ``-N(E)/tau_esc(E)``.

Escape is selected by a string name from :data:`ESCAPE_MODELS`; ``'off'``.
Bohm (``index = 1``), Kraichnan (1/2), Kolmogorov (1/3) and
energy-independent (0) are special cases of the continuous forms
(``'powerlaw'``, ``'diffusion'``).

``norm`` is ALWAYS a dimensionless O(1) multiplier on a strategy-specific fixed
reference scale. ``index`` is the energy slope (interpreted per strategy;
ignored by the fixed-form models).

    tau_esc(E_gev, B_g, R_cm, v_cms, t_yr, norm, index) -> seconds
"""

from __future__ import annotations

import numpy as np

from . import constants as C

# --- Fixed reference scales (documented; not fitted) -----------------
E_REF_GEV = 1.0e3            # 1 TeV reference energy [GeV]
B_REF_G = 1.0e-5            # 10 uG reference field [Gauss]
R_REF_CM = C.pc             # 1 pc reference radius [cm]
D_REF = 1.0e27             # diffusion-coefficient reference [cm^2/s]
TAU_REF = 1.0e4 * C.yr        # timescale reference [s]  (1e4 yr)
KAPPA = 6.0               # tau = R^2/(kappa D)

# m_e c^2 [GeV]
MEC2_GEV = C.mec2_keV * 1.0e-6

_V_FLOOR = 1.0e-30          # cm/s floor


def _tau_off(E_gev, B_g, R_cm, v_cms, t_yr, norm, index):
    """No escape: tau = +inf so the survival factor is exactly 1."""
    return np.full_like(E_gev, np.inf)


def _tau_powerlaw(E_gev, B_g, R_cm, v_cms, t_yr, norm, index):
    """Phenomenological power law: tau = norm.TAU_REF.(R/R_REF)^2.(E/E_REF)^-index.

    Keeps the physical R^2 growth; no explicit B coupling. ``index`` is the
    energy slope ``s`` (any real value).
    """
    return (norm * TAU_REF * (R_cm / R_REF_CM) ** 2
            * (E_gev / E_REF_GEV) ** (-index))


def _tau_diffusion(E_gev, B_g, R_cm, v_cms, t_yr, norm, index):
    """Diffusion form tau = R^2/(KAPPA.D), D = norm.D_REF.(E/E_REF)^index.(B/B_REF)^-1.

    ``index`` is the diffusion energy index ``delta`` (delta = 1 -> Bohm).
    """
    D = norm * D_REF * (E_gev / E_REF_GEV) ** index * (B_g / B_REF_G) ** (-1.0)
    return R_cm ** 2 / (KAPPA * np.maximum(D, 1e-300))


def _tau_bohm(E_gev, B_g, R_cm, v_cms, t_yr, norm, index):
    """Literal TIDE Bohm escape (fixed delta=1): tau = e B R^2 / (2 gamma m_e c^3).

    ``norm`` is an efficiency multiplier (norm = 1 -> pure Bohm; larger ->
    faster escape). ``index`` is ignored.
    """
    gamma = E_gev / MEC2_GEV
    tau = ((C.e_charge * B_g * R_cm ** 2)
           / (2.0 * gamma * C.m_electron * C.c_light ** 3))
    return tau / norm


def _tau_constant(E_gev, B_g, R_cm, v_cms, t_yr, norm, index):
    """Energy-independent escape time tau = norm.TAU_REF. ``index`` ignored."""
    return np.full_like(E_gev, norm * TAU_REF)


def _tau_advective(E_gev, B_g, R_cm, v_cms, t_yr, norm, index):
    """Advective crossing time tau = norm.R/|v|. Energy-independent."""
    return np.full_like(E_gev, norm * R_cm / max(abs(v_cms), _V_FLOOR))


ESCAPE_MODELS = {
    'off': _tau_off,
    'powerlaw': _tau_powerlaw,
    'diffusion': _tau_diffusion,
    'bohm': _tau_bohm,
    'constant': _tau_constant,
    'advective': _tau_advective,
}


def escape_tau_seconds(name, E_gev, B_g, R_cm, v_cms, t_yr, norm, index):
    """Escape timescale [s] for strategy *name*, evaluated over energies *E_gev*.

    Parameters
    ----------
    name : str
        Strategy key in :data:`ESCAPE_MODELS`.
    E_gev : array_like
        Electron energies [GeV].
    B_g : float
        PWN magnetic field [Gauss].
    R_cm : float
        PWN radius [cm].
    v_cms : float
        PWN expansion velocity [cm/s] (used by ``'advective'``).
    norm : float
        Dimensionless normalization multiplier.
    index : float
        Energy index (ignored by fixed-form models).

    Returns
    -------
    ndarray
        Escape timescale [s], shaped like ``E_gev``.
    """
    fn = ESCAPE_MODELS.get(name)
    if fn is None:
        raise ValueError(
            f"unknown escape_model {name!r}; choose from "
            f"{sorted(ESCAPE_MODELS)}"
        )
    return fn(np.asarray(E_gev, dtype=float), B_g, R_cm, v_cms, t_yr,
              norm, index)
