"""
Pulsar spin-down physics.

Energy injection into the PWN as a function of time.
"""

from __future__ import annotations

import numpy as np

from . import constants as C


def initial_pwn_energy(
    time_yr: float, spindown_index: float, esd_40: float, tau_sd: float
) -> float:
    """Cumulative energy deposited by the pulsar into the PWN up to *time_yr*.

    Parameters
    ----------
    time_yr : float
        Time since birth [years].
    spindown_index : float
        n = (p+1)/(p-1) where p is the braking index.
    esd_40 : float
        Total rotational energy E_sd in units of 10^40 erg.
    tau_sd : float
        Spin-down timescale [years].

    Returns
    -------
    float
        Energy in units of 10^40 erg.
    """
    x = time_yr / tau_sd
    if spindown_index == 2:
        epsilon = np.log(1.0 + x) / x - 1.0 / (x + 1.0)
    else:
        n = spindown_index
        epsilon = (
            (1.0 + x) ** (1.0 - n) / (1.0 - n)
            - (1.0 + x) ** (2.0 - n) / (x * (1.0 - n) * (2.0 - n))
            + 1.0 / (x * (1.0 - n) * (2.0 - n))
        )
    return esd_40 * epsilon


def true_age_yr(
    period: float, p_dot: float, braking_index: float, tau_sd: float
) -> float:
    """True pulsar age [years] from timing parameters.

    Deliberately uses the 365-day year (365*86400 s), not ``C.yr``,
    matching the original convention; do not "fix" it.
    """
    tau_char = period / (2 * p_dot) / (365.0 * 86400.0)
    return (2 * tau_char / (braking_index - 1)) - tau_sd


def tau_sd_from_age_yr(
    period: float, p_dot: float, braking_index: float, age: float
) -> float:
    """Spin-down timescale [years] that gives a known true age.

    Inverse of :func:`true_age_yr`.
    """
    tau_char = period / (2 * p_dot) / (365.0 * 86400.0)
    return (2 * tau_char / (braking_index - 1)) - age


def e0_37_from_timing(
    period: float, p_dot: float, braking_index: float, tau_sd: float
) -> float:
    """Initial spin-down luminosity [10^37 erg/s] from timing parameters."""
    e_dot_current = 4.0 * np.pi**2 * C.I_ns * p_dot / period**3
    n_idx = (braking_index + 1) / (braking_index - 1)
    age = true_age_yr(period, p_dot, braking_index, tau_sd)
    return e_dot_current * (1 + age / tau_sd) ** n_idx / 1.0e37
