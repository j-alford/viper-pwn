"""
Supernova remnant dynamics following Truelove & McKee (1999).

Provides functions for the radius, velocity, reverse shock position,
and pressure profile of an SNR through the free-expansion and
Sedov-Taylor evolutionary phases.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from . import constants as C


def calc_vt(esn_51: float, mej: float) -> float:
    """Transition velocity v_t of the SNR ejecta [cm/s].

    Parameters
    ----------
    esn_51 : float
        Supernova energy in units of 10^51 erg.
    mej : float
        Ejecta mass in solar masses.
    """
    m = C.m_ejecta_index
    vt_const = (10.0 * m - 50.0) / (3.0 * m - 9.0)
    log_vt = 0.5 * (
        np.log10(vt_const) + 51.0 + np.log10(esn_51)
        - np.log10(mej) - np.log10(C.M_sun)
    )
    return 10.0**log_vt


def calc_a(esn_51: float, vt: float) -> float:
    """Ejecta density normalization A [g cm^{-3} (cm/s)^m].

    Parameters
    ----------
    esn_51 : float
        Supernova energy [10^51 erg].
    vt : float
        Transition velocity [cm/s].
    """
    m = C.m_ejecta_index
    a_const = (5.0 * m - 25.0) / (2.0 * np.pi * m)
    log_a = np.log10(a_const) + 51.0 + np.log10(esn_51) - 5.0 * np.log10(vt)
    return 10.0**log_a


def calc_characteristic_scales(
    esn_51: float, mej: float, n_ism: float
) -> tuple[float, float, float]:
    """Return (r_ch [cm], t_ch [s], v_ch [cm/s]) characteristic scales."""
    rho_ism = n_ism * C.m_proton * C.mu_ism
    r_ch = (mej * C.M_sun) ** (1.0 / 3.0) * rho_ism ** (-1.0 / 3.0)
    t_ch = (
        esn_51 ** (-0.5) * 10.0 ** (-51.0 / 2.0)
        * (mej * C.M_sun) ** (5.0 / 6.0)
        * rho_ism ** (-1.0 / 3.0)
    )
    v_ch = r_ch / t_ch
    return r_ch, t_ch, v_ch


def calc_trad(esn_51: float, n_ism: float) -> float:
    """Time when the SNR becomes radiative [years]."""
    return 2.9e4 * esn_51 ** (4.0 / 17.0) * n_ism ** (-9.0 / 17.0)


def calc_rad_snr(time_yr: float, t_ch: float, r_ch: float) -> float:
    """SNR forward shock radius [cm].

    Parameters
    ----------
    time_yr : float
        Current time [years].
    t_ch : float
        Characteristic time [s].
    r_ch : float
        Characteristic radius [cm].
    """
    m = C.m_ejecta_index
    t_star = time_yr * C.yr / t_ch
    if t_star < C.tst_star:
        rad_const = (
            27.0 / (4.0 * np.pi)
            * 1.0 / (m * (m - 3.0))
            * C.l_ed ** (m - 2.0)
            / C.phi_ed
            * (10.0 / 3.0 * (m - 5.0) / (m - 3.0)) ** ((m - 3.0) / 2.0)
        )
        rad_const = rad_const ** (1.0 / m)
        r_star = rad_const * t_star ** ((m - 3.0) / m)
    else:
        inner = C.rst_star**2.5 + C.psi_0**0.5 * (t_star - C.tst_star)
        r_star = inner**0.4
    return r_star * r_ch


def calc_vel_snr(time_yr: float, t_ch: float, v_ch: float) -> float:
    """SNR forward shock velocity [cm/s].
    """
    m = C.m_ejecta_index
    t_star = time_yr * C.yr / t_ch
    if t_star < C.tst_star:
        vel_const = (
            27.0 / (4.0 * np.pi)
            * 1.0 / (m * (m - 3.0))
            * C.l_ed ** (m - 2.0)
            / C.phi_ed
            * (10.0 / 3.0 * (m - 5.0) / (m - 3.0)) ** ((m - 3.0) / 2.0)
        )
        v_star = (m - 3.0) / m * vel_const ** (1.0 / m) * t_star ** (-3.0 / m)
    else:
        inner = C.rst_star**2.5 + C.psi_0**0.5 * (t_star - C.tst_star)
        v_star = 0.4 * C.psi_0**0.5 * inner ** (-0.6)
    return v_star * v_ch


def calc_rad_rs(
    time_yr: float, t_ch: float, r_ch: float, rad_snr: float
) -> float:
    """Reverse shock radius [cm].
    """
    t_star = time_yr * C.yr / t_ch
    if t_star < C.tcore_star:
        return rad_snr / C.l_ed
    else:
        r_star = t_star * (
            C.rrscore_star / C.tcore_star
            - C.arscore_star * (t_star - C.tcore_star)
            - (C.vrscore_star - C.arscore_star * C.tcore_star)
            * np.log(t_star / C.tcore_star)
        )
        return r_star * r_ch


def calc_prs(
    time_yr: float,
    t_ch: float,
    v_ch: float,
    a: float,
    vel_snr: float,
    p_snr: float,
    n_ism: float,
) -> float:
    """Pressure just behind the reverse shock [dyne cm^-2].
    """
    t_star = time_yr * C.yr / t_ch
    if t_star < C.tcore_star:
        return C.phi_ed * p_snr
    else:
        v_rs_star = C.vrscore_star + C.arscore_star * (t_star - C.tcore_star)
        v_rs = v_rs_star * v_ch
        log_rho_ej_rs = np.log10(a) - 3.0 * np.log10(time_yr * C.yr)
        log_phi = (
            log_rho_ej_rs
            + 2.0 * np.log10(v_rs)
            - np.log10(n_ism)
            - np.log10(C.m_proton)
            - 2.0 * np.log10(vel_snr)
        )
        log_prs = log_phi + np.log10(p_snr)
        return 10.0**log_prs


def calc_mass_env(rho_core: float, rad_core: float, rad_pwn: float) -> float:
    """Mass of the SNR envelope between rad_core and rad_pwn [g].
    """
    m = C.m_ejecta_index
    log_term1 = np.log10(4.0 * np.pi) + np.log10(rho_core) + m * np.log10(rad_core)
    term1 = 10.0**log_term1
    log_term2 = (3.0 - m) * np.log10(rad_pwn)
    term2 = 10.0**log_term2
    log_term3 = (3.0 - m) * np.log10(rad_core)
    term3 = 10.0**log_term3
    return term1 * (term2 - term3) / (3.0 - m)


def calc_snr_interior_pressure(
    psi: float,
    p_snr: float,
    vel_snr: float,
    n_ism: float,
    optimize_vtilda: bool = False,
) -> tuple[float, float, float, float]:
    """Compute pressure and density inside the SNR at position psi = R_pwn/R_snr.

    Uses the Bandiera (1984) self-similar solution for a Sedov-Taylor SNR.

    Parameters
    ----------
    optimize_vtilda : bool
        If True, use scipy.optimize.minimize_scalar for vtilda instead
        of the 25-point grid search.

    Returns
    -------
    p_snr_pwn : float
        Pressure at the PWN boundary [dyne cm^-2].
    rho_ej_pwn : float
        Density at the PWN boundary [g cm^-3].
    vel_ej_pwn : float
        Velocity of ejecta at the PWN boundary [cm/s].
    best_vtilda : float
        Best-fit dimensionless velocity parameter.
    """
    gamma = C.gamma_snr
    smla = -1.0 / (-2.0 * gamma - 1.0)
    smlb = (gamma + 1.0) / (3.0 * gamma - 1.0)
    smlc = gamma / (3.0 * (gamma - 1.0) - 3.0)

    min_vtilda = (gamma + 1.0) / (2.0 * gamma)
    max_vtilda = 1.0

    if optimize_vtilda:
        def _psi_residual(vtilda):
            bigd = (
                2.0 * (3.0 * gamma - 1.0) / (gamma - 7.0)
                * (vtilda - (gamma + 1.0) / 2.0 * 5.0 / (3.0 * gamma - 1.0))
            )
            bigb = (
                2.0 * gamma / (gamma - 1.0)
                * (vtilda - (gamma + 1.0) / (2.0 * gamma))
                / bigd
            )
            biga = vtilda / bigd
            psi_val = (biga ** (-0.4)
                       * bigb ** (smla * (gamma - 1.0))
                       * bigd ** (-smlb))
            return (psi_val - psi) ** 2

        res = minimize_scalar(
            _psi_residual, bounds=(min_vtilda, max_vtilda), method='bounded',
        )
        best_vtilda = res.x
    else:
        # original 25-point grid search
        n_vtilda = 25
        del_vtilda = (max_vtilda - min_vtilda) / n_vtilda

        best_psi_diff = psi * 10.0
        best_vtilda = min_vtilda

        for i in range(n_vtilda):
            vtilda = min_vtilda + i * del_vtilda
            bigd = (
                2.0 * (3.0 * gamma - 1.0) / (gamma - 7.0)
                * (vtilda - (gamma + 1.0) / 2.0 * 5.0 / (3.0 * gamma - 1.0))
            )
            bigb = (
                2.0 * gamma / (gamma - 1.0)
                * (vtilda - (gamma + 1.0) / (2.0 * gamma))
                / bigd
            )
            biga = vtilda / bigd
            psi_val = (biga ** (-0.4)
                       * bigb ** (smla * (gamma - 1.0))
                       * bigd ** (-smlb))
            psi_diff = abs(psi_val - psi)
            if psi_diff < best_psi_diff:
                best_psi_diff = psi_diff
                best_vtilda = vtilda

    bigd = (
        2.0 * (3.0 * gamma - 1.0) / (gamma - 7.0)
        * (best_vtilda - (gamma + 1.0) / 2.0 * 5.0 / (3.0 * gamma - 1.0))
    )
    bigc = (
        2.0 / (gamma - 1.0) * ((gamma + 1.0) / 2.0 - best_vtilda) / bigd
    )
    bigb = (
        2.0 * gamma / (gamma - 1.0)
        * (best_vtilda - (gamma + 1.0) / (2.0 * gamma))
        / bigd
    )
    biga = best_vtilda / bigd

    phi = biga**1.2 * bigc ** (3.0 * smlc) * bigd ** (1.0 + smlb)
    p_snr_pwn = phi * p_snr

    rho_ej_pwn = (
        (gamma + 1.0) / (gamma - 1.0)
        * n_ism
        * bigb ** (1.0 - 2.0 * (gamma - 1.0) * smla)
        * bigc ** (3.0 * smlc - 1.0)
        * bigd ** (3.0 * smlb - 1.0)
        * C.m_proton
    )

    vel_ej_pwn = best_vtilda * psi * 2.0 / (gamma + 1.0) * vel_snr

    return p_snr_pwn, rho_ej_pwn, vel_ej_pwn, best_vtilda
