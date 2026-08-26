"""Radiation physics: synchrotron and inverse-Compton emission."""

from __future__ import annotations

import math
import logging

import numpy as np
from scipy.special import kv as bessel_kv

from . import constants as C
from .grids import SpectralGrid, LOG_EMPTY, LOG_EMPTY_CUT

logger = logging.getLogger(__name__)

# --- Optional Numba import with graceful fallback ---
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        """No-op decorator: runs function as plain Python."""
        if args and callable(args[0]):
            return args[0]
        return lambda f: f
    prange = range

# --- Synchrotron F(x) lookup table ---

def _build_fx_table(n_points: int = 2000):
    """Build a lookup table for the synchrotron function F(x).

    F(x) = x ∫_x^∞ K_{5/3}(ξ) dξ, via trapezoidal summation of K_{5/3}
    on a fine log grid.
    """
    x_vals = np.logspace(-6, np.log10(C.max_synch_x), n_points)
    chi_max = 3.0 * C.max_synch_x
    n_fine = int(2000 * np.log10(chi_max / x_vals[0])) + 1
    fine = np.logspace(np.log10(x_vals[0]), np.log10(chi_max), n_fine)
    master = np.union1d(x_vals, fine)     # keeps the x nodes exactly
    kvals = bessel_kv(5.0 / 3.0, master)
    seg = 0.5 * (kvals[1:] + kvals[:-1]) * np.diff(master)
    # tail[i] = ∫_{master[i]}^{chi_max} K_{5/3}
    tail = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
    fx_vals = x_vals * tail[np.searchsorted(master, x_vals)]
    return np.log10(x_vals), fx_vals


# Lazy build keeps import fast.
_FX_TABLE_X = None
_FX_TABLE_Y = None


def _get_fx_table():
    """Return (log_x, F(x)) lookup arrays, building them on first use."""
    global _FX_TABLE_X, _FX_TABLE_Y
    if _FX_TABLE_X is None:
        _FX_TABLE_X, _FX_TABLE_Y = _build_fx_table()
    return _FX_TABLE_X, _FX_TABLE_Y


def calc_fx(x: float) -> float:
    """Synchrotron function F(x) via lookup table interpolation."""
    if x <= 0 or x > C.max_synch_x:
        return 0.0
    table_x, table_y = _get_fx_table()
    return float(np.interp(np.log10(x), table_x, table_y))


def calc_fx_array(x: np.ndarray) -> np.ndarray:
    """Vectorized F(x) via lookup table."""
    result = np.zeros_like(x)
    mask = (x > 0) & (x <= C.max_synch_x)
    if np.any(mask):
        table_x, table_y = _get_fx_table()
        result[mask] = np.interp(np.log10(x[mask]), table_x, table_y)
    return result


# --- Synchrotron helpers ---

def calc_synch_freq(log_energy: float, bpwn: float) -> float:
    """Critical synchrotron frequency [Hz].
    """
    if bpwn <= 0:
        return 0.0
    log4pi = np.log10(4.0 * np.pi)
    log_freq = (
        np.log10(0.29) + 0.5 * C.log_3 + C.log_e_charge
        - log4pi - 3.0 * C.log_m_electron - 5.0 * C.log_c
        + np.log10(bpwn) + 2.0 * (log_energy - C.log_erg) + 0.5 * C.log_two
    )
    return 10.0 ** log_freq


def csynch_calc(log_energy: float, bpwn: float, n_elec_40: float,
                log_rad: float) -> float:
    """Synchrotron cooling coefficient [1/s/erg]. Returns 0 if optically thick.
    """
    log_tau_const = np.log10(4.2e7)
    nu_crit = calc_synch_freq(log_energy, bpwn)
    if nu_crit <= 0:
        return 0.0
    log_nu_crit = np.log10(nu_crit)
    log_n_los = (C.log_3 + np.log10(max(n_elec_40, 1e-300)) + 40.0
                 - np.log10(8.0 * np.pi) - 2.0 * log_rad)
    log_tau = (log_tau_const + 1.5 * np.log10(bpwn) - 2.5 * log_nu_crit
               + log_n_los + np.log10(1.1) + 0.75 * (C.log_two - C.log_3))
    if log_tau < 0:
        log_csynch = (2.0 * C.log_two + 4.0 * C.log_e_charge
                      - 2.0 * C.log_3 - 4.0 * C.log_m_electron
                      - 7.0 * C.log_c + 2.0 * np.log10(bpwn))
        return 10.0 ** log_csynch
    return 0.0


def bpwn_calc(mag_energy_40: float, log_volume: float) -> float:
    """PWN magnetic field [Gauss] from magnetic energy and volume.
    """
    log_bpwn = 0.5 * (np.log10(8.0 * np.pi) + np.log10(max(mag_energy_40, 1e-300))
                       + 40.0 - log_volume)
    return 10.0 ** log_bpwn


# --- Vectorized synchrotron cooling + energy update ---

_LOG_TAU_CONST = np.log10(4.2e7)
_LOG_8PI = np.log10(8.0 * np.pi)
_LOG_1P1 = np.log10(1.1)
_LOG_SYNCH_FREQ_OFFSET = (
    np.log10(0.29) + 0.5 * C.log_3 + C.log_e_charge
    - np.log10(4.0 * np.pi) - 3.0 * C.log_m_electron - 5.0 * C.log_c
    + 0.5 * C.log_two
)
_LOG_CSYNCH_VAL = (
    2.0 * C.log_two + 4.0 * C.log_e_charge
    - 2.0 * C.log_3 - 4.0 * C.log_m_electron - 7.0 * C.log_c
)


def csynch_calc_vec(log_energies: np.ndarray, bpwn: float,
                    n_elec_40: float, log_rad: float) -> np.ndarray:
    """Vectorized synchrotron cooling coefficient for an array of log10(energies).

    Returns array of cooling coefficients [1/s/erg], with 0.0 where
    optically thick or B-field is negligible.
    """
    n = len(log_energies)
    result = np.zeros(n)
    if bpwn <= 0:
        return result

    log_bpwn = np.log10(bpwn)

    log_nu_crit = (
        _LOG_SYNCH_FREQ_OFFSET + log_bpwn
        + 2.0 * (log_energies - C.log_erg)
    )

    log_n_los = (C.log_3 + np.log10(max(n_elec_40, 1e-300)) + 40.0
                 - _LOG_8PI - 2.0 * log_rad)
    log_tau = (
        _LOG_TAU_CONST + 1.5 * log_bpwn - 2.5 * log_nu_crit
        + log_n_los + _LOG_1P1 + 0.75 * (C.log_two - C.log_3)
    )

    thin = log_tau < 0
    if np.any(thin):
        result[thin] = 10.0 ** (_LOG_CSYNCH_VAL + 2.0 * log_bpwn)

    return result


def new_energy_calc_vec(energies: np.ndarray, log_c_rad: np.ndarray,
                        log_rad: float, log_old_rad: float,
                        ln_rad_rat: float, log_vel: float,
                        vel_neg: bool) -> np.ndarray:
    """Vectorized new particle energy after radiative + adiabatic losses.

    Parameters
    ----------
    energies : array of particle energies [erg-like]
    log_c_rad : array of log10(cooling coefficient) for each bin
    Others : scalars, same as the scalar version.

    Returns array of new energies.
    """
    rad_rat = np.exp(-ln_rad_rat)
    abs_e_max = rad_rat * energies

    local_ln = -ln_rad_rat if vel_neg else ln_rad_rat
    if local_ln <= 0:
        return np.clip(abs_e_max, 0.0, None)

    log_energies = np.log10(energies)

    log_term1 = log_vel + log_old_rad + log_energies - C.log_erg
    term1 = 10.0 ** log_term1

    term3 = 10.0 ** (log_rad + log_vel)

    log_local_ln = np.log10(local_ln)
    log_term2 = (log_c_rad + log_rad + log_local_ln
                 + log_old_rad + log_energies - C.log_erg)
    term2 = 10.0 ** log_term2

    denom = term2 + term3

    new_e = np.where(
        denom > 0,
        np.minimum(term1 / np.maximum(denom, 1e-300) * C.GeV_per_erg, abs_e_max),
        abs_e_max,
    )
    return np.clip(new_e, 0.0, None)


# --- Inverse Compton: fused cross-section + power computation ---

def calc_pow_ic(phot_grid: SpectralGrid,
                pow_nu_ic: np.ndarray) -> np.ndarray:
    """Total IC power P(E) for each electron energy bin [erg/s]. Vectorized."""
    # valid mask: (n_elec, n_phot)
    valid = pow_nu_ic > LOG_EMPTY
    log_terms = pow_nu_ic + phot_grid.log_width[np.newaxis, :]
    safe = np.where(valid, 10.0 ** log_terms, 0.0)
    return safe.sum(axis=1)


def calc_sigma_and_pow_nu_ic(
    elec_grid: SpectralGrid, phot_grid: SpectralGrid,
    target_grid: SpectralGrid,
    ic_early_break: bool = False,
    ic_j_break: bool = False,
    ic_skip_empty_bins: bool = True,
) -> np.ndarray:
    """Compute pow_nu_ic[i, j] directly, fusing cross-section and summation.

    Returns pow_nu_ic array of shape (ne, np).
    """
    if HAS_NUMBA:
        return _calc_sigma_and_pow_nu_ic_numba(elec_grid, phot_grid, target_grid,
                                               ic_early_break=ic_early_break,
                                               ic_j_break=ic_j_break,
                                               ic_skip_empty_bins=ic_skip_empty_bins)
    else:
        return _calc_sigma_and_pow_nu_ic_numpy(
            elec_grid, phot_grid, target_grid,
            ic_skip_empty_bins=ic_skip_empty_bins,
        )


# --- Backend 1: Numba JIT kernel ---

@njit(cache=True, parallel=True)
def _ic_kernel(
    elec_log_avg,       # (ne,)  log10(bin center energy) [keV]
    phot_log_avg,       # (np,)  log10(bin center freq) [Hz]
    phot_bin_avg,       # (np,)  bin center freq [Hz]
    target_log_avg,     # (nt,)  log10(bin center freq) [Hz]
    target_bin_avg,     # (nt,)  bin center freq [Hz]
    target_log_number,  # (nt,)  log10(photon number density) [cm^-3 Hz^-1]
    target_log_width,   # (nt,)  log10(bin width) [Hz]
    ne, np_, nt,
    log_mec2, log_hev, log_sigT, log_075, prefactor,
    ic_early_break,     # break k at kinematic cutoff
    ic_j_break,         # break j at electron limit
    ic_skip_empty_bins, # skip empty target bins
):
    """Compute IC power spectrum via scalar triple loop (Numba JIT target).

        P_ν(i, j) = c · h · ν_j · Σ_k  (dσ/dν_t)(i,j,k) · n_target(k) · Δν(k)

    with dσ/dν_t the Klein-Nishina differential cross-section
    (Blumenthal & Gould 1970).
    Parameters are all plain arrays/scalars for Numba compatibility.
    Returns (ne, np_) array of log10(P_ν) values, LOG_EMPTY where no IC.
    """
    result = np.full((ne, np_), LOG_EMPTY)

    # ε = E[keV] / 511 keV
    log_epsilon = np.empty(ne)
    epsilon = np.empty(ne)
    for i in range(ne):
        log_epsilon[i] = elec_log_avg[i] - log_mec2 + 6.0
        epsilon[i] = 10.0 ** log_epsilon[i]

    # hν_s / (m_e c²)
    log_hnus_mc2 = np.empty(np_)
    hnus_mc2 = np.empty(np_)
    for j in range(np_):
        log_hnus_mc2[j] = log_hev + phot_log_avg[j] - 3.0 - log_mec2
        hnus_mc2[j] = 10.0 ** log_hnus_mc2[j]

    # log10(n(ν) · Δν)
    target_contrib = np.empty(nt)
    for k in range(nt):
        target_contrib[k] = target_log_number[k] + target_log_width[k]

    # Γ_base = 4 hν_t / (m_e c²)
    log4 = math.log10(4.0)
    log_gamma_base = np.empty(nt)
    for k in range(nt):
        log_gamma_base[k] = log4 + log_hev + target_log_avg[k] - 3.0 - log_mec2

    # prange: iterations independent, race-free
    for i in prange(1, ne):
        if log_epsilon[i] < log_hnus_mc2[1]:
            continue

        eps_i = epsilon[i]
        log_eps_i = log_epsilon[i]
        log_term2_base = log_sigT - 2.0 * log_eps_i

        for j in range(1, np_):
            # sorted grid: remaining j all fail
            if eps_i <= hnus_mc2[j]:
                if ic_j_break:
                    break
                continue

            # gamma_q = hν_s / (ε - hν_s) in m_e c² units
            gamma_q = hnus_mc2[j] / (eps_i - hnus_mc2[j])
            log_gamma_q = math.log10(gamma_q)

            total = 0.0

            for k in range(1, nt - 1):
                if ic_skip_empty_bins and target_log_number[k] < LOG_EMPTY_CUT:
                    continue

                # target must be lower frequency
                if target_bin_avg[k] >= phot_bin_avg[j]:
                    if ic_early_break:
                        break  # sorted grid: remaining k fail
                    continue

                log_gamma_i_k = log_gamma_base[k] + log_eps_i
                log_q = log_gamma_q - log_gamma_i_k
                q = 10.0 ** log_q

                if q > 1.0 or q <= 0.0:
                    continue

                # Klein-Nishina combination factor
                combo = (2.0 * q * math.log(q)
                         + (1.0 - q) * (1.0 + 2.0 * q
                            + 0.5 * gamma_q * gamma_q / (1.0 + gamma_q)))

                if combo <= 0.0:
                    continue

                log_sigma = (log_075 + log_term2_base
                             - target_log_avg[k] + math.log10(combo))
                total += 10.0 ** (log_sigma + target_contrib[k])

            if total > 0.0:
                result[i, j] = prefactor + phot_log_avg[j] + math.log10(total)

    return result


def _calc_sigma_and_pow_nu_ic_numba(
    elec_grid: SpectralGrid, phot_grid: SpectralGrid,
    target_grid: SpectralGrid,
    ic_early_break: bool = False,
    ic_j_break: bool = False,
    ic_skip_empty_bins: bool = True,
) -> np.ndarray:
    """Wrapper: extract arrays from grid objects, call _ic_kernel."""
    return _ic_kernel(
        elec_grid.log_avg,
        phot_grid.log_avg, phot_grid.bin_avg,
        target_grid.log_avg, target_grid.bin_avg,
        target_grid.log_number, target_grid.log_width,
        elec_grid.n_bins, phot_grid.n_bins, target_grid.n_bins,
        math.log10(C.mec2_keV), math.log10(C.h_eV),
        math.log10(C.sigma_T), math.log10(0.75),
        C.log_c + C.log_h_cgs,
        ic_early_break,
        ic_j_break,
        ic_skip_empty_bins,
    )


# --- Backend 2: NumPy fallback ---

def _calc_sigma_and_pow_nu_ic_numpy(
    elec_grid: SpectralGrid, phot_grid: SpectralGrid,
    target_grid: SpectralGrid,
    ic_skip_empty_bins: bool = True,
) -> np.ndarray:
    """NumPy fallback for fused IC computation (no Numba required).

    Results match the Numba kernel.
    """
    ne = elec_grid.n_bins
    np_ = phot_grid.n_bins
    nt = target_grid.n_bins
    result = np.full((ne, np_), LOG_EMPTY)

    log_mec2 = np.log10(C.mec2_keV)
    log_hev = np.log10(C.h_eV)

    log_epsilon = elec_grid.log_avg - log_mec2 + 6.0
    epsilon = 10.0 ** log_epsilon

    log_hnus_mc2 = log_hev + phot_grid.log_avg - 3.0 - log_mec2
    hnus_mc2 = 10.0 ** log_hnus_mc2

    k_range = np.arange(1, nt - 1)

    if ic_skip_empty_bins:
        k_populated = target_grid.log_number[k_range] > LOG_EMPTY_CUT
        k_range = k_range[k_populated]

    nk = len(k_range)
    target_log_k = target_grid.log_avg[k_range]
    target_avg_k = target_grid.bin_avg[k_range]
    phot_avg = phot_grid.bin_avg

    target_contrib = (target_grid.log_number[k_range]
                      + target_grid.log_width[k_range])

    log_gamma_base = np.log10(4.0) + log_hev + target_log_k - 3.0 - log_mec2
    log_075 = np.log10(0.75)
    log_sigT = np.log10(C.sigma_T)
    prefactor = C.log_c + C.log_h_cgs

    nj_max = np_ - 1
    j_indices = np.arange(1, np_)
    hnus_j = hnus_mc2[j_indices]

    k_valid_by_j = target_avg_k[np.newaxis, :] < phot_avg[j_indices][:, np.newaxis]
    log_term2_target = -target_log_k

    # work buffers reused per i
    log_q = np.empty((nj_max, nk))
    q_ic = np.empty((nj_max, nk))
    combo = np.empty((nj_max, nk))
    log_sigma = np.empty((nj_max, nk))
    summands = np.empty((nj_max, nk))

    for i in range(1, ne):
        if log_epsilon[i] < log_hnus_mc2[1]:
            continue

        eps_i = epsilon[i]
        log_eps_i = log_epsilon[i]

        j_mask = eps_i > hnus_j
        if not np.any(j_mask):
            continue

        j_sel = np.where(j_mask)[0]
        nj = len(j_sel)

        gq_1d = hnus_j[j_sel] / (eps_i - hnus_j[j_sel])
        log_gq_1d = np.log10(gq_1d)

        log_gamma_i = log_gamma_base + log_eps_i

        lq = log_q[:nj, :nk]
        np.subtract(log_gq_1d[:, np.newaxis], log_gamma_i[np.newaxis, :], out=lq)
        np.power(10.0, lq, out=q_ic[:nj, :nk])
        qi = q_ic[:nj, :nk]

        valid = k_valid_by_j[j_sel] & (qi <= 1)
        if not np.any(valid):
            continue

        q = np.where(valid, qi, 0.5)
        gq = gq_1d[:, np.newaxis]
        cb = combo[:nj, :nk]
        np.multiply(2.0, q, out=cb)
        cb *= np.log(np.maximum(q, 1e-300))
        cb += (1.0 - q) * (1.0 + 2.0 * q + 0.5 * gq ** 2 / (1.0 + gq))

        valid = valid & (cb > 0)
        if not np.any(valid):
            continue

        log_term2_i = log_sigT - 2.0 * log_eps_i + log_term2_target
        ls = log_sigma[:nj, :nk]
        np.log10(np.maximum(cb, 1e-300), out=ls)
        ls += log_075
        ls += log_term2_i[np.newaxis, :]

        lt = ls + target_contrib[np.newaxis, :]
        sm = summands[:nj, :nk]
        np.power(10.0, lt, out=sm)
        sm[~valid] = 0.0
        totals = sm.sum(axis=1)

        got = totals > 0
        if np.any(got):
            j_idx = j_indices[j_sel[got]]
            result[i, j_idx] = prefactor + phot_grid.log_avg[j_idx] + np.log10(totals[got])

    return result


# --- Synchrotron self-Compton (SSC) ---

# 2.24 uniform-sphere factor (Atoyan & Aharonian 1996; Martin et al. 2012).
SSC_GEOM_FACTOR = 2.24

# true critical frequency: 0.29 peak factor removed
_LOG_NU_CRIT_OFFSET = _LOG_SYNCH_FREQ_OFFSET - np.log10(0.29)

# p_ν = 10**(_LOG_SYNCH_CONST_B0 + log10(B)) · F(ν/ν_crit) [erg/s/Hz]
_LOG_SYNCH_CONST_B0 = (np.log10(2.0 / 3.0 * 3.0 ** 0.5)
                       + 3.0 * C.log_e_charge
                       - C.log_m_electron - 2.0 * C.log_c)


@njit(cache=True, parallel=True)
def _ic_cooling_matrix(
    elec_log_avg,       # (ne,)  log10(bin center energy) [keV]
    phot_log_avg,       # (np,)  log10(bin center freq) [Hz]
    phot_bin_avg,       # (np,)  bin center freq [Hz]
    phot_log_width,     # (np,)  log10(bin width) [Hz]
    target_log_avg,     # (nt,)  log10(bin center freq) [Hz]
    target_bin_avg,     # (nt,)  bin center freq [Hz]
    ne, np_, nt,
    log_mec2, log_hev, log_sigT, log_075, prefactor,
):
    """K2[i, k]: IC cooling power per electron per unit target photon
    column, K2 @ (n·Δν) == calc_pow_ic(kernel(target=n)) [erg/s].

    Same triple loop as _ic_kernel with the j and k roles swapped.
    """
    k2 = np.zeros((ne, nt))

    log_epsilon = np.empty(ne)
    epsilon = np.empty(ne)
    for i in range(ne):
        log_epsilon[i] = elec_log_avg[i] - log_mec2 + 6.0
        epsilon[i] = 10.0 ** log_epsilon[i]

    log_hnus_mc2 = np.empty(np_)
    hnus_mc2 = np.empty(np_)
    for j in range(np_):
        log_hnus_mc2[j] = log_hev + phot_log_avg[j] - 3.0 - log_mec2
        hnus_mc2[j] = 10.0 ** log_hnus_mc2[j]

    log4 = math.log10(4.0)
    log_gamma_base = np.empty(nt)
    for k in range(nt):
        log_gamma_base[k] = log4 + log_hev + target_log_avg[k] - 3.0 - log_mec2

    # log10(c·h·ν_j·Δν_j)
    w_j = np.empty(np_)
    for j in range(np_):
        w_j[j] = prefactor + phot_log_avg[j] + phot_log_width[j]

    for i in prange(1, ne):
        if log_epsilon[i] < log_hnus_mc2[1]:
            continue

        eps_i = epsilon[i]
        log_eps_i = log_epsilon[i]
        log_term2_base = log_sigT - 2.0 * log_eps_i

        for j in range(1, np_):
            if eps_i <= hnus_mc2[j]:
                break  # sorted grid: remaining j fail

            gamma_q = hnus_mc2[j] / (eps_i - hnus_mc2[j])
            log_gamma_q = math.log10(gamma_q)
            base_ij = w_j[j] + log_075 + log_term2_base

            for k in range(1, nt - 1):
                # target must be lower frequency
                if target_bin_avg[k] >= phot_bin_avg[j]:
                    break  # sorted grid: remaining k fail

                log_q = log_gamma_q - (log_gamma_base[k] + log_eps_i)
                q = 10.0 ** log_q

                if q > 1.0 or q <= 0.0:
                    continue

                combo = (2.0 * q * math.log(q)
                         + (1.0 - q) * (1.0 + 2.0 * q
                            + 0.5 * gamma_q * gamma_q / (1.0 + gamma_q)))

                if combo <= 0.0:
                    continue

                k2[i, k] += 10.0 ** (base_ij - target_log_avg[k]
                                     + math.log10(combo))

    return k2


def _ic_cooling_matrix_numpy(elec_grid: SpectralGrid,
                             phot_grid: SpectralGrid,
                             target_grid: SpectralGrid) -> np.ndarray:
    """NumPy fallback for the K2 cooling matrix (no Numba required)."""
    ne = elec_grid.n_bins
    np_ = phot_grid.n_bins
    nt = target_grid.n_bins
    k2 = np.zeros((ne, nt))

    log_mec2 = np.log10(C.mec2_keV)
    log_hev = np.log10(C.h_eV)
    log_075 = np.log10(0.75)
    log_sigT = np.log10(C.sigma_T)
    prefactor = C.log_c + C.log_h_cgs

    log_epsilon = elec_grid.log_avg - log_mec2 + 6.0
    epsilon = 10.0 ** log_epsilon
    log_hnus_mc2 = log_hev + phot_grid.log_avg - 3.0 - log_mec2
    hnus_mc2 = 10.0 ** log_hnus_mc2

    j_idx = np.arange(1, np_)
    k_idx = np.arange(1, nt - 1)
    target_log_k = target_grid.log_avg[k_idx]
    log_gamma_base = np.log10(4.0) + log_hev + target_log_k - 3.0 - log_mec2
    w_j = (prefactor + phot_grid.log_avg[j_idx]
           + phot_grid.log_width[j_idx])
    kinematic = (target_grid.bin_avg[k_idx][np.newaxis, :]
                 < phot_grid.bin_avg[j_idx][:, np.newaxis])

    for i in range(1, ne):
        if log_epsilon[i] < log_hnus_mc2[1]:
            continue
        eps_i = epsilon[i]
        log_eps_i = log_epsilon[i]

        j_mask = eps_i > hnus_mc2[j_idx]
        if not np.any(j_mask):
            continue
        gq = hnus_mc2[j_idx][j_mask] / (eps_i - hnus_mc2[j_idx][j_mask])
        log_gq = np.log10(gq)

        log_q = log_gq[:, np.newaxis] - (log_gamma_base + log_eps_i)
        q = 10.0 ** log_q
        valid = kinematic[j_mask] & (q <= 1.0) & (q > 0.0)

        qs = np.where(valid, q, 0.5)
        gq2 = gq[:, np.newaxis]
        combo = (2.0 * qs * np.log(qs)
                 + (1.0 - qs) * (1.0 + 2.0 * qs
                    + 0.5 * gq2 ** 2 / (1.0 + gq2)))
        valid &= combo > 0

        log_sigma = (log_075 + log_sigT - 2.0 * log_eps_i
                     - target_log_k[np.newaxis, :]
                     + np.log10(np.maximum(combo, 1e-300)))
        terms = 10.0 ** (w_j[j_mask][:, np.newaxis] + log_sigma)
        terms[~valid] = 0.0
        k2[i, k_idx] = terms.sum(axis=0)

    return k2


def ic_cooling_matrix(elec_grid: SpectralGrid, phot_grid: SpectralGrid,
                      target_grid: SpectralGrid) -> np.ndarray:
    """K2 cooling matrix (ne, nt) [erg/s per unit n·Δν]; grids only.

    Only the grid geometry of ``target_grid`` is read; its ``number``
    content is ignored.
    """
    if HAS_NUMBA:
        return _ic_cooling_matrix(
            elec_grid.log_avg,
            phot_grid.log_avg, phot_grid.bin_avg, phot_grid.log_width,
            target_grid.log_avg, target_grid.bin_avg,
            elec_grid.n_bins, phot_grid.n_bins, target_grid.n_bins,
            math.log10(C.mec2_keV), math.log10(C.h_eV),
            math.log10(C.sigma_T), math.log10(0.75),
            C.log_c + C.log_h_cgs,
        )
    return _ic_cooling_matrix_numpy(elec_grid, phot_grid, target_grid)


@njit(cache=True)
def _synch_seed_lum(elec_log_avg, elec_number, log_bpwn, target_log_avg,
                    ne, nt, log_nu_crit_offset, log_erg,
                    table_x0, table_dx, table_n, table_y, table_x_end):
    """Σ_i N_i·F(ν_k/ν_crit,i) at each target bin (F(x)-table part).

    Matches calc_fx_array's semantics: 0 above C.max_synch_x, clamped
    to the first table value below the table range.  Serial on purpose.
    """
    out = np.zeros(nt)
    for i in range(ne):
        if elec_number[i] <= 0.0:
            continue
        log_nu_c = (log_nu_crit_offset + log_bpwn
                    + 2.0 * (elec_log_avg[i] - log_erg))
        n_i = elec_number[i]
        for k in range(nt):
            lx = target_log_avg[k] - log_nu_c
            if lx > table_x_end:
                continue
            if lx <= table_x0:
                fx = table_y[0]
            else:
                pos = (lx - table_x0) / table_dx
                i0 = int(pos)
                if i0 >= table_n - 1:
                    fx = table_y[table_n - 1]
                else:
                    frac = pos - i0
                    fx = table_y[i0] + (table_y[i0 + 1] - table_y[i0]) * frac
            if fx > 0.0:
                out[k] += n_i * fx
    return out


def _synch_seed_lum_numpy(elec_log_avg, elec_number, log_bpwn,
                          target_log_avg, log_nu_crit_offset,
                          log_erg) -> np.ndarray:
    """NumPy fallback: same sum via calc_fx_array on a 2-D x matrix."""
    occ = elec_number > 0
    if not np.any(occ):
        return np.zeros(len(target_log_avg))
    log_nu_c = (log_nu_crit_offset + log_bpwn
                + 2.0 * (elec_log_avg[occ] - log_erg))
    x = 10.0 ** (target_log_avg[np.newaxis, :] - log_nu_c[:, np.newaxis])
    fx = calc_fx_array(x)
    return elec_number[occ] @ fx


def synch_seed_lum(elec_grid: SpectralGrid, bpwn: float,
                   target_grid: SpectralGrid) -> np.ndarray:
    """Instantaneous synchrotron L_ν at the target bins [10^40 erg/s/Hz].

    L_ν(ν_k) = Σ_i N_i · p_synch(E_i, ν_k); units follow
    ``elec_grid.number``'s counts/1e40 scaling.
    """
    nt = target_grid.n_bins
    if bpwn <= 0:
        return np.zeros(nt)
    log_bpwn = np.log10(bpwn)
    if HAS_NUMBA:
        table_x, table_y = _get_fx_table()
        n_table = len(table_x)
        raw = _synch_seed_lum(
            elec_grid.log_avg, elec_grid.number, log_bpwn,
            target_grid.log_avg, elec_grid.n_bins, nt,
            _LOG_NU_CRIT_OFFSET, C.log_erg,
            table_x[0], (table_x[-1] - table_x[0]) / (n_table - 1),
            n_table, table_y, table_x[-1],
        )
    else:
        raw = _synch_seed_lum_numpy(
            elec_grid.log_avg, elec_grid.number, log_bpwn,
            target_grid.log_avg, _LOG_NU_CRIT_OFFSET, C.log_erg,
        )
    return 10.0 ** (_LOG_SYNCH_CONST_B0 + log_bpwn) * raw


def ssc_seed_density(elec_grid: SpectralGrid, bpwn: float, rad_pwn: float,
                     target_grid: SpectralGrid) -> np.ndarray:
    """SSC seed photon number density n_ν [cm^-3 Hz^-1] at the target bins.

    One-zone uniform-sphere mean density (Atoyan & Aharonian 1996):
    ``n_ν = SSC_GEOM_FACTOR · L_ν / (4π R² c · hν)``; ``bpwn`` [G],
    ``rad_pwn`` [cm].
    """
    l_nu_40 = synch_seed_lum(elec_grid, bpwn, target_grid)
    if rad_pwn <= 0:
        return np.zeros_like(l_nu_40)
    u_nu = (SSC_GEOM_FACTOR * l_nu_40 * 1.0e40
            / (4.0 * np.pi * rad_pwn ** 2 * C.c_light))
    return u_nu / (C.h_cgs * target_grid.bin_avg)


def blackbody_number_density(freq: float, temp: float) -> float:
    """Photon number density [cm^-3 Hz^-1] for a blackbody.
    """
    log_expon = np.log10(C.h_cgs) + np.log10(freq) - np.log10(C.k_B) - np.log10(temp)
    expon = 10.0 ** log_expon
    if expon > 500.0:
        return 0.0  # negligible at high hv/kT
    denom = np.exp(expon) - 1.0
    if denom <= 0:
        return 0.0
    log_num = (np.log10(8.0 * np.pi) - 3.0 * C.log_c
               + 2.0 * np.log10(freq) - np.log10(denom))
    return 10.0 ** log_num


# --- New energy after cooling + adiabatic losses ---

def new_energy_calc(energy, log_c_rad, log_rad, log_old_rad,
                    ln_rad_rat, log_vel, vel_neg):
    """New particle energy after radiative + adiabatic losses.
    """
    log_energy = np.log10(energy)
    rad_rat = np.exp(-ln_rad_rat)
    abs_e_max = rad_rat * energy

    log_term1 = log_vel + log_old_rad + log_energy - C.log_erg
    term1 = 10.0 ** log_term1

    log_term3 = log_rad + log_vel
    term3 = 10.0 ** log_term3

    local_ln = -ln_rad_rat if vel_neg else ln_rad_rat
    if local_ln <= 0:
        return max(0.0, min(abs_e_max, rad_rat * energy))
    log_term2 = (log_c_rad + log_rad + np.log10(local_ln)
                 + log_old_rad + log_energy - C.log_erg)
    term2 = 10.0 ** log_term2

    denom = term2 + term3
    if denom <= 0:
        return max(0.0, abs_e_max)

    new_e = min(term1 / denom * C.GeV_per_erg, abs_e_max)
    return max(new_e, 0.0)


# --- Photon spectrum calculation ---

def _calc_photon_spectrum_instantaneous(elec_grid, phot_grid, bpwn,
                                        min_ic_bin, pow_nu_ic, pow_ic_elec):
    """Snapshot emission: every electron bin emits at its instantaneous
    theoretical rate (the titer→0 limit of the secant formula).
    Modifies phot_grid."""
    n_elec = elec_grid.n_bins

    log_synch_const = (np.log10(2.0 / 3.0 * 3.0 ** 0.5)
                       + 3.0 * C.log_e_charge + np.log10(bpwn)
                       - C.log_m_electron - 2.0 * C.log_c)

    for i in range(n_elec):
        if elec_grid.number[i] <= 0:
            continue

        # Synchrotron
        nu_crit = calc_synch_freq(elec_grid.log_avg[i], bpwn) / 0.29
        max_synch_freq = nu_crit * C.max_synch_x
        max_bin = _calc_bin(max_synch_freq, phot_grid)
        if max_bin >= 0:
            x_vals = phot_grid.bin_avg[:max_bin + 1] / nu_crit
            fx_vals = calc_fx_array(x_vals)
            valid = fx_vals > 0
            if np.any(valid):
                log_power = (log_synch_const + np.log10(fx_vals[valid])
                             + phot_grid.log_width[:max_bin + 1][valid]
                             + elec_grid.log_number[i])
                phot_grid.number[:max_bin + 1][valid] += 10.0 ** log_power

        # IC
        if i >= min_ic_bin and pow_ic_elec[i] > 0:
            valid_j = pow_nu_ic[i, :] > LOG_EMPTY
            if np.any(valid_j):
                log_emit = (pow_nu_ic[i, valid_j] + elec_grid.log_number[i]
                            + phot_grid.log_width[valid_j])
                phot_grid.number[valid_j] += 10.0 ** log_emit

    # Convert to luminosity density
    mask = phot_grid.number > 0
    phot_grid.log_number[mask] = np.log10(phot_grid.number[mask])
    phot_grid.number[mask] /= phot_grid.bin_width[mask]
    phot_grid.log_number[mask] -= phot_grid.log_width[mask]


def calc_photon_spectrum(elec_grid, phot_grid, bpwn, old_radius, new_radius,
                         velocity, min_ic_bin, pow_nu_ic, pow_ic_elec, titer,
                         snapshot_mode='secant'):
    """Compute broadband photon spectrum (synchrotron + IC). Modifies phot_grid.

    snapshot_mode:
      'secant' (default): each electron bin emits at the last-step
        secant-average power (energy radiated over ``titer`` divided
        by ``titer``).
      'instantaneous': the titer→0 limit; each bin emits at its
        instantaneous theoretical rate.
    """
    if snapshot_mode not in ('secant', 'instantaneous'):
        raise ValueError(
            f"snapshot_mode must be 'secant' or 'instantaneous', "
            f"got {snapshot_mode!r}")
    if snapshot_mode == 'instantaneous':
        _calc_photon_spectrum_instantaneous(
            elec_grid, phot_grid, bpwn, min_ic_bin, pow_nu_ic, pow_ic_elec)
        return

    n_elec = elec_grid.n_bins
    n_phot = phot_grid.n_bins

    log_new_rad = np.log10(new_radius)
    log_old_rad = np.log10(old_radius)
    ln_rad_rat = np.log(new_radius / old_radius)
    rad_rat = old_radius / new_radius

    log_synch_const = (np.log10(2.0 / 3.0 * 3.0 ** 0.5)
                       + 3.0 * C.log_e_charge + np.log10(bpwn)
                       - C.log_m_electron - 2.0 * C.log_c)

    vel_neg = velocity < 0
    log_vel = np.log10(abs(velocity)) if velocity != 0 else -999999.0
    nelec_40 = np.sum(elec_grid.number)

    for i in range(n_elec):
        if elec_grid.number[i] <= 0:
            continue

        c_synch = csynch_calc(elec_grid.log_avg[i], bpwn, nelec_40, log_old_rad)

        if i >= min_ic_bin and pow_ic_elec[i] > 0:
            log_cic = (np.log10(pow_ic_elec[i])
                       - 2.0 * (elec_grid.log_avg[i] - C.log_erg))
            cic = 10.0 ** log_cic
            ic_frac = cic / (c_synch + cic)
        else:
            cic = 0.0
            ic_frac = 0.0

        synch_frac = max(1.0 - ic_frac, 0.0)
        c_total = c_synch + cic

        if c_total <= 0:
            ne_avg = rad_rat * elec_grid.bin_avg[i]
        else:
            ne_avg = new_energy_calc(
                elec_grid.bin_avg[i], np.log10(c_total),
                log_new_rad, log_old_rad, ln_rad_rat, log_vel, vel_neg)
            if ne_avg < 0:
                ne_avg = 0.0

        e_max_bin = rad_rat * elec_grid.bin_avg[i]
        if ne_avg > e_max_bin:
            ne_avg = e_max_bin
        rl_avg = e_max_bin - ne_avg
        if rl_avg <= 0:
            continue

        log_rl_bin = np.log10(rl_avg) - C.log_erg + elec_grid.log_number[i]

        # Synchrotron emission
        if synch_frac > 0 and c_synch > 0:
            log_sp_act = (np.log10(synch_frac) + log_rl_bin
                          - elec_grid.log_number[i] - np.log10(titer * C.yr))
            log_sp_theory = (np.log10(c_synch)
                             + 2.0 * (elec_grid.log_avg[i] - C.log_erg))
            log_sp_frac = min(log_sp_act - log_sp_theory, 0.0)

            nu_crit = calc_synch_freq(elec_grid.log_avg[i], bpwn) / 0.29
            max_synch_freq = nu_crit * C.max_synch_x

            max_bin = _calc_bin(max_synch_freq, phot_grid)

            if max_bin >= 0:
                x_vals = phot_grid.bin_avg[:max_bin + 1] / nu_crit
                fx_vals = calc_fx_array(x_vals)
                valid = fx_vals > 0
                if np.any(valid):
                    log_power = (log_synch_const + np.log10(fx_vals[valid])
                                 + log_sp_frac
                                 + phot_grid.log_width[:max_bin + 1][valid]
                                 + elec_grid.log_number[i])
                    phot_grid.number[:max_bin + 1][valid] += 10.0 ** log_power

        # IC emission
        if ic_frac > 0 and pow_ic_elec[i] > 0:
            log_pic_bin = (np.log10(ic_frac) + np.log10(rl_avg)
                           - C.log_erg - np.log10(titer) - C.log_yr)
            log_pic_frac = log_pic_bin - np.log10(pow_ic_elec[i])

            valid_j = pow_nu_ic[i, :] > LOG_EMPTY
            if np.any(valid_j):
                log_emit = (pow_nu_ic[i, valid_j] + elec_grid.log_number[i]
                            + phot_grid.log_width[valid_j] + log_pic_frac)
                phot_grid.number[valid_j] += 10.0 ** log_emit

    # Convert to luminosity density
    mask = phot_grid.number > 0
    phot_grid.log_number[mask] = np.log10(phot_grid.number[mask])
    phot_grid.number[mask] /= phot_grid.bin_width[mask]
    phot_grid.log_number[mask] -= phot_grid.log_width[mask]


def _calc_bin(value, grid):
    """Find bin index for value.
    """
    n = grid.n_bins
    log_bin_min = grid.log_avg[0]
    log_bin_max = grid.log_avg[n - 2]
    log_bin_iter = (log_bin_max - log_bin_min) / (n - 1)
    if value > 0:
        guess = int(round((np.log10(value) - log_bin_min) / log_bin_iter)) + 5
        return max(0, min(n - 1, guess))
    return -1
