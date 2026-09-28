"""Main PWN evolution loop.

Adaptive timestep calculation and the time-stepping logic that couples
the dynamical and spectral evolution of a PWN inside a non-radiative
supernova remnant.

Reference: Gelfand, Slane & Zhang (2009), ApJ 703, 2051.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from . import constants as C
from .grids import SpectralGrid, make_log_grid, LOG_EMPTY, LOG_EMPTY_CUT
from .particles import (
    inject_powerlaw,
    inject_broken_powerlaw,
    inject_maxwellian,
    integrate_electron_energy,
    place_in_bins_batch,
)
from .pulsar import initial_pwn_energy
from .escape import ESCAPE_MODELS, escape_tau_seconds
from .injection import get_injection_shape, injection_bin_counts
from .radiation import (
    HAS_NUMBA,
    njit,
    bpwn_calc,
    blackbody_number_density,
    calc_pow_ic,
    calc_sigma_and_pow_nu_ic,
    calc_photon_spectrum,
    csynch_calc,
    csynch_calc_vec,
    ic_cooling_matrix,
    new_energy_calc,
    new_energy_calc_vec,
    ssc_seed_density,
    _LOG_TAU_CONST,
    _LOG_8PI,
    _LOG_1P1,
    _LOG_SYNCH_FREQ_OFFSET,
    _LOG_CSYNCH_VAL,
)
from .snr import (
    calc_vt,
    calc_a,
    calc_characteristic_scales,
    calc_trad,
    calc_rad_snr,
    calc_vel_snr,
    calc_rad_rs,
    calc_prs,
    calc_mass_env,
    calc_snr_interior_pressure,
)

logger = logging.getLogger(__name__)


@dataclass
class DynState:
    """Dynamical state at one snapshot time.

    Attributes
    ----------
    time : float
        Age [yr].
    rad_pwn, rad_snr, rad_rs, rad_psr : float
        PWN radius, SNR forward-shock radius, reverse-shock radius,
        pulsar offset from the explosion center [pc].
    bpwn : float
        PWN magnetic field [µG].
    ad_lum_40, synch_lum_40, ic_lum_40, edot_40 : float
        Adiabatic, synchrotron, inverse-Compton, spin-down
        luminosities [10^40 erg/s].
    epwn_50, epwnb_50, epwnp_50 : float
        Total, magnetic, particle energy of the PWN [10^50 erg].
    vpwn, vejpwn, vsnr : float
        PWN shell, ejecta-at-PWN-boundary, SNR shock velocities [km/s].
    mswpwn : float
        Ejecta mass swept into the PWN shell [M_sun].
    ppwn, psnrpwn : float
        PWN interior pressure, SNR pressure at the PWN boundary
        [dyn cm^-2].
    rhoejpwn : float
        Ejecta density at the PWN boundary [g cm^-3].
    eparfudge : float
        Ratio of the two particle-energy accountings (diagnostic).
    delta_t : float
        Timestep that produced this state [yr].
    """
    time: float = 0.0
    rad_pwn: float = 0.0
    rad_snr: float = 0.0
    rad_rs: float = 0.0
    rad_psr: float = 0.0
    bpwn: float = 0.0
    ad_lum_40: float = 0.0
    synch_lum_40: float = 0.0
    ic_lum_40: float = 0.0
    edot_40: float = 0.0
    epwn_50: float = 0.0
    epwnb_50: float = 0.0
    epwnp_50: float = 0.0
    vpwn: float = 0.0
    vejpwn: float = 0.0
    mswpwn: float = 0.0
    ppwn: float = 0.0
    psnrpwn: float = 0.0
    rhoejpwn: float = 0.0
    vsnr: float = 0.0
    eparfudge: float = 0.0
    delta_t: float = 0.0


@dataclass
class PWNModelResult:
    """Full model output."""
    dyninfo: list[DynState] = field(default_factory=list)
    elecinfo: list = field(default_factory=list)   # (time, spectrum) snapshots
    radinfo: list = field(default_factory=list)     # (time, spectrum) snapshots
    # parallel to elecinfo
    escinfo: list = field(default_factory=list)
    elec_grid: Optional[SpectralGrid] = None
    phot_grid: Optional[SpectralGrid] = None
    electron_spectrum: Optional[np.ndarray] = None
    photon_spectrum: Optional[np.ndarray] = None
    photon_freqs: Optional[np.ndarray] = None
    electron_energies: Optional[np.ndarray] = None
    # cumulative; parallel to electron_energies
    escaped_spectrum: Optional[np.ndarray] = None
    # empty unless timestep_events
    events: list = field(default_factory=list)
    # continuous-injection energy range [GeV] (None for a pure Maxwellian);
    # lets plotters give the bin that straddles e_max its true width
    e_min_inj: Optional[float] = None
    e_max_inj: Optional[float] = None


# === Adaptive timestep ===

def calc_timestep(
    log_e_max: float, bpwn: float, n_elec_40: float, r_pwn: float,
    v_pwn: float, r_rs: float, time: float, age: float, r_snr: float,
    f_pwn: float, m_pwn: float,
) -> float:
    """Compute the adaptive timestep [years].

    log_e_max is the physical injection-bound energy, not the grid bound.
    """
    n_elec_bins = C.N_ELEC_BINS
    min_titer = 0.5
    max_synch_frac = 0.1
    max_dr_frac = 0.1
    max_dv_frac = 0.1
    max_r_snr = 5.0
    n_step = 5.0 * n_elec_bins

    titer5 = age / n_step
    titer8 = abs(r_pwn / v_pwn / C.yr) if v_pwn != 0 else 1e30

    # (1) Synchrotron loss timescale
    c_synch = csynch_calc(log_e_max, bpwn, n_elec_40, np.log10(r_pwn))
    if c_synch > 0:
        log_titer1 = (
            np.log10(1.0 - max_synch_frac) - np.log10(c_synch) - C.log_yr
            - log_e_max + C.log_erg
        )
        titer = 10.0 ** log_titer1
    else:
        titer = min_titer

    # (2) Radius change fraction
    if v_pwn != 0:
        titer2 = abs(r_pwn * (max_dr_frac - 1.0) / v_pwn / C.yr)
        if titer2 < titer:
            titer = titer2

    # (3) Time to reach reverse shock
    if r_pwn < r_rs:
        if v_pwn > 0:
            titer3 = (r_rs - r_pwn) / v_pwn / C.yr
            if 0 < titer3 < titer:
                titer = titer3
    else:
        if v_pwn != 0:
            titer6 = abs((r_snr - r_pwn) / max_r_snr / v_pwn / C.yr)
            if titer6 < titer:
                titer = titer6

    # (7) Velocity change fraction
    if f_pwn != 0 and m_pwn != 0:
        titer7 = abs(max_dv_frac * v_pwn * m_pwn / f_pwn / C.yr)
        if not np.isnan(titer7) and titer7 < titer:
            titer = titer7

    # (4) Time to reach final age
    if time < age:
        titer4 = age - time
        if titer4 < titer:
            titer = titer4

    if titer <= 0:
        titer4 = age - time if time < age else min_titer
        titer = min(titer4, min_titer)

    if titer < min_titer:
        titer = titer5 if titer5 < min_titer else min_titer

    if v_pwn < 0 and titer > titer8:
        titer = min(0.5 * titer8, min_titer)

    return titer


# === Electron spectrum evolution ===

def evolve_spectrum(
    old_grid: SpectralGrid, new_grid: SpectralGrid,
    bpwn: float, old_radius: float, new_radius: float, velocity: float,
    n_elec_40: float, min_ic_bin: int, pow_ic_elec: np.ndarray,
    scratch: Optional[tuple] = None,
) -> tuple[float, float]:
    """Evolve the electron spectrum by one timestep.

    Uses the fused numba kernel when available, else the
    vectorized-numpy path; the two are kept bit-identical.
    ``scratch`` optionally supplies four preallocated float64 arrays
    of length ``old_grid.n_bins``.
    Returns (new_energy_40, rad_loss_40).
    """
    if not HAS_NUMBA:
        return _evolve_spectrum_numpy(
            old_grid, new_grid, bpwn, old_radius, new_radius, velocity,
            n_elec_40, min_ic_bin, pow_ic_elec, scratch=scratch,
        )

    n_bins = old_grid.n_bins
    log_new_rad = np.log10(new_radius)
    log_old_rad = np.log10(old_radius)
    ln_rad_rat = np.log(new_radius / old_radius)
    rad_rat = old_radius / new_radius
    log_vel = np.log10(abs(velocity)) if velocity != 0 else -999999.0
    vel_neg = velocity < 0

    # not bit-equal to rad_rat
    rad_rat_e = np.exp(-ln_rad_rat)
    local_ln = -ln_rad_rat if vel_neg else ln_rad_rat
    early_adiab = local_ln <= 0
    log_local_ln = np.log10(local_ln) if not early_adiab else 0.0
    term3 = 10.0 ** (log_new_rad + log_vel)

    # csynch_calc_vec per-call scalars
    bpwn_pos = bpwn > 0
    if bpwn_pos:
        log_bpwn = np.log10(bpwn)
        csynch_thin = 10.0 ** (_LOG_CSYNCH_VAL + 2.0 * log_bpwn)
        log_n_los = (C.log_3 + np.log10(max(n_elec_40, 1e-300)) + 40.0
                     - _LOG_8PI - 2.0 * log_old_rad)
    else:
        log_bpwn = 0.0
        csynch_thin = 0.0
        log_n_los = 0.0

    if scratch is None:
        scratch = (np.empty(n_bins), np.empty(n_bins),
                   np.empty(n_bins), np.empty(n_bins))
    new_e_min, new_e_max, energy_terms, rad_loss_terms = scratch

    _evolve_spectrum_kernel(
        old_grid.bin_min, old_grid.bin_max, old_grid.bin_avg,
        old_grid.log_min, old_grid.log_max, old_grid.log_avg,
        old_grid.number, old_grid.log_number, pow_ic_elec[:n_bins],
        min_ic_bin, bpwn_pos, log_bpwn, log_n_los, csynch_thin,
        rad_rat, rad_rat_e, early_adiab, log_vel, log_new_rad,
        log_old_rad, log_local_ln, term3,
        C.log_erg, C.GeV_per_erg, C.log_two, C.log_3,
        _LOG_TAU_CONST, _LOG_1P1, _LOG_SYNCH_FREQ_OFFSET,
        new_e_min, new_e_max, energy_terms, rad_loss_terms,
    )

    new_energy = np.sum(energy_terms)
    rad_loss_total = np.sum(rad_loss_terms)

    occupied = old_grid.number > 0
    has_particles = (new_e_min > 0) | (new_e_max > 0)
    batch_mask = occupied & has_particles
    overflow_mask = occupied & ~has_particles

    _place_in_bins_kernel(new_e_min, new_e_max, old_grid.number,
                          batch_mask, new_grid.bin_min, new_grid.bin_max,
                          new_grid.number)

    # cooled to zero -> bin 0
    if np.any(overflow_mask):
        new_grid.number[0] += np.sum(old_grid.number[overflow_mask])

    new_grid.update_log_number()
    return new_energy, rad_loss_total


def _evolve_spectrum_numpy(
    old_grid: SpectralGrid, new_grid: SpectralGrid,
    bpwn: float, old_radius: float, new_radius: float, velocity: float,
    n_elec_40: float, min_ic_bin: int, pow_ic_elec: np.ndarray,
    scratch: Optional[tuple] = None,
) -> tuple[float, float]:
    """Vectorized-numpy implementation of :func:`evolve_spectrum`.

    Returns (new_energy_40, rad_loss_40).
    """
    n_bins = old_grid.n_bins
    log_new_rad = np.log10(new_radius)
    log_old_rad = np.log10(old_radius)
    ln_rad_rat = np.log(new_radius / old_radius)
    rad_rat = old_radius / new_radius

    vel_neg = velocity < 0
    log_vel = np.log10(abs(velocity)) if velocity != 0 else -999999.0

    # --- Vectorized cooling coefficients for bin min and max edges ---
    c_synch_min = csynch_calc_vec(old_grid.log_min, bpwn, n_elec_40, log_old_rad)
    c_synch_max = csynch_calc_vec(old_grid.log_max, bpwn, n_elec_40, log_old_rad)

    # Add IC cooling where applicable
    c_total_min = c_synch_min.copy()
    c_total_max = c_synch_max.copy()

    ic_mask = (np.arange(n_bins) >= min_ic_bin) & (pow_ic_elec[:n_bins] > 0)
    if np.any(ic_mask):
        log_cic = (np.log10(np.maximum(pow_ic_elec[:n_bins][ic_mask], 1e-300))
                   - 2.0 * (old_grid.log_avg[ic_mask] - C.log_erg))
        cic = 10.0 ** log_cic
        c_total_min[ic_mask] += cic
        c_total_max[ic_mask] += cic

    # --- Vectorized new energies at bin edges ---

    # Min edges
    has_cool_min = c_total_min > 0
    new_e_min = rad_rat * old_grid.bin_min.copy()  # default: adiabatic only
    if np.any(has_cool_min):
        log_c_min = np.log10(np.maximum(c_total_min[has_cool_min], 1e-300))
        new_e_min[has_cool_min] = new_energy_calc_vec(
            old_grid.bin_min[has_cool_min], log_c_min,
            log_new_rad, log_old_rad, ln_rad_rat, log_vel, vel_neg,
        )
    new_e_min = np.where(np.isnan(new_e_min), 0.0, new_e_min)

    # Max edges
    has_cool_max = c_total_max > 0
    new_e_max = rad_rat * old_grid.bin_max.copy()  # default: adiabatic only
    if np.any(has_cool_max):
        log_c_max = np.log10(np.maximum(c_total_max[has_cool_max], 1e-300))
        new_e_max[has_cool_max] = new_energy_calc_vec(
            old_grid.bin_max[has_cool_max], log_c_max,
            log_new_rad, log_old_rad, ln_rad_rat, log_vel, vel_neg,
        )
    new_e_max = np.where(np.isnan(new_e_max), 0.0, new_e_max)

    # --- Vectorized average new energy and energy/loss tracking ---
    both_pos = (new_e_min > 0) & (new_e_max > 0)
    new_e_avg = np.where(
        both_pos,
        10.0 ** (0.5 * (np.log10(np.maximum(new_e_min, 1e-300))
                        + np.log10(np.maximum(new_e_max, 1e-300)))),
        (new_e_min + new_e_max) / 2.0,
    )
    new_e_avg = np.where(np.isnan(new_e_avg), 0.0, new_e_avg)
    adiabatic_avg = rad_rat * old_grid.bin_avg
    new_e_avg = np.minimum(new_e_avg, adiabatic_avg)

    # Energy tracking
    occupied = old_grid.number > 0
    safe_avg = np.maximum(new_e_avg, 1e-300)
    energy_terms = np.where(
        occupied,
        10.0 ** (old_grid.log_number + np.log10(safe_avg) - C.log_erg),
        0.0,
    )
    new_energy = np.sum(energy_terms)

    # Radiative losses
    rl_min = np.maximum(rad_rat * old_grid.bin_min - new_e_min, 0.0)
    rl_max = np.maximum(rad_rat * old_grid.bin_max - new_e_max, 0.0)

    both_rl_pos = (rl_min > 0) & (rl_max > 0)
    rl_avg = np.where(
        both_rl_pos,
        10.0 ** (0.5 * (np.log10(np.maximum(rl_min, 1e-300))
                        + np.log10(np.maximum(rl_max, 1e-300)))),
        (rl_min + rl_max) / 2.0,
    )
    rad_loss_terms = np.where(
        occupied & (rl_avg > 0),
        10.0 ** (np.log10(np.maximum(rl_avg, 1e-300)) - C.log_erg
                 + old_grid.log_number),
        0.0,
    )
    rad_loss_total = np.sum(rad_loss_terms)

    # --- Batch place evolved particles into new bins ---
    has_particles = (new_e_min > 0) | (new_e_max > 0)
    batch_mask = occupied & has_particles
    overflow_mask = occupied & ~has_particles

    place_in_bins_batch(new_e_min, new_e_max, old_grid.number, batch_mask, new_grid)

    # cooled to zero -> bin 0
    if np.any(overflow_mask):
        new_grid.number[0] += np.sum(old_grid.number[overflow_mask])

    new_grid.update_log_number()

    if scratch is not None:
        # export per-bin work arrays
        np.copyto(scratch[0], new_e_min)
        np.copyto(scratch[1], new_e_max)
        np.copyto(scratch[2], energy_terms)
        np.copyto(scratch[3], rad_loss_terms)

    return new_energy, rad_loss_total


@njit(cache=True, error_model='numpy')
def _kernel_new_energy(e, log_c, rad_rat_e, early_adiab, log_vel,
                       log_new_rad, log_old_rad, log_local_ln, term3,
                       c_log_erg, c_gev_per_erg):
    """Scalar transcription of new_energy_calc_vec for one energy edge.

    Expression trees match the vectorized source line for line;
    reassociating or folding scalar subexpressions would change bits.
    """
    abs_e_max = rad_rat_e * e
    if early_adiab:
        return np.maximum(abs_e_max, 0.0)
    log_e = np.log10(e)
    term1 = 10.0 ** (log_vel + log_old_rad + log_e - c_log_erg)
    term2 = 10.0 ** (log_c + log_new_rad + log_local_ln + log_old_rad
                     + log_e - c_log_erg)
    denom = term2 + term3
    if denom > 0.0:
        new_e = np.minimum(term1 / np.maximum(denom, 1e-300)
                           * c_gev_per_erg, abs_e_max)
    else:
        new_e = abs_e_max
    return np.maximum(new_e, 0.0)


@njit(cache=True, error_model='numpy')
def _evolve_spectrum_kernel(
    bin_min, bin_max, bin_avg, log_min, log_max, log_avg,
    number, log_number, pow_ic,
    min_ic_bin, bpwn_pos, log_bpwn, log_n_los, csynch_thin,
    rad_rat, rad_rat_e, early_adiab, log_vel, log_new_rad, log_old_rad,
    log_local_ln, term3,
    c_log_erg, c_gev_per_erg, c_log_two, c_log_3,
    c_log_tau_const, c_log_1p1, c_synch_freq_offset,
    new_e_min, new_e_max, energy_terms, rad_loss_terms,
):
    """Fused serial per-bin transcription of _evolve_spectrum_numpy.

    Reads no module globals (cache=True bakes them in); expression order
    matches the numpy path; reductions stay in the dispatcher.
    """
    n = bin_min.shape[0]
    for i in range(n):
        # --- cooling coefficient at each bin edge: csynch + IC ---
        c_min = 0.0
        c_max = 0.0
        if bpwn_pos:
            log_nu_crit = (c_synch_freq_offset + log_bpwn
                           + 2.0 * (log_min[i] - c_log_erg))
            log_tau = (c_log_tau_const + 1.5 * log_bpwn
                       - 2.5 * log_nu_crit + log_n_los + c_log_1p1
                       + 0.75 * (c_log_two - c_log_3))
            if log_tau < 0.0:
                c_min = csynch_thin
            log_nu_crit = (c_synch_freq_offset + log_bpwn
                           + 2.0 * (log_max[i] - c_log_erg))
            log_tau = (c_log_tau_const + 1.5 * log_bpwn
                       - 2.5 * log_nu_crit + log_n_los + c_log_1p1
                       + 0.75 * (c_log_two - c_log_3))
            if log_tau < 0.0:
                c_max = csynch_thin
        if i >= min_ic_bin and pow_ic[i] > 0.0:
            cic = 10.0 ** (np.log10(np.maximum(pow_ic[i], 1e-300))
                           - 2.0 * (log_avg[i] - c_log_erg))
            c_min = c_min + cic
            c_max = c_max + cic

        # --- new energy at each edge (cooled or pure adiabatic) ---
        if c_min > 0.0:
            ne_min = _kernel_new_energy(
                bin_min[i], np.log10(np.maximum(c_min, 1e-300)),
                rad_rat_e, early_adiab, log_vel, log_new_rad,
                log_old_rad, log_local_ln, term3,
                c_log_erg, c_gev_per_erg)
        else:
            ne_min = rad_rat * bin_min[i]
        if np.isnan(ne_min):
            ne_min = 0.0

        if c_max > 0.0:
            ne_max = _kernel_new_energy(
                bin_max[i], np.log10(np.maximum(c_max, 1e-300)),
                rad_rat_e, early_adiab, log_vel, log_new_rad,
                log_old_rad, log_local_ln, term3,
                c_log_erg, c_gev_per_erg)
        else:
            ne_max = rad_rat * bin_max[i]
        if np.isnan(ne_max):
            ne_max = 0.0

        # --- average new energy, capped at adiabatic-only ---
        if ne_min > 0.0 and ne_max > 0.0:
            avg = 10.0 ** (0.5 * (np.log10(np.maximum(ne_min, 1e-300))
                                  + np.log10(np.maximum(ne_max, 1e-300))))
        else:
            avg = (ne_min + ne_max) / 2.0
        if np.isnan(avg):
            avg = 0.0
        avg = np.minimum(avg, rad_rat * bin_avg[i])

        # --- energy tracking term ---
        if number[i] > 0.0:
            energy_terms[i] = 10.0 ** (log_number[i]
                                       + np.log10(np.maximum(avg, 1e-300))
                                       - c_log_erg)
        else:
            energy_terms[i] = 0.0

        # --- radiative-loss term ---
        rl_min = np.maximum(rad_rat * bin_min[i] - ne_min, 0.0)
        rl_max = np.maximum(rad_rat * bin_max[i] - ne_max, 0.0)
        if rl_min > 0.0 and rl_max > 0.0:
            rl_avg = 10.0 ** (0.5 * (np.log10(np.maximum(rl_min, 1e-300))
                                     + np.log10(np.maximum(rl_max, 1e-300))))
        else:
            rl_avg = (rl_min + rl_max) / 2.0
        if number[i] > 0.0 and rl_avg > 0.0:
            rad_loss_terms[i] = 10.0 ** (np.log10(np.maximum(rl_avg, 1e-300))
                                         - c_log_erg + log_number[i])
        else:
            rad_loss_terms[i] = 0.0

        new_e_min[i] = ne_min
        new_e_max[i] = ne_max


@njit(cache=True, error_model='numpy')
def _place_in_bins_kernel(new_mins, new_maxs, values, batch_mask,
                          grid_bin_min, grid_bin_max, number):
    """Fused transcription of place_in_bins_batch (particles.py).

    Pass order (span-0, span-1 lower, span-1 upper, span>=2) and the
    frac quirks are kept verbatim for bit-identical accumulation.
    """
    n_bins = grid_bin_min.shape[0]
    abs_max = grid_bin_max[n_bins - 1]
    n_src = new_mins.shape[0]

    nm_buf = np.empty(n_src)
    nx_buf = np.empty(n_src)
    val_buf = np.empty(n_src)
    lo_buf = np.empty(n_src, dtype=np.int64)
    hi_buf = np.empty(n_src, dtype=np.int64)

    m = 0
    for k in range(n_src):
        if batch_mask[k]:
            nm = new_mins[k]
            nx = np.minimum(new_maxs[k], abs_max - 1.0)
            if nx < nm:
                t = nm
                nm = nx
                nx = t
            lo = np.searchsorted(grid_bin_max, nm, side='left')
            if lo > n_bins - 1:
                lo = n_bins - 1
            hi = np.searchsorted(grid_bin_min, nx, side='right') - 1
            if hi < 0:
                hi = 0
            if hi > n_bins - 1:
                hi = n_bins - 1
            nm_buf[m] = nm
            nx_buf[m] = nx
            val_buf[m] = values[k]
            lo_buf[m] = lo
            hi_buf[m] = hi
            m += 1

    for j in range(m):                      # pass 1: span == 0
        if hi_buf[j] - lo_buf[j] == 0:
            i = lo_buf[j]
            w = nx_buf[j] - nm_buf[j]
            copy_min = np.maximum(nm_buf[j], grid_bin_min[i])
            copy_max = np.minimum(nx_buf[j], grid_bin_max[i])
            if w > 0:
                frac = (copy_max - copy_min) / w
            else:
                frac = 1.0
            frac = np.minimum(np.maximum(frac, 0.0), 1.0)
            number[i] += frac * val_buf[j]

    for j in range(m):                      # pass 2: span == 1, lower bins
        if hi_buf[j] - lo_buf[j] == 1:
            i = lo_buf[j]
            w = nx_buf[j] - nm_buf[j]
            lo_copy_min = np.maximum(nm_buf[j], grid_bin_min[i])
            lo_copy_max = grid_bin_max[i]
            if w > 0:
                frac = (lo_copy_max - lo_copy_min) / w
            else:
                frac = 0.5
            frac = np.minimum(np.maximum(frac, 0.0), 1.0)
            number[i] += frac * val_buf[j]

    for j in range(m):                      # pass 3: span == 1, upper bins
        if hi_buf[j] - lo_buf[j] == 1:
            i = hi_buf[j]
            w = nx_buf[j] - nm_buf[j]
            hi_copy_min = grid_bin_min[i]
            hi_copy_max = np.minimum(nx_buf[j], grid_bin_max[i])
            if w > 0:
                frac = (hi_copy_max - hi_copy_min) / w
            else:
                frac = 0.5
            frac = np.minimum(np.maximum(frac, 0.0), 1.0)
            number[i] += frac * val_buf[j]

    for j in range(m):                      # pass 4: span >= 2 (rare)
        if hi_buf[j] - lo_buf[j] >= 2:
            w = nx_buf[j] - nm_buf[j]
            v = val_buf[j]
            nmin_k = nm_buf[j]
            nmax_k = nx_buf[j]
            for i in range(lo_buf[j], hi_buf[j] + 1):
                if nmin_k <= grid_bin_max[i] and nmax_k >= grid_bin_min[i]:
                    copy_min = (nmin_k if nmin_k > grid_bin_min[i]
                                else grid_bin_min[i])
                    copy_max = (nmax_k if nmax_k < grid_bin_max[i]
                                else grid_bin_max[i])
                    if w > 0:
                        frac = (copy_max - copy_min) / w
                    else:
                        frac = 1.0
                    if frac > 0.0 and frac <= 1.0:
                        number[i] += frac * v


# === Fill target photon field ===

def _fill_target_field(target_grid, n_ic_target, n_ic, ic_temps, ic_norms):
    """Fill the IC target photon field with CMB + extra backgrounds.

    ic_norms are total energy densities [eV/cm^3], converted here to
    the internal dimensionless blackbody multiplier.
    """
    for i in range(n_ic_target):
        target_grid.number[i] += blackbody_number_density(
            target_grid.bin_avg[i], C.T_CMB
        )
    if n_ic > 0:
        for j in range(n_ic):
            T = ic_temps[j]
            u_evcm3 = ic_norms[j]
            if T > 0.0 and u_evcm3 > 0.0:
                norm_dim = u_evcm3 / (
                    C.radiation_const * T**4 * C.eV_per_erg
                )
            else:
                norm_dim = 0.0
            for i in range(n_ic_target):
                target_grid.number[i] += norm_dim * blackbody_number_density(
                    target_grid.bin_avg[i], T
                )
    target_grid.update_log_number()


# === CMB IC lookup ===

_CMB_LOOKUP_CACHE: dict = {}  # {constants_source: (array, meta_dict)}


def _load_cmb_lookup():
    """Load CMB pow_nu_ic table for the current constants source."""
    src = C.CONSTANTS_SOURCE
    if src in _CMB_LOOKUP_CACHE:
        return _CMB_LOOKUP_CACHE[src]
    import importlib.resources
    filename = f'cmb_pow_nu_ic_{src}.npz'
    try:
        ref = importlib.resources.files('viper.data').joinpath(filename)
        with ref.open('rb') as f:
            npz = np.load(f, allow_pickle=False)
            arr = npz['pow_nu_ic']
            meta = {k: npz[k] for k in npz.files if k != 'pow_nu_ic'}
        _CMB_LOOKUP_CACHE[src] = (arr, meta)
        return _CMB_LOOKUP_CACHE[src]
    except (FileNotFoundError, KeyError):
        return None


def _cmb_lookup_compatible(use_cmb_lookup, cpp_compat, n_ic, f_max,
                            n_elec_bins, n_phot_bins, n_ic_target):
    """Return (compatible: bool, reason: str) for lookup activation."""
    if not use_cmb_lookup:
        return False, "use_cmb_lookup=False"
    if cpp_compat:
        return False, "cpp_compat=True forbids lookup"
    if n_ic != 0:
        return False, f"n_ic={n_ic} (lookup is CMB-only)"
    if f_max == 1:
        return False, "f_max==1 (Maxwellian-only)"
    if n_elec_bins != C.ELEC_GRID_FIXED_NBINS:
        return False, f"n_elec_bins={n_elec_bins} != {C.ELEC_GRID_FIXED_NBINS}"
    if n_phot_bins != C.N_PHOT_FREQ:
        return False, f"n_phot_bins={n_phot_bins} != {C.N_PHOT_FREQ}"
    if n_ic_target != C.N_IC_TARGET:
        return False, f"n_ic_target={n_ic_target} != {C.N_IC_TARGET}"
    loaded = _load_cmb_lookup()
    if loaded is None:
        return False, (f"lookup file for constants_source="
                        f"{C.CONSTANTS_SOURCE} not present")
    _arr, meta = loaded
    file_src = meta.get('constants_source', '<unknown>')
    if hasattr(file_src, 'item'):
        file_src = file_src.item()
    file_src = str(file_src)
    if file_src != C.CONSTANTS_SOURCE:
        return False, (f"lookup file constants_source={file_src} != "
                        f"runtime C.CONSTANTS_SOURCE={C.CONSTANTS_SOURCE} "
                        f"(should not happen; _load_cmb_lookup should "
                        f"pick the right file)")
    if int(meta.get('elec_grid_nbins', -1)) != C.ELEC_GRID_FIXED_NBINS:
        return False, (f"lookup file nbins={int(meta['elec_grid_nbins'])} "
                        f"!= C.ELEC_GRID_FIXED_NBINS={C.ELEC_GRID_FIXED_NBINS}; "
                        f"regenerate via tools/build_cmb_lookup.py --force")
    return True, "ok"


# === Main evolution function ===

_COOLING_LOSS_FRAC = 0.1
_COOLING_PREFACTOR = 1.0 / (1.0 - _COOLING_LOSS_FRAC) - 1.0


def _cooling_bound(elec_grid, bpwn, nelec_40, rad_pwn, pow_ic_elec,
                   floor_yr):
    """Timestep bound from the per-bin 10%-loss cooling time."""
    n = elec_grid.n_bins
    occ = elec_grid.number[:n] > 0
    if not np.any(occ):
        return 1e30
    log_avg = elec_grid.log_avg[:n][occ]
    cs = csynch_calc_vec(log_avg, bpwn, nelec_40, np.log10(rad_pwn))
    e_erg = 10.0 ** (log_avg - C.log_erg)
    pie = pow_ic_elec[:n][occ]
    cic = np.where(pie > 0, pie / e_erg ** 2, 0.0)
    ctot = cs + cic
    valid = ctot > 0
    if not np.any(valid):
        return 1e30
    t10 = _COOLING_PREFACTOR / (ctot[valid] * e_erg[valid]) / C.yr
    return max(float(t10.min()), floor_yr)


def _apply_escape(elec_grid, esc_grid, escape_model, esc_norm, esc_index,
                  bpwn, rad_pwn, vel_pwn, time, titer):
    """One-zone escape sink: deplete each occupied bin by exp(-dt/tau_esc(E)).

    esc_grid is a cumulative reservoir, not reset per step.
    Returns the escaped energy this step [10^40 erg]; 0.0 when off.
    """
    if escape_model == 'off':
        return 0.0
    occ = elec_grid.number > 0
    if not np.any(occ):
        return 0.0
    E = elec_grid.bin_avg[occ]
    tau = escape_tau_seconds(escape_model, E, bpwn, rad_pwn, vel_pwn,
                             time, esc_norm, esc_index)
    surv = np.exp(-(titer * C.yr) / np.maximum(tau, 1e-300))
    N = elec_grid.number[occ]
    esc_count = N * (1.0 - surv)
    esc_energy_40 = float(np.sum(esc_count * E) / 10.0 ** C.log_erg)
    esc_grid.number[occ] += esc_count
    elec_grid.number[occ] = N * surv
    elec_grid.update_log_number()
    return esc_energy_40


class _EventStepper:
    """Event-resolved timestep capping around reverse-shock crushes.

    cap_dt may only shorten a step: it bisection-localizes the PWN/RS
    contact while tcol == -1, and caps dt <= (r/abs(v))/CAP_FRACTION
    during crush episodes, floored at floor_yr. Episodes are recorded
    as dicts {t_enter, t_min, r_min, t_exit, n_steps, outcome}
    (yr, pc) on PWNModelResult.events.
    """

    CAP_FRACTION = 20.0          # dt <= (r/|v|)/CAP_FRACTION
    HYSTERESIS = 2.0
    EXPANSION_STREAK = 50
    BUDGET_STEPS = 20000

    def __init__(self, floor_yr, t_ch, r_ch):
        self.floor_yr = floor_yr
        self.t_ch = t_ch
        self.r_ch = r_ch
        self.episodes = []
        self._ep = None
        self._streak = 0

    # -- contact localization ------------------------------------

    def _contact_cap(self, titer, time, rad_pwn, vel_pwn):
        def gap(dt):
            r_fore = rad_pwn + vel_pwn * dt * C.yr
            rsnr = calc_rad_snr(time + dt, self.t_ch, self.r_ch)
            rrs = calc_rad_rs(time + dt, self.t_ch, self.r_ch, rsnr)
            return r_fore - rrs

        if gap(0.0) >= 0 or gap(titer) <= 0:
            return titer
        lo, hi = 0.0, titer
        while hi - lo > self.floor_yr:
            mid = 0.5 * (lo + hi)
            if gap(mid) > 0:
                hi = mid
            else:
                lo = mid
        return max(hi, self.floor_yr)

    # -- crush-episode capping -----------------------------------

    def cap_dt(self, titer, time, rad_pwn, vel_pwn, tcol):
        if tcol == -1:
            titer = self._contact_cap(titer, time, rad_pwn, vel_pwn)
        if self._ep is None and vel_pwn < 0:
            self._ep = dict(t_enter=time, t_min=time, r_min=rad_pwn,
                            n_steps=0, outcome=None)
            self._streak = 0
        if self._ep is None or vel_pwn == 0:
            return titer
        cap = max(rad_pwn / abs(vel_pwn) / C.yr / self.CAP_FRACTION,
                  self.floor_yr)
        if (vel_pwn < 0 and cap == self.floor_yr
                and rad_pwn < abs(vel_pwn) * self.floor_yr * C.yr):
            self._close(time, 'collapse_at_floor')
            logger.warning(
                "event stepper: model-level collapse at the dt floor "
                "(r=%.3e cm < |v|*floor at t=%.1f yr); episode "
                "terminated, cap released", rad_pwn, time)
            return titer
        return min(titer, cap)

    # -- episode records (post-step) -------------------------------

    def observe(self, time, rad_pwn, vel_pwn):
        if self._ep is None:
            return
        ep = self._ep
        ep['n_steps'] += 1
        if rad_pwn < ep['r_min']:
            ep['r_min'] = rad_pwn
            ep['t_min'] = time
        self._streak = self._streak + 1 if vel_pwn > 0 else 0
        if vel_pwn > 0 and rad_pwn > self.HYSTERESIS * ep['r_min']:
            self._close(time, 'reexpansion')
        elif self._streak >= self.EXPANSION_STREAK:
            self._close(time, 'sustained_expansion')
        elif ep['n_steps'] >= self.BUDGET_STEPS:
            logger.warning(
                "event stepper: episode step budget (%d) exhausted at "
                "t=%.1f yr; cap released (outcome=budget_release)",
                self.BUDGET_STEPS, time)
            self._close(time, 'budget_release')

    def finalize(self, time):
        if self._ep is not None:
            self._close(time, 'run_end')

    def _close(self, time, outcome):
        ep = self._ep
        ep['t_exit'] = time
        ep['outcome'] = outcome
        ep['r_min'] = ep['r_min'] / C.pc
        self.episodes.append(ep)
        self._ep = None
        self._streak = 0


def _snr_boundary_state(time, rad_pwn, old_rad_pwn, old_msw_pwn, vel_pwn,
                        tcol, log_vol_pwn, a, m, rad_core, rad_rs, rad_snr,
                        t_ch, v_ch, tst, vel_snr, n_ism,
                        psnr_pwn, msw_pwn, vel_ej_pwn, rho_ej_pwn,
                        p_snr, prs, optimize_vtilda, corrector):
    """SNR-side boundary state at the current PWN radius.

    Shared by the predictor and the pc_full corrector. corrector=True
    keeps the predictor's rho_ej_pwn in the pre-collision branch and
    does not recompute p_snr/prs.
    """
    if tcol == -1:
        # Pre-collision: ejecta in free expansion
        psnr_pwn = 0.0
        if rad_pwn <= rad_core:
            log_rho_core = np.log10(a) - 3.0 * np.log10(time * C.yr)
            if not corrector:
                rho_ej_pwn = 10.0 ** log_rho_core
            log_msw = log_rho_core + log_vol_pwn
            msw_pwn = 10.0 ** log_msw
        else:
            log_rho_core = np.log10(a) - 3.0 * np.log10(time * C.yr)
            log_mass_core = (
                log_rho_core + np.log10(4.0 / 3.0 * np.pi)
                + 3.0 * np.log10(rad_core)
            )
            mass_core = 10.0 ** log_mass_core
            mass_env = calc_mass_env(
                10.0 ** log_rho_core, rad_core, rad_pwn
            )
            if not corrector:
                rho_ej_pwn = (
                    10.0 ** log_rho_core
                    * (rad_core / rad_pwn) ** (-1.0 * m)
                )
            msw_pwn = mass_core + mass_env
        vel_ej_pwn = rad_pwn / time / C.yr
    else:
        # Post-collision: use SNR interior solution
        if not corrector:
            log_p_snr = (
                np.log10(0.75) + np.log10(n_ism) + np.log10(C.m_proton)
                + 2.0 * np.log10(vel_snr)
            )
            p_snr = 10.0 ** log_p_snr
            prs = calc_prs(time, t_ch, v_ch, a, vel_snr, p_snr, n_ism)

        if time * C.yr < tst and rad_rs > rad_core:
            # Free-expansion + Chevalier approximation
            rad_cd = rad_snr / 1.14
            if rad_pwn > rad_cd:
                psnr_pwn = p_snr
            else:
                psnr_pwn = prs
            msw_pwn = old_msw_pwn
        else:
            # Bandiera (1984) self-similar solution
            pwn_psi = rad_pwn / rad_snr
            psnr_pwn_val, rho_val, vej_val, _ = (
                calc_snr_interior_pressure(
                    pwn_psi, p_snr, vel_snr, n_ism,
                    optimize_vtilda=optimize_vtilda,
                )
            )
            psnr_pwn = psnr_pwn_val
            if time * C.yr < tst and psnr_pwn < prs:
                psnr_pwn = prs
            rho_ej_pwn = rho_val
            vel_ej_pwn = vej_val
            if vel_ej_pwn < vel_pwn:
                msw_pwn = old_msw_pwn + (
                    4.0 / 3.0 * np.pi
                    * (rad_pwn ** 3 - old_rad_pwn ** 3)
                    * rho_ej_pwn
                )

    return psnr_pwn, msw_pwn, vel_ej_pwn, rho_ej_pwn, p_snr, prs


def _momentum_update(ppwn, psnr_pwn, rad_pwn, msw_pwn, old_msw_pwn,
                     v_old, vel_ej_pwn, titer, vel_pwn):
    """Thin-shell momentum update; returns ``(for_pwn, vel_pwn)``.

    v_old is the velocity entering the update (predictor: current
    vel_pwn; corrector: old_vel_pwn). If neither mass is positive the
    incoming vel_pwn is returned unchanged.
    """
    for_pwn = (ppwn - psnr_pwn) * 4.0 * np.pi * rad_pwn ** 2
    del_m = max(msw_pwn - old_msw_pwn, 0.0)

    if msw_pwn > 0:
        vel_pwn = (
            old_msw_pwn * v_old + for_pwn * titer * C.yr
            + del_m * vel_ej_pwn
        ) / msw_pwn
    elif old_msw_pwn > 0:
        vel_pwn = (
            old_msw_pwn * v_old + for_pwn * titer * C.yr
        ) / old_msw_pwn

    return for_pwn, vel_pwn


def _make_dynstate(time, rad_pwn, rad_snr, rad_rs, rad_psr, bpwn, vel_pwn,
                   epwn_40, epwnmag_40, epwnkin_40, vel_ej_pwn, msw_pwn,
                   ppwn, psnr_pwn, rho_ej_pwn, vel_snr, eparfudge, titer,
                   e0_37, tau_sd, braking_index, depsr_40):
    """Build a DynState snapshot."""
    return DynState(
        time=time,
        rad_pwn=rad_pwn / C.pc,
        rad_snr=rad_snr / C.pc,
        rad_rs=rad_rs / C.pc,
        rad_psr=rad_psr / C.pc,
        bpwn=bpwn * 1e6,
        ad_lum_40=(vel_pwn / rad_pwn * epwn_40 if rad_pwn > 0 else 0.0),
        synch_lum_40=0.0,
        ic_lum_40=0.0,
        edot_40=(
            e0_37 * 1e-3
            * (1 + time / tau_sd)
            ** (-(braking_index + 1) / (braking_index - 1))
            if depsr_40 > 0 else 0.0
        ),
        epwn_50=epwn_40 / 1e10,
        epwnb_50=epwnmag_40 / 1e10,
        epwnp_50=epwnkin_40 / 1e10,
        vpwn=vel_pwn / 1e5,
        vejpwn=vel_ej_pwn / 1e5,
        mswpwn=msw_pwn / C.M_sun,
        ppwn=ppwn,
        psnrpwn=psnr_pwn,
        rhoejpwn=rho_ej_pwn,
        vsnr=vel_snr / 1e5,
        eparfudge=eparfudge,
        delta_t=titer,
    )


def _make_continuous_injector(injection_model, grid, e_min, e_max,
                              e_break, p1, p2):
    """Build the per-step injector for the continuous spectral component.

    'broken_powerlaw' calls the analytic injectors (e_break == 0 or
    p2 == 0 selects a single power law); other registered shapes are
    integrated over the bins once, then scaled by the injected energy
    each step. Both paths no-op when energy_40 <= 0.
    """
    if injection_model == 'broken_powerlaw':
        def inject(g, energy_40):
            if e_break == 0 or p2 == 0:
                inject_powerlaw(g, energy_40, e_min, e_max, p1)
            else:
                inject_broken_powerlaw(
                    g, energy_40, e_min, e_max, e_break, p1, p2)
        return inject

    counts = injection_bin_counts(
        injection_model, grid, e_min, e_max, e_break, p1, p2)

    def inject(g, energy_40):
        if energy_40 <= 0:
            return
        g.number += energy_40 * counts
        g.update_log_number()
    return inject


def evolve_pwn(
    esn_51: float, mej: float, n_ism: float,
    braking_index: float, tau_sd: float, age: float, e0_37: float,
    vel_psr: float, eta_g: float, eta_b: float,
    e_min: float, e_max: float, e_break: float,
    p1: float, p2: float,
    f_max: float = 0.0, kT_max: float = 0.0,
    n_ic: int = 0,
    ic_temps: Optional[list[float]] = None,
    ic_norms: Optional[list[float]] = None,
    n_elec_bins: int = C.N_ELEC_BINS,
    n_phot_bins: int = C.N_PHOT_FREQ,
    n_ic_target: int = C.N_IC_TARGET,
    phot_freq_min: float = C.PHOT_FREQ_MIN,
    phot_freq_max: float = C.PHOT_FREQ_MAX,
    target_freq_min: float = C.TARGET_FREQ_MIN,
    target_freq_max: float = C.TARGET_FREQ_MAX,
    dyn_step: int = 1,
    elec_step: int = -1,
    phot_step: int = -1,
    integration: str = 'pc_full',
    cpp_compat: bool = False,
    optimize_vtilda: Optional[bool] = None,
    ic_early_break: Optional[bool] = None,
    ic_j_break: Optional[bool] = None,
    ic_skip_empty_bins: bool = True,
    use_cmb_lookup: bool = False,
    replay_dt_sequence: Optional[Sequence[float]] = None,
    timestep_events: bool = False,
    event_dt_floor_yr: float = 0.5,
    snapshot_mode: str = 'secant',
    timestep_cooling: str = 'compat',
    escape_model: str = 'off',
    esc_norm: float = 1.0,
    esc_index: float = 0.0,
    injection_model: str = 'broken_powerlaw',
    ssc_model: str = 'off',
    snapshot_injection: str = 'included',
) -> PWNModelResult:
    """Evolve a PWN inside an SNR.  Main entry point.

    Parameters
    ----------
    esn_51 : float
        Supernova kinetic energy [10^51 erg].
    mej : float
        Ejecta mass [M_sun].
    n_ism : float
        Ambient ISM number density [cm^-3].
    braking_index : float
        Pulsar braking index n (must not equal 1).
    tau_sd : float
        Initial spin-down timescale [yr].
    age : float
        Age at which the evolution stops [yr].
    e0_37 : float
        Initial spin-down luminosity [10^37 erg/s].
    vel_psr : float
        Pulsar space velocity [cm/s]; 0 keeps the pulsar at the center.
    eta_g : float
        Fraction of spin-down power lost upstream of the termination
        shock.
    eta_b : float
        Fraction of spin-down power injected as magnetic energy.
    e_min, e_max : float
        Minimum and maximum injected particle energy [GeV].
    e_break : float
        Injection break energy [GeV]; 0 selects a single power law.
    p1, p2 : float
        Injection indices below/above the break; p2 = 0 selects a
        single power law of index p1.
    f_max : float
        Fraction of injected energy in a relativistic Maxwellian
        component (0 = none).
    kT_max : float
        Maxwellian temperature [GeV].
    n_ic : int
        Number of extra IC target photon fields (CMB always included).
    ic_temps, ic_norms : list of float
        Blackbody temperature [K] and total energy density [eV cm^-3]
        of each extra field.
    dyn_step : int
        Dynamical-snapshot cadence: 1 = every timestep (default),
        N > 0 = every N steps, -1 = final only.
    elec_step : int
        Electron-spectrum snapshot cadence: -1 = final only (default),
        N > 0 = every N steps.
    phot_step : int
        Photon-spectrum snapshot cadence (expensive): -1 = final only
        (default), N > 0 = every N steps.
    integration : str
        'euler' (forward Euler), 'pc_radius' (predictor-corrector on
        radius), or 'pc_full' (radius and velocity; default).
    cpp_compat : bool
        If True, reproduce the original C++ implementation's
        conventions (forces Euler integration; newer options off).
        Bit-level reproduction also needs VIPER_CONSTANTS=gsz2009
        set before import.
    use_cmb_lookup : bool, default False
        Use the precomputed CMB IC table when compatible (CMB-only,
        default grids, not cpp_compat, lookup file present); otherwise
        falls through to live IC, reason logged at INFO.
    replay_dt_sequence : sequence of float, optional
        Replay these dt values verbatim instead of calling
        :func:`calc_timestep`; falls back to live calls once
        exhausted. Benchmarking/A-B comparison only, not for
        production fits.
    timestep_events : bool
        Event-resolved stepping: contact bisection plus a
        crush-episode cap dt <= (r/abs(v))/20; episodes recorded on
        ``result.events``. Capped replayed steps hand their remainder
        back to the sequence. Forced False under cpp_compat.
    event_dt_floor_yr : float
        Floor for the event cap [yr] (default 0.5).
    snapshot_injection : {'included', 'excluded'}
        What the recorded snapshots (``elecinfo``/``radinfo``, the
        final ``electron_spectrum``/``photon_spectrum``) contain.
        ``'included'`` (default): the grid as it stands at the end of
        the step, i.e. cooled particles plus the particles injected
        during that step.  ``'excluded'``: the step's own deposit is
        subtracted first, so the snapshot is the resident (cooled)
        population.  Near ``e_max`` the fresh deposit scales with the
        timestep and dominates the top bins; ``'excluded'`` removes that
        dt dependence.  Display option -- fits keep the default.  Forced
        ``'included'`` under ``cpp_compat``.
    timestep_cooling : str
        'compat' (default; synchrotron-only bound inside
        calc_timestep) or 'total_loss10' (also cap each step at the
        per-bin 10%-loss time under synchrotron + IC cooling, floored
        at event_dt_floor_yr). Forced 'compat' under cpp_compat.
    snapshot_mode : str
        'secant' (default; last-step average, flux deficit linear in
        the final dt) or 'instantaneous' (the titer->0 limit). Forced
        'secant' under cpp_compat.
    escape_model : str
        One-zone escape sink (escape.py registry): 'off' (default,
        exact no-op), 'powerlaw'/'diffusion' (use esc_index),
        'bohm'/'constant'/'advective' (fixed forms). Escaped particles
        accumulate in ``result.escaped_spectrum`` and their energy is
        removed from the particle budget. Forced 'off' under
        cpp_compat.
    esc_norm : float
        Dimensionless O(1) escape normalization.
    esc_index : float
        Escape energy index; used by 'powerlaw' and 'diffusion'.
    injection_model : str
        Continuous injection shape (injection.py registry):
        'broken_powerlaw' (default, analytic), 'log_parabola',
        'powerlaw_cutoff', or user-registered; non-default shapes
        reuse the e_break/p1/p2 slots. The Maxwellian component
        applies on top of any form. Forced 'broken_powerlaw' under
        cpp_compat.
    ssc_model : str
        'off' (default, exact no-op) or 'uniform' (one-zone
        synchrotron self-Compton seed, Atoyan & Aharonian 1996 factor
        2.24; cooling seed lagged one step, snapshots use a live
        kernel call). The 'compat' timestep bound is blind to SSC
        cooling; use 'total_loss10' with SSC on. Forced 'off' under
        cpp_compat.
    """
    if cpp_compat:
        integration = 'euler'
        timestep_events = False
        snapshot_mode = 'secant'
        timestep_cooling = 'compat'
        snapshot_injection = 'included'
        escape_model = 'off'
        injection_model = 'broken_powerlaw'
        ssc_model = 'off'
        if C.CONSTANTS_SOURCE != 'gsz2009':
            logger.warning(
                "cpp_compat=True but constants are '%s', not 'gsz2009'. "
                "Set VIPER_CONSTANTS=gsz2009 before import for "
                "bit-identical compat-mode output.",
                C.CONSTANTS_SOURCE,
            )

    # optimize_vtilda stays dormant in both modes
    if optimize_vtilda is None:
        optimize_vtilda = False
    if ic_early_break is None:
        ic_early_break = not cpp_compat
    if ic_j_break is None:
        ic_j_break = not cpp_compat

    if integration not in ('euler', 'pc_radius', 'pc_full'):
        raise ValueError(
            f"integration must be 'euler', 'pc_radius', or 'pc_full', "
            f"got {integration!r}"
        )
    if snapshot_mode not in ('secant', 'instantaneous'):
        raise ValueError(
            f"snapshot_mode must be 'secant' or 'instantaneous', "
            f"got {snapshot_mode!r}"
        )
    if timestep_cooling not in ('compat', 'total_loss10'):
        raise ValueError(
            f"timestep_cooling must be 'compat' or 'total_loss10', "
            f"got {timestep_cooling!r}"
        )
    if escape_model not in ESCAPE_MODELS:
        raise ValueError(
            f"escape_model must be one of {sorted(ESCAPE_MODELS)}, "
            f"got {escape_model!r}"
        )
    get_injection_shape(injection_model)  # fail fast on unknown names
    if ssc_model not in ('off', 'uniform'):
        raise ValueError(
            f"ssc_model must be 'off' or 'uniform', got {ssc_model!r}"
        )
    if snapshot_injection not in ('included', 'excluded'):
        raise ValueError(
            f"snapshot_injection must be 'included' or 'excluded', "
            f"got {snapshot_injection!r}"
        )
    if ssc_model != 'off' and timestep_cooling == 'compat':
        logger.warning(
            "ssc_model=%r with timestep_cooling='compat': the compat "
            "timestep bound is synchrotron-only and "
            "will not see strong SSC cooling; 'total_loss10' is "
            "recommended for SSC runs.", ssc_model)

    if ic_temps is None:
        ic_temps = []
    if ic_norms is None:
        ic_norms = []

    use_lookup, _lookup_reason = _cmb_lookup_compatible(
        use_cmb_lookup, cpp_compat, n_ic, f_max,
        n_elec_bins, n_phot_bins, n_ic_target,
    )
    if use_cmb_lookup and not use_lookup:
        logger.info("CMB lookup requested but skipped: %s", _lookup_reason)
    elif use_lookup:
        logger.info("Using CMB lookup table (fixed electron grid, no live IC kernel)")

    m = C.m_ejecta_index
    result = PWNModelResult()
    if f_max < 1:
        result.e_min_inj = float(e_min)
        result.e_max_inj = float(e_max)

    # --- SNR characteristic scales ---
    vt = calc_vt(esn_51, mej)
    a = calc_a(esn_51, vt)
    r_ch, t_ch, v_ch = calc_characteristic_scales(esn_51, mej, n_ism)
    tst = C.tst_star * t_ch
    trad = calc_trad(esn_51, n_ism)

    # --- Pulsar ---
    spindown_index = (braking_index + 1.0) / (braking_index - 1.0)
    esd_40 = e0_37 * 1.0e-3 * tau_sd * C.yr

    tmax = min(age, trad)
    if age > trad:
        logger.debug("age = %.1f > trad = %.1f, capping", age, trad)

    # --- Build grids ---
    if use_lookup:
        ebin_min = C.ELEC_GRID_FIXED_MIN
        ebin_max = C.ELEC_GRID_FIXED_MAX
    else:
        ebin_min = e_min / 1.0e5
        ebin_max = e_max * 2.0
        if f_max == 1:
            ebin_min = kT_max / 1000.0
            ebin_max = kT_max * 25.0
    log_ebin_max = np.log10(ebin_max)

    # injection bound, not the grid bound
    if f_max == 1:
        e_max_for_timestep = kT_max * 25.0
    else:
        e_max_for_timestep = e_max * 2.0
    log_e_max_for_timestep = np.log10(e_max_for_timestep)

    ne = n_elec_bins + 2
    np_ = n_phot_bins + 2
    nt = n_ic_target + 2

    elec_grid = make_log_grid(ebin_min, ebin_max, ne)
    phot_grid = make_log_grid(phot_freq_min, phot_freq_max, np_)
    target_grid = make_log_grid(target_freq_min, target_freq_max, nt)

    # cumulative; not reset per step
    esc_grid = make_log_grid(ebin_min, ebin_max, ne)

    # --- IC setup ---
    _fill_target_field(target_grid, n_ic_target, n_ic, ic_temps, ic_norms)

    uic = np.sum(
        target_grid.number * target_grid.bin_avg * C.h_eV * target_grid.bin_width
    )
    logger.info("Background photon field energy density = %.4f eV/cm^3", uic)

    if use_lookup:
        _arr, _meta = _load_cmb_lookup()
        assert _arr.shape == (ne, np_), (
            f"lookup pow_nu_ic shape {_arr.shape} != expected ({ne}, {np_}); "
            f"regenerate via tools/build_cmb_lookup.py --force"
        )
        pow_nu_ic = _arr
    else:
        logger.info("Computing IC cross-section + power tables (fused)...")
        pow_nu_ic = calc_sigma_and_pow_nu_ic(
            elec_grid, phot_grid, target_grid,
            ic_early_break=ic_early_break,
            ic_j_break=ic_j_break,
            ic_skip_empty_bins=ic_skip_empty_bins)
    pow_ic_elec = calc_pow_ic(phot_grid, pow_nu_ic)

    from .radiation import HAS_NUMBA
    logger.info("IC backend: %s", "Numba JIT" if HAS_NUMBA else "NumPy (install numba for ~10x speedup)")

    min_ic_bin = 0
    for i in range(ne - 1, -1, -1):
        if pow_ic_elec[i] > 0:
            min_ic_bin = i

    # --- SSC setup ---
    # _ssc_tgt: static target never mutated
    _ssc_on = ssc_model != 'off'
    if _ssc_on:
        logger.info("SSC on (ssc_model=%r): building K2 cooling matrix",
                    ssc_model)
        _k2_ssc = ic_cooling_matrix(elec_grid, phot_grid, target_grid)
        _tgt_dnu = target_grid.bin_width.copy()
        _ssc_tgt = make_log_grid(target_freq_min, target_freq_max, nt)

    def _snapshot_ic_arrays():
        """Full 2-D IC arrays for calc_photon_spectrum.

        SSC on: one live kernel call on the instantaneous seed, then a
        linear (not log) sum with the static arrays; pow_ic_elec and
        min_ic_bin are rederived from the combined table.
        """
        if not _ssc_on:
            return pow_nu_ic, pow_ic_elec, min_ic_bin
        _ssc_tgt.number[:] = ssc_seed_density(
            elec_grid, bpwn, rad_pwn, _ssc_tgt)
        _ssc_tgt.update_log_number()
        pnu_ssc = calc_sigma_and_pow_nu_ic(
            elec_grid, phot_grid, _ssc_tgt,
            ic_early_break=ic_early_break,
            ic_j_break=ic_j_break,
            ic_skip_empty_bins=ic_skip_empty_bins,
        )
        lin = (np.where(pow_nu_ic > LOG_EMPTY_CUT, 10.0 ** pow_nu_ic, 0.0)
               + np.where(pnu_ssc > LOG_EMPTY_CUT, 10.0 ** pnu_ssc, 0.0))
        pnu_tot = np.where(
            lin > 0.0, np.log10(np.maximum(lin, 1e-300)), LOG_EMPTY)
        pie_tot = calc_pow_ic(phot_grid, pnu_tot)
        nz = np.nonzero(pie_tot > 0)[0]
        return pnu_tot, pie_tot, (int(nz[0]) if nz.size else 0)

    # --- Initial conditions ---
    titer = 0.5
    titer = min(titer, tau_sd / 100.0)
    time = titer

    rad_psr = vel_psr * time * C.yr
    log_rad_pwn = (
        np.log10(1.50)
        + 0.2 * np.log10(m / (m - 5.0))
        + 0.5 * np.log10((m - 5.0) / (m - 3.0))
        + 0.1 * (
            3.0 * (51.0 + np.log10(esn_51))
            + 2.0 * (37.0 + np.log10(e0_37))
            - 5.0 * (np.log10(C.M_sun) + np.log10(mej))
        )
        + 1.2 * np.log10(time * C.yr)
    )
    rad_pwn = 10.0 ** log_rad_pwn
    log_vol_pwn = np.log10(4.0 / 3.0 * np.pi) + 3.0 * np.log10(rad_pwn)

    log_vel_pwn = (
        np.log10(1.50)
        + 0.2 * np.log10(m / (m - 5.0))
        + 0.5 * np.log10((m - 5.0) / (m - 3.0))
        + 0.1 * (
            3.0 * 51.0 + 3.0 * np.log10(esn_51)
            + 2.0 * 37.0 + 2.0 * np.log10(e0_37)
            - 5.0 * np.log10(C.M_sun) - 5.0 * np.log10(mej)
        )
        + np.log10(1.2) + 0.2 * np.log10(time * C.yr)
    )
    vel_pwn = 10.0 ** log_vel_pwn

    epwn_40 = (1.0 - eta_g) * initial_pwn_energy(
        time, spindown_index, esd_40, tau_sd
    )
    epwnkin_40 = (1.0 - eta_b) * epwn_40
    epwnkin2_40 = epwnkin_40
    epwnmag_40 = eta_b * epwn_40
    bpwn = bpwn_calc(epwnmag_40, log_vol_pwn)

    # Initial particle injection
    _inject_continuous = (
        _make_continuous_injector(
            injection_model, elec_grid, e_min, e_max, e_break, p1, p2)
        if f_max < 1 else None)
    if f_max < 1:
        _inject_continuous(elec_grid, epwnkin_40 * (1.0 - f_max))
    if f_max > 0:
        inject_maxwellian(elec_grid, epwnkin_40 * f_max, kT_max)

    # Energies deposited by the most recent step (for snapshot_injection).
    _last_pl_e = 0.0
    _last_mx_e = 0.0

    def _snapshot_grid():
        """Grid to record/radiate at a snapshot (see snapshot_injection)."""
        if snapshot_injection != 'excluded' or (
                _last_pl_e <= 0.0 and _last_mx_e <= 0.0):
            return elec_grid
        dep = elec_grid.copy()
        dep.reset_number(0.0)
        if f_max < 1 and _last_pl_e > 0.0:
            _inject_continuous(dep, _last_pl_e)
        if f_max > 0 and _last_mx_e > 0.0:
            inject_maxwellian(dep, _last_mx_e, kT_max)
        snap = elec_grid.copy()
        snap.number = np.maximum(elec_grid.number - dep.number, 0.0)
        snap.update_log_number()
        return snap

    rad_snr = calc_rad_snr(time, t_ch, r_ch)
    vel_snr = calc_vel_snr(time, t_ch, v_ch)
    log_p_snr = (
        np.log10(0.75) + np.log10(n_ism)
        + np.log10(C.m_proton) + 2.0 * np.log10(vel_snr)
    )
    p_snr = 10.0 ** log_p_snr
    rad_rs = calc_rad_rs(time, t_ch, r_ch, rad_snr)
    prs = calc_prs(time, t_ch, v_ch, a, vel_snr, p_snr, n_ism)

    # State variables
    msw_pwn = 0.0
    vel_ej_pwn = 0.0
    rho_ej_pwn = 0.0
    for_pwn = 0.0
    psnr_pwn = 0.0
    ppwn = 0.0
    tcol = -1.0
    depsr_40 = 0.0
    old_rad_pwn = 0.0
    old_msw_pwn = 0.0
    nelec_40 = 0.0
    eparfudge = 0.0
    stepnum = 1

    # Replay mode (benchmarks only)
    _replay_dts = (list(replay_dt_sequence)
                    if replay_dt_sequence is not None else None)
    _replay_idx = 0

    # stepper may only shorten steps
    _stepper = (_EventStepper(event_dt_floor_yr, t_ch, r_ch)
                if timestep_events else None)

    # geometry arrays shared read-only
    old_elec = SpectralGrid(
        bin_min=elec_grid.bin_min, bin_max=elec_grid.bin_max,
        bin_avg=elec_grid.bin_avg, bin_width=elec_grid.bin_width,
        log_min=elec_grid.log_min, log_max=elec_grid.log_max,
        log_avg=elec_grid.log_avg, log_width=elec_grid.log_width,
        number=elec_grid.number.copy(),
        log_number=elec_grid.log_number.copy(),
    )

    _es_scratch = (np.empty(ne), np.empty(ne), np.empty(ne), np.empty(ne))

    # --- Main loop ---
    while time < tmax:
        stepnum += 1

        # Count electrons before the copy
        nelec_40 = float(elec_grid.number[:n_elec_bins].sum())

        # SSC seed lagged one step
        if _ssc_on:
            _n_ssc = ssc_seed_density(elec_grid, bpwn, rad_pwn, _ssc_tgt)
            pow_ic_elec_step = pow_ic_elec + _k2_ssc @ (_n_ssc * _tgt_dnu)
            _nz = np.nonzero(pow_ic_elec_step > 0)[0]
            min_ic_bin_step = int(_nz[0]) if _nz.size else 0
        else:
            pow_ic_elec_step = pow_ic_elec
            min_ic_bin_step = min_ic_bin

        # Adaptive timestep
        _from_replay = (_replay_dts is not None
                        and _replay_idx < len(_replay_dts))
        if _from_replay:
            titer = float(_replay_dts[_replay_idx])
            _replay_idx += 1
        else:
            titer = calc_timestep(
                log_e_max_for_timestep, bpwn, nelec_40, rad_pwn, vel_pwn,
                rad_rs, time, tmax, rad_snr, for_pwn, msw_pwn,
            )
        if _stepper is not None or timestep_cooling == 'total_loss10':
            _capped = titer
            if timestep_cooling == 'total_loss10':
                _capped = min(_capped, _cooling_bound(
                    elec_grid, bpwn, nelec_40, rad_pwn, pow_ic_elec_step,
                    event_dt_floor_yr))
            if _stepper is not None:
                _capped = _stepper.cap_dt(_capped, time, rad_pwn,
                                          vel_pwn, tcol)
            if _capped < titer:
                if _from_replay:
                    # hand remainder back to sequence
                    _replay_dts.insert(_replay_idx, titer - _capped)
                titer = _capped
        time += titer

        # Save old state
        old_rad_pwn = rad_pwn
        old_msw_pwn = msw_pwn
        old_vel_pwn = vel_pwn
        eparfudge = epwnkin2_40 / epwnkin_40 if epwnkin_40 != 0 else 0.0

        np.copyto(old_elec.number, elec_grid.number)
        np.copyto(old_elec.log_number, elec_grid.log_number)
        # consumers reset phot/target grids
        elec_grid.reset_number()

        # Update positions
        rad_psr = vel_psr * time * C.yr
        rad_pwn = old_rad_pwn + vel_pwn * titer * C.yr

        if rad_pwn < 0:
            logger.warning("rad_pwn = %.3e < 0 at time = %.1f", rad_pwn, time)
            break

        log_vol_pwn = np.log10(4.0 / 3.0 * np.pi) + 3.0 * np.log10(rad_pwn)

        # Evolve particles
        epwnkin_40, rad_loss_40 = evolve_spectrum(
            old_elec, elec_grid, bpwn, old_rad_pwn, rad_pwn,
            vel_pwn, nelec_40, min_ic_bin_step, pow_ic_elec_step,
            scratch=_es_scratch,
        )

        # Escape applied before fresh injection
        esc_energy_40 = _apply_escape(
            elec_grid, esc_grid, escape_model, esc_norm, esc_index,
            bpwn, rad_pwn, vel_pwn, time, titer,
        )
        epwnkin_40 -= esc_energy_40

        # Adiabatic energy tracking
        if epwnkin2_40 > 0:
            log_ad = (
                np.log10(old_rad_pwn) - np.log10(rad_pwn)
                + np.log10(epwnkin2_40)
            )
            epwnkin2_40 = 10.0 ** log_ad - rad_loss_40
        else:
            log_ad = np.log10(old_rad_pwn) - np.log10(rad_pwn)
            epwnkin2_40 = 10.0 ** log_ad * epwnkin2_40 - rad_loss_40
        epwnkin2_40 -= esc_energy_40

        # Pulsar energy injection
        if rad_psr > rad_pwn and tcol != -1:
            depsr_40 = 0.0
            _last_pl_e = 0.0
            _last_mx_e = 0.0
        else:
            if spindown_index == 1:
                depsr_40 = (1.0 - eta_g) * esd_40 * (
                    np.log(1 + time / tau_sd)
                    - np.log(1 + (time - titer) / tau_sd)
                )
            else:
                depsr_40 = (
                    (1.0 - eta_g) * esd_40 / (1.0 - spindown_index)
                    * (
                        (1.0 + time / tau_sd) ** (1.0 - spindown_index)
                        - (1.0 + (time - titer) / tau_sd)
                        ** (1.0 - spindown_index)
                    )
                )

            # Inject new particles
            _last_pl_e = 0.0
            _last_mx_e = 0.0
            if f_max < 1:
                pl_e = depsr_40 * (1.0 - eta_b) * (1.0 - f_max)
                _inject_continuous(elec_grid, pl_e)
                _last_pl_e = pl_e
            if f_max > 0:
                _last_mx_e = depsr_40 * (1.0 - eta_b) * f_max
                inject_maxwellian(elec_grid, _last_mx_e, kT_max)

        epwnkin_40 += (1.0 - eta_b) * depsr_40
        epwnkin2_40 += (1.0 - eta_b) * depsr_40
        epwnmag_40_pre = epwnmag_40
        epwnmag_40 = epwnmag_40 * old_rad_pwn / rad_pwn + eta_b * depsr_40
        bpwn = bpwn_calc(epwnmag_40, log_vol_pwn)
        epwn_40 = epwnkin_40 + epwnmag_40

        log_pkin = (
            np.log10(max(epwnkin_40, 1e-300)) + 40.0 - C.log_3 - log_vol_pwn
        )
        log_pmag = np.log10(max(epwnmag_40, 1e-300)) + 40.0 - log_vol_pwn
        ppwn = 10.0 ** log_pkin + 10.0 ** log_pmag

        # --- SNR properties ---
        rad_snr = calc_rad_snr(time, t_ch, r_ch)
        if rad_pwn > rad_snr:
            break

        rad_rs = calc_rad_rs(time, t_ch, r_ch, rad_snr)
        if rad_pwn > rad_rs and tcol == -1:
            tcol = time
            logger.info("PWN collides with RS at t=%.1f yr", tcol)

        rad_core = vt * time * C.yr
        vel_snr = calc_vel_snr(time, t_ch, v_ch)

        # Pressure and mass outside PWN
        (psnr_pwn, msw_pwn, vel_ej_pwn, rho_ej_pwn,
         p_snr, prs) = _snr_boundary_state(
            time, rad_pwn, old_rad_pwn, old_msw_pwn, vel_pwn,
            tcol, log_vol_pwn, a, m, rad_core, rad_rs, rad_snr,
            t_ch, v_ch, tst, vel_snr, n_ism,
            psnr_pwn, msw_pwn, vel_ej_pwn, rho_ej_pwn, p_snr, prs,
            optimize_vtilda, corrector=False,
        )

        # Momentum equation
        for_pwn, vel_pwn = _momentum_update(
            ppwn, psnr_pwn, rad_pwn, msw_pwn, old_msw_pwn,
            vel_pwn, vel_ej_pwn, titer, vel_pwn,
        )

        # Predictor-corrector: correct radius (and optionally velocity)
        if integration != 'euler':
            rad_pwn_corrected = (
                old_rad_pwn
                + 0.5 * (old_vel_pwn + vel_pwn) * titer * C.yr
            )
            if rad_pwn_corrected > 0:
                rad_pwn = rad_pwn_corrected
                log_vol_pwn = (
                    np.log10(4.0 / 3.0 * np.pi)
                    + 3.0 * np.log10(rad_pwn)
                )
                epwnmag_40 = (
                    epwnmag_40_pre * old_rad_pwn / rad_pwn
                    + eta_b * depsr_40
                )
                bpwn = bpwn_calc(epwnmag_40, log_vol_pwn)
                epwn_40 = epwnkin_40 + epwnmag_40
                log_pkin = (
                    np.log10(max(epwnkin_40, 1e-300))
                    + 40.0 - C.log_3 - log_vol_pwn
                )
                log_pmag = (
                    np.log10(max(epwnmag_40, 1e-300))
                    + 40.0 - log_vol_pwn
                )
                ppwn = 10.0 ** log_pkin + 10.0 ** log_pmag

                # Full PC: also correct velocity at corrected radius
                if integration == 'pc_full':
                    (psnr_pwn, msw_pwn, vel_ej_pwn, rho_ej_pwn,
                     p_snr, prs) = _snr_boundary_state(
                        time, rad_pwn, old_rad_pwn, old_msw_pwn, vel_pwn,
                        tcol, log_vol_pwn, a, m, rad_core, rad_rs,
                        rad_snr, t_ch, v_ch, tst, vel_snr, n_ism,
                        psnr_pwn, msw_pwn, vel_ej_pwn, rho_ej_pwn,
                        p_snr, prs, optimize_vtilda, corrector=True,
                    )
                    for_pwn, vel_pwn = _momentum_update(
                        ppwn, psnr_pwn, rad_pwn, msw_pwn, old_msw_pwn,
                        old_vel_pwn, vel_ej_pwn, titer, vel_pwn,
                    )

        if _stepper is not None:
            _stepper.observe(time, rad_pwn, vel_pwn)

        # --- Intermediate snapshots ---
        if dyn_step > 0 and stepnum % dyn_step == 0:
            dyn = _make_dynstate(
                time, rad_pwn, rad_snr, rad_rs, rad_psr, bpwn, vel_pwn,
                epwn_40, epwnmag_40, epwnkin_40, vel_ej_pwn, msw_pwn,
                ppwn, psnr_pwn, rho_ej_pwn, vel_snr, eparfudge, titer,
                e0_37, tau_sd, braking_index, depsr_40,
            )
            result.dyninfo.append(dyn)

        if (elec_step > 0 and stepnum % elec_step == 0) or (
                phot_step > 0 and stepnum % phot_step == 0):
            _snap = _snapshot_grid()

        if elec_step > 0 and stepnum % elec_step == 0:
            result.elecinfo.append((time, _snap.number.copy()))
            if escape_model != 'off':
                result.escinfo.append((time, esc_grid.number.copy()))

        if phot_step > 0 and stepnum % phot_step == 0:
            phot_grid.reset_number()
            orad = (old_rad_pwn if old_rad_pwn != rad_pwn
                    else 0.99 * rad_pwn)
            _pnu_snap, _pie_snap, _mib_snap = _snapshot_ic_arrays()
            calc_photon_spectrum(
                _snap, phot_grid, bpwn, orad, rad_pwn,
                vel_pwn, _mib_snap, _pnu_snap, _pie_snap, titer,
                snapshot_mode=snapshot_mode,
            )
            result.radinfo.append((time, phot_grid.number.copy()))

    if _stepper is not None:
        _stepper.finalize(time)
        result.events = _stepper.episodes

    # --- Final output ---
    dyn = _make_dynstate(
        time, rad_pwn, rad_snr, rad_rs, rad_psr, bpwn, vel_pwn,
        epwn_40, epwnmag_40, epwnkin_40, vel_ej_pwn, msw_pwn,
        ppwn, psnr_pwn, rho_ej_pwn, vel_snr, eparfudge, titer,
        e0_37, tau_sd, braking_index, depsr_40,
    )
    result.dyninfo.append(dyn)

    # Final photon spectrum
    target_grid.reset_number()
    _fill_target_field(target_grid, n_ic_target, n_ic, ic_temps, ic_norms)

    phot_grid.reset_number()
    orad = old_rad_pwn if old_rad_pwn != rad_pwn else 0.99 * rad_pwn
    _pnu_snap, _pie_snap, _mib_snap = _snapshot_ic_arrays()
    _snap = _snapshot_grid()
    calc_photon_spectrum(
        _snap, phot_grid, bpwn, orad, rad_pwn,
        vel_pwn, _mib_snap, _pnu_snap, _pie_snap, titer,
        snapshot_mode=snapshot_mode,
    )

    result.elec_grid = elec_grid
    result.phot_grid = phot_grid
    result.electron_spectrum = _snap.number.copy()
    result.electron_energies = elec_grid.bin_avg.copy()
    result.photon_spectrum = phot_grid.number.copy()
    result.photon_freqs = phot_grid.bin_avg.copy()
    result.escaped_spectrum = esc_grid.number.copy()

    # Append final snapshot to history lists
    if elec_step > 0:
        result.elecinfo.append((time, result.electron_spectrum.copy()))
        if escape_model != 'off':
            result.escinfo.append((time, result.escaped_spectrum.copy()))
    if phot_step > 0:
        result.radinfo.append((time, result.photon_spectrum.copy()))

    logger.info("Evolution complete: %d steps, final t=%.1f yr, "
                "R_pwn=%.2f pc, B=%.2f uG",
                stepnum, time, rad_pwn / C.pc, bpwn * 1e6)

    return result
