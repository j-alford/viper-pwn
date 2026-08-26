"""
Particle injection spectra and bin manipulation.

Handles power-law, broken power-law, and relativistic Maxwellian
injection into the electron energy grid.
"""

from __future__ import annotations

import numpy as np
from scipy.special import kn as bessel_kn

from . import constants as C
from .grids import SpectralGrid


def inject_powerlaw(
    grid: SpectralGrid,
    energy_40: float,
    e_min: float,
    e_max: float,
    p: float,
) -> None:
    """Inject a power-law dN/dE ~ E^{-p} particle spectrum into *grid*.

    Parameters
    ----------
    grid : SpectralGrid
        Electron spectrum grid (modified in-place).
    energy_40 : float
        Total energy to inject [10^40 erg].
    e_min, e_max : float
        Min/max injection energy [GeV].
    p : float
        Spectral index.
    """
    if energy_40 <= 0:
        return
    erg = C.GeV_per_erg
    if p == 2:
        n0 = energy_40 * erg / np.log(e_max / e_min)
    else:
        n0 = energy_40 * erg * (2.0 - p) / (e_max ** (2.0 - p) - e_min ** (2.0 - p))

    for i in range(grid.n_bins):
        if grid.bin_max[i] > e_min and grid.bin_min[i] < e_max:
            lo = max(grid.bin_min[i], e_min)
            hi = min(grid.bin_max[i], e_max)
            if p == 1:
                grid.number[i] += n0 * np.log(hi / lo)
            else:
                grid.number[i] += n0 / (1.0 - p) * (hi ** (1.0 - p) - lo ** (1.0 - p))

    grid.update_log_number()


def inject_broken_powerlaw(
    grid: SpectralGrid,
    energy_40: float,
    e_min: float,
    e_max: float,
    e_break: float,
    p1: float,
    p2: float,
) -> None:
    """Inject a broken power-law particle spectrum.

    Below e_break: dN/dE ~ n_break * (E_break)^{p1} * E^{-p1}
    Above e_break: dN/dE ~ n_break * (E_break)^{p2} * E^{-p2}

    Parameters
    ----------
    grid : SpectralGrid
        Electron spectrum grid (modified in-place).
    energy_40 : float
        Total energy to inject [10^40 erg].
    e_min, e_max : float
        Min/max injection energy [GeV].
    e_break : float
        Break energy [GeV].
    p1, p2 : float
        Spectral indices below and above the break.
    """
    if energy_40 <= 0:
        return
    erg = C.GeV_per_erg

    if p1 == 2:
        n1 = np.log(e_break / e_min)
    else:
        n1 = (e_break ** (2.0 - p1) - e_min ** (2.0 - p1)) / (2.0 - p1)

    if p2 == 2:
        n2 = np.log(e_max / e_break)
    else:
        n2 = (e_max ** (2.0 - p2) - e_break ** (2.0 - p2)) / (2.0 - p2)

    n_break = energy_40 * erg / (e_break**p1 * n1 + e_break**p2 * n2)

    bmin = grid.bin_min
    bmax = grid.bin_max

    mask1 = (bmax >= e_min) & (bmin < e_break)
    if np.any(mask1):
        lo1 = np.maximum(bmin[mask1], e_min)
        hi1 = np.minimum(bmax[mask1], e_break)
        if p1 == 1:
            grid.number[mask1] += n_break * e_break * np.log(hi1 / lo1)
        else:
            grid.number[mask1] += (
                n_break * e_break**p1 / (1.0 - p1)
                * (hi1 ** (1.0 - p1) - lo1 ** (1.0 - p1))
            )

    mask2 = (bmax >= e_break) & (bmin <= e_max)
    if np.any(mask2):
        lo2 = np.maximum(bmin[mask2], e_break)
        hi2 = np.minimum(bmax[mask2], e_max)
        if p2 == 1:
            grid.number[mask2] += n_break * e_break * np.log(hi2 / lo2)
        else:
            grid.number[mask2] += (
                n_break * e_break**p2 / (1.0 - p2)
                * (hi2 ** (1.0 - p2) - lo2 ** (1.0 - p2))
            )

    grid.update_log_number()


def inject_maxwellian(
    grid: SpectralGrid,
    energy_40: float,
    kT: float,
) -> None:
    """Inject a relativistic Maxwellian particle spectrum.

    Parameters
    ----------
    grid : SpectralGrid
        Electron spectrum grid (modified in-place).
    energy_40 : float
        Total energy to inject [10^40 erg].
    kT : float
        Temperature [GeV].
    """
    if energy_40 <= 0:
        return

    mec2_gev = C.mec2_keV * 1.0e-6  # GeV
    theta = kT / mec2_gev  # dimensionless temperature

    # Norm integral: int gamma sqrt(gamma^2-1) exp(-gamma/theta) dgamma
    inv_theta = 1.0 / theta
    norm_int = (bessel_kn(4, inv_theta) - bessel_kn(0, inv_theta)) / 8.0
    log_norm_int = np.log10(norm_int)

    log_max_norm = np.log10(energy_40) + C.log_erg - np.log10(mec2_gev) - log_norm_int
    max_norm = 10.0**log_max_norm

    for i in range(grid.n_bins - 1):
        gamma = grid.bin_avg[i] / mec2_gev + 1.0
        dgamma = grid.bin_width[i] / mec2_gev
        grid.number[i] += (
            max_norm * gamma * np.sqrt(gamma**2 - 1.0) * np.exp(-gamma / theta) * dgamma
        )

    grid.update_log_number()


def inject_particles(
    grid: SpectralGrid,
    energy_40: float,
    e_min: float,
    e_max: float,
    e_break: float,
    p1: float,
    p2: float,
    f_max: float,
    kT_max: float,
    injection_model: str = 'broken_powerlaw',
) -> None:
    """Inject particles according to the combined spectrum specification.

    ``injection_model`` selects the continuous component's shape; the
    default ``'broken_powerlaw'`` uses the analytic path, with the
    ``e_break == 0 or p2 == 0`` single-power-law convention.
    """
    if f_max < 1:
        pl_energy = energy_40 * (1.0 - f_max)
        if injection_model == 'broken_powerlaw':
            if e_break == 0 or p2 == 0:
                inject_powerlaw(grid, pl_energy, e_min, e_max, p1)
            else:
                inject_broken_powerlaw(grid, pl_energy, e_min, e_max, e_break, p1, p2)
        elif pl_energy > 0:
            from .injection import injection_bin_counts
            grid.number += pl_energy * injection_bin_counts(
                injection_model, grid, e_min, e_max, e_break, p1, p2)
            grid.update_log_number()

    if f_max > 0:
        inject_maxwellian(grid, energy_40 * f_max, kT_max)


def place_in_bins_batch(
    new_mins: np.ndarray,
    new_maxs: np.ndarray,
    values: np.ndarray,
    occupied: np.ndarray,
    grid: SpectralGrid,
) -> None:
    """Batch redistribute cooled electron bins onto a new grid.

    After cooling, each old bin's energy range shifts to
    [new_min, new_max]; its particle count is distributed across the
    new-grid bins it overlaps, weighted by fractional overlap.

    Parameters
    ----------
    new_mins, new_maxs : arrays of new energy edges for each old bin.
    values : particle counts per old bin.
    occupied : boolean mask of bins with particles.
    grid : destination SpectralGrid (modified in-place).
    """
    n_bins = grid.n_bins
    abs_max = grid.bin_max[n_bins - 1]
    grid_bin_min = grid.bin_min
    grid_bin_max = grid.bin_max

    occ_idx = np.where(occupied)[0]
    if len(occ_idx) == 0:
        return

    nm = new_mins[occ_idx].copy()
    nx = new_maxs[occ_idx].copy()
    vals = values[occ_idx]

    nx = np.minimum(nx, abs_max - 1.0)
    swap = nx < nm
    nm[swap], nx[swap] = nx[swap].copy(), nm[swap].copy()
    widths = nx - nm

    idx_lo = np.clip(np.searchsorted(grid_bin_max, nm, side='left'), 0, n_bins - 1)
    idx_hi = np.clip(np.searchsorted(grid_bin_min, nx, side='right') - 1, 0, n_bins - 1)

    span = idx_hi - idx_lo

    # --- Fast path 1: single-bin (span == 0) ---
    s1 = span == 0
    if np.any(s1):
        lo1 = idx_lo[s1]
        copy_min = np.maximum(nm[s1], grid_bin_min[lo1])
        copy_max = np.minimum(nx[s1], grid_bin_max[lo1])
        w1 = widths[s1]
        with np.errstate(divide='ignore', invalid='ignore'):
            frac = np.where(w1 > 0, (copy_max - copy_min) / w1, 1.0)
        np.add.at(grid.number, lo1, np.clip(frac, 0.0, 1.0) * vals[s1])

    # --- Fast path 2: two-bin (span == 1) ---
    s2 = span == 1
    if np.any(s2):
        lo2 = idx_lo[s2]
        hi2 = idx_hi[s2]
        w2 = widths[s2]
        nm2 = nm[s2]
        nx2 = nx[s2]
        v2 = vals[s2]

        lo_copy_min = np.maximum(nm2, grid_bin_min[lo2])
        lo_copy_max = grid_bin_max[lo2]
        frac_lo = np.where(w2 > 0, (lo_copy_max - lo_copy_min) / w2, 0.5)
        frac_lo = np.clip(frac_lo, 0.0, 1.0)

        hi_copy_min = grid_bin_min[hi2]
        hi_copy_max = np.minimum(nx2, grid_bin_max[hi2])
        frac_hi = np.where(w2 > 0, (hi_copy_max - hi_copy_min) / w2, 0.5)
        frac_hi = np.clip(frac_hi, 0.0, 1.0)

        np.add.at(grid.number, lo2, frac_lo * v2)
        np.add.at(grid.number, hi2, frac_hi * v2)

    # --- Slow path: span >= 2 (rare) ---
    s3 = span >= 2
    multi_idx = np.where(s3)[0]
    for k in multi_idx:
        lo = idx_lo[k]
        hi = idx_hi[k]
        w = widths[k]
        v = vals[k]
        nmin_k = nm[k]
        nmax_k = nx[k]

        for i in range(lo, hi + 1):
            if nmin_k <= grid_bin_max[i] and nmax_k >= grid_bin_min[i]:
                copy_min = nmin_k if nmin_k > grid_bin_min[i] else grid_bin_min[i]
                copy_max = nmax_k if nmax_k < grid_bin_max[i] else grid_bin_max[i]
                if w > 0:
                    frac = (copy_max - copy_min) / w
                else:
                    frac = 1.0
                if 0 < frac <= 1:
                    grid.number[i] += frac * v


def place_in_bins(
    new_min: float,
    new_max: float,
    value: float,
    grid: SpectralGrid,
) -> None:
    """Distribute *value* across bins proportional to overlap with [new_min, new_max].
    """
    n_bins = grid.n_bins
    abs_max = grid.bin_max[n_bins - 1]

    if new_max > abs_max:
        new_max = abs_max - 1.0

    if new_max < new_min:
        new_min, new_max = new_max, new_min

    new_width = new_max - new_min

    log_bin_min = grid.log_max[0]
    log_bin_max = grid.log_max[n_bins - 2]
    log_bin_iter = (log_bin_max - log_bin_min) / (n_bins - 1)
    bin_buf = 3

    if new_min > 0:
        guess_min = int(round((np.log10(new_min) - log_bin_min) / log_bin_iter))
        guess_min = max(0, guess_min - bin_buf)
        guess_min = min(n_bins - 1, guess_min)
    else:
        guess_min = 0

    guess_max = int(round((np.log10(new_max) - log_bin_min) / log_bin_iter))
    guess_max = max(0, guess_max + bin_buf)
    guess_max = min(n_bins - 1, guess_max)

    for i in range(guess_min, guess_max + 1):
        if new_min <= grid.bin_max[i] and new_max >= grid.bin_min[i]:
            copy_min = max(new_min, grid.bin_min[i])
            copy_max = min(new_max, grid.bin_max[i])
            if new_width > 0:
                frac = (copy_max - copy_min) / new_width
            else:
                frac = 1.0
            if 0 < frac <= 1:
                grid.number[i] += frac * value
                if grid.number[i] > 0:
                    grid.log_number[i] = np.log10(grid.number[i])


def integrate_electron_energy(grid: SpectralGrid) -> float:
    """Total energy in the electron spectrum [10^40 erg].
    """
    mask = grid.number > 0
    if not np.any(mask):
        return 0.0
    log_e = grid.log_avg[mask] - C.log_erg + grid.log_number[mask]
    return np.sum(10.0**log_e)
