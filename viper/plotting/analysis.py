"""Synchrotron/IC luminosity split from a model result."""

from __future__ import annotations

import numpy as np

from .style import _h_eV




# ===================================================================
# Luminosity computation (synchrotron / IC separation)
# ===================================================================

def compute_luminosities(result, print_summary: bool = True) -> dict:
    """Compute total, synchrotron, and IC luminosities from model output.

    Integrates L_nu over the photon spectrum, splitting at the valley
    between the synchrotron and IC humps.

    Parameters
    ----------
    result : PWNModelResult
        Model output from ``PWNModel.run()`` or ``evolve_pwn()``.
    print_summary : bool
        If True, print a formatted summary table.

    Returns
    -------
    dict with keys:
        L_total    - total bolometric luminosity [erg/s]
        L_synch    - synchrotron luminosity [erg/s]
        L_ic       - inverse Compton luminosity [erg/s]
        nu_split   - crossover frequency [Hz]
        E_split_eV - crossover energy [eV]
    """
    freq = result.photon_freqs
    spec = result.photon_spectrum
    phot_grid = result.phot_grid

    mask = spec > 0
    if not np.any(mask):
        empty = dict(L_total=0.0, L_synch=0.0, L_ic=0.0,
                     nu_split=0.0, E_split_eV=0.0)
        if print_summary:
            print("No photon spectrum available.")
        return empty

    lnu = spec * 1e40
    dnu = phot_grid.bin_width

    l_total = np.sum(lnu[mask] * dnu[mask])

    nulnu = lnu * freq
    nulnu_masked = np.where(mask, nulnu, 0.0)

    log_nulnu = np.full_like(nulnu, -np.inf)
    log_nulnu[mask] = np.log10(nulnu[mask])

    i_valley = _find_sed_valley(freq, log_nulnu, mask)
    nu_split = freq[i_valley]

    lo = mask & (np.arange(len(freq)) <= i_valley)
    hi = mask & (np.arange(len(freq)) > i_valley)

    l_synch = np.sum(lnu[lo] * dnu[lo]) if np.any(lo) else 0.0
    l_ic = np.sum(lnu[hi] * dnu[hi]) if np.any(hi) else 0.0

    out = dict(
        L_total=l_total,
        L_synch=l_synch,
        L_ic=l_ic,
        nu_split=nu_split,
        E_split_eV=nu_split * _h_eV,
    )

    if print_summary:
        print(f"{'Luminosity component':<25s} {'erg/s':>14s}")
        print("-" * 42)
        print(f"{'Synchrotron':<25s} {l_synch:>14.4e}")
        print(f"{'Inverse Compton':<25s} {l_ic:>14.4e}")
        print(f"{'Total':<25s} {l_total:>14.4e}")
        e_split = nu_split * _h_eV
        if e_split >= 1e9:
            split_str = f'{e_split/1e9:.1f} GeV'
        elif e_split >= 1e6:
            split_str = f'{e_split/1e6:.1f} MeV'
        elif e_split >= 1e3:
            split_str = f'{e_split/1e3:.1f} keV'
        else:
            split_str = f'{e_split:.1f} eV'
        print(f"{'Split energy':<25s} {split_str:>14s}")

    return out


def _find_sed_valley(freq, log_nulnu, mask):
    """Find the frequency of the valley between synch and IC humps.

    Searches for the minimum of nu*L_nu between the two peaks.
    Falls back to 1e18 Hz if no clear valley is found.
    """
    valid = np.where(mask & (log_nulnu > -np.inf))[0]
    if len(valid) < 10:
        default_idx = np.argmin(np.abs(np.log10(freq) - 18.0))
        return default_idx

    vals = log_nulnu[valid]
    freqs = freq[valid]

    i_peak1 = np.argmax(vals[:len(vals) // 2])
    i_peak2 = len(vals) // 2 + np.argmax(vals[len(vals) // 2:])

    if i_peak2 <= i_peak1 + 2:
        default_idx = np.argmin(np.abs(np.log10(freq) - 18.0))
        return default_idx

    search = vals[i_peak1:i_peak2 + 1]
    i_min_local = np.argmin(search)
    i_valley_valid = i_peak1 + i_min_local

    return valid[i_valley_valid]
