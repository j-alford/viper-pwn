#!/usr/bin/env python3
"""Generate cached CMB pow_nu_ic for evolve_pwn(use_cmb_lookup=True).

Usage (from repo root):
    VIPER_CONSTANTS=gsz2009 python3 tools/build_cmb_lookup.py
    python3 tools/build_cmb_lookup.py --force  # overwrite existing

Output: viper/data/cmb_pow_nu_ic_{gsz2009,astropy}.npz, ~2 MB each.
This file ships with the package via [tool.setuptools.package-data].
Regenerate any time the fixed-grid constants or T_CMB change.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, REPO)

from viper import constants as C
from viper.grids import make_log_grid
from viper.radiation import calc_sigma_and_pow_nu_ic
from viper.evolution import _fill_target_field

OUT = os.path.join(REPO, 'viper', 'data',
                    f'cmb_pow_nu_ic_{C.CONSTANTS_SOURCE}.npz')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    if os.path.exists(OUT) and not args.force:
        print(f"{OUT} exists. Use --force to overwrite.")
        sys.exit(1)

    ne = C.ELEC_GRID_FIXED_NBINS + 2
    np_ = C.N_PHOT_FREQ + 2
    nt = C.N_IC_TARGET + 2

    elec_grid = make_log_grid(C.ELEC_GRID_FIXED_MIN, C.ELEC_GRID_FIXED_MAX, ne)
    phot_grid = make_log_grid(C.PHOT_FREQ_MIN, C.PHOT_FREQ_MAX, np_)
    target_grid = make_log_grid(C.TARGET_FREQ_MIN, C.TARGET_FREQ_MAX, nt)
    _fill_target_field(target_grid, C.N_IC_TARGET, n_ic=0, ic_temps=[], ic_norms=[])

    print(f"Computing IC table for fixed grid: {ne}x{np_}x{nt}, T_CMB={C.T_CMB} K...")
    t0 = time.perf_counter()
    pow_nu_ic = calc_sigma_and_pow_nu_ic(
        elec_grid, phot_grid, target_grid,
        ic_early_break=True, ic_j_break=True, ic_skip_empty_bins=True,
    )
    dt = time.perf_counter() - t0
    print(f"  Done in {dt:.1f} s. Shape={pow_nu_ic.shape}, "
          f"size={pow_nu_ic.nbytes / 1e6:.2f} MB")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(
        OUT, pow_nu_ic=pow_nu_ic,
        elec_grid_min=C.ELEC_GRID_FIXED_MIN,
        elec_grid_max=C.ELEC_GRID_FIXED_MAX,
        elec_grid_nbins=C.ELEC_GRID_FIXED_NBINS,
        phot_freq_min=C.PHOT_FREQ_MIN, phot_freq_max=C.PHOT_FREQ_MAX,
        n_phot_freq=C.N_PHOT_FREQ,
        target_freq_min=C.TARGET_FREQ_MIN, target_freq_max=C.TARGET_FREQ_MAX,
        n_ic_target=C.N_IC_TARGET,
        t_cmb=C.T_CMB,
        constants_source=C.CONSTANTS_SOURCE,
    )
    print(f"Wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == '__main__':
    main()
