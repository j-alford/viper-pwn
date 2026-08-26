"""Regression test: today's Python output vs the committed grid snapshots.

Regenerate: tools/build_grid_python.py --force
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
GRID = os.path.join(REPO, 'tests', 'reference', 'grid')

sys.path.insert(0, REPO)
from viper import constants as C
from viper.scalar_params import PWNScalarParams
from viper.evolution import evolve_pwn

# snapshots are gsz2009-only
pytestmark = pytest.mark.skipif(
    C.CONSTANTS_SOURCE != 'gsz2009',
    reason='requires VIPER_CONSTANTS=gsz2009 (gsz2009 snapshots)',
)

REGRESSION_REL_TOL = 1e-10
REGRESSION_ABS_TOL = 1e-300   # effectively just relative

DYN_FIELDS = [
    'time', 'rad_pwn', 'rad_snr', 'rad_rs', 'rad_psr',
    'bpwn', 'ad_lum_40', 'synch_lum_40', 'ic_lum_40',
    'edot_40', 'epwn_50', 'epwnb_50', 'epwnp_50',
    'vpwn', 'vejpwn', 'mswpwn', 'ppwn', 'psnrpwn',
    'rhoejpwn', 'vsnr', 'eparfudge', 'delta_t',
]


def pytest_generate_tests(metafunc):
    """Parametrize across all grid points that have python snapshots."""
    if 'point_name' in metafunc.fixturenames:
        path = os.path.join(GRID, 'grid.json')
        if not os.path.exists(path):
            metafunc.parametrize('point_name', [], ids=[])
            return
        manifest = json.load(open(path))
        names = []
        for p in manifest['points']:
            npz = os.path.join(GRID, 'python', f'{p["name"]}.npz')
            if os.path.exists(npz):
                names.append(p['name'])
        metafunc.parametrize('point_name', names, ids=names)


@pytest.fixture(scope='module')
def grid_points():
    path = os.path.join(GRID, 'grid.json')
    manifest = json.load(open(path))
    units = manifest.get('ic_norm_units')
    if units != 'eV_per_cm3':
        raise RuntimeError(
            f'grid.json ic_norm_units={units!r}, expected "eV_per_cm3". '
            f'The manifest predates the eV/cm^3 migration and the snapshots '
            f'would be regressed against the wrong IC-field semantics. '
            f'Run tools/migrate_grid_ic_norms.py and regenerate.'
        )
    return {p['name']: p for p in manifest['points']}


def _run_python(point: dict):
    params = PWNScalarParams(
        esn=point['esn'], mej=point['mej'], n_ism=point['n_ism'],
        braking_index=point['braking_index'], tau_sd=point['tau_sd'],
        period=point['period'], p_dot=point['p_dot'],
        eta_b=point['eta_b'],
        e_min=point['e_min'], e_max=point['e_max'],
        e_break=point['e_break'], p1=point['p1'], p2=point['p2'],
        f_max=point.get('f_max', 0.0), kT_max=point.get('kT_max', 0.0),
        eta_g=point.get('eta_g', 0.0), vel_psr=point.get('vel_psr', 0.0),
        n_ic=len(point.get('ic_temps', [])),
        ic_temps=list(point.get('ic_temps', [])),
        ic_norms=list(point.get('ic_norms', [])),
    )
    return evolve_pwn(
        cpp_compat=True,
        esn_51=params.esn, mej=params.mej, n_ism=params.n_ism,
        braking_index=params.braking_index, tau_sd=params.tau_sd,
        age=params.true_age, e0_37=params.e0_37,
        vel_psr=params.vel_psr, eta_g=params.eta_g, eta_b=params.eta_b,
        e_min=params.e_min, e_max=params.e_max,
        e_break=params.e_break, p1=params.p1, p2=params.p2,
        f_max=params.f_max, kT_max=params.kT_max,
        n_ic=params.n_ic,
        ic_temps=params.ic_temps, ic_norms=params.ic_norms,
        dyn_step=-1, elec_step=-1, phot_step=-1,
    )


def test_python_regression(point_name, grid_points):
    snapshot_path = os.path.join(GRID, 'python', f'{point_name}.npz')
    snapshot = np.load(snapshot_path)

    result = _run_python(grid_points[point_name])

    failures = []
    # Dynamics
    dyn = result.dyninfo[-1]
    for f in DYN_FIELDS:
        old = float(snapshot[f'dyn_{f}'])
        new = float(getattr(dyn, f))
        if not np.isclose(new, old,
                           rtol=REGRESSION_REL_TOL,
                           atol=REGRESSION_ABS_TOL,
                           equal_nan=True):
            failures.append(
                f'  dyn.{f}: snapshot={old!r} now={new!r} '
                f'rel={abs(new-old)/max(abs(old), 1e-300):.2e}'
            )

    # Spectra
    for arr_name, attr_name in [
        ('elec_spectrum',  'electron_spectrum'),
        ('phot_spectrum',  'photon_spectrum'),
    ]:
        old = snapshot[arr_name]
        new = getattr(result, attr_name)
        n = min(len(old), len(new))
        if not np.allclose(new[:n], old[:n],
                            rtol=REGRESSION_REL_TOL,
                            atol=REGRESSION_ABS_TOL,
                            equal_nan=True):
            diff = np.abs(new[:n] - old[:n])
            rel_diff = diff / np.maximum(np.abs(old[:n]), 1e-300)
            failures.append(
                f'  {arr_name}: max abs diff {diff.max():.2e}, '
                f'max rel diff {rel_diff.max():.2e} '
                f'(at index {int(np.argmax(rel_diff))})'
            )

    assert not failures, (
        f'\n{point_name}: Python regression: output drifted from '
        f'committed snapshot:\n' + '\n'.join(failures) +
        '\n\nIf this is intentional, regenerate the snapshot with:\n'
        f'  VIPER_CONSTANTS=gsz2009 python3 tools/build_grid_python.py '
        f'--names {point_name} --force\n'
    )
