"""Observed orders of convergence in timestep, electron bins and photon bins."""
import json
from pathlib import Path

import numpy as np
import pytest

import viper.constants as C
from viper.evolution import evolve_pwn
from viper.scalar_params import PWNScalarParams

from _analytic import (ORIGINAL_SCHEME, loguniform_dts, pair_orders,
                       refine_dts)

pytestmark = pytest.mark.slow

GRID_JSON = Path(__file__).resolve().parent / 'reference' / 'grid' / 'grid.json'
INTEGRATORS = ['euler', 'pc_full']
BANDS = {'radio': (1e9, 1e10), 'xray': (2.4e17, 2.4e18), 'tev': (2.4e26, 2.4e27)}
NUS = (1e9, 1e12, 1e15, 1e18, 1e24, 1e26)


def _base_kw():
    with open(GRID_JSON) as f:
        pt = {x['name']: x for x in json.load(f)['points']}['g54_baseline']
    keys = ('esn', 'mej', 'n_ism', 'braking_index', 'tau_sd', 'period', 'p_dot',
            'eta_b', 'e_min', 'e_max', 'e_break', 'p1', 'p2', 'f_max', 'kT_max',
            'eta_g', 'vel_psr', 'ic_temps', 'ic_norms')
    p = PWNScalarParams(**{k: pt[k] for k in keys})
    return dict(esn_51=p.esn, mej=p.mej, n_ism=p.n_ism,
                braking_index=p.braking_index, tau_sd=p.tau_sd,
                age=float(p.true_age), e0_37=float(p.e0_37), vel_psr=0.0,
                eta_g=0.0, eta_b=p.eta_b, e_min=p.e_min, e_max=p.e_max,
                e_break=p.e_break, p1=p.p1, p2=p.p2, f_max=0.0, kT_max=0.0,
                n_ic=p.n_ic, ic_temps=p.ic_temps, ic_norms=p.ic_norms,
                dyn_step=1, elec_step=-1, phot_step=-1,
                snapshot_mode='instantaneous', **ORIGINAL_SCHEME)


BASE = _base_kw()
# free expansion, constant Edot, no radiation
CLEAN = dict(BASE, tau_sd=1e9, n_ism=1e-5, eta_b=1e-9, e_min=0.1, e_max=1e3,
             e_break=100.0, n_ic=0, ic_temps=[], ic_norms=[])


def _functionals(r):
    d = r.dyninfo[-1]
    g = r.phot_grid
    lw = r.photon_spectrum * g.bin_width
    out = dict(R=d.rad_pwn, v=d.vpwn, B=d.bpwn, Ep=d.epwnp_50, Eb=d.epwnb_50,
               P=d.ppwn, Ltot=lw.sum())
    for b, (lo, hi) in BANDS.items():
        out[b] = lw[(g.bin_avg > lo) & (g.bin_avg < hi)].sum()
    return out


def _native_dts(kw, integration):
    r = evolve_pwn(integration=integration, **kw)
    return [float(x.delta_t) for x in r.dyninfo][:-1]


def _ladder(kw, integration, dts, ks):
    return {k: _functionals(evolve_pwn(integration=integration,
                                       replay_dt_sequence=refine_dts(dts, k),
                                       **kw)) for k in ks}


def _orders(q, ks, field):
    return pair_orders([q[k][field] for k in ks])


# --- timestep ----------------------------------------------------------------

KS = (1, 2, 4, 8, 16, 32, 64)


@pytest.mark.parametrize('integration', INTEGRATORS)
@pytest.mark.parametrize('eta_b', [1e-9, 0.5])
def test_everything_is_first_order_in_dt(integration, eta_b):
    """Energy recurrences are first order; pc_full gains no order."""
    kw = dict(CLEAN, eta_b=eta_b)
    dts = loguniform_dts(0.5, kw['age'], 0.1)
    q = _ladder(kw, integration, dts, KS)
    eb = _orders(q, KS, 'Eb')
    assert np.all((eb[2:] > 0.9) & (eb[2:] < 1.15)), eb
    deepest = {f: _orders(q, KS, f)[-1] for f in ('R', 'v', 'Ep', 'Eb', 'P')}
    assert all(0.7 < o for o in deepest.values()), deepest
    if integration == 'pc_full':
        assert max(deepest.values()) < 1.35, deepest


@pytest.mark.parametrize('integration', INTEGRATORS)
def test_native_timestep_error(integration):
    """Known: error of the native dt sequence against its dt -> 0 limit."""
    ks = (1, 2, 4, 8, 16, 32)
    dts = _native_dts(BASE, integration)
    q = _ladder(BASE, integration, dts, ks)
    err = {f: abs(q[1][f] / q[32][f] - 1.0) for f in q[1]}
    assert err['R'] < 5e-3 and err['v'] < 1e-3 and err['B'] < 1.5e-2, err
    assert 1e-2 < err['Eb'] < 4e-2, err
    assert 3e-2 < err['Ltot'] < 1e-1, err
    assert 3e-2 < err['xray'] < 1.2e-1, err
    eb = _orders(q, ks, 'Eb')
    assert np.all((eb > 0.9) & (eb < 1.05)), eb


# --- bins --------------------------------------------------------------------

@pytest.fixture(scope='module')
def native():
    return _native_dts(BASE, 'euler')


def test_electron_bin_ladder(native):
    """Band fluxes, E_p and B converge as 1/n_bins; 500-bin offsets < 1.5 %."""
    nbs = (250, 500, 1000, 2000, 4000)
    q = {nb: _functionals(evolve_pwn(integration='euler',
                                     replay_dt_sequence=native,
                                     n_elec_bins=nb, **BASE)) for nb in nbs}
    for f in ('radio', 'xray', 'tev', 'Ep', 'B'):
        v = np.array([q[nb][f] for nb in nbs])
        off = np.abs(v / (2.0 * v[-1] - v[-2]) - 1.0)
        assert np.all(np.diff(off[:-1]) < 0), (f, off)
        assert off[1] < 1.5e-2 and off[3] < 2e-3, (f, off)
    for f in ('Ep', 'B'):
        v = np.array([q[nb][f] for nb in nbs])
        off = v / (2.0 * v[-1] - v[-2]) - 1.0
        ratio = off[1:-1] / off[:-2]
        assert np.all((ratio > 0.25) & (ratio < 0.65)), (f, ratio)


def test_photon_bin_ladder(native):
    """L_nu at fixed nu is second order in photon binning; 1e-3 at 200 bins."""
    nbs = (100, 200, 400, 800, 1600, 3200)
    lnu, dyn = {}, {}
    for nb in nbs:
        r = evolve_pwn(integration='euler', replay_dt_sequence=native,
                       n_phot_bins=nb, **BASE)
        g = r.phot_grid
        m = r.photon_spectrum > 0
        lnu[nb] = np.interp(np.log10(NUS), np.log10(g.bin_avg[m]),
                            np.log10(r.photon_spectrum[m]))
        dyn[nb] = _functionals(r)
    off = {nb: np.abs(10.0 ** (lnu[nb] - lnu[3200]) - 1.0) for nb in nbs[:-1]}
    assert off[200].max() < 1e-3, off[200]
    radio = np.array([off[nb][0] for nb in nbs[:-1]])
    ratio = radio[1:] / radio[:-1]
    assert np.all((ratio > 0.18) & (ratio < 0.32)), ratio
    for f in ('Ep', 'B'):
        assert abs(dyn[100][f] / dyn[3200][f] - 1.0) < 1e-5
