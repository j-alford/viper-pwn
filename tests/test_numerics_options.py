"""Numerical corrections, off by default: each removes a known error."""
import numpy as np
import pytest

import viper.constants as C
from viper import evolution as ev
from viper.evolution import evolve_pwn, evolve_spectrum
from viper.grids import make_log_grid
from viper.particles import inject_powerlaw
from viper.radiation import csynch_calc_vec
from viper.radiative import kernels as K
from viper.model import PWNModel
from viper.fitting import build_fit_spec
from viper.source import OPTION_DEFAULTS

from _analytic import (CoolingHarness, band_ratio, powerlaw_norm,
                       steady_state_counts, synch_b_coeff)

ALL_ON = dict(injection_credit='midstep', remap='powerlaw',
              threshold_bin='centre', ic_cooling='exact')
B = 1e-4
E_LO, E_HI, P = 10.0, 1e5, 2.3


# --- defaults and validation ---------------------------------------------------

def test_option_defaults():
    for k in ALL_ON:
        assert OPTION_DEFAULTS[k] in ('end', 'uniform', 'edges', 'table')
    m = PWNModel()
    assert (m.injection_credit, m.remap, m.threshold_bin, m.ic_cooling) == (
        'end', 'uniform', 'edges', 'table')


@pytest.mark.parametrize('name,bad', [('injection_credit', 'half'),
                                      ('remap', 'linear'),
                                      ('threshold_bin', 'mean'),
                                      ('ic_cooling', 'kn')])
def test_rejects_unknown_value(name, bad):
    kw = dict(esn_51=1.0, mej=8.0, n_ism=0.1, braking_index=3.0, tau_sd=500.0,
              age=100.0, e0_37=1.0, vel_psr=0.0, eta_g=0.0, eta_b=0.01,
              e_min=1.0, e_max=1e3, e_break=10.0, p1=1.5, p2=2.5)
    with pytest.raises(ValueError):
        evolve_pwn(**kw, **{name: bad})
    with pytest.raises(ValueError):
        PWNModel().apply_run_config({name: bad})


def test_model_and_fit_spec_carry_options():
    m = PWNModel()
    m.apply_run_config(dict(ALL_ON))
    assert (m.injection_credit, m.remap, m.threshold_bin, m.ic_cooling) == (
        'midstep', 'powerlaw', 'centre', 'exact')
    m.eta_b.frozen = False
    m.eta_b.bounds = (1e-4, 0.5)
    spec = build_fit_spec(m, [])
    assert (spec.injection_credit, spec.remap, spec.threshold_bin,
            spec.ic_cooling) == ('midstep', 'powerlaw', 'centre', 'exact')


# --- spectral update: numpy and numba paths agree --------------------------------

@pytest.mark.parametrize('remap,thr', [('powerlaw', 'edges'),
                                       ('uniform', 'centre'),
                                       ('powerlaw', 'centre')])
def test_backends_agree(remap, thr):
    if not ev.HAS_NUMBA:
        pytest.skip('numba not available')
    g = make_log_grid(1.0, 1e6, 302)
    inject_powerlaw(g, 1.0, 10.0, 1e5, 2.3)
    out = []
    for use_numba in (True, False):
        ev.HAS_NUMBA = use_numba
        try:
            new = g.copy()
            new.reset_number()
            res = evolve_spectrum(g, new, 3e-5, 1e18, 1.02e18, 1e7, 1e9, 302,
                                  np.zeros(302), remap=remap, threshold_bin=thr)
        finally:
            ev.HAS_NUMBA = True
        out.append((res, new.number.copy()))
    assert out[0][0] == out[1][0]
    np.testing.assert_array_equal(out[0][1], out[1][1])
    assert new.number.sum() == pytest.approx(g.number.sum(), rel=1e-12)


# --- 6a: slope-aware remap ---------------------------------------------------------

@pytest.mark.parametrize('nb', [252, 502])
def test_remap_removes_steady_state_bias(nb):
    """Uniform in-bin density inflates the cooled band by (p+1)L/2; power law does not."""
    ex = {}
    for remap in ('uniform', 'powerlaw'):
        b = synch_b_coeff(B)
        h = CoolingHarness(nb, B, dt_s=0.5 * C.yr, remap=remap)
        g = h.grid
        inj = g.copy()
        inj.reset_number(0.0)
        inject_powerlaw(inj, 1.0, E_LO, E_HI, P)
        h.run(3000, inject=inj.number)
        ana = steady_state_counts(g, powerlaw_norm(1.0, P, E_LO, E_HI) / h.dt,
                                  P, E_LO, E_HI, b)
        ex[remap] = band_ratio(g, h.cur.number, ana, 3e3, 1e5) - 1.0
    assert ex['uniform'] > 0.08
    assert abs(ex['powerlaw']) < 5e-3


# --- 6d: self-absorption threshold at the bin centre ------------------------------

def test_threshold_bin_identity_closes():
    b, ne, r0 = 3.0, 1.0e9, 0.005 * C.pc
    grid = make_log_grid(1e-4, 1e6, 502)
    e = np.logspace(-4, 6, 10001)
    e_thr = e[np.argmax(csynch_calc_vec(np.log10(e), b, ne, np.log10(r0)) > 0)]
    ibin = int(np.searchsorted(grid.bin_max, e_thr))
    dt, r1 = 0.5 * C.yr, 0.8 * r0
    res = {}
    for thr in ('edges', 'centre'):
        old = grid.copy()
        old.reset_number(0.0)
        old.number[ibin] = 1.0
        old.update_log_number()
        new = grid.copy()
        new.reset_number(0.0)
        new_e, rad = evolve_spectrum(old, new, b, r0, r1, (r1 - r0) / dt, ne,
                                     grid.n_bins, np.zeros(grid.n_bins),
                                     threshold_bin=thr)
        e_old = grid.bin_avg[ibin] / C.GeV_per_erg * r0 / r1
        res[thr] = (e_old - rad - new_e) / e_old
    assert res['edges'] > 0.1
    assert abs(res['centre']) < 1e-3


# --- 6e: exact IC loss rate ------------------------------------------------------------

def test_exact_ic_loss_matches_closed_form():
    grid = make_log_grid(1e-3, 1e8, 502)
    pie = ev._exact_ic_loss(grid, 1, [3000.0], [0.5])
    e_erg = grid.bin_avg / C.GeV_per_erg
    want = (K.ic_loss_rate_blackbody(e_erg, C.T_CMB)
            + K.ic_loss_rate_blackbody(e_erg, 3000.0, 0.5 / C.eV_per_erg))
    np.testing.assert_allclose(pie, want, rtol=1e-12)
    assert np.all(pie > 0)


# --- 6c: mid-step credit (slow, full runs) --------------------------------------------

FREE = dict(esn_51=1.0, mej=8.0, n_ism=0.1, braking_index=3.0, tau_sd=1e9,
            age=3000.0, e0_37=4.1728412947090015e-06, vel_psr=0.0, eta_g=0.0,
            eta_b=1e-3, e_min=10.0, e_max=1e3, e_break=100.0, p1=1.5, p2=2.5,
            n_elec_bins=500, n_ic=0, ic_temps=[], ic_norms=[],
            dyn_step=1, elec_step=-1, phot_step=-1,
            snapshot_mode='instantaneous')


def _free_expansion_excess(integration, credit):
    r = evolve_pwn(integration=integration, injection_credit=credit, **FREE)
    rows = [x for x in r.dyninfo if x.time > 0]
    g = lambda a: np.array([getattr(x, a) for x in rows])
    t = g('time') * C.yr
    R = g('rad_pwn') * C.pc
    e = (g('epwnp_50') + g('epwnb_50')) * 1e50
    edot = FREE['e0_37'] * 1e37
    er = e[0] * R[0] + np.concatenate(
        [[0.0], np.cumsum(0.5 * (R[1:] + R[:-1]) * np.diff(t))]) * edot
    return e[-1] / (er[-1] / R[-1]) - 1.0


@pytest.mark.slow
@pytest.mark.parametrize('integration', ['euler', 'pc_full'])
def test_midstep_credit_closes_free_expansion_budget(integration):
    """End-of-step credit leaves 10-20 % at native dt; mid-step credit < 2 %."""
    end = _free_expansion_excess(integration, 'end')
    mid = _free_expansion_excess(integration, 'midstep')
    assert 0.08 < end < 0.25
    assert abs(mid) < 2e-2


@pytest.mark.slow
def test_midstep_credit_reduces_magnetic_energy_error():
    """g54-like run: native-dt error of E_b against the dt -> 0 limit."""
    from test_order_of_convergence import BASE, _native_dts, _functionals
    from _analytic import refine_dts
    dts = _native_dts(BASE, 'euler')
    err = {}
    for credit in ('end', 'midstep'):
        kw = dict(BASE, injection_credit=credit)
        q = {k: _functionals(evolve_pwn(integration='euler',
                                        replay_dt_sequence=refine_dts(dts, k),
                                        **kw)) for k in (1, 16)}
        err[credit] = abs(q[1]['Eb'] / q[16]['Eb'] - 1.0)
    assert err['end'] > 1.5e-2
    assert err['midstep'] < 0.3 * err['end']


@pytest.mark.slow
def test_all_options_run_and_conserve_number():
    r0 = evolve_pwn(**dict(FREE, eta_b=0.01, e_max=1e5))
    r1 = evolve_pwn(**dict(FREE, eta_b=0.01, e_max=1e5), **ALL_ON)
    assert r1.dyninfo[-1].rad_pwn == pytest.approx(r0.dyninfo[-1].rad_pwn,
                                                   rel=3e-2)
    n0, n1 = r0.electron_spectrum.sum(), r1.electron_spectrum.sum()
    assert n1 == pytest.approx(n0, rel=1e-6)
