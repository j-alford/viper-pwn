"""Numerical corrections: each removes a known error of the original scheme."""
import json
from pathlib import Path

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
from viper.scalar_params import PWNScalarParams

from _analytic import (ORIGINAL_SCHEME, CoolingHarness, band_ratio,
                       powerlaw_norm,
                       steady_state_counts, synch_b_coeff)

ALL_ON = dict(injection_credit='midstep', remap='powerlaw',
              threshold_bin='centre', ic_cooling='exact',
              cooling_field='mean')
B = 1e-4
E_LO, E_HI, P = 10.0, 1e5, 2.3


# --- defaults and validation ---------------------------------------------------

def test_option_defaults():
    m = PWNModel()
    for k, v in ALL_ON.items():
        assert OPTION_DEFAULTS[k] == v
        assert getattr(m, k) == v


@pytest.mark.parametrize('name,bad', [('injection_credit', 'half'),
                                      ('remap', 'linear'),
                                      ('threshold_bin', 'mean'),
                                      ('ic_cooling', 'kn'),
                                      ('cooling_field', 'end')])
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
    m.apply_run_config(dict(ORIGINAL_SCHEME))
    assert (m.injection_credit, m.remap, m.threshold_bin, m.ic_cooling,
            m.cooling_field) == ('end', 'uniform', 'edges', 'table', 'start')
    m.eta_b.frozen = False
    m.eta_b.bounds = (1e-4, 0.5)
    spec = build_fit_spec(m, [])
    assert (spec.injection_credit, spec.remap, spec.threshold_bin,
            spec.ic_cooling, spec.cooling_field) == (
        'end', 'uniform', 'edges', 'table', 'start')


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


# --- slope-aware remap ---------------------------------------------------------

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


# --- self-absorption threshold at the bin centre ------------------------------

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


# --- exact IC loss rate ------------------------------------------------------------

def test_exact_ic_loss_matches_closed_form():
    grid = make_log_grid(1e-3, 1e8, 502)
    pie = ev._exact_ic_loss(grid, 1, [3000.0], [0.5])
    e_erg = grid.bin_avg / C.GeV_per_erg
    want = (K.ic_loss_rate_blackbody(e_erg, C.T_CMB)
            + K.ic_loss_rate_blackbody(e_erg, 3000.0, 0.5 / C.eV_per_erg))
    np.testing.assert_allclose(pie, want, rtol=1e-12)
    assert np.all(pie > 0)


# --- mid-step injection (slow, full runs) --------------------------------------------

FREE = dict(esn_51=1.0, mej=8.0, n_ism=0.1, braking_index=3.0, tau_sd=1e9,
            age=3000.0, e0_37=4.1728412947090015e-06, vel_psr=0.0, eta_g=0.0,
            eta_b=1e-3, e_min=10.0, e_max=1e3, e_break=100.0, p1=1.5, p2=2.5,
            n_elec_bins=500, n_ic=0, ic_temps=[], ic_norms=[],
            dyn_step=1, elec_step=-1, phot_step=-1,
            snapshot_mode='instantaneous')


def _free_expansion_excess(integration, credit, **kw):
    r = evolve_pwn(integration=integration, injection_credit=credit, **FREE,
                   **kw)
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
    """End-of-step credit leaves 10-20 % at the original steps; mid-step < 2 %."""
    end = _free_expansion_excess(integration, 'end', radius_step_frac=None)
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


def _native_vs_limit(kw, ks=(8, 16)):
    """Native-timestep result over its Richardson limit, minus one."""
    from test_order_of_convergence import _functionals
    from _analytic import refine_dts
    kw = {k: v for k, v in kw.items() if k not in ORIGINAL_SCHEME}
    base = evolve_pwn(integration='euler', **kw, **ORIGINAL_SCHEME)
    dts = [float(x.delta_t) for x in base.dyninfo][:-1]
    ref = [evolve_pwn(integration='euler', **kw, **ALL_ON,
                      replay_dt_sequence=refine_dts(dts, k)) for k in ks]
    on = evolve_pwn(integration='euler', **kw, **ALL_ON,
                    replay_dt_sequence=dts)

    def err(f):
        a, b = (f(r) for r in ref)
        lim = 2.0 * b - a
        return f(base) / lim - 1.0, f(on) / lim - 1.0
    return err, _functionals, on.elec_grid.bin_avg


@pytest.mark.slow
def test_all_options_remove_native_timestep_error():
    """g54-like run: X-ray, TeV and particle energy against the dt -> 0 limit."""
    from test_order_of_convergence import BASE
    err, fn, _ = _native_vs_limit(BASE)
    for name, floor, tol in (('xray', 5e-2, 1.5e-2), ('tev', 2e-2, 1e-2),
                             ('Ep', 1.5e-2, 1e-2), ('Eb', 1.5e-2, 1e-2)):
        off, on = err(lambda r: fn(r)[name])
        assert abs(off) > floor, (name, off)
        assert abs(on) < tol, (name, on)


@pytest.mark.slow
def test_injection_above_a_pev_is_cooled_within_the_step():
    """Young source, e_max = 4 PeV: top half decade of the electron spectrum."""
    kw = dict(esn_51=1.0, mej=8.0, n_ism=0.1, braking_index=3.0, tau_sd=700.0,
              age=950.0, e0_37=300.0, vel_psr=0.0, eta_g=0.0, eta_b=0.05,
              e_min=1.0, e_max=4e6, e_break=500.0, p1=1.5, p2=2.4,
              n_ic=0, ic_temps=[], ic_norms=[], dyn_step=1, elec_step=-1,
              phot_step=-1, snapshot_mode='instantaneous')
    err, _, e = _native_vs_limit(kw)
    top = (e > 1.26e6) & (e < 4e6)
    off, on = err(lambda r: r.electron_spectrum[top].sum())
    assert off > 1.0, off
    assert abs(on) < 0.1, on


@pytest.mark.slow
def test_all_options_run_and_conserve_number():
    r0 = evolve_pwn(**dict(FREE, eta_b=0.01, e_max=1e5), **ORIGINAL_SCHEME)
    r1 = evolve_pwn(**dict(FREE, eta_b=0.01, e_max=1e5), **ALL_ON,
                    radius_step_frac=None)
    assert r1.dyninfo[-1].rad_pwn == pytest.approx(r0.dyninfo[-1].rad_pwn,
                                                   rel=3e-2)
    n0, n1 = r0.electron_spectrum.sum(), r1.electron_spectrum.sum()
    assert n1 == pytest.approx(n0, rel=1e-6)


# --- radius-change limit on the timestep ------------------------------------------

_GRID_JSON = Path(__file__).resolve().parent / 'reference' / 'grid' / 'grid.json'


def _grid_run(name, eps=0.0, **kw):
    pt = {p['name']: p for p in json.load(open(_GRID_JSON))['points']}[name]
    p = PWNScalarParams(
        esn=pt['esn'], mej=pt['mej'] * (1.0 + eps), n_ism=pt['n_ism'],
        braking_index=pt['braking_index'], tau_sd=pt['tau_sd'],
        period=pt['period'], p_dot=pt['p_dot'], eta_b=pt['eta_b'],
        e_min=pt['e_min'], e_max=pt['e_max'], e_break=pt['e_break'],
        p1=pt['p1'], p2=pt['p2'])
    return evolve_pwn(
        esn_51=p.esn, mej=p.mej, n_ism=p.n_ism,
        braking_index=p.braking_index, tau_sd=p.tau_sd,
        age=float(p.true_age), e0_37=float(p.e0_37), vel_psr=0.0,
        eta_g=0.0, eta_b=p.eta_b, e_min=p.e_min, e_max=p.e_max,
        e_break=p.e_break, p1=p.p1, p2=p.p2, dyn_step=1, **kw)


def test_radius_step_default_and_validation():
    assert OPTION_DEFAULTS['radius_step_frac'] == 0.01
    assert PWNModel().radius_step_frac == 0.01
    m = PWNModel()
    m.apply_run_config({'radius_step_frac': None})
    assert m.radius_step_frac is None
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            PWNModel().apply_run_config({'radius_step_frac': bad})


@pytest.mark.slow
def test_radius_step_limits_every_step():
    d = _grid_run('lhs_077').dyninfo
    t = np.array([x.time for x in d])
    r = np.array([x.rad_pwn for x in d]) * C.pc
    v = np.array([x.vpwn for x in d]) * 1e5
    frac = np.abs(v[:-1]) * np.diff(t) * C.yr / r[:-1]
    assert frac[1:].max() < 0.0101
    d0 = _grid_run('lhs_077', radius_step_frac=None).dyninfo
    t0 = np.array([x.time for x in d0])
    # original rule: steps of a third of the age
    assert (np.diff(t0)[1:] / t0[1:-1]).max() > 0.2


@pytest.mark.slow
def test_radius_step_makes_compressed_model_reproducible():
    eps = (-1e-3, 0.0, 1e-3)

    def spread(**kw):
        r = [_grid_run('lhs_077', e, **kw).dyninfo[-1].rad_pwn for e in eps]
        return max(r) / min(r)

    assert spread() < 1.05
    # known error of the original stepping
    assert spread(radius_step_frac=None) > 1.5


@pytest.mark.slow
def test_cpp_compat_ignores_radius_step():
    a = _grid_run('g54_baseline', cpp_compat=True)
    b = _grid_run('g54_baseline', cpp_compat=True, radius_step_frac=None)
    assert len(a.dyninfo) == len(b.dyninfo)
    assert a.dyninfo[-1].rad_pwn == b.dyninfo[-1].rad_pwn
