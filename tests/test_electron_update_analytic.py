"""Analytic limits of the per-step spectral update."""
import numpy as np
import pytest

import viper.constants as C
from viper.grids import make_log_grid
from viper.particles import inject_powerlaw
from viper.radiation import csynch_calc, csynch_calc_vec

from _analytic import (CoolingHarness, synch_b_coeff, powerlaw_counts,
                       powerlaw_norm, steady_state_counts, band_ratio,
                       bins_per_decade, logslope, translation_excess)

B = 100.0e-6
P = 2.2
E_LO, E_HI = 0.1, 1.0e6
NBS = (252, 502, 1002)


def _ln_width(grid):
    return np.log(10.0) / bins_per_decade(grid)


def _powerlaw_harness(nb, b_gauss, dt_s, p=P, **kw):
    h = CoolingHarness(nb, b_gauss, dt_s=dt_s, **kw)
    g = h.grid
    h.set_counts(powerlaw_counts(g.bin_min, g.bin_max, 1.0, p, E_LO, E_HI))
    return h


# --- coefficient ---------------------------------------------------------

def test_synch_coefficient_is_thomson_form():
    b = synch_b_coeff(B)
    vec = csynch_calc_vec(np.array([0.0, 3.0, 6.0]), B, 1e-30, np.log10(C.pc))
    assert np.allclose(vec, b, rtol=1e-12)
    assert np.isclose(csynch_calc(3.0, B, 1e-30, np.log10(C.pc)), b, rtol=1e-12)


# --- delta function ------------------------------------------------------

@pytest.mark.parametrize('nb,tol', [(502, 0.03), (1002, 0.005)])
def test_delta_cools_on_the_closed_form(nb, tol):
    b = synch_b_coeff(B)
    h = CoolingHarness(nb, B, dt_s=0.0)
    g = h.grid
    i0 = int(np.searchsorted(g.bin_max, 1.0e4))
    h.dt = 0.1 / (b * g.bin_avg[i0] / C.GeV_per_erg)
    cnt = np.zeros(nb)
    cnt[i0] = 1.0
    h.set_counts(cnt)
    n_steps = 30
    h.run(n_steps)
    width = g.bin_max[i0] - g.bin_min[i0]
    ana = h.analytic_counts(1.0 / width, 0.0, g.bin_min[i0], g.bin_max[i0], b)
    num = h.cur.number
    assert abs(num.sum() - 1.0) < 1e-12
    occ = np.nonzero(num)[0]
    assert set(np.nonzero(ana)[0]) <= set(occ)
    assert occ.max() - occ.min() + 1 <= n_steps + 2
    e_num = (num * g.bin_avg).sum()
    e_ana = (ana * g.bin_avg).sum() / ana.sum()
    assert abs(e_num / e_ana - 1.0) < tol


# --- Kardashev, no injection ---------------------------------------------

def _kardashev(nb, n_steps=100):
    b = synch_b_coeff(B)
    h = _powerlaw_harness(nb, B, 0.5 * C.yr)
    h.run(n_steps)
    g = h.grid
    ana = h.analytic_counts(1.0, P, E_LO, E_HI, b)
    num = h.cur.number
    e_break = C.GeV_per_erg / (b * h.theta * h.r)
    frac = g.bin_avg / h.preimage(g.bin_avg, b)
    sel = (frac > 0.3) & (frac < 0.9) & (ana > 0)
    excess = np.median(num[sel] / ana[sel]) - 1.0
    leak = num[g.bin_min > e_break].sum() / num[g.bin_avg > 1e3].sum()
    return dict(num=num, ana=ana, excess=excess, leak=leak, grid=g)


@pytest.fixture(scope='module')
def kardashev():
    return {nb: _kardashev(nb) for nb in NBS}


@pytest.mark.parametrize('nb', NBS)
def test_kardashev_conservation_and_uncooled_band(kardashev, nb):
    r = kardashev[nb]
    assert abs(r['num'].sum() / r['ana'].sum() - 1.0) < 1e-8
    assert abs(band_ratio(r['grid'], r['num'], r['ana'], 1.0, 100.0) - 1.0) < 1e-4


def test_kardashev_cooled_band_converges(kardashev):
    ex = [kardashev[nb]['excess'] for nb in NBS]
    leak = [kardashev[nb]['leak'] for nb in NBS]
    assert ex[1] < 0.008 and ex[2] < 0.005
    assert all(e > 0 for e in ex)
    for a, b_ in zip(ex, ex[1:]):
        assert 1.4 < a / b_ < 2.6
    assert leak[1] < 2e-3 and leak[2] < leak[1]


def test_thomson_ic_cooling_adds_to_synchrotron():
    """Thomson IC at b equals synchrotron at B sqrt(2)."""
    b = synch_b_coeff(B)
    ref = _powerlaw_harness(502, B * np.sqrt(2.0), 0.5 * C.yr)
    g = ref.grid
    pow_ic = b * (g.bin_avg / C.GeV_per_erg) ** 2
    ic = _powerlaw_harness(502, B, 0.5 * C.yr, pow_ic=pow_ic)
    ref.run(100)
    ic.run(100)
    sel = ref.cur.number > 0
    assert np.allclose(ic.cur.number[sel], ref.cur.number[sel], rtol=1e-9, atol=0)


# --- steady injection ----------------------------------------------------

def _steady(nb, p, n_steps=3000):
    b = synch_b_coeff(B)
    h = CoolingHarness(nb, B, dt_s=0.5 * C.yr)
    g = h.grid
    inj = g.copy()
    inj.reset_number(0.0)
    inject_powerlaw(inj, 1.0, E_LO, E_HI, p)
    h.run(n_steps, inject=inj.number)
    k_rate = powerlaw_norm(1.0, p, E_LO, E_HI) / h.dt
    ana = steady_state_counts(g, k_rate, p, E_LO, E_HI, b)
    excess = band_ratio(g, h.cur.number, ana, 3e3, 1e5) - 1.0
    return h, ana, excess


@pytest.fixture(scope='module')
def steady():
    return {nb: _steady(nb, P) for nb in NBS}


@pytest.mark.parametrize('nb', NBS)
def test_steady_state_cooled_slope_and_bias_law(steady, nb):
    h, ana, excess = steady[nb]
    g = h.grid
    pred = (P + 1.0) * _ln_width(g) / 2.0
    assert abs(excess / pred - 1.0) < 0.12
    s_num = logslope(g.bin_avg, h.cur.number / g.bin_width, 5e3, 3e5)
    assert abs(s_num + (P + 1.0)) < 0.08
    s_ana = logslope(g.bin_avg, ana / g.bin_width, 5e3, 3e5)
    assert abs(s_ana + (P + 1.0)) < 0.06


def test_steady_state_bias_is_first_order(steady):
    ex = [steady[nb][2] for nb in NBS]
    for a, b_ in zip(ex, ex[1:]):
        assert 1.8 < a / b_ < 2.3


@pytest.mark.parametrize('p', [1.5, 3.0])
def test_steady_state_bias_scales_with_slope(p):
    h, ana, excess = _steady(502, p)
    pred = (p + 1.0) * _ln_width(h.grid) / 2.0
    assert abs(excess / pred - 1.0) < 0.10


# --- adiabatic only ------------------------------------------------------

def _adiabatic(nb, p, n_steps):
    h = _powerlaw_harness(nb, 0.0, 1.0, p=p, eps=2.0 ** (1.0 / n_steps) - 1.0)
    h.run(n_steps)
    g = h.grid
    ana = h.analytic_counts(1.0, p, E_LO, E_HI, 0.0)
    num = h.cur.number
    sel = (g.bin_avg >= 1.0) & (g.bin_avg <= 1e5)
    ratio = num[sel] / ana[sel]
    return h, ana, ratio


@pytest.mark.parametrize('nb', NBS)
def test_adiabatic_shift_conserves_number_and_slope(nb):
    h, ana, ratio = _adiabatic(nb, P, 1000)
    g = h.grid
    assert abs(h.r / h.r0 - 2.0) < 1e-9
    assert abs(h.cur.number.sum() / ana.sum() - 1.0) < 1e-12
    assert abs(logslope(g.bin_avg, h.cur.number / g.bin_width, 1.0, 1e5) + P) < 1e-3
    assert ratio.max() - ratio.min() < 1e-6


@pytest.mark.parametrize('nb', NBS)
@pytest.mark.parametrize('p', [1.5, P, 3.0])
def test_adiabatic_shift_excess_matches_scheme_prediction(nb, p):
    n_steps = 1000
    h, ana, ratio = _adiabatic(nb, p, n_steps)
    delta = 1.0 / bins_per_decade(h.grid)
    pred = translation_excess(p, delta, np.log(2.0), n_steps)
    assert abs((ratio[0] - 1.0) / pred - 1.0) < 0.02


def test_adiabatic_excess_independent_of_step_size():
    _, _, r200 = _adiabatic(502, P, 200)
    _, _, r1000 = _adiabatic(502, P, 1000)
    assert abs((r200[0] - 1.0) / (r1000[0] - 1.0) - 1.0) < 0.06


# --- per-step identities -------------------------------------------------

def _step_residuals(nb):
    h = _powerlaw_harness(nb, B, 0.5 * C.yr)
    g = h.grid
    old = h.cur.copy()
    e_old = (old.number * g.bin_avg).sum() / C.GeV_per_erg
    h.step()
    e_new, rl = h.last
    e_grid = (h.cur.number * g.bin_avg).sum() / C.GeV_per_erg
    return dict(
        dn=h.cur.number.sum() / old.number.sum() - 1.0,
        closure=(e_old * h.rad_rat() - e_new - rl) / rl,
        placement=e_grid / e_new - 1.0,
    )


def test_step_identities_and_their_order():
    res = [_step_residuals(nb) for nb in (252, 502, 1002, 2002)]
    for r in res:
        assert abs(r['dn']) < 1e-12
    cl = [r['closure'] for r in res]
    pl = [r['placement'] for r in res]
    assert abs(cl[0]) < 3e-3 and abs(pl[0]) < 3e-4
    for a, b_ in zip(cl, cl[1:]):
        assert 3.0 < a / b_ < 5.0
    for a, b_ in zip(pl, pl[1:]):
        assert 2.5 < a / b_ < 4.0
