"""Integrator-independent physics invariants: energy identities, closures, analytic limits."""
import json
from pathlib import Path

import numpy as np
import pytest

import viper.constants as C
from viper.evolution import evolve_pwn
from viper.scalar_params import PWNScalarParams

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]
GRID_JSON = ROOT / 'tests' / 'reference' / 'grid' / 'grid.json'

POINTS = ['g54_baseline', 'cta1_aliu_pre', 'corner_dense_ism']
INTEGRATORS = ['euler', 'pc_full']

# radiation-negligible closure override
RADLESS = dict(eta_b=1e-9, e_min=0.1, e_max=1.0e4, e_break=100.0,
               ic_temps=[], ic_norms=[])

# free-expansion config: constant Edot
FREE_EXP = dict(esn=1.0, mej=8.0, n_ism=0.1, braking_index=3.0,
                tau_sd=1.0e9, period=0.1, p_dot=1e-14, eta_b=1e-3,
                e_min=10.0, e_max=1e3, e_break=100.0, p1=1.5, p2=2.5,
                ic_temps=[], ic_norms=[])


def _params(point=None, overrides=None):
    pt = {}
    if point:
        with open(GRID_JSON) as f:
            d = json.load(f)
        pt = dict({p['name']: p for p in d['points']}[point])
    pt.update(overrides or {})
    return PWNScalarParams(
        esn=pt['esn'], mej=pt['mej'], n_ism=pt['n_ism'],
        braking_index=pt['braking_index'], tau_sd=pt['tau_sd'],
        period=pt['period'], p_dot=pt['p_dot'], eta_b=pt['eta_b'],
        e_min=pt['e_min'], e_max=pt['e_max'], e_break=pt['e_break'],
        p1=pt['p1'], p2=pt['p2'], f_max=pt.get('f_max', 0.0),
        kT_max=pt.get('kT_max', 0.0), eta_g=pt.get('eta_g', 0.0),
        vel_psr=pt.get('vel_psr', 0.0),
        n_ic=len(pt.get('ic_temps', [])),
        ic_temps=list(pt.get('ic_temps', [])),
        ic_norms=list(pt.get('ic_norms', [])),
    )


@pytest.fixture(scope='module')
def runner():
    """Memoized evolve_pwn runner shared across the suite."""
    cache = {}

    def _run(point=None, integration='euler', overrides=None, age=None):
        key = (point, integration,
               json.dumps(overrides or {}, sort_keys=True), age)
        if key not in cache:
            params = _params(point, overrides)
            t_age = age if age is not None else float(params.true_age)
            r = evolve_pwn(
                integration=integration,
                esn_51=params.esn, mej=params.mej, n_ism=params.n_ism,
                braking_index=params.braking_index, tau_sd=params.tau_sd,
                age=t_age, e0_37=float(params.e0_37),
                vel_psr=params.vel_psr,
                eta_g=params.eta_g, eta_b=params.eta_b,
                e_min=params.e_min, e_max=params.e_max,
                e_break=params.e_break,
                p1=params.p1, p2=params.p2, f_max=params.f_max,
                kT_max=params.kT_max,
                n_ic=params.n_ic, ic_temps=params.ic_temps,
                ic_norms=params.ic_norms,
                dyn_step=1, elec_step=-1, phot_step=-1,
            )
            cache[key] = (params, r)
        return cache[key]

    return _run


def _series(r):
    rows = r.dyninfo
    t = np.array([x.time for x in rows])
    keep = np.ones(len(rows), bool)
    keep[1:] = np.diff(t) > 0          # drop the post-loop duplicate row
    rows = [x for x, k in zip(rows, keep) if k]
    g = lambda a: np.array([getattr(x, a) for x in rows])
    return dict(t=g('time'), dt=g('delta_t'), r=g('rad_pwn'),
                rsnr=g('rad_snr'), epwnb=g('epwnb_50'),
                epwnp=g('epwnp_50'), epf=g('eparfudge'))


def _depsr_steps(params, t, dt, eta_g=0.0):
    """Closed-form spin-down energy per step, 1e40 erg."""
    esd_40 = float(params.e0_37) * 1.0e-3 * params.tau_sd * C.yr
    idx = (params.braking_index + 1.0) / (params.braking_index - 1.0)
    tau = params.tau_sd
    t0 = t - dt
    if idx == 1:
        de = esd_40 * (np.log(1 + t / tau) - np.log(1 + t0 / tau))
    else:
        de = esd_40 / (1.0 - idx) * (
            (1 + t / tau) ** (1.0 - idx) - (1 + t0 / tau) ** (1.0 - idx))
    return (1.0 - eta_g) * de


def _logslope(t, y, lo, hi):
    m = (t >= lo) & (t <= hi) & (y > 0)
    assert m.sum() >= 4, f'window [{lo},{hi}] has {m.sum()} points'
    return np.polyfit(np.log10(t[m]), np.log10(y[m]), 1)[0]


# ---------------------------------------------------------------------------
# Magnetic-energy identity: exact, all regimes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('integration', INTEGRATORS)
@pytest.mark.parametrize('point', POINTS)
def test_magnetic_energy_identity(runner, point, integration):
    """epwnb[i+1] == epwnb[i]·r[i]/r[i+1] + eta_b·ΔE_sd, every step."""
    params, r = runner(point, integration)
    s = _series(r)
    de = _depsr_steps(params, s['t'], s['dt'])[1:]
    pred = (s['epwnb'][:-1] * s['r'][:-1] / s['r'][1:]
            + params.eta_b * de * 1e-10)
    act = s['epwnb'][1:]
    res = np.abs(act - pred) / np.maximum(np.abs(act), 1e-30)
    assert res.max() < 1e-12, (
        f'magnetic-energy identity violated: max rel residual '
        f'{res.max():.3e} at step {int(np.argmax(res))}')


# ---------------------------------------------------------------------------
# Particle-energy closure: strict before the collision, tracked after
# ---------------------------------------------------------------------------

def _particle_gap(params, s, eta_b):
    de = _depsr_steps(params, s['t'], s['dt'])[1:] * 1e-10
    pred = s['epwnp'][0]
    for i in range(len(de)):
        pred = pred * s['r'][i] / s['r'][i + 1] + (1 - eta_b) * de[i]
    return abs(s['epwnp'][-1] - pred) / s['epwnp'][-1]


@pytest.mark.parametrize('integration,tol', [('euler', 2e-3),
                                             ('pc_full', 3e-2)])
def test_particle_closure_pre_collision(runner, integration, tol):
    """Electron-spectrum energy follows adiabatic+injection (radiation-negligible)."""
    params, r = runner('g54_baseline', integration, overrides=RADLESS)
    s = _series(r)
    gap = _particle_gap(params, s, RADLESS['eta_b'])
    assert gap < tol, f'particle closure gap {gap:.3e} > {tol:g}'


@pytest.mark.parametrize('integration', INTEGRATORS)
def test_eparfudge_pre_collision(runner, integration):
    """eparfudge stays within [0.95, 1.001] pre-collision."""
    params, r = runner('g54_baseline', integration)
    epf = _series(r)['epf']
    epf = epf[epf != 0.0]              # initial rows before tracking starts
    assert epf.min() > 0.95 and epf.max() < 1.001, (
        f'eparfudge left pre-collision band: [{epf.min():.4f}, '
        f'{epf.max():.4f}]')


def test_post_collision_energy_divergence_registered(runner):
    """Post-collision energy-accounting divergence must stay present."""
    params, r = runner('cta1_aliu_pre', 'euler')
    epf = _series(r)['epf']
    epf = epf[epf != 0.0]
    assert epf.min() < 0.9, (
        f'post-collision eparfudge min is now {epf.min():.4f} (was '
        f'0.005); energy accounting through the RS collision changed; '
        f're-measure and re-pin')


# ---------------------------------------------------------------------------
# Analytic limits
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('integration', INTEGRATORS)
def test_pwn_free_expansion_slope(runner, integration):
    """R_pwn → t^(6/5) for constant Edot; band [1.16, 1.21] off startup transient."""
    params, r = runner(None, integration, overrides=FREE_EXP, age=2000.0)
    s = _series(r)
    slope = _logslope(s['t'], s['r'], 500, 1900)
    assert 1.16 < slope < 1.21, f'R_pwn free-expansion slope {slope:.4f}'


@pytest.mark.parametrize('integration', INTEGRATORS)
def test_snr_ed_phase_slope(runner, integration):
    """Ejecta-dominated R_snr = t^(2/3) exactly (TM99 n=9 envelope)."""
    params, r = runner(None, integration, overrides=FREE_EXP, age=2000.0)
    s = _series(r)
    slope = _logslope(s['t'], s['rsnr'], 20, 200)
    assert abs(slope - 2.0 / 3.0) < 0.005, (
        f'SNR ED-phase slope {slope:.4f} != 2/3')


@pytest.mark.parametrize('point,lo,hi', [
    ('cta1_aliu_pre', 8000, 15000),      # measured 0.4291
    ('corner_dense_ism', 4000, 8500),    # measured 0.4246
])
def test_snr_sedov_slope(runner, point, lo, hi):
    """Late Sedov R_snr → t^(2/5); slope approaches 0.4 from above, band [0.40, 0.44]."""
    params, r = runner(point, 'euler')
    s = _series(r)
    slope = _logslope(s['t'], s['rsnr'], lo, hi)
    assert 0.40 < slope < 0.44, f'Sedov slope {slope:.4f}'
