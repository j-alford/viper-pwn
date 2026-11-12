"""Per-step particle-energy budget of the evolution loop, through the reverse-shock crush."""
import json
from pathlib import Path

import numpy as np
import pytest

import viper.constants as C
import viper.evolution as ev
from viper.grids import make_log_grid
from viper.radiation import csynch_calc_vec
from viper.scalar_params import PWNScalarParams

ROOT = Path(__file__).resolve().parents[1]
GRID_JSON = ROOT / 'tests' / 'reference' / 'grid' / 'grid.json'


def _point(name):
    pt = {p['name']: p for p in json.load(open(GRID_JSON))['points']}[name]
    return PWNScalarParams(
        esn=pt['esn'], mej=pt['mej'], n_ism=pt['n_ism'],
        braking_index=pt['braking_index'], tau_sd=pt['tau_sd'],
        period=pt['period'], p_dot=pt['p_dot'], eta_b=pt['eta_b'],
        e_min=pt['e_min'], e_max=pt['e_max'], e_break=pt['e_break'],
        p1=pt['p1'], p2=pt['p2'], n_ic=0, ic_temps=[], ic_norms=[])


def _budget_run(monkeypatch, name, integration):
    """Record (new_energy, rad_loss) of every spectral update and align with dyninfo."""
    p = _point(name)
    rec = []
    orig = ev.evolve_spectrum

    def spy(old_grid, new_grid, bpwn, r0, r1, *a, **k):
        out = orig(old_grid, new_grid, bpwn, r0, r1, *a, **k)
        rec.append((out[0], out[1], r0, r1))
        return out

    monkeypatch.setattr(ev, 'evolve_spectrum', spy)
    r = ev.evolve_pwn(
        integration=integration, esn_51=p.esn, mej=p.mej, n_ism=p.n_ism,
        braking_index=p.braking_index, tau_sd=p.tau_sd, age=float(p.true_age),
        e0_37=float(p.e0_37), vel_psr=0.0, eta_g=0.0, eta_b=p.eta_b,
        e_min=p.e_min, e_max=p.e_max, e_break=p.e_break, p1=p.p1, p2=p.p2,
        n_ic=0, ic_temps=[], ic_norms=[], dyn_step=1, elec_step=-1,
        phot_step=-1, snapshot_mode='instantaneous')
    rows = list(r.dyninfo)
    t = np.array([x.time for x in rows])
    keep = np.ones(len(t), bool)
    keep[1:] = np.diff(t) > 0
    rows = [x for x, k in zip(rows, keep) if k]
    g = lambda a: np.array([getattr(x, a) for x in rows])
    t, R, rrs, ep = g('time'), g('rad_pwn'), g('rad_rs'), g('epwnp_50') * 1e10
    n = len(rows) - 1
    assert len(rec) == n + 1                     # one setup call, one per step
    new_e = np.array([x[0] for x in rec[1:]])
    rad = np.array([x[1] for x in rec[1:]])
    r_used = np.array([x[3] for x in rec[1:]]) / C.pc     # spectral-update radius
    esd = float(p.e0_37) * 1e-3 * p.tau_sd * C.yr
    idx = (p.braking_index + 1) / (p.braking_index - 1)
    cum = lambda tt: esd / (1 - idx) * ((1 + tt / p.tau_sd) ** (1 - idx) - 1)
    inj = (cum(t[1:]) - cum(t[:-1])) * (1 - p.eta_b)
    rho = R[:-1] / r_used
    return dict(t=t, R=R, icol=int(np.argmax(R > rrs)),
                credit=(ep[1:] - new_e - inj) / ep[1:],
                edge=(rho * ep[:-1] - rad - new_e) / ep[:-1],
                frac=rad / (rho * ep[:-1]),
                r_mismatch=r_used / R[1:] - 1.0)


@pytest.fixture(scope='module', params=['euler', 'pc_full'])
def cta1(request):
    mp = pytest.MonkeyPatch()
    try:
        yield _budget_run(mp, 'cta1_aliu_pre', request.param)
    finally:
        mp.undo()


@pytest.mark.slow
def test_injection_credit_is_exact(cta1):
    assert np.abs(cta1['credit']).max() < 1e-10


@pytest.mark.slow
def test_edge_map_closes_pre_collision(cta1):
    e = cta1['edge'][:cta1['icol']]
    assert np.median(np.abs(e)) < 5e-4
    assert np.abs(e[cta1['t'][1:cta1['icol'] + 1] > 100]).max() < 1e-3


@pytest.mark.slow
def test_edge_map_breaks_only_in_the_deep_crush(cta1):
    """Registered: self-absorption threshold bin loses its energy accounting (WP6)."""
    e, R = cta1['edge'], cta1['R'][1:]
    bad = np.abs(e) > 1e-2
    assert 1 <= bad.sum() <= 40
    assert np.all(R[bad] < 0.05)
    assert np.abs(e).max() > 0.02


@pytest.mark.slow
def test_corrector_radius_differs_from_spectral_update_radius(cta1, request):
    """Registered: pc_full evolves the spectrum at the predictor radius."""
    m = np.abs(cta1['r_mismatch'])
    if request.node.callspec.params['cta1'] == 'euler':
        assert m.max() < 1e-12
    else:
        assert m.max() > 0.1 and (m > 1e-2).sum() <= 5


# --- mechanism --------------------------------------------------------------

def _thin_threshold(b_gauss, n_elec_40, r_cm):
    e = np.logspace(-4, 6, 10001)
    c = csynch_calc_vec(np.log10(e), b_gauss, n_elec_40, np.log10(r_cm))
    return e[np.argmax(c > 0)]


def _step_identity(grid, ibin, b_gauss, n_elec_40, r0, r1, dt_s):
    old = grid.copy()
    old.reset_number(0.0)
    old.number[ibin] = 1.0
    old.update_log_number()
    new = grid.copy()
    new.reset_number(0.0)
    v = (r1 - r0) / dt_s
    new_e, rad = ev.evolve_spectrum(old, new, b_gauss, r0, r1, v, n_elec_40,
                                    grid.n_bins, np.zeros(grid.n_bins))
    e_old = grid.bin_avg[ibin] / C.GeV_per_erg * r0 / r1
    return (e_old - rad - new_e) / e_old, rad / e_old


def test_threshold_bin_breaks_the_edge_map_identity():
    """Lower edge thick (no loss), upper edge thin: the bin's energy is misbooked."""
    b, ne, r0 = 3.0, 1.0e9, 0.005 * C.pc
    grid = make_log_grid(1e-4, 1e6, 502)
    e_thr = _thin_threshold(b, ne, r0)
    ibin = int(np.searchsorted(grid.bin_max, e_thr))
    assert grid.bin_min[ibin] < e_thr < grid.bin_max[ibin]
    dt = 0.5 * C.yr
    res_thr, frac_thr = _step_identity(grid, ibin, b, ne, r0, 0.8 * r0, dt)
    res_thin, frac_thin = _step_identity(grid, ibin + 3, b, ne, r0, 0.8 * r0, dt)
    res_thick, frac_thick = _step_identity(grid, ibin - 3, b, ne, r0, 0.8 * r0, dt)
    assert frac_thin > 0.5 and frac_thick == 0.0
    assert abs(res_thin) < 1e-3 and abs(res_thick) < 1e-3
    assert res_thr > 0.1                           # registered
