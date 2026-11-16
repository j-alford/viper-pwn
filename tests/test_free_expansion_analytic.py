"""Constant-Edot PWN in a uniform ejecta core: the thin-shell similarity solution."""
import numpy as np
import pytest

import viper.constants as C
from viper import snr
from viper.evolution import evolve_pwn

pytestmark = pytest.mark.slow

ESN, MEJ = 1.0, 8.0
E0_37 = 4.1728412947090015e-06          # from P, Pdot
EDOT = E0_37 * 1.0e37
ETA_B = 1.0e-3
ALPHA = 6.0 / 5.0
EPS = 5.0 / 11.0                        # 1 / (1 + alpha)


def amplitude():
    """R = A t^(6/5): A^5 = 125 Edot / (132 pi a), rho_core = a / t^3."""
    vt = snr.calc_vt(ESN, MEJ)
    a = snr.calc_a(ESN, vt)
    return (125.0 * EDOT / (132.0 * np.pi * a)) ** 0.2


def _kw(n_ism, nb):
    return dict(esn_51=ESN, mej=MEJ, n_ism=n_ism, braking_index=3.0,
                tau_sd=1.0e9, age=1.0e4, e0_37=E0_37, vel_psr=0.0, eta_g=0.0,
                eta_b=ETA_B, e_min=10.0, e_max=1.0e3, e_break=100.0, p1=1.5,
                p2=2.5, n_ic=0, ic_temps=[], ic_norms=[], dyn_step=1,
                elec_step=-1, phot_step=-1, snapshot_mode='instantaneous',
                n_elec_bins=nb)


def _rows(result):
    rows = [x for x in result.dyninfo if x.time > 0]
    t = np.array([x.time for x in rows])
    keep = np.ones(len(t), bool)
    keep[1:] = np.diff(t) > 0
    return [x for x, k in zip(rows, keep) if k]


def _run(integration, k=16, nb=500, n_ism=0.1):
    """Native dt sequence split k ways and replayed."""
    native = evolve_pwn(integration=integration, **_kw(n_ism, nb))
    dts = [float(x.delta_t) for x in native.dyninfo][:-1]
    seq = [d / k for d in dts for _ in range(k)]
    r = evolve_pwn(integration=integration, replay_dt_sequence=seq,
                   **_kw(n_ism, nb))
    rows = _rows(r)
    g = lambda a: np.array([getattr(x, a) for x in rows])
    t = g('time') * C.yr
    R = g('rad_pwn') * C.pc
    ep, eb = g('epwnp_50') * 1e50, g('epwnb_50') * 1e50
    e_tot = ep + eb
    er = e_tot[0] * R[0] + np.concatenate(
        [[0.0], np.cumsum(0.5 * (R[1:] + R[:-1]) * np.diff(t))]) * EDOT
    return dict(t=t, R=R, ep=ep, eb=eb, e=e_tot, e_exact=er / R,
                p=g('ppwn'), dt=g('delta_t') * C.yr, native=native)


@pytest.fixture(scope='module', params=['euler', 'pc_full'])
def fine(request):
    return _run(request.param)


def test_similarity_exponent(fine):
    t, R = fine['t'], fine['R']
    m = t > t[-1] / 10.0
    slope = np.polyfit(np.log(t[m]), np.log(R[m]), 1)[0]
    assert abs(slope - ALPHA) < 2e-3


def test_similarity_amplitude(fine):
    t, R = fine['t'], fine['R']
    assert abs(R[-1] / (amplitude() * t[-1] ** ALPHA) - 1.0) < 1e-2


def test_energy_equation_closes_along_trajectory(fine):
    """d(ER)/dt = Edot R integrated on the code's R(t) gives 5/11."""
    t, e_exact = fine['t'], fine['e_exact']
    assert abs(e_exact[-1] / (EDOT * t[-1]) / EPS - 1.0) < 1e-3


def test_pressure_identity(fine):
    vol = 4.0 / 3.0 * np.pi * fine['R'] ** 3
    p = (fine['ep'] / 3.0 + fine['eb']) / vol
    assert np.allclose(fine['p'], p, rtol=1e-12, atol=0)


def test_magnetic_energy_step_is_exact(fine):
    R, eb, dt = fine['R'], fine['eb'], fine['dt']
    pred = eb[:-1] * R[:-1] / R[1:] + ETA_B * EDOT * dt[1:]
    assert np.allclose(eb[1:], pred, rtol=1e-6, atol=0)


def test_particle_energy_excess_is_bookkeeping(fine):
    """End-of-step injection credit: excess ~ alpha dt/(2t) plus the bin bias."""
    e, e_exact = fine['e'], fine['e_exact']
    excess = e[-1] / e_exact[-1] - 1.0
    assert 0.005 < excess < 0.03


def test_particle_energy_excess_shrinks_with_bins():
    r500 = _run('euler', nb=500)
    r2000 = _run('euler', nb=2000)
    x500 = r500['e'][-1] / r500['e_exact'][-1] - 1.0
    x2000 = r2000['e'][-1] / r2000['e_exact'][-1] - 1.0
    assert x2000 < 0.6 * x500


def test_native_stepping_energy_excess_known():
    """Quiet config steps at dt/t ~ 0.25 and carries +15 % energy (injection_credit='end')."""
    for integration in ('euler', 'pc_full'):
        r = _run(integration, k=1)
        t, dt = r['t'], r['dt']
        assert 0.15 < np.median(dt[t > t[-1] / 10] / t[t > t[-1] / 10]) < 0.35
        excess = r['e'][-1] / r['e_exact'][-1] - 1.0
        assert 0.10 < excess < 0.20


def test_pre_collision_independent_of_ism():
    a = _run('euler', n_ism=0.1)
    b = _run('euler', n_ism=1.0e-3)
    n = min(len(a['R']), len(b['R']))
    assert np.array_equal(a['R'][:n], b['R'][:n])
    assert np.array_equal(a['e'][:n], b['e'][:n])
