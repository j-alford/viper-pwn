"""Hadronic and bremsstrahlung tests."""

import numpy as np
import pytest
import astropy.units as u
from scipy import integrate

from _radiative_common import NE, RNG as RNG_E, RNG_P
from viper.radiative import Bremsstrahlung, PionDecay, PowerLaw
from viper.radiative import hadronic as H

GEV = H._GEV
MEC2 = H.MEC2
SIG0 = H.ALPHA_FS * H.R0**2 / MEC2
MODELS = ("Geant4", "Pythia8", "SIBYLL", "QGSJET")


# e-p bremsstrahlung
@pytest.mark.parametrize("y", [1e-4, 1e-2, 0.3, 0.7, 0.99])
def test_bh_ultrarelativistic_limit(y):
    g = 1e6
    k = y * g
    ur = 4.0 / k * (1 + (1 - y) ** 2 - 2 / 3 * (1 - y)) \
        * (np.log(2 * g * (g - k) / k) - 0.5)
    val = H.brems_ep_sigma(k * MEC2, g, elwert=False) / SIG0
    assert val == pytest.approx(ur, rel=1e-6)


@pytest.mark.parametrize("f", [0.01, 0.5, 0.9])
def test_bh_nonrelativistic_limit(f):
    T0 = 1e-6
    p0 = np.sqrt(T0 * (T0 + 2))
    k = f * T0
    T = T0 - k
    p = np.sqrt(T * (T + 2))
    nr = 16 / 3 / (k * p0**2) * np.log((p0 + p) / (p0 - p))
    val = H.brems_ep_sigma(k * MEC2, 1 + T0, elwert=False) / SIG0
    assert val == pytest.approx(nr, rel=1e-5)


@pytest.mark.parametrize("g", [1e5, 1e7])
def test_bh_energy_loss(g):
    T0 = g - 1

    def f(y):
        e = y * T0 * MEC2
        return e * H.brems_ep_sigma(e, g, elwert=False) * T0 * MEC2

    v, _ = integrate.quad(f, 0, 1, limit=1000, epsrel=1e-12,
                          points=[1 - 10 / g, 1 - 1 / g])
    bg70 = 4 * H.ALPHA_FS * H.R0**2 * g * MEC2 * (np.log(2 * g) - 1 / 3)
    assert v == pytest.approx(bg70, rel=1e-4)


def test_bh_endpoint_behaviour():
    g = 3.0
    T = np.geomspace(1e-9, 1e-5, 9)
    k = (g - 1 - T) * MEC2
    born = H.brems_ep_sigma(k, g, elwert=False)
    p = np.sqrt(T * (T + 2))
    r = born / p
    assert np.all(np.abs(r / r[0] - 1) < 1e-3)          # Born ~ p
    el = H.brems_ep_sigma(k, g, elwert=True)
    assert np.all(np.abs(el / el[0] - 1) < 1e-2)        # Elwert finite
    assert el[0] > 0
    assert H.brems_ep_sigma((g - 1) * MEC2 * 1.0001, g) == 0.0
    big = H.brems_ep_sigma(1e3 * MEC2, 1e6, elwert=True)
    assert big == pytest.approx(H.brems_ep_sigma(1e3 * MEC2, 1e6, elwert=False),
                                rel=1e-6)


def test_bh_charge_scaling():
    a = H.brems_ep_sigma(10 * MEC2, 1e3, Z=1.0, elwert=False)
    b = H.brems_ep_sigma(10 * MEC2, 1e3, Z=2.0, elwert=False)
    assert b == pytest.approx(4 * a, rel=1e-14)


# e-e bremsstrahlung
def test_ee_endpoint_kinematics():
    g = np.array([1.001, 1.5, 3.0, 10.0, 1e4])
    T = g - 1
    beta_cm = np.sqrt(T / (T + 2))
    eps_star = T / np.sqrt(2 * (g + 1))
    g_cm = np.sqrt((g + 1) / 2)
    exact = g_cm * (1 + beta_cm) * eps_star
    assert np.allclose(H.brems_ee_eps_max(g), exact, rtol=1e-13)
    w = H.brems_ee_eps_max(g)
    assert np.allclose(H._ee_T_min(w), T, rtol=1e-12)
    gg = np.array([20.0, 1e3])
    wm = H.brems_ee_eps_max(gg)
    assert np.all(H.brems_ee_sigma(wm * MEC2 * (1 + 1e-9), gg) == 0)
    assert np.all(H.brems_ee_sigma(wm * MEC2 * (1 - 1e-6), gg) > 0)


@pytest.mark.parametrize("w", [1e3, 1e5])
def test_ee_approaches_ep_ultrarelativistic(w):
    g = 1e6
    r = H.brems_ee_sigma(w * MEC2, g) / H.brems_ep_sigma(w * MEC2, g)
    assert r == pytest.approx(1.0, abs=2e-3)


def test_ee_small_eps_series():
    w = np.array([1e-3, 0.0199, 0.0201])
    direct = (-1 / w**2 + 3 / w - 4 - 4 * w - 8 * w**2
              - 2 * (1 - 2 * w) * np.log1p(-2 * w)
              * (0.25 / w**3 - 0.5 / w**2 + 3 / w - 2 + 4 * w))
    assert np.allclose(H._a3_bracket(w), direct, rtol=1e-9)
    tiny = H._a3_bracket(np.array([1e-9]))
    assert tiny[0] == pytest.approx(28 / 3, rel=1e-8)


def test_ee_nonneg_and_regime():
    g = np.linspace(1.01, 4.9, 40)[:, None]
    w = np.geomspace(1e-5, 4.0, 300)[None, :]
    s = H.brems_ee_sigma(w * MEC2, g, "nr")
    assert np.all(s >= 0) and np.all(np.isfinite(s))
    with pytest.raises(ValueError):
        H.brems_ee_sigma(1.0, 10.0, "bad")


def test_brems_ee_ep_ratio():
    E = np.array([0.1, 100.0]) * u.GeV
    ee = Bremsstrahlung(NE, weight_ep=0.0, **RNG_E).flux(E)
    ep = Bremsstrahlung(NE, weight_ee=0.0, **RNG_E).flux(E)
    assert np.allclose((ee / ep).value, 1.088 / 1.263, rtol=1e-3)  # Baring99 0.86


def test_brems_weights_sum():
    E = np.array([0.1, 100.0]) * u.GeV
    tot = Bremsstrahlung(NE, **RNG_E).flux(E)
    ee = Bremsstrahlung(NE, weight_ep=0.0, **RNG_E).flux(E)
    ep = Bremsstrahlung(NE, weight_ee=0.0, **RNG_E).flux(E)
    np.testing.assert_allclose((ee + ep).value, tot.value, rtol=1e-12)
    half = Bremsstrahlung(NE, weight_ee=0.544, weight_ep=0.0, **RNG_E).flux(E)
    np.testing.assert_allclose((half / ee).value, 0.5, rtol=1e-12)


def test_brems_vs_direct_quadrature():
    m = Bremsstrahlung(NE, **RNG_E, rtol=1e-6)
    eps = np.array([1e-3, 1.0, 1e3, 3e4]) * u.GeV
    ours = m.flux(eps, distance=0).to_value(1 / (u.s * u.eV))
    for e, o in zip(eps.to_value(u.erg), ours):
        def f(lnE):
            E = np.exp(lnE)
            g = E / MEC2
            v = H.C.c_light * np.sqrt(1 - 1 / g**2)
            s = (1.088 * H.brems_ee_sigma(e, g) + 1.263 * H.brems_ep_sigma(e, g))
            return E * NE(E * u.erg).to_value(1 / u.erg) * v * s
        lo = np.log(max(m.Eemin.value, e + MEC2 * 1.0000001))
        hi = np.log(m.Eemax.value)
        jump = np.log(MEC2 * (1 + H._ee_T_min(e / MEC2)))
        v, _ = integrate.quad(f, lo, hi, limit=500, epsrel=1e-8,
                              points=[jump] if lo < jump < hi else None)
        assert o == pytest.approx(v / u.erg.to(u.eV), rel=1e-5)


# p-p cross sections
def test_threshold_and_masses():
    assert H._TTH == pytest.approx(0.2797, abs=1e-4)                # KATV14 Eq. 1
    assert H.TP_TH == pytest.approx(H._TTH * GEV, rel=1e-15)
    assert H.pp_inelastic_sigma(0.9 * H.TP_TH) == 0
    assert H.pp_pi0_sigma(0.999 * H.TP_TH) == 0
    assert H.pion_gamma_dsigma(0.06 * GEV, 0.99 * H.TP_TH) == 0


@pytest.mark.parametrize("Tp, mb", [(1.0, 22.519), (10.0, 29.470),
                                    (1e3, 34.895), (1e6, 57.200)])
def test_inelastic_sigma_eq1(Tp, mb):
    L = np.log(Tp / H._TTH)
    ref = (30.7 - 0.96 * L + 0.18 * L**2) * (1 - (H._TTH / Tp) ** 1.9) ** 3
    val = H.pp_inelastic_sigma(Tp * GEV) / 1e-27
    assert val == pytest.approx(ref, rel=1e-12)
    assert val == pytest.approx(mb, rel=1e-4)


def test_inelastic_sigma_shape():
    T = np.geomspace(0.281, 1e7, 400) * GEV
    s = H.pp_inelastic_sigma(T) / 1e-27
    assert np.all(s > 0)
    assert 20 < s[np.argmin(np.abs(T / GEV - 1))] < 25
    assert np.all(np.diff(s[T / GEV > 200]) > 0)                   # log rise


def test_single_pion_channel_vs_data():
    Tp = np.array(1.25)
    s2 = 5.7 / (1 + np.exp(-9.3 * (Tp - 1.4)))
    s1 = H._sigma_pi_gev(Tp, "Pythia8") - s2
    assert s1 == pytest.approx(3.74, rel=0.15)                      # KATV14 Table I


def test_multiplicity_qgsjet_excess():
    m = {k: H.pp_pi0_multiplicity(100 * GEV, k) for k in MODELS}
    for k in ("Geant4", "Pythia8", "SIBYLL"):
        assert 1.4 < m["QGSJET"] / m[k] < 2.0                       # KATV14 ~1.7


def test_model_spread():
    for T in (1e3, 1e4, 1e5):
        s = np.array([H.pp_pi0_sigma(T * GEV, k) for k in MODELS])
        assert np.all(np.abs(s / s.mean() - 1) < 0.3)
    for T in (1e5, 1e6):
        s = np.array([H.pp_pi0_sigma(T * GEV, k) for k in MODELS[1:]])
        assert np.all(np.abs(s / s.mean() - 1) < 0.3)


def test_low_energy_models_identical():
    T = np.geomspace(0.3, 49.0, 50) * GEV
    eg = 0.1 * GEV
    ref = H.pion_gamma_dsigma(eg, T, "Geant4")
    for k in MODELS[1:]:
        assert np.array_equal(H.pion_gamma_dsigma(eg, T, k), ref)


@pytest.mark.parametrize("model", MODELS)
def test_photon_number_is_two_per_pion(model):
    for T in (0.3, 0.9, 3.0, 10.0, 200.0, 1e4, 1e6):
        Tp = T * GEV
        emax = H.pion_gamma_emax(Tp)
        emin = (H.M_PI0 * GEV) ** 2 / (4 * emax)
        v, _ = integrate.quad(
            lambda le: np.exp(le) * H.pion_gamma_dsigma(np.exp(le), Tp, model),
            np.log(emin), np.log(emax), limit=400, epsrel=1e-8,
            points=[np.log(H.M_PI0 * GEV / 2)])
        assert 0.8 < v / (2 * H.pp_pi0_sigma(Tp, model)) < 1.25    # KATV14 20%


def test_dsigma_symmetry_and_endpoint():
    Tp = np.array([0.5, 5.0, 500.0]) * GEV
    r = np.geomspace(1.01, 30, 7)[:, None]
    half = 0.5 * H.M_PI0 * GEV
    a = H.pion_gamma_dsigma(half * r, Tp)
    b = H.pion_gamma_dsigma(half / r, Tp)
    assert np.allclose(a, b, rtol=1e-10, atol=0)
    emax = H.pion_gamma_emax(Tp)
    assert np.all(H.pion_gamma_dsigma(emax * (1 + 1e-9), Tp) == 0)
    assert np.all(H.pion_gamma_dsigma(emax * (1 - 1e-6), Tp) > 0)
    emin = half**2 / emax
    assert np.all(H.pion_gamma_dsigma(emin * (1 - 1e-9), Tp) == 0)


def test_tp_min_inverts_emax():
    Tp = np.geomspace(0.3, 1e6, 30) * GEV
    back = H.pion_tp_min(H.pion_gamma_emax(Tp))
    assert np.allclose(back, Tp, rtol=1e-9)
    assert np.all(H.pion_tp_min(np.array([0.06, 0.07]) * GEV) == H.TP_TH)


def test_amax_over_inelastic():
    T = np.geomspace(1e2, 1e6, 9) * GEV
    r = H.pion_gamma_amax(T, "Pythia8") / H.pp_inelastic_sigma(T) * GEV
    assert np.all((4.5 < r) & (r < 6.0))                            # KATV14 Fig. 9


def test_nuclear_enhancement():
    eps = H.pp_nuclear_enhancement(np.array([0.5, 1.0, 10.0, 1e3, 1e6]) * GEV)
    assert eps[2] == pytest.approx(1.79, abs=0.01)                  # KATV14 Sec. IV
    assert eps[0] == eps[1]
    assert np.all(np.diff(eps[1:]) < 0)
    a = H.pion_gamma_dsigma(0.3 * GEV, 10 * GEV, nuclear_enhancement=True)
    b = H.pion_gamma_dsigma(0.3 * GEV, 10 * GEV)
    c = H.pion_gamma_dsigma(0.3 * GEV, 10 * GEV, nuclear_enhancement=1.5)
    assert a / b == pytest.approx(eps[2], rel=1e-12)
    assert c / b == pytest.approx(1.5, rel=1e-12)


# models
def test_pion_vs_direct_quadrature():
    m = PionDecay(NE, rtol=1e-6, **RNG_P)
    eps = np.array([0.03, 0.0675, 0.5, 10.0, 1e3, 3e4]) * GEV
    ours = m._rate(eps)
    br = [0.56, 1.0, 2.0, 4.0, 5.0, 20.0, 50.0, 100.0, 1e3]
    for e, o in zip(eps, ours):
        def f(lnT):
            T = np.exp(lnT)
            E = T * GEV + H.MPC2
            beta = np.sqrt(T * (T + 2 * H._MP)) / (T + H._MP)
            ds = H.pion_gamma_dsigma(e, T * GEV, "Pythia8", True)
            return T * GEV * NE(E * u.erg).to_value(1 / u.erg) * beta * H.C.c_light * ds
        lo = np.log(H.pion_tp_min(e) / GEV)
        hi = np.log(1e6 - H._MP)
        pts = [np.log(b) for b in br if lo < np.log(b) < hi]
        v, _ = integrate.quad(f, lo, hi, points=pts, limit=1000, epsrel=1e-9)
        assert o == pytest.approx(v, rel=1e-5)


@pytest.mark.parametrize("cls, kw, E", [
    (Bremsstrahlung, RNG_E, np.geomspace(1e6, 1e15, 40)),
    (PionDecay, RNG_P, np.geomspace(1e7, 1e15, 40))])
def test_quadrature_converges(cls, kw, E):
    E = E * u.eV
    a = cls(NE, **kw)
    sa = a.sed(E).value
    assert a.quadrature.converged and a.quadrature.rel_error <= 1e-4
    b = cls(NE, rtol=1e-8, **kw)
    sb = b.sed(E).value
    assert b.quadrature.rel_error <= 1e-8
    m = sb > 1e-6 * sb.max()
    assert np.max(np.abs(sa[m] / sb[m] - 1)) < 3e-4


def test_pythia8_vs_geant4():
    E = np.geomspace(0.1, 1e4, 11) * u.GeV
    r = (PionDecay(NE, model="Pythia8", **RNG_P).sed(E)
         / PionDecay(NE, model="Geant4", **RNG_P).sed(E)).value
    assert np.all(np.abs(r[:3] - 1) < 0.03)        # shared below 50 GeV
    assert np.all(np.abs(r - 1) < 0.4)                              # model spread


def test_units_and_scaling():
    E = np.array([0.1, 1.0, 10.0]) * u.GeV
    for cls, kw, dens in ((Bremsstrahlung, RNG_E, "n0"), (PionDecay, RNG_P, "nh")):
        m1 = cls(NE, **kw)
        m2 = cls(NE, **{dens: 3 / u.cm**3}, **kw)
        f1 = m1.flux(E, distance=1 * u.kpc)
        assert f1.unit == u.Unit("1/(cm2 s eV)")
        f2 = m1.flux(E, distance=2 * u.kpc)
        assert np.allclose((f1 / f2).value, 4.0, rtol=1e-12)
        assert np.allclose((m2.flux(E) / f1).value, 3.0, rtol=1e-12)
        L = m1.luminosity(E)
        assert L.unit == u.erg / u.s
        d = (1 * u.kpc).to(u.cm)
        assert np.allclose((m1.sed(E) * 4 * np.pi * d**2).to_value(u.erg / u.s),
                           L.value, rtol=1e-10)
        assert np.allclose(m1.flux(E.to(u.MeV)).value, f1.value, rtol=1e-12)


def test_pion_kinematic_cutoffs():
    m = PionDecay(NE, Epmin=1 * u.GeV, Epmax=10 * u.GeV)
    emax = H.pion_gamma_emax(10 * u.GeV.to(u.erg) - H.MPC2) * u.erg
    f = m.flux(np.array([0.99, 1.01]) * emax)
    assert f[0] > 0 and f[1] == 0
    low = PionDecay(NE, Epmin=0.5 * u.GeV, Epmax=(H.MPC2 + 0.99 * H.TP_TH) * u.erg)
    assert np.all(low.flux(np.array([0.05, 0.07, 0.1]) * u.GeV).value == 0)


def test_wp_and_energy_conventions():
    pl = PowerLaw(1e30 / u.GeV, 1 * u.GeV, 2.0)
    m = PionDecay(pl, Epmin=10 * u.GeV, Epmax=1e4 * u.GeV)
    exact = 1e30 * np.log(1e3) * u.GeV
    assert m.Wp.to_value(u.GeV) == pytest.approx(exact.value, rel=1e-10)
    d = PionDecay(NE)
    assert d.Epmin.to_value(u.erg) == pytest.approx(H.MPC2 + H.TP_TH, rel=1e-14)
    assert d.Epmax.to(u.PeV).value == pytest.approx(10.0)
    assert d.Epmin is d.Emin and d.Wp == d.W
    d.Epmax = 1 * u.PeV
    assert d.Emax.to_value(u.PeV) == pytest.approx(1.0)


def test_hadronic_bad_inputs():
    with pytest.raises(ValueError):
        PionDecay(NE, model="EPOS")
    with pytest.raises(ValueError):
        H.pp_pi0_sigma(1.0, "EPOS")
    with pytest.raises(u.UnitsError):
        PionDecay(lambda E: np.ones(np.shape(E)) * u.one)
    with pytest.raises(ValueError):
        PionDecay(NE, Epmin=10 * u.TeV, Epmax=1 * u.TeV)
