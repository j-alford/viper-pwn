"""Kernel accuracy tests."""

import numpy as np
import pytest
from scipy import integrate, special

from viper.radiative import kernels as K

LN = np.log


# reference integrals
def F_ref(x):
    """Reference F(x)."""
    f = lambda lt: special.kve(5.0 / 3.0, np.exp(lt)) * np.exp(-(np.exp(lt) - x)) * np.exp(lt)
    val, _ = integrate.quad(f, LN(x), LN(x + 80.0), epsabs=0, epsrel=1e-12, limit=400)
    return x * np.exp(-x) * val


def G_pitch_ref(x):
    """Pitch-angle-averaged F."""
    f = lambda a: np.sin(a) ** 2 * F_ref(x / np.sin(a))
    lo = 0.0 if x < 1 else np.arcsin(min(1.0, x / (x + 80.0)))
    val, _ = integrate.quad(f, lo, np.pi / 2, epsabs=0, epsrel=1e-10, limit=200)
    return val


def akp10_D7(x):
    """AKP10 D7."""
    cb = np.cbrt(x)
    return (1.808 * cb / np.sqrt(1 + 3.4 * cb**2)
            * (1 + 2.210 * cb**2 + 0.347 * cb**4)
            / (1 + 1.353 * cb**2 + 0.217 * cb**4) * np.exp(-x))


def f_i_scaled(x0, i):
    """Scaled f_i, ZP13."""
    if i == "ln":
        g = lambda t: np.log(x0 + t)
    else:
        g = lambda t: (x0 + t) ** i
    f = lambda lt: g(np.exp(lt)) * np.exp(-np.exp(lt)) / (-np.expm1(-(x0 + np.exp(lt)))) * np.exp(lt)
    val, _ = integrate.quad(f, LN(1e-14 * min(x0, 1.0)), LN(80.0), epsabs=0,
                            epsrel=1e-13, limit=500)
    return val


def kak_F_ref(x0):
    """Reference F1..F4, KAK14 Eqs. 12-16."""
    f1 = f_i_scaled(x0, 1)
    fm1 = f_i_scaled(x0, -1)
    fln = f_i_scaled(x0, "ln")
    f0 = -np.log1p(-np.exp(-x0)) * np.exp(x0)
    F1 = f1
    F2 = f1 + 2 * x0**2 * fm1 - 2 * x0 * f0
    F3 = f1 - x0 * f0
    F4 = F3 + 2 * x0 * (np.log(x0) * f0 - fln) + 2 * x0 * (f0 - x0 * fm1)
    return np.array([F1, F2, F3, F4])


def kak_G12(x, p):
    alpha, a, beta, b = p
    return (np.pi**2 / 6 + x) * np.exp(-x) / (1 + a * x**alpha / (1 + b * x**beta))


def kak_G34(x, p):
    alpha, a, beta, b, c = p
    g0 = np.pi**2 / 6 * (1 + c * x) / (1 + np.pi**2 / 6 * c * x) * np.exp(-x)
    return g0 / (1 + a * x**alpha / (1 + b * x**beta))


KAK_P = {1: (0.857, 0.153, 1.840, 0.254), 2: (0.691, 1.330, 1.668, 0.534),
         3: (0.606, 0.443, 1.481, 0.540, 0.319), 4: (0.461, 0.726, 1.457, 0.382, 6.62)}


# synchrotron
@pytest.mark.parametrize("x", [1e-9, 1e-5, 1e-3, 0.1, 0.29, 1.0, 5.0, 20.0, 100.0, 500.0])
def test_F_table_vs_quadrature(x):
    assert K.synch_F(x) == pytest.approx(F_ref(x), rel=1e-6)


def test_G_table_vs_closed_form():
    x = np.geomspace(1e-9, 600.0, 4001)
    rel = np.abs(K.synch_G(x) / K.synch_G_exact(x) - 1)
    assert rel.max() < 1e-6


@pytest.mark.parametrize("x", [1e-4, 1e-2, 0.229, 1.0, 4.0, 12.0, 30.0])
def test_G_vs_pitch_angle_quadrature(x):
    """G vs pitch-angle quadrature."""
    assert K.synch_G(x) == pytest.approx(G_pitch_ref(x), rel=1e-4)


def test_akp10_D7_within_0p3_percent():
    """D7 vs exact G."""
    x = np.geomspace(1e-5, 30.0, 3001)
    rel = np.abs(akp10_D7(x) / K.synch_G(x) - 1)
    assert rel.max() < 3e-3
    print(f"\nAKP10 D7 vs exact G: max {rel.max():.2e} at x = {x[rel.argmax()]:.3g}")


def test_G_peak_location():
    x = np.geomspace(0.1, 0.5, 40001)
    assert x[np.argmax(K.synch_G(x))] == pytest.approx(0.2292, abs=2e-4)


@pytest.mark.parametrize("kernel", ["isotropic", "fixed", "turbulent"])
@pytest.mark.parametrize("gamma", [1e3, 1e7])
def test_synch_total_power(kernel, gamma):
    B = 1e-4
    nuc = K.synch_nu_c(gamma, B)
    lnnu = np.linspace(LN(nuc * 1e-12), LN(nuc * 300.0), 200001)
    nu = np.exp(lnnu)
    P = K.synch_power_per_electron(nu, gamma, B, kernel=kernel)
    tot = integrate.simpson(P * nu, x=lnnu)
    assert tot / K.synch_loss_rate(gamma, B) == pytest.approx(1.0, abs=1e-3)


def test_synch_fixed_pitch_angle_scaling():
    """Power scales as 1.5 sin^2."""
    B, gamma, s = 1e-4, 1e5, 0.4
    nuc = K.synch_nu_c(gamma, B)
    lnnu = np.linspace(LN(nuc * 1e-12), LN(nuc * 300.0), 200001)
    nu = np.exp(lnnu)
    P = K.synch_power_per_electron(nu, gamma, B, kernel="fixed", sin_alpha=s)
    tot = integrate.simpson(P * nu, x=lnnu)
    assert tot / K.synch_loss_rate(gamma, B) == pytest.approx(1.5 * s**2, rel=1e-3)


def test_turbulent_normalisation_identity():
    """int x Q dx = 3/4."""
    val, _ = integrate.quad(lambda lx: np.exp(2 * lx) * K.synch_Q_turbulent(np.exp(lx)),
                            LN(1e-12), LN(200.0), epsabs=0, epsrel=1e-12, limit=400)
    assert val == pytest.approx(0.75, rel=1e-10)


def test_synch_photon_rate_matches_power():
    B, g = 3e-5, np.array([1e4, 1e6, 1e8])
    eps = np.geomspace(1e-20, 1e-4, 50)
    for kern in ("isotropic", "fixed", "turbulent"):
        rate = K.synch_photon_rate(eps, g, B, kern)
        P = K.synch_power_per_electron((eps / K.C.h_cgs)[None, :], g[:, None], B, kern)
        want = P / (K.C.h_cgs * eps)
        big = want > 1e-250
        np.testing.assert_allclose(rate[big], want[big], rtol=1e-12)


# blackbody IC
@pytest.mark.parametrize("x0", [1e-10, 1e-6, 1e-3, 0.037, 0.5, 3.3, 10.0, 100.0, 500.0])
def test_kak_F_exact_vs_definitions(x0):
    ref = kak_F_ref(x0)
    got = np.array([K.ic_bb_F(k, x0) for k in (1, 2, 3, 4)]) * np.exp(x0)
    np.testing.assert_allclose(got, ref, rtol=1e-7)


def test_kak_F_asymptotes():
    for k in (1, 2, 3, 4):
        assert K.ic_bb_F(k, 1e-14) == pytest.approx(np.pi**2 / 6, rel=1e-9)
    x0 = 600.0
    assert K.ic_bb_F(3, x0) * np.exp(x0) == pytest.approx(1.0, rel=1e-9)
    assert K.ic_bb_F(4, x0) * np.exp(x0) == pytest.approx(1 - 2 / x0 + 8 / x0**2, rel=1e-6)
    assert K.ic_bb_F(1, x0) * np.exp(x0) / x0 == pytest.approx(1 + 1 / x0, rel=1e-9)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_kak_fits_within_stated_1_percent(k):
    """KAK14 fits vs exact."""
    x0 = np.geomspace(1e-3, 1e2, 2001)
    fit = kak_G12(x0, KAK_P[k]) if k <= 2 else kak_G34(x0, KAK_P[k])
    rel = fit / K.ic_bb_F(k, x0) - 1
    i = np.argmax(np.abs(rel))
    print(f"\nKAK14 G{k}/F{k} - 1: max |dev| {abs(rel[i]):.2e} at x0 = {x0[i]:.3g}")
    assert np.max(np.abs(rel)) < 1.05e-2


def _jones_planck_ref(eps, gamma, T, u):
    """Jones x Planck quadrature."""
    z = eps / (gamma * K.MEC2)
    lo = LN(z / (4 * gamma * (1 - z)) * K.MEC2 * (1 + 1e-14))
    hi = min(LN(z * gamma / (1 - z) * K.MEC2), LN(200 * K.C.k_B * T))
    f = lambda le: (K.planck_number_density(np.exp(le), T, u) * np.exp(le)
                    * K.ic_jones_kernel(eps, gamma, np.exp(le)))
    val, _ = integrate.quad(f, lo, hi, epsabs=0, epsrel=1e-12, limit=1000)
    return val


@pytest.mark.parametrize("gamma,eps_eV,T", [
    (1e4, 1e7, 3000.0), (1e6, 1e11, 3000.0), (1e9, 1e14, 3000.0),
    (0.9999e9, 3.5e14, 3000.0), (2e5, 1e8, 2.72548), (1e8, 4e13, 30.0)])
def test_bb_kernel_vs_planck_quadrature(gamma, eps_eV, T):
    u = 1.602e-12
    eps = eps_eV * 1.602176634e-12
    got = K.ic_blackbody_kernel(np.array([eps]), np.array([gamma]), T, u)[0, 0]
    assert got == pytest.approx(_jones_planck_ref(eps, gamma, T, u), rel=1e-4)


def _aniso_planck_ref(eps, gamma, T, theta):
    """KAK14 Eq. 2 x Planck."""
    E = gamma
    w = eps / K.MEC2
    z = w / E
    omc = 1 - np.cos(theta)

    def f(le):
        w0 = np.exp(le) / K.MEC2
        bt = 2 * w0 * E * omc
        rate = (4 * np.pi * K.C.c_light * K.R0**2 * omc / (bt * E)
                * (1 + z**2 / (2 * (1 - z)) - 2 * z / (bt * (1 - z))
                   + 2 * z**2 / (bt**2 * (1 - z) ** 2)))
        return K.planck_number_density(np.exp(le), T) * np.exp(le) * rate / K.MEC2

    lo = LN(z / ((1 - z) * 2 * E * omc) * K.MEC2 * (1 + 1e-14))
    val, _ = integrate.quad(f, lo, LN(200 * K.C.k_B * T), epsabs=0, epsrel=1e-12, limit=1000)
    return val


@pytest.mark.parametrize("gamma,eps_eV,theta", [
    (1e6, 1e11, np.pi), (1e8, 1e13, 2.0), (1e5, 1e9, 0.7)])
def test_bb_aniso_kernel_vs_planck_quadrature(gamma, eps_eV, theta):
    T = 3000.0
    eps = eps_eV * 1.602176634e-12
    got = K.ic_blackbody_kernel(np.array([eps]), np.array([gamma]), T, theta=theta)[0, 0]
    assert got == pytest.approx(_aniso_planck_ref(eps, gamma, T, theta), rel=1e-4)


def test_bb_kernel_2d_gamma_equals_outer():
    eps = np.geomspace(1e-6, 1e2, 7)
    g = np.geomspace(1e3, 1e9, 5)
    outer = K.ic_blackbody_kernel(eps, g, 30.0, 1e-12)
    g2 = np.repeat(g[:, None], eps.size, axis=1)
    np.testing.assert_array_equal(K.ic_blackbody_kernel(eps, g2, 30.0, 1e-12), outer)


# tabulated seeds
def _planck_table(T, u, n=300):
    e0 = np.geomspace(1e-5, 60.0, n) * K.C.k_B * T
    return K.SeedTable(e0, K.planck_number_density(e0, T, u))


def test_table_seed_kernel_vs_blackbody():
    T, u = 3000.0, 1.602e-12
    st = _planck_table(T, u)
    eps = np.geomspace(1e-7, 1e2, 40)                 # 0.06 MeV-60 PeV
    g = np.geomspace(1e4, 1e9, 30)
    a = K.ic_seed_kernel(eps, g, st)
    b = K.ic_blackbody_kernel(eps, g, T, u)
    sig = b > 1e-8 * b.max(axis=1, keepdims=True)
    rel = np.abs(a[sig] / b[sig] - 1)
    assert rel.max() < 1e-3


def test_seed_table_energy_density():
    T, u = 30.0, 0.5 * 1.602176634e-12
    assert _planck_table(T, u).energy_density() == pytest.approx(u, rel=1e-4)


# loss rates
def kak_Giso(t):
    """KAK14 Eqs. 38-39."""
    alpha, a, beta, b, c = 0.682, -0.362, 1.281, 0.826, 5.68
    g0 = c * t * np.log(1 + 0.722 * t / c) / (1 + c * t / 0.822)
    return g0 / (1 + a * t**alpha / (1 + b * t**beta))


def test_loss_function_thomson_limit():
    for t in (1e-9, 1e-7, 1e-6):
        assert K.ic_loss_function_iso(t) == pytest.approx(np.pi**4 / 135 * t**2, rel=5e-5)


def test_loss_function_kn_slope():
    """KN slope pi^2/12."""
    t1, t2 = 1e8, 1e9
    slope = (K.ic_loss_function_iso(t2) - K.ic_loss_function_iso(t1)) / np.log(t2 / t1)
    assert slope == pytest.approx(np.pi**2 / 12, rel=2e-3)


def test_loss_function_vs_kak_fit():
    t = np.geomspace(1e-2, 1e4, 601)
    rel = np.abs(kak_Giso(t) / K.ic_loss_function_iso(t) - 1)
    print(f"\nKAK14 G_iso vs exact F_iso: max {rel.max():.2e} at t = {t[rel.argmax()]:.3g}")
    assert rel.max() < 1.5e-2


def test_loss_function_vs_direct_double_integral():
    """KAK14 Eq. 36, 2-D."""
    def J(z, b):
        return (1 + z**2 / (2 * (1 - z)) + z / (b * (1 - z)) - 2 * z**2 / (b**2 * (1 - z) ** 2)
                - z**3 / (2 * b * (1 - z) ** 2) - 2 * z / (b * (1 - z)) * np.log(b * (1 - z) / z))

    for t in (0.3, 30.0):
        def inner(lb):
            b = np.exp(lb)
            zmax = b / (1 + b)
            v, _ = integrate.quad(lambda z: z * J(z, b), 0, zmax, epsabs=0, epsrel=1e-11, limit=200)
            return b * b * v / np.expm1(b / t)
        val, _ = integrate.quad(inner, LN(1e-6 * t), LN(80 * t), epsabs=0, epsrel=1e-10, limit=200)
        assert K.ic_loss_function_iso(t) == pytest.approx(val / t**2, rel=1e-5)


def test_bb_loss_rate_thomson():
    T, u = 2.72548, None
    g = np.array([3.0, 10.0])
    uu = K.A_RAD * T**4
    th = 4 / 3 * K.SIGMA_T * K.C.c_light * uu * g**2
    np.testing.assert_allclose(K.ic_loss_rate_blackbody(g * K.MEC2, T, u), th, rtol=1e-6)


@pytest.mark.parametrize("gamma", [1e3, 1e6, 1e9])
def test_bb_emission_integrates_to_loss(gamma):
    """Emission integrates to loss."""
    T, u = 3000.0, 1.602e-12
    E = gamma * K.MEC2
    le = np.linspace(LN(E * 1e-16), LN(E * (1 - 1e-12)), 40001)
    eps = np.exp(le)
    k = K.ic_blackbody_kernel(eps, np.array([gamma]), T, u)[0]
    emitted = integrate.simpson(k * eps * eps, x=le)
    assert emitted == pytest.approx(float(K.ic_loss_rate_blackbody(E, T, u)), rel=1e-4)


def test_table_seed_loss_vs_blackbody():
    T, u = 3000.0, 1.602e-12
    st = _planck_table(T, u)
    E = np.geomspace(1e-3, 1e3, 13) * 1.602176634          # 1 GeV-1 PeV
    np.testing.assert_allclose(K.ic_loss_rate_seed(E, st),
                               K.ic_loss_rate_blackbody(E, T, u), rtol=1e-4)
