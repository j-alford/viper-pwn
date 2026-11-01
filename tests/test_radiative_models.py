"""Model tests."""

import os
import subprocess
import sys
import time
import warnings

import numpy as np
import pytest
import astropy.units as u

from _radiative_common import E_IC, E_SYN, NE, RNG, RNG_P
from viper import constants as C
from viper.radiative import (Blackbody, Bremsstrahlung, BrokenPowerLaw,
                             InverseCompton, PionDecay, PowerLaw,
                             QuadratureWarning, SeedTable, SSC, SumModel,
                             Synchrotron, TableDistribution, TableSeed,
                             ssc_seed, trapz_loglog)
from viper.radiative import kernels as K
from viper.radiative.quadrature import integrate_piecewise, integrate_population
from viper.radiative.seeds import Seed, as_seed


# quadrature
@pytest.mark.parametrize("p", [-2.3, -1.0, 0.0, 1.7])
def test_trapz_loglog_exact_for_power_laws(p):
    x = np.geomspace(1.0, 1e4, 7)
    y = x**p
    exact = np.log(1e4) if p == -1 else (1e4 ** (p + 1) - 1) / (p + 1)
    assert trapz_loglog(y, x) == pytest.approx(exact, rel=1e-12)


def test_integrate_population_power_law_exact():
    res, info = integrate_population(
        lambda E, e: np.ones((E.size, e.size)), lambda E: E**-2.0,
        1.0, 1e6, np.array([1.0, 2.0]), rtol=1e-12)
    assert res == pytest.approx(1 - 1e-6, rel=1e-13)
    assert info.converged


def test_integrate_piecewise_matches_single_piece():
    kern = lambda E, e, k: np.exp(-E[:, None] * e[None, :] / 1e3)
    dens = lambda E: E**-1.5
    eps = np.array([0.5, 1.0, 3.0])
    one, _ = integrate_piecewise(kern, dens, [1.0, 1e4], eps, rtol=1e-10)
    two, _ = integrate_piecewise(kern, dens, [1.0, 30.0, 1e4], eps, rtol=1e-10)
    np.testing.assert_allclose(two, one, rtol=1e-9)


@pytest.mark.parametrize("model", ["syn_iso", "syn_turb", "ic"])
def test_achieved_error_meets_rtol(model):
    def build(rtol):
        if model == "syn_iso":
            return Synchrotron(NE, B=100 * u.uG, rtol=rtol, max_per_decade=1e5, **RNG), E_SYN
        if model == "syn_turb":
            return Synchrotron(NE, B=100 * u.uG, kernel="turbulent", rtol=rtol,
                               max_per_decade=1e5, **RNG), E_SYN
        return InverseCompton(NE, seeds=["CMB", "FIR", "NIR"], rtol=rtol,
                              max_per_decade=1e5, **RNG), E_IC

    m, E = build(1e-4)
    a = m.sed(E).value
    assert m.quadrature.converged and m.quadrature.rel_error <= 1e-4
    ref, _ = build(1e-9)
    b = ref.sed(E).value
    sig = b > 1e-6 * b.max()
    assert np.max(np.abs(a[sig] / b[sig] - 1)) < 1e-4


# convergence signalling
@pytest.mark.parametrize("make", [
    lambda: Synchrotron(NE, B=100 * u.uG, max_per_decade=4, **RNG),
    lambda: InverseCompton(NE, seeds=["CMB"], max_per_decade=4, **RNG),
    lambda: Bremsstrahlung(NE, max_per_decade=4, **RNG),
    lambda: PionDecay(NE, max_per_decade=4, **RNG_P)])
def test_unconverged_warns(make):
    E = np.geomspace(1e9, 1e13, 5) * u.eV
    m = make()
    with pytest.warns(QuadratureWarning, match="rel_error .* points per decade"):
        m.sed(E)
    assert not m.quadrature.converged
    assert m.quadrature.per_decade < 32


def test_on_unconverged_modes():
    E = np.geomspace(1e-3, 1e3, 5) * u.eV
    kw = dict(B=100 * u.uG, max_per_decade=4, **RNG)
    with pytest.raises(QuadratureWarning):
        Synchrotron(NE, on_unconverged="raise", **kw).sed(E)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = Synchrotron(NE, on_unconverged="ignore", **kw)
        m.sed(E)
    assert not m.quadrature.converged
    with pytest.raises(ValueError):
        Synchrotron(NE, on_unconverged="shout", **kw)


def test_seed_kernel_unconverged_warns():
    e0 = np.geomspace(1e-5, 60.0, 300) * K.C.k_B * 3000.0
    st = SeedTable(e0, K.planck_number_density(e0, 3000.0, 1.6e-12))
    with pytest.warns(QuadratureWarning, match="seed kernel"):
        K.ic_seed_kernel(np.geomspace(1e-7, 1e2, 20), np.geomspace(1e4, 1e9, 10),
                         st, rtol=1e-12, max_panels=8)


def test_sum_model_quadrature_is_worst_part():
    syn = Synchrotron(NE, B=100 * u.uG, **RNG)
    bad = Synchrotron(NE, B=100 * u.uG, max_per_decade=4, on_unconverged="ignore", **RNG)
    tot = syn + bad
    assert tot.quadrature is None
    tot.sed(np.geomspace(1e-3, 1e3, 5) * u.eV)
    assert tot.quadrature is bad.quadrature
    assert not tot.quadrature.converged


# interface
def test_flux_sed_units_and_scaling():
    syn = Synchrotron(NE, B=10 * u.uG, **RNG)
    E = np.geomspace(1e-3, 1e4, 9) * u.eV
    f1 = syn.flux(E, distance=1 * u.kpc)
    f2 = syn.flux(E, distance=2 * u.kpc)
    L = syn.flux(E, distance=0)
    assert f1.unit == u.Unit("1/(s cm2 eV)") and L.unit == u.Unit("1/(s eV)")
    np.testing.assert_allclose((f1 / f2).value, 4.0, rtol=1e-12)
    np.testing.assert_allclose((L / (4 * np.pi * (1 * u.kpc) ** 2) / f1).decompose().value,
                               1.0, rtol=1e-12)
    sed = syn.sed(E, distance=1 * u.kpc)
    np.testing.assert_allclose(sed.to_value("erg/(cm2 s)"),
                               (f1 * E**2).to_value("erg/(cm2 s)"), rtol=1e-12)
    assert syn.luminosity(E).unit == u.erg / u.s
    nu = E.to(u.Hz, equivalencies=u.spectral())
    np.testing.assert_allclose(syn.flux(nu).value, f1.value, rtol=1e-12)


def test_distance_none_and_negative():
    syn = Synchrotron(NE, B=10 * u.uG, **RNG)
    E = np.geomspace(1e-3, 1e4, 5) * u.eV
    np.testing.assert_array_equal(syn.flux(E, distance=None).value,
                                  syn.flux(E, distance=0).value)
    assert syn.sed(E, distance=None).unit == u.erg / u.s
    with pytest.raises(ValueError):
        syn.flux(E, distance=-1 * u.kpc)


def test_sum_and_seed_selection():
    ic = InverseCompton(NE, seeds=["CMB", Blackbody(30 * u.K, 0.5 * u.eV / u.cm**3, name="dust")], **RNG)
    E = np.geomspace(1e9, 1e14, 11) * u.eV
    tot = ic.sed(E).value
    parts = ic.sed(E, seed="CMB").value + ic.sed(E, seed="dust").value
    np.testing.assert_allclose(tot, parts, rtol=2e-4)
    np.testing.assert_allclose(ic.sed(E, seed=1).value, ic.sed(E, seed="dust").value, rtol=1e-12)
    syn = Synchrotron(NE, B=10 * u.uG, **RNG)
    both = (syn + ic).sed(E).value
    np.testing.assert_allclose(both, syn.sed(E).value + tot, rtol=1e-12)
    with pytest.raises(ValueError):
        ic.flux(E, seed="nope")


def test_sum_model_nesting_flattens():
    a = Synchrotron(NE, B=10 * u.uG, **RNG)
    b = InverseCompton(NE, **RNG)
    c = Bremsstrahlung(NE, **RNG)
    s = (a + b) + c
    assert isinstance(s, SumModel) and s.models == [a, b, c]
    s2 = SumModel([SumModel([a, SumModel([b])]), c])
    assert s2.models == [a, b, c]


def test_inverse_compton_bare_string_seed():
    ic = InverseCompton(NE, seeds="CMB", **RNG)
    assert ic.seed_names == ["CMB"]
    E = np.geomspace(1e9, 1e13, 5) * u.eV
    np.testing.assert_array_equal(ic.sed(E).value,
                                  InverseCompton(NE, seeds=["CMB"], **RNG).sed(E).value)


# scaling laws
def test_synchrotron_B_scaling_power_law():
    pl = PowerLaw(1e30 / u.erg, 1 * u.TeV, 2.0)
    E = np.array([1e-2, 1.0, 100.0]) * u.eV
    r = (Synchrotron(pl, B=20 * u.uG, **RNG).flux(E)
         / Synchrotron(pl, B=10 * u.uG, **RNG).flux(E)).value
    np.testing.assert_allclose(r, 2**1.5, rtol=1e-2)


def test_linearity_in_amplitude():
    pl = PowerLaw(1e30 / u.erg, 1 * u.TeV, 2.0)
    pl3 = PowerLaw(3e30 / u.erg, 1 * u.TeV, 2.0)
    E = np.array([1e-2, 1.0, 100.0]) * u.eV
    r = (Synchrotron(pl3, B=10 * u.uG, **RNG).flux(E)
         / Synchrotron(pl, B=10 * u.uG, **RNG).flux(E)).value
    np.testing.assert_allclose(r, 3.0, rtol=1e-12)


def test_ic_proportional_to_energy_density():
    E = np.geomspace(1e9, 1e13, 5) * u.eV
    a = InverseCompton(NE, seeds=[Blackbody(30 * u.K, 0.5 * u.eV / u.cm**3)], **RNG).flux(E)
    b = InverseCompton(NE, seeds=[Blackbody(30 * u.K, 1.5 * u.eV / u.cm**3)], **RNG).flux(E)
    np.testing.assert_allclose((b / a).value, 3.0, rtol=1e-12)


def test_ssc_inverse_square_radius():
    E = np.geomspace(1e9, 1e13, 5) * u.eV
    s1 = SSC(NE, B=100 * u.uG, R=1 * u.pc, **RNG)
    s2 = SSC(NE, B=100 * u.uG, R=2 * u.pc, **RNG)
    np.testing.assert_allclose((s1.flux(E) / s2.flux(E)).value, 4.0, rtol=1e-12)


def test_ssc_seed_follows_mutation():
    E = np.geomspace(1e9, 1e13, 5) * u.eV
    ssc = SSC(NE, B=100 * u.uG, R=1 * u.pc, **RNG)
    f0 = ssc.flux(E).value
    ssc.synchrotron.B = 200 * u.uG
    f1 = ssc.flux(E).value
    assert np.all(f1 > 2.5 * f0)
    np.testing.assert_array_equal(f1, SSC(NE, B=200 * u.uG, R=1 * u.pc, **RNG).flux(E).value)
    ssc.R = 2 * u.pc
    np.testing.assert_array_equal(ssc.flux(E).value,
                                  SSC(NE, B=200 * u.uG, R=2 * u.pc, **RNG).flux(E).value)
    ssc.B = 100 * u.uG
    np.testing.assert_array_equal(ssc.flux(E).value,
                                  SSC(NE, B=100 * u.uG, R=2 * u.pc, **RNG).flux(E).value)


def test_ssc_extra_seeds_and_seed_options():
    E = np.geomspace(1e9, 1e13, 5) * u.eV
    ssc = SSC(NE, B=100 * u.uG, R=1 * u.pc, extra_seeds=["CMB"], **RNG)
    assert ssc.seed_names == ["SSC", "CMB"]
    cmb = InverseCompton(NE, seeds=["CMB"], **RNG).sed(E).value
    np.testing.assert_allclose(ssc.sed(E, seed="CMB").value, cmb, rtol=1e-12)
    syn = Synchrotron(NE, B=100 * u.uG, **RNG)
    energy = np.geomspace(1e-6, 1e5, 400) * u.eV
    s1 = ssc_seed(syn, 1 * u.pc, energy=energy, factor=1.0)
    s2 = ssc_seed(syn, 1 * u.pc, energy=energy)
    assert (s2.energy_density / s1.energy_density).value == pytest.approx(K.SSC_UNIFORM_SPHERE, rel=1e-12)
    assert s1.table.eps.size == energy.size
    with pytest.raises(ValueError):
        ssc_seed(syn, 1 * u.pc, geometry="shell")


# distributions
def test_broken_power_law_continuity_and_slopes():
    bpl = BrokenPowerLaw(1e30 / u.erg, 1 * u.TeV, 10 * u.TeV, 1.5, 2.5)
    eb = 10 * u.TeV
    assert (bpl(eb * (1 - 1e-12)) / bpl(eb * (1 + 1e-12))).value == pytest.approx(1.0, rel=1e-10)
    assert (bpl(1 * u.TeV) / bpl(2 * u.TeV)).value == pytest.approx(2**1.5, rel=1e-12)
    assert (bpl(20 * u.TeV) / bpl(40 * u.TeV)).value == pytest.approx(2**2.5, rel=1e-12)


@pytest.mark.parametrize("mode,ic_tol,syn_tol", [
    ("pchip", 3e-4, 1e-5), ("cubic", 3e-4, 1e-5), ("linear", None, None)])
def test_table_distribution_modes(mode, ic_tol, syn_tol):
    Ee = np.geomspace(1 * u.GeV, 1 * u.PeV, 200)
    td = TableDistribution(Ee, NE(Ee), interpolation=mode)
    a = Synchrotron(td, B=100 * u.uG).sed(E_SYN).value
    b = Synchrotron(NE, B=100 * u.uG, **RNG).sed(E_SYN).value
    sig = b > 1e-6 * b.max()
    d_syn = np.max(np.abs(a[sig] / b[sig] - 1))
    ic = InverseCompton(td, seeds=["CMB", "FIR", "NIR"])
    a = ic.sed(E_IC).value
    t0 = time.perf_counter()
    for _ in range(3):
        ic.sed(E_IC)
    ms = (time.perf_counter() - t0) / 3 * 1e3
    b = InverseCompton(NE, seeds=["CMB", "FIR", "NIR"], **RNG).sed(E_IC).value
    sig = b > 1e-6 * b.max()
    d_ic = np.max(np.abs(a[sig] / b[sig] - 1))
    print(f"\nTableDistribution {mode}: IC {ms:.1f} ms, dev IC {d_ic:.2e} syn {d_syn:.2e}")
    if mode == "linear":
        assert d_ic > 2e-3 and d_syn > 2e-3
    else:
        assert d_ic < ic_tol and d_syn < syn_tol


def test_table_distribution_step_no_overshoot():
    Et = np.geomspace(1, 1e6, 60) * u.GeV
    v = np.where(Et.value < 1e3, 1e30, 1e28) / u.GeV
    td = TableDistribution(Et, v)
    y = td(np.geomspace(1, 1e6, 100001) * u.GeV).to_value(1 / u.GeV)
    assert y.max() <= 1e30 * (1 + 1e-12) and y.min() >= 1e28 * (1 - 1e-12)
    cub = TableDistribution(Et, v, interpolation="cubic")
    assert cub(np.geomspace(1, 1e6, 100001) * u.GeV).to_value(1 / u.GeV).max() > 1e30 * 1.01


def test_table_distribution_zeros_and_errors():
    Et = np.geomspace(1, 1e6, 60) * u.GeV
    v = np.where(Et.value < 1e3, 1e30, 0.0) / u.GeV
    td = TableDistribution(Et, v)
    assert td.Emax == Et[Et.value < 1e3][-1].to(u.erg)
    assert td(td.Emax).value > 0 and td(Et[-1]).value == 0 and td(2e3 * u.GeV).value == 0
    np.testing.assert_allclose(td(Et[:3]).to_value(1 / u.GeV), 1e30, rtol=1e-12)
    with pytest.raises(ValueError):
        TableDistribution(Et, v, interpolation="spline")
    with pytest.raises(ValueError):
        TableDistribution(Et[::-1], v)
    with pytest.raises(ValueError):
        TableDistribution(Et, np.zeros(60) / u.GeV)


def test_blackbody_presets_and_energy_density():
    assert Blackbody("CMB").T.value == pytest.approx(C.T_CMB)
    nir = Blackbody("NIR")
    assert nir.energy_density.to_value(u.eV / u.cm**3) == pytest.approx(1.0)
    cmb_u = Blackbody("CMB").energy_density.to_value(u.eV / u.cm**3)
    assert cmb_u == pytest.approx(0.26, rel=0.02)


def test_model_bad_inputs():
    with pytest.raises(ValueError):
        Synchrotron(NE, kernel="legacy")
    with pytest.raises(u.UnitsError):
        Synchrotron(lambda E: E.to(u.erg))
    with pytest.raises(ValueError):
        Blackbody("XYZ")
    with pytest.raises(ValueError):
        Synchrotron(NE, Eemin=1 * u.TeV, Eemax=1 * u.GeV)


# seeds
def test_seed_protocol():
    for s in (Blackbody("CMB"), TableSeed([1, 2, 3] * u.eV, [1, 2, 1] / (u.cm**3 * u.eV))):
        assert isinstance(s, Seed) and as_seed(s) is s
        assert s.eps_max > 0 and s.energy_density.unit.is_equivalent(u.erg / u.cm**3)
    with pytest.raises(TypeError):
        as_seed(3.0)
    with pytest.raises(NotImplementedError):
        Blackbody(30 * u.K, theta=1 * u.rad).loss_rate(np.array([1.0]))
    with pytest.raises(NotImplementedError):
        InverseCompton(NE, seeds=[Blackbody(30 * u.K, theta=1 * u.rad)], **RNG).energy_loss_rate(1 * u.TeV)


def test_table_seed_interpolation_options():
    e0 = np.geomspace(1e-5, 60.0, 300) * K.C.k_B * 3000.0 * u.erg
    n = K.planck_number_density(e0.value, 3000.0, 1.6e-12) / (u.cm**3 * u.erg)
    lin = TableSeed(e0, n, interpolation="linear")
    cub = TableSeed(e0, n)
    assert lin.energy_density.value == pytest.approx(cub.energy_density.value, rel=1e-3)
    with pytest.raises(ValueError):
        TableSeed(e0, n, interpolation="quintic")
    with pytest.raises(ValueError):
        SeedTable(np.array([1.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        SeedTable(np.array([1.0, 2.0]), np.array([1.0, 0.0]))
    with pytest.raises(u.UnitsError):
        TableSeed(e0, n.value * u.K)


# energy conservation
@pytest.mark.parametrize("kernel", ["isotropic", "fixed", "turbulent"])
def test_synchrotron_population_energy_conservation(kernel):
    pl = PowerLaw(1e30 / u.erg, 1 * u.TeV, 2.0)
    syn = Synchrotron(pl, B=50 * u.uG, kernel=kernel, Eemin=10 * u.GeV, Eemax=100 * u.TeV)
    E = np.geomspace(1e-14, 1e10, 1200) * u.eV
    L = syn.sed(E, distance=0).to_value(u.erg / u.s)
    emitted = np.trapezoid(L, np.log(E.value))
    Ee = np.geomspace(10, 1e5, 4001) * u.GeV
    lost = trapz_loglog((pl(Ee) * syn.energy_loss_rate(Ee)).to_value(1 / u.s), Ee.to_value(u.erg))
    assert emitted == pytest.approx(lost, rel=2e-3)


def test_ic_population_energy_conservation():
    ic = InverseCompton(NE, seeds=["CMB", "NIR"], **RNG)
    E = np.geomspace(1e3, 1e15, 1500) * u.eV
    L = ic.sed(E, distance=0).to_value(u.erg / u.s)
    emitted = np.trapezoid(L, np.log(E.value))
    Ee = np.geomspace(1 * u.GeV, 1 * u.PeV, 4001)
    lost = trapz_loglog((NE(Ee) * ic.energy_loss_rate(Ee)).to_value(1 / u.s), Ee.to_value(u.erg))
    assert emitted == pytest.approx(lost, rel=2e-3)


def test_table_seed_model_vs_blackbody():
    T, uu = 3000.0, 1.0 * u.eV / u.cm**3
    e0 = np.geomspace(1e-5, 60.0, 300) * K.C.k_B * T * u.erg
    n = K.planck_number_density(e0.value, T, uu.to_value(u.erg / u.cm**3))
    ts = TableSeed(e0, n / (u.cm**3 * u.erg))
    a = InverseCompton(NE, seeds=[ts], **RNG).sed(E_IC).value
    b = InverseCompton(NE, seeds=[Blackbody(T * u.K, uu)], **RNG).sed(E_IC).value
    sig = b > 1e-6 * b.max()
    assert np.max(np.abs(a[sig] / b[sig] - 1)) < 1e-3
    ts2 = TableSeed(e0, (n * e0.value**2) * u.erg / u.cm**3)
    np.testing.assert_allclose(ts2.table.dnde, ts.table.dnde, rtol=1e-12)


def test_anisotropic_blackbody_head_on_vs_isotropic():
    E = np.geomspace(1e6, 1e9, 7) * u.eV
    iso = InverseCompton(NE, seeds=[Blackbody(2.72548 * u.K)], **RNG).sed(E).value
    head = InverseCompton(NE, seeds=[Blackbody(2.72548 * u.K, theta=np.pi * u.rad)], **RNG).sed(E).value
    side = InverseCompton(NE, seeds=[Blackbody(2.72548 * u.K, theta=np.pi / 2 * u.rad)], **RNG).sed(E).value
    assert np.all(head > iso)
    # (1-cos)^((p+1)/2)
    np.testing.assert_allclose(head / side, 2 ** 1.65, rtol=5e-3)


def test_ssc_seed_density():
    syn = Synchrotron(NE, B=100 * u.uG, **RNG)
    R = 1 * u.pc
    seed = ssc_seed(syn, R)
    E = np.geomspace(1e-8, 1e8, 3201) * u.eV
    L = syn.sed(E, distance=0).to_value(u.erg / u.s)
    Ltot = np.trapezoid(L, np.log(E.value))
    expect = K.SSC_UNIFORM_SPHERE * Ltot / (4 * np.pi * R.to_value(u.cm) ** 2 * K.C.c_light)
    assert seed.energy_density.to_value(u.erg / u.cm**3) == pytest.approx(expect, rel=2e-3)
    ssc = SSC(NE, B=100 * u.uG, R=R, **RNG)
    s = ssc.sed(np.geomspace(1e9, 1e13, 5) * u.eV).value
    assert np.all(s > 0) and ssc.quadrature.converged


# constants set
def test_astropy_constants_subprocess():
    code = (
        "import numpy as np, astropy.units as u\n"
        "from viper import constants as C\n"
        "from viper.radiative import ExponentialCutoffPowerLaw, InverseCompton, Synchrotron\n"
        "from viper.radiative import kernels as K\n"
        "assert C.CONSTANTS_SOURCE == 'astropy' and K.T_CMB == C.T_CMB\n"
        "ne = ExponentialCutoffPowerLaw(1e36 / u.eV, 1 * u.TeV, 2.3, 50 * u.TeV)\n"
        "rng = dict(Eemin=1 * u.GeV, Eemax=1 * u.PeV)\n"
        "m = Synchrotron(ne, B=100 * u.uG, **rng) + InverseCompton(ne, seeds=['CMB', 'FIR'], **rng)\n"
        "s = m.sed(np.geomspace(1e-3, 1e13, 30) * u.eV).value\n"
        "assert np.all(np.isfinite(s)) and np.all(s > 0) and m.quadrature.converged\n"
        "print(C.T_CMB)\n")
    env = dict(os.environ, VIPER_CONSTANTS="astropy")
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                          text=True, check=True)
    assert float(proc.stdout.strip()) == pytest.approx(2.7255, abs=1e-3)
