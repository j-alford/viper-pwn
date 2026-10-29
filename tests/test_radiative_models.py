"""Model tests."""


import numpy as np
import pytest
import astropy.units as u

from viper.radiative import (Blackbody, ExponentialCutoffPowerLaw,
                             InverseCompton, PowerLaw, SSC, Synchrotron,
                             TableDistribution, TableSeed, ssc_seed,
                             trapz_loglog)
from viper.radiative import kernels as K
from viper.radiative.quadrature import integrate_population

NE = ExponentialCutoffPowerLaw(1e36 / u.eV, 1 * u.TeV, 2.3, 50 * u.TeV)
RNG = dict(Eemin=1 * u.GeV, Eemax=1 * u.PeV)
E_SYN = np.geomspace(1e-7, 1e7, 100) * u.eV
E_IC = np.geomspace(1e6, 1e15, 100) * u.eV


# quadrature
@pytest.mark.parametrize("p", [-2.3, -1.0, 0.0, 1.7])
def test_trapz_loglog_exact_for_power_laws(p):
    x = np.geomspace(1.0, 1e4, 7)
    y = x**p
    exact = np.log(1e4) if p == -1 else (1e4 ** (p + 1) - 1) / (p + 1)
    assert trapz_loglog(y, x) == pytest.approx(exact, rel=1e-12)


def test_integrate_population_power_law_exact():
    """Power law integrated exactly."""
    res, info = integrate_population(
        lambda E, e: np.ones((E.size, e.size)), lambda E: E**-2.0,
        1.0, 1e6, np.array([1.0, 2.0]), rtol=1e-12)
    assert res == pytest.approx(1 - 1e-6, rel=1e-13)
    assert info.converged


@pytest.mark.parametrize("model", ["syn_iso", "syn_turb", "ic"])
def test_achieved_error_meets_rtol(model):
    """Achieved error within rtol."""
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
    # frequency input
    nu = E.to(u.Hz, equivalencies=u.spectral())
    np.testing.assert_allclose(syn.flux(nu).value, f1.value, rtol=1e-12)


def test_sum_and_seed_selection():
    ic = InverseCompton(NE, seeds=["CMB", Blackbody(30 * u.K, 0.5 * u.eV / u.cm**3, name="dust")], **RNG)
    E = np.geomspace(1e9, 1e14, 11) * u.eV
    tot = ic.sed(E).value
    parts = ic.sed(E, seed="CMB").value + ic.sed(E, seed="dust").value
    np.testing.assert_allclose(tot, parts, rtol=2e-4)
    syn = Synchrotron(NE, B=10 * u.uG, **RNG)
    both = (syn + ic).sed(E).value
    np.testing.assert_allclose(both, syn.sed(E).value + tot, rtol=1e-12)
    with pytest.raises(ValueError):
        ic.flux(E, seed="nope")


def test_table_distribution_matches_analytic():
    Ee = np.geomspace(1 * u.GeV, 1 * u.PeV, 600)
    td = TableDistribution(Ee, NE(Ee))
    E = np.geomspace(1e-5, 1e5, 21) * u.eV
    a = Synchrotron(td, B=100 * u.uG).sed(E).value
    b = Synchrotron(NE, B=100 * u.uG, **RNG).sed(E).value
    sig = b > 1e-6 * b.max()
    assert np.max(np.abs(a[sig] / b[sig] - 1)) < 2e-4


def test_blackbody_presets_and_energy_density():
    assert Blackbody("CMB").T.value == pytest.approx(2.72548)
    nir = Blackbody("NIR")
    assert nir.energy_density.to_value(u.eV / u.cm**3) == pytest.approx(1.0)
    cmb_u = Blackbody("CMB").energy_density.to_value(u.eV / u.cm**3)
    assert cmb_u == pytest.approx(0.26, rel=0.02)


def test_bad_inputs():
    with pytest.raises(ValueError):
        Synchrotron(NE, kernel="legacy")
    with pytest.raises(u.UnitsError):
        Synchrotron(lambda E: E.to(u.erg))
    with pytest.raises(ValueError):
        Blackbody("XYZ")


# energy conservation
@pytest.mark.parametrize("kernel", ["isotropic", "fixed", "turbulent"])
def test_synchrotron_population_energy_conservation(kernel):
    """Population energy conservation."""
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
    # E^2 dn/dE input
    ts2 = TableSeed(e0, (n * e0.value**2) * u.erg / u.cm**3)
    np.testing.assert_allclose(ts2.table.dnde, ts.table.dnde, rtol=1e-12)


def test_anisotropic_blackbody_head_on_vs_isotropic():
    """Anisotropic Thomson scaling."""
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
    expect = 2.24 * Ltot / (4 * np.pi * R.to_value(u.cm) ** 2 * K.C.c_light)
    assert seed.energy_density.to_value(u.erg / u.cm**3) == pytest.approx(expect, rel=2e-3)
    ssc = SSC(NE, B=100 * u.uG, R=R, **RNG)
    s = ssc.sed(np.geomspace(1e9, 1e13, 5) * u.eV).value
    assert np.all(s > 0) and ssc.quadrature.converged
