"""Cross-check that the gammapy wrapper agrees with the standard SED path."""

import numpy as np
import pytest

pytest.importorskip("gammapy")

import astropy.units as u

from viper import constants as C
from viper.model import PWNModel
from viper.gammapy import PWNSpectralModel


def _build_pair():
    """PWNModel and PWNSpectralModel with identical physical parameters."""
    period = 0.31686
    p_dot = 3.614e-13
    esn_erg = 1.0e51
    mej, n_ism, distance_pc = 6.0, 0.4, 1400.0
    braking_index, tau_sd_yr, eta_b = 2.5, 3000.0, 0.01
    e_min, e_max, e_break, p1, p2 = 3.0, 5e5, 50.0, 0.8, 2.5

    m = PWNModel()
    m.period.value        = period * u.s
    m.p_dot.value         = p_dot
    m.esn.value           = esn_erg * u.erg
    m.mej.value           = mej * u.solMass
    m.n_ism.value         = n_ism * u.Unit("1 / cm3")
    m.distance.value      = distance_pc * u.pc
    m.braking_index.value = braking_index
    m.tau_sd.value        = tau_sd_yr * u.yr
    m.eta_b.value         = eta_b
    m.e_min.value         = e_min * u.GeV
    m.e_max.value         = e_max * u.GeV
    m.e_break.value       = e_break * u.GeV
    m.p1.value, m.p2.value = p1, p2

    m2 = PWNSpectralModel(
        period=period, p_dot=p_dot,
        esn=esn_erg * 1e-51,
        mej=mej * u.solMass,
        n_ism=n_ism * u.Unit("1 / cm3"),
        distance=distance_pc * u.pc,
        braking_index=braking_index,
        tau_sd=tau_sd_yr * u.yr,
        eta_b=eta_b,
        e_min=e_min * u.GeV,
        e_max=e_max * u.GeV,
        e_break=e_break * u.GeV,
        p1=p1, p2=p2,
    )
    return m, m2


def test_gammapy_wrapper_matches_standard_sed():
    """E^2 dN/dE from gammapy wrapper must match nuFnu within 1e-4."""
    m, m2 = _build_pair()
    res = m.run(dyn_step=-1)

    # reference conversion needs package constants
    d_cm = m.distance.value * C.pc
    freq = res.photon_freqs
    nufnu = 1e40 * res.photon_spectrum * freq / (4.0 * np.pi * d_cm**2)

    erg_per_tev = 1.0e12 / C.eV_per_erg
    e_tev = C.h_cgs * freq / erg_per_tev
    dnde_q = m2(e_tev * u.TeV)
    e2dnde_erg = (e_tev ** 2) * dnde_q.to_value("cm-2 s-1 TeV-1") * erg_per_tev

    floor = 1e-25
    mask = (nufnu > floor) & (e2dnde_erg > floor)
    assert mask.sum() > 100, (
        f"Too few comparable bins ({mask.sum()}); check for empty SED "
        "or zero-flux in one path."
    )
    ratio = e2dnde_erg[mask] / nufnu[mask]
    worst = float(np.max(np.abs(1.0 - ratio)))
    assert worst < 1.0e-4, (
        f"Gammapy wrapper SED disagrees with PWNModel SED by "
        f"{worst:.3e} relative.  Symptom of a unit bug in "
        "viper/gammapy.py."
    )


# -----------------------------------------------------------------------
# Unit equivalence at the gammapy API
# -----------------------------------------------------------------------

def _common_kwargs():
    """Shared physical parameters for unit-equivalence tests."""
    return dict(
        period=0.31686, p_dot=3.614e-13,
        esn=1.0,
        mej=6.0 * u.solMass,
        n_ism=0.4 * u.Unit("1 / cm3"),
        distance=1400.0 * u.pc,
        braking_index=2.5,
        tau_sd=3000.0 * u.yr,
        eta_b=0.01,
        e_min=3.0 * u.GeV,
        e_max=5e5 * u.GeV,
        e_break=50.0 * u.GeV,
        p1=0.8, p2=2.5,
    )


@pytest.mark.parametrize("name, base, alt", [
    ("e_min",    3.0    * u.GeV,    3000.0    * u.MeV),
    ("e_min",    3.0    * u.GeV,    0.003     * u.TeV),
    ("e_max",    5e5    * u.GeV,    500.0     * u.TeV),
    ("e_break",  50.0   * u.GeV,    50000.0   * u.MeV),
    ("distance", 1400.0 * u.pc,     1.4       * u.kpc),
    ("tau_sd",   3000.0 * u.yr,     1095750.0 * u.d),
])
def test_unit_equivalence_at_gammapy_api(name, base, alt):
    """A Quantity in any compatible unit must give the same SED."""
    kw_base = _common_kwargs()
    kw_alt  = dict(kw_base, **{name: alt})
    kw_base[name] = base

    m_base = PWNSpectralModel(**kw_base)
    m_alt  = PWNSpectralModel(**kw_alt)

    energy = np.logspace(-9, 2, 40) * u.TeV
    sed_base = m_base(energy).value
    sed_alt  = m_alt(energy).value

    mask = (sed_base > 0) & (sed_alt > 0)
    assert mask.sum() > 20, "Too few non-zero bins to compare."

    ratio = sed_alt[mask] / sed_base[mask]
    worst = float(np.max(np.abs(1.0 - ratio)))
    assert worst < 1.0e-10, (
        f"SED with {name}={alt} differs from SED with {name}={base} "
        f"by {worst:.3e} relative. _INTERNAL_UNITS likely missing "
        f"or wrong for {name!r}."
    )


# -----------------------------------------------------------------------
# Inverse-square distance scaling
# -----------------------------------------------------------------------

def test_distance_scales_dnde_as_inverse_square():
    """Doubling the source distance must drop dN/dE by exactly 4x."""
    common = _common_kwargs()
    common.pop("distance")
    m_near = PWNSpectralModel(distance=1000.0 * u.pc, **common)
    m_far  = PWNSpectralModel(distance=2000.0 * u.pc, **common)

    energy = np.logspace(-9, 2, 40) * u.TeV
    near = m_near(energy).value
    far  = m_far(energy).value

    mask = (near > 0) & (far > 0)
    assert mask.sum() > 20

    ratio = far[mask] / near[mask]
    worst = float(np.max(np.abs(ratio - 0.25)))
    assert worst < 1e-12, (
        f"Doubling distance gave dN/dE ratio differing from 0.25 by "
        f"{worst:.3e} (worst bin). Check _PC_CM, the 4 pi d^2 factor, "
        f"and the distance entry in _INTERNAL_UNITS."
    )


# -----------------------------------------------------------------------
# Presets (README advertises them)
# -----------------------------------------------------------------------

def test_for_cta1_preset_runs():
    """for_cta1() must build and evaluate to a finite, non-empty SED."""
    m = PWNSpectralModel.for_cta1()
    energy = np.logspace(-9, 2, 40) * u.TeV
    dnde = m(energy).value
    assert np.all(np.isfinite(dnde))
    assert (dnde > 0).sum() > 20


def test_for_g54_preset_ic_not_cmb_only():
    """for_g54()'s NIR field must be migrated (eV/cm^3) and contribute."""
    m = PWNSpectralModel.for_g54()
    assert m.ic_temp_1.quantity.to_value(u.K) == pytest.approx(
        2753.05883, rel=1e-6)
    assert m.ic_norm_1.quantity.to_value(u.eV / u.cm**3) == pytest.approx(
        23.6136836610, rel=1e-6)

    energy = np.logspace(-3, 1, 20) * u.TeV
    with_nir = m(energy).value

    cmb_only = PWNSpectralModel.for_g54()
    cmb_only.ic_norm_1.value = 0.0
    without = cmb_only(energy).value

    mask = (with_nir > 0) & (without > 0)
    assert mask.sum() > 5
    boost = float(np.max(with_nir[mask] / without[mask]))
    assert boost > 3.0, (
        f"for_g54 SED barely changes when the NIR field is zeroed "
        f"(max boost {boost:.2f}x); the preset's ic_norm_1 is not "
        f"contributing; check the eV/cm^3 migration."
    )


# -----------------------------------------------------------------------
# Serialization
# -----------------------------------------------------------------------

def test_registered_in_spectral_model_registry():
    """YAML tag resolution requires registry membership."""
    from gammapy.modeling.models import SPECTRAL_MODEL_REGISTRY
    assert SPECTRAL_MODEL_REGISTRY.get_cls("pwn") is PWNSpectralModel
    assert (SPECTRAL_MODEL_REGISTRY.get_cls("PWNSpectralModel")
            is PWNSpectralModel)


def test_serialization_roundtrip_preserves_physics():
    """to_dict/from_dict must carry period/p_dot/injection/ssc."""
    m = PWNSpectralModel.for_cta1()
    m2 = PWNSpectralModel.from_dict(m.to_dict())
    assert m2.period == pytest.approx(0.31686)
    assert m2.p_dot == pytest.approx(3.614e-13)
    assert m2.injection_model == 'broken_powerlaw'
    assert m2.ssc_model == 'off'
    for p in m.parameters:
        p2 = m2.parameters[p.name]
        assert p2.value == pytest.approx(p.value, rel=1e-12)
        assert p2.frozen == p.frozen


# -----------------------------------------------------------------------
# Parameter conditioning: interp="log" + frozen-by-default alignment
# -----------------------------------------------------------------------

def test_log_interp_on_decade_spanning_parameters():
    """The six log10-sampled PWNModel params must be interp='log' here too."""
    m = PWNSpectralModel()
    for name in ("n_ism", "tau_sd", "eta_b", "e_min", "e_max", "e_break"):
        assert m.parameters[name].interp == "log", name
    for name in ("esn", "mej", "braking_index", "p1", "p2", "distance",
                 "ic_temp_1", "ic_norm_1", "ic_temp_2", "ic_norm_2"):
        assert m.parameters[name].interp == "lin", name


def test_all_parameters_frozen_by_default():
    """Matches PWNModel: everything frozen, user unfreezes the fit subset."""
    m = PWNSpectralModel()
    assert all(p.frozen for p in m.parameters)
    assert len(m.parameters.free_parameters) == 0
