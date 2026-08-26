"""Tests for the generalized injection spectra (viper/injection.py)."""

import pickle

import numpy as np
import pytest
import astropy.units as u

from viper.model import PWNModel
from viper.evolution import evolve_pwn
from viper.grids import make_log_grid
from viper.injection import (
    INJECTION_MODELS,
    get_injection_shape,
    injection_bin_counts,
    register_injection_model,
)
from viper.particles import (
    inject_powerlaw,
    inject_broken_powerlaw,
    inject_particles,
    integrate_electron_energy,
)
from viper.fitting import build_fit_spec, _theta_to_evolve_args


def _cta1_like_model():
    """A small CTA-1-ish model (matches tests/test_escape.py)."""
    m = PWNModel()
    m.period.value        = 0.31686 * u.s
    m.p_dot.value         = 3.614e-13
    m.esn.value           = 1.0e51 * u.erg
    m.mej.value           = 6.0 * u.solMass
    m.n_ism.value         = 0.4 * u.Unit("1 / cm3")
    m.distance.value      = 1400.0 * u.pc
    m.braking_index.value = 2.5
    m.tau_sd.value        = 3000.0 * u.yr
    m.eta_b.value         = 0.01
    m.e_min.value         = 3.0 * u.GeV
    m.e_max.value         = 5e5 * u.GeV
    m.e_break.value       = 50.0 * u.GeV
    m.p1.value            = 0.8
    m.p2.value            = 2.5
    return m


_TINY_EVOLVE = dict(
    esn_51=1.0, mej=8.0, n_ism=0.1, braking_index=3.0, tau_sd=700.0,
    age=2000.0, e0_37=3.0, vel_psr=0.0, eta_g=0.0, eta_b=0.001,
    e_min=1.0, e_max=1e6, e_break=100.0, p1=1.5, p2=2.5,
    n_elec_bins=200, n_phot_bins=200, n_ic_target=60,
)


# ===================================================================
# Registry-level unit tests (fast; no evolution)
# ===================================================================

def test_registry_builtins_present():
    for name in ('broken_powerlaw', 'log_parabola', 'powerlaw_cutoff'):
        assert name in INJECTION_MODELS
        assert callable(get_injection_shape(name))


def test_registry_unknown_raises():
    with pytest.raises(ValueError, match="unknown injection_model"):
        get_injection_shape('no_such_shape')


def test_register_custom_and_reregister():
    def shape(e_gev, e_break, p1, p2):
        return np.exp(-0.5 * (np.log10(e_gev / e_break) / 0.5) ** 2)

    register_injection_model('_test_gauss', shape)
    assert get_injection_shape('_test_gauss') is shape
    register_injection_model('_test_gauss', shape)


def test_register_cannot_override_builtin():
    with pytest.raises(ValueError, match="built-in"):
        register_injection_model('log_parabola', lambda e, eb, p1, p2: e)


def test_register_rejects_non_callable():
    with pytest.raises(TypeError, match="callable"):
        register_injection_model('_test_bogus', 42)


# ===================================================================
# Generic quadrature injector vs the analytic legacy injectors
# ===================================================================

def test_quadrature_matches_analytic_broken_powerlaw():
    """Pins quadrature accuracy and the kink split at e_break."""
    g_ref = make_log_grid(1e-3, 1e8, 300)
    g_gen = make_log_grid(1e-3, 1e8, 300)
    inject_broken_powerlaw(g_ref, 1.0, 3.0, 5e5, 50.0, 1.6, 2.5)
    g_gen.number += injection_bin_counts(
        'broken_powerlaw', g_gen, 3.0, 5e5, 50.0, 1.6, 2.5)
    mask = g_ref.number > 0
    assert np.any(mask)
    np.testing.assert_allclose(
        g_gen.number[mask], g_ref.number[mask], rtol=1e-10)
    np.testing.assert_allclose(g_gen.number[~mask], 0.0, atol=0.0)


def test_log_parabola_beta_zero_is_powerlaw():
    g_ref = make_log_grid(1e-3, 1e8, 300)
    g_gen = make_log_grid(1e-3, 1e8, 300)
    inject_powerlaw(g_ref, 1.0, 3.0, 5e5, 2.2)
    g_gen.number += injection_bin_counts(
        'log_parabola', g_gen, 3.0, 5e5, 123.0, 2.2, 0.0)
    mask = g_ref.number > 0
    np.testing.assert_allclose(
        g_gen.number[mask], g_ref.number[mask], rtol=1e-10)


def test_log_parabola_curves_downward():
    g = make_log_grid(1e-3, 1e8, 500)
    c0 = injection_bin_counts('log_parabola', g, 3.0, 5e5, 1e3, 2.0, 0.0)
    c1 = injection_bin_counts('log_parabola', g, 3.0, 5e5, 1e3, 2.0, 0.3)
    i_piv = np.argmin(np.abs(g.bin_avg - 1e3))
    r0 = c0 / c0[i_piv]
    r1 = c1 / c1[i_piv]
    far = (g.bin_avg > 1e5) & (c0 > 0)   # two decades above the pivot
    assert np.all(r1[far] < r0[far])


def test_energy_normalization_all_forms():
    """Injected energy equals energy_40 up to bin discretization."""
    for name in ('broken_powerlaw', 'log_parabola', 'powerlaw_cutoff'):
        g = make_log_grid(1e-3, 1e8, 500)
        g.number += 2.5 * injection_bin_counts(
            name, g, 3.0, 5e5, 1e3, 1.8, 0.3)
        g.update_log_number()
        assert integrate_electron_energy(g) == pytest.approx(
            2.5, rel=1e-3), name


def test_counts_scale_linearly_with_energy():
    g = make_log_grid(1e-3, 1e8, 300)
    c = injection_bin_counts('log_parabola', g, 3.0, 5e5, 1e3, 2.0, 0.2)
    g1 = make_log_grid(1e-3, 1e8, 300)
    g2 = make_log_grid(1e-3, 1e8, 300)
    inject_particles(g1, 1.0, 3.0, 5e5, 1e3, 2.0, 0.2, 0.0, 0.0,
                     injection_model='log_parabola')
    inject_particles(g2, 7.0, 3.0, 5e5, 1e3, 2.0, 0.2, 0.0, 0.0,
                     injection_model='log_parabola')
    np.testing.assert_allclose(g2.number, 7.0 * g1.number, rtol=1e-12)
    np.testing.assert_allclose(g1.number, c, rtol=1e-12)


def test_support_truncation():
    """No counts outside [e_min, e_max]; partial edge bins allowed."""
    g = make_log_grid(1e-3, 1e8, 300)
    c = injection_bin_counts('powerlaw_cutoff', g, 10.0, 1e4, 1e3, 2.0, 0.0)
    outside = (g.bin_max <= 10.0) | (g.bin_min >= 1e4)
    assert np.all(c[outside] == 0.0)
    assert np.any(c > 0)


def test_bad_shape_values_raise():
    register_injection_model(
        '_test_negative', lambda e, eb, p1, p2: -np.ones_like(e))
    with pytest.raises(ValueError, match="negative or non-finite"):
        injection_bin_counts('_test_negative', make_log_grid(1e-3, 1e8, 100),
                             3.0, 5e5, 1e3, 2.0, 0.0)
    register_injection_model(
        '_test_zero', lambda e, eb, p1, p2: np.zeros_like(e))
    with pytest.raises(ValueError, match="zero or non-finite"):
        injection_bin_counts('_test_zero', make_log_grid(1e-3, 1e8, 100),
                             3.0, 5e5, 1e3, 2.0, 0.0)


def test_bad_support_raises():
    g = make_log_grid(1e-3, 1e8, 100)
    with pytest.raises(ValueError, match="e_min"):
        injection_bin_counts('log_parabola', g, 0.0, 5e5, 1e3, 2.0, 0.0)
    with pytest.raises(ValueError, match="e_max"):
        injection_bin_counts('log_parabola', g, 10.0, 10.0, 1e3, 2.0, 0.0)


def test_log_parabola_requires_positive_pivot():
    g = make_log_grid(1e-3, 1e8, 100)
    with pytest.raises(ValueError, match="pivot"):
        injection_bin_counts('log_parabola', g, 3.0, 5e5, 0.0, 2.0, 0.2)


def test_maxwellian_mix_with_generic_form():
    """f_max splits injected energy between Maxwellian and continuous form."""
    from viper.particles import inject_maxwellian

    g_mix = make_log_grid(1e-4, 1e4, 400)
    inject_particles(g_mix, 1.0, 1.0, 1e3, 10.0, 2.0, 0.2, 0.4, 0.005,
                     injection_model='log_parabola')

    g_sum = make_log_grid(1e-4, 1e4, 400)
    inject_particles(g_sum, 0.6, 1.0, 1e3, 10.0, 2.0, 0.2, 0.0, 0.0,
                     injection_model='log_parabola')
    inject_maxwellian(g_sum, 0.4, 0.005)

    np.testing.assert_allclose(g_mix.number, g_sum.number,
                               rtol=1e-12, atol=0.0)
    g_cont = make_log_grid(1e-4, 1e4, 400)
    inject_particles(g_cont, 0.6, 1.0, 1e3, 10.0, 2.0, 0.2, 0.0, 0.0,
                     injection_model='log_parabola')
    g_cont.update_log_number()
    assert integrate_electron_energy(g_cont) == pytest.approx(0.6, rel=1e-3)


# ===================================================================
# End-to-end injection_model selection
# ===================================================================

def test_default_dispatch_bit_identical():
    r1 = evolve_pwn(**_TINY_EVOLVE)
    r2 = evolve_pwn(**_TINY_EVOLVE, injection_model='broken_powerlaw')
    assert np.array_equal(r1.photon_spectrum, r2.photon_spectrum)
    assert np.array_equal(r1.electron_spectrum, r2.electron_spectrum)
    assert len(r1.dyninfo) == len(r2.dyninfo)


def test_forms_are_physically_distinct():
    specs = {}
    for name in ('broken_powerlaw', 'log_parabola', 'powerlaw_cutoff'):
        specs[name] = evolve_pwn(
            **_TINY_EVOLVE, injection_model=name).photon_spectrum
    # atol=0: default atol would equate spectra
    assert not np.allclose(
        specs['broken_powerlaw'], specs['log_parabola'],
        rtol=1e-3, atol=0.0)
    assert not np.allclose(
        specs['broken_powerlaw'], specs['powerlaw_cutoff'],
        rtol=1e-3, atol=0.0)


def test_custom_form_end_to_end():
    def _lognormal(e_gev, e_break, p1, p2):
        return np.exp(-0.5 * (np.log10(e_gev / e_break) / 1.0) ** 2)

    register_injection_model('_test_lognormal_e2e', _lognormal)
    r = evolve_pwn(**_TINY_EVOLVE, injection_model='_test_lognormal_e2e')
    assert np.all(np.isfinite(r.photon_spectrum))
    assert r.photon_spectrum.max() > 0


def test_evolve_unknown_name_fails_fast():
    with pytest.raises(ValueError, match="unknown injection_model"):
        evolve_pwn(**_TINY_EVOLVE, injection_model='typo')


def test_cpp_compat_forces_broken_powerlaw():
    r_compat = evolve_pwn(**_TINY_EVOLVE, cpp_compat=True,
                          injection_model='log_parabola')
    r_ref = evolve_pwn(**_TINY_EVOLVE, cpp_compat=True)
    assert np.array_equal(r_compat.photon_spectrum, r_ref.photon_spectrum)


# ===================================================================
# Model / run_config / MCMC-spec consistency
# ===================================================================

def test_model_attribute_and_run_roundtrip():
    m = _cta1_like_model()
    assert m.injection_model == 'broken_powerlaw'
    m.injection_model = 'log_parabola'
    m.validate()  # registered name passes validation
    m.injection_model = 'nope'
    with pytest.raises(ValueError, match="injection_model"):
        m.validate()


def test_apply_run_config_key():
    m = _cta1_like_model()
    m.apply_run_config({'injection_model': 'powerlaw_cutoff'})
    assert m.injection_model == 'powerlaw_cutoff'
    m2 = _cta1_like_model()
    m2.apply_run_config({})
    assert m2.injection_model == 'broken_powerlaw'
    with pytest.raises(ValueError, match="injection_model"):
        _cta1_like_model().apply_run_config({'injection_model': 'nope'})


def test_fit_spec_carries_injection_model_and_pickles():
    m = _cta1_like_model()
    m.injection_model = 'log_parabola'
    m.tau_sd.frozen = False
    m.tau_sd.bounds = (100.0 * u.yr, 1e5 * u.yr)
    spec = build_fit_spec(m, observables=[])
    assert spec.injection_model == 'log_parabola'
    # only the string is pickled
    spec2 = pickle.loads(pickle.dumps(spec))
    assert spec2.injection_model == 'log_parabola'
    theta = np.array([np.log10(3000.0)])
    kw = _theta_to_evolve_args(theta, spec2)
    assert kw['injection_model'] == 'log_parabola'


def test_fit_spec_default_is_legacy():
    m = _cta1_like_model()
    m.tau_sd.frozen = False
    m.tau_sd.bounds = (100.0 * u.yr, 1e5 * u.yr)
    spec = build_fit_spec(m, observables=[])
    assert spec.injection_model == 'broken_powerlaw'
    kw = _theta_to_evolve_args(np.array([np.log10(3000.0)]), spec)
    assert kw['injection_model'] == 'broken_powerlaw'
