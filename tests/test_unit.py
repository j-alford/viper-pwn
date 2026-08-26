"""Fast unit tests: conversions, parameters, observables, fitting infrastructure."""

import subprocess
import sys

import numpy as np
import pytest

from viper import constants as C
from viper.scalar_params import PWNScalarParams
from viper.observables import (
    Observable, ObsType, _to_hz, _ENERGY_TO_HZ, _ANGLE_FROM_RAD,
    _extract_angular_size, _extract_spectral_index,
    _model_flux_density_cgs, _extract_flux_density,
    extract_observable, extract_all,
)
from viper.fitting import (
    build_fit_spec, log_prior, log_probability,
    _init_walkers, _theta_to_evolve_args,
)
from viper.model import PWNModel
from viper.parameter import Parameter, ParameterSet
import astropy.units as u


# ===================================================================
# Helpers: reusable fixtures
# ===================================================================

_BASE_PARAMS = dict(
    esn=1.0, mej=6.0, n_ism=0.4,
    braking_index=2.5, tau_sd=3000.0,
    period=0.31686, p_dot=3.614e-13,
    eta_b=0.01, e_min=3.0, e_max=500000.0,
    e_break=50.0, p1=0.8, p2=2.5,
    distance=1400.0,
)


def _make_params(**overrides):
    """Create a PWNScalarParams with sensible defaults, overriding as needed."""
    kw = dict(_BASE_PARAMS)
    kw.update(overrides)
    return PWNScalarParams(**kw)


def _fake_spectrum():
    """Power-law photon spectrum (photon index 1.5) for extraction tests."""
    freqs = np.logspace(8, 28, 500)  # Hz
    lum_40 = 1e-10 * (freqs / 1e15) ** (-0.5)  # 10^40 erg/s/Hz
    return freqs, lum_40


# ===================================================================
# 1. Constants checks
# ===================================================================

class TestConstants:
    """Verify key physical constants haven't been accidentally changed."""

    def test_parsec(self):
        # gsz2009 rounded; astropy precise
        expected = 3.09e18 if C.CONSTANTS_SOURCE == 'gsz2009' else 3.0857e18
        assert C.pc == pytest.approx(expected, rel=1e-3)

    def test_solar_mass(self):
        assert C.M_sun == pytest.approx(1.99e33, rel=1e-3)

    def test_speed_of_light(self):
        assert C.c_light == pytest.approx(3.0e10, rel=1e-3)

    def test_planck_constant_cgs(self):
        assert C.h_cgs == pytest.approx(6.626e-27, rel=1e-3)

    def test_planck_constant_eV(self):
        assert C.h_eV == pytest.approx(4.136e-15, rel=1e-3)

    def test_thomson_cross_section(self):
        assert C.sigma_T == pytest.approx(6.652e-25, rel=1e-3)

    def test_ns_moment_of_inertia(self):
        assert C.I_ns == 1.0e45


# ===================================================================
# 2. Unit conversions in observables.py
# ===================================================================

class TestUnitConversions:
    """Verify energy/frequency → Hz conversions."""

    def test_hz_identity(self):
        assert _to_hz(1.4e9, 'Hz') == 1.4e9

    def test_ghz(self):
        assert _to_hz(1.4, 'GHz') == pytest.approx(1.4e9)

    def test_mhz(self):
        assert _to_hz(1400, 'MHz') == pytest.approx(1.4e9)

    def test_eV_to_hz(self):
        expected = 1.0 / C.h_eV  # 1 eV in Hz
        assert _to_hz(1.0, 'eV') == pytest.approx(expected, rel=1e-6)

    def test_keV_to_hz(self):
        expected = 1.0e3 / C.h_eV
        assert _to_hz(1.0, 'keV') == pytest.approx(expected, rel=1e-6)

    def test_GeV_to_hz(self):
        expected = 1.0e9 / C.h_eV
        assert _to_hz(1.0, 'GeV') == pytest.approx(expected, rel=1e-6)

    def test_TeV_to_hz(self):
        expected = 1.0e12 / C.h_eV
        assert _to_hz(1.0, 'TeV') == pytest.approx(expected, rel=1e-6)

    def test_case_insensitive_energy(self):
        assert _to_hz(1.0, 'gev') == _to_hz(1.0, 'GeV')
        assert _to_hz(1.0, 'tev') == _to_hz(1.0, 'TeV')

    def test_consistency_keV_eV(self):
        assert _to_hz(1.0, 'keV') == pytest.approx(1000 * _to_hz(1.0, 'eV'), rel=1e-10)

    def test_consistency_GeV_MeV(self):
        assert _to_hz(1.0, 'GeV') == pytest.approx(1000 * _to_hz(1.0, 'MeV'), rel=1e-10)

    def test_unknown_unit_raises(self):
        with pytest.raises(ValueError, match="Unknown frequency"):
            _to_hz(1.0, 'parsec')


# ===================================================================
# 3. Observable constructors
# ===================================================================

class TestObservableConstructors:

    def test_angular_size_snr(self):
        obs = Observable.angular_size_snr(45.0, 10.0, 'arcmin')
        assert obs.obs_type == ObsType.ANGULAR_SIZE_SNR
        assert obs.value == 45.0
        assert obs.error == 10.0
        assert obs._angle_unit == 'arcmin'

    def test_angular_size_pwn(self):
        obs = Observable.angular_size_pwn(12.0, 6.0, 'arcmin')
        assert obs.obs_type == ObsType.ANGULAR_SIZE_PWN
        assert obs.value == 12.0
        assert obs.error == 6.0

    def test_flux_density_radio(self):
        obs = Observable.flux_density(1.4, 'GHz', 0.080, 0.010, 'Jy')
        assert obs.obs_type == ObsType.FLUX_DENSITY
        assert obs._freq_hz == pytest.approx(1.4e9)
        assert obs._flux_unit == 'Jy'

    def test_flux_density_gamma(self):
        obs = Observable.flux_density(
            1.0, 'TeV', 3.5e-13, 1.0e-13, 'photons/cm2/s/TeV'
        )
        assert obs._freq_hz == pytest.approx(1e12 / C.h_eV, rel=1e-6)
        assert obs._flux_unit == 'photons/cm2/s/TeV'

    def test_flux_density_upper_limit(self):
        obs = Observable.flux_density(1.4, 'GHz', 0.1, 0.0, 'Jy', upper_limit=True)
        assert obs.upper_limit is True
        assert '[UL]' in obs.label

    def test_spectral_index(self):
        obs = Observable.spectral_index(0.5, 7.0, 'keV', 1.85, 0.11)
        assert obs.obs_type == ObsType.SPECTRAL_INDEX
        assert obs.value == 1.85
        assert obs.error == 0.11
        assert obs._freq_lo_hz < obs._freq_hi_hz

    def test_band_flux(self):
        obs = Observable.band_flux(2.0, 10.0, 'keV', 5.43e-12, 0.035e-12)
        assert obs.obs_type == ObsType.BAND_FLUX
        assert obs._freq_lo_hz < obs._freq_hi_hz

    def test_weight_default(self):
        obs = Observable.angular_size_snr(45.0, 10.0, 'arcmin')
        assert obs.weight == 1.0

    def test_bad_angle_unit_raises(self):
        with pytest.raises(ValueError, match="Unknown angle unit"):
            Observable.angular_size_snr(45.0, 10.0, 'furlongs')


# ===================================================================
# 4. Observable extraction (unit conversion math)
# ===================================================================

class TestObservableExtraction:

    def test_angular_size_small_angle(self):
        """1 pc at 1000 pc → ~3.44 arcmin."""
        theta = _extract_angular_size(1.0, 1000.0, 'arcmin')
        expected = (1.0 / 1000.0) * (180.0 / np.pi * 60.0)
        assert theta == pytest.approx(expected, rel=1e-6)

    def test_angular_size_deg(self):
        theta = _extract_angular_size(1.0, 1000.0, 'deg')
        expected = (1.0 / 1000.0) * (180.0 / np.pi)
        assert theta == pytest.approx(expected, rel=1e-6)

    def test_flux_density_interpolation(self):
        """Verify log-log interpolation of a power-law spectrum."""
        freqs, lum_40 = _fake_spectrum()
        d_pc = 1000.0

        nu_test = 1e15  # Hz
        f_nu = _model_flux_density_cgs(nu_test, freqs, lum_40, d_pc)

        d_cm = d_pc * C.pc
        expected = 1e-10 * 1e40 / (4.0 * np.pi * d_cm**2)
        assert f_nu == pytest.approx(expected, rel=1e-2)

    def test_flux_density_outside_range_returns_zero(self):
        freqs, lum_40 = _fake_spectrum()
        assert _model_flux_density_cgs(1.0, freqs, lum_40, 1000.0) == 0.0
        assert _model_flux_density_cgs(1e30, freqs, lum_40, 1000.0) == 0.0

    def test_flux_jy_conversion(self):
        """Verify Jy conversion factor: 1 Jy = 1e-23 erg/s/cm²/Hz."""
        freqs, lum_40 = _fake_spectrum()
        d_pc = 1000.0
        nu = 1e15

        f_cgs = _model_flux_density_cgs(nu, freqs, lum_40, d_pc)
        f_jy = _extract_flux_density(nu, freqs, lum_40, d_pc, 'Jy')
        assert f_jy == pytest.approx(f_cgs * 1e23, rel=1e-6)

    def test_flux_mjy_conversion(self):
        freqs, lum_40 = _fake_spectrum()
        f_jy = _extract_flux_density(1e15, freqs, lum_40, 1000.0, 'Jy')
        f_mjy = _extract_flux_density(1e15, freqs, lum_40, 1000.0, 'mJy')
        assert f_mjy == pytest.approx(f_jy * 1e3, rel=1e-6)

    def test_spectral_index_power_law(self):
        """A pure ν^{-0.5} spectrum should give Γ = 1.5."""
        freqs, lum_40 = _fake_spectrum()
        nu_lo = _to_hz(0.5, 'keV')
        nu_hi = _to_hz(7.0, 'keV')
        gamma = _extract_spectral_index(nu_lo, nu_hi, freqs, lum_40)
        assert gamma == pytest.approx(1.5, abs=0.01)

    def test_spectral_index_different_band(self):
        """Same power law measured in a different band → same Γ."""
        freqs, lum_40 = _fake_spectrum()
        gamma_xray = _extract_spectral_index(
            _to_hz(2.0, 'keV'), _to_hz(10.0, 'keV'), freqs, lum_40
        )
        gamma_gamma = _extract_spectral_index(
            _to_hz(1.0, 'GeV'), _to_hz(100.0, 'GeV'), freqs, lum_40
        )
        assert gamma_xray == pytest.approx(gamma_gamma, abs=0.02)

    def test_extract_observable_dispatches_angular_size(self):
        freqs, lum_40 = _fake_spectrum()
        obs = Observable.angular_size_snr(45.0, 10.0, 'arcmin')
        pred = extract_observable(obs, freqs, lum_40, 5.0, 15.0, 1400.0)
        expected = _extract_angular_size(15.0, 1400.0, 'arcmin')
        assert pred == pytest.approx(expected)

    def test_extract_all_length(self):
        freqs, lum_40 = _fake_spectrum()
        obs_list = [
            Observable.angular_size_snr(45.0, 10.0, 'arcmin'),
            Observable.angular_size_pwn(12.0, 6.0, 'arcmin'),
            Observable.flux_density(1.4, 'GHz', 0.08, 0.01, 'Jy'),
        ]
        result = extract_all(obs_list, freqs, lum_40, 5.0, 15.0, 1400.0)
        assert len(result) == 3

    def test_photon_flux_density_tev(self):
        """photons/cm2/s/TeV conversion: dN/dE = F_nu / (h^2 nu)."""
        freqs, lum_40 = _fake_spectrum()
        d_pc = 1000.0
        E_TeV = 1.0
        nu = _to_hz(E_TeV, 'TeV')

        dnde_tev = _extract_flux_density(nu, freqs, lum_40, d_pc, 'photons/cm2/s/TeV')

        f_nu = _model_flux_density_cgs(nu, freqs, lum_40, d_pc)
        dnde_per_erg = f_nu / (C.h_cgs * C.h_cgs * nu)
        # erg/TeV from active constants, not hardcoded
        erg_per_TeV = 1.0e12 / C.eV_per_erg
        dnde_expected = dnde_per_erg * erg_per_TeV

        assert dnde_tev == pytest.approx(dnde_expected, rel=1e-4)

    def test_unknown_flux_unit_raises(self):
        freqs, lum_40 = _fake_spectrum()
        with pytest.raises(ValueError, match="Unknown flux unit"):
            _extract_flux_density(1e15, freqs, lum_40, 1000.0, 'bananas/cm2/s')


# ===================================================================
# 5. PWNScalarParams
# ===================================================================

class TestPWNScalarParams:

    def test_valid_construction(self):
        p = _make_params()
        assert p.esn == 1.0
        assert p.mej == 6.0

    def test_true_age_positive(self):
        p = _make_params()
        assert p.true_age > 0

    def test_true_age_formula(self):
        """Check true_age = 2τ_char/(n-1) - τ_sd."""
        p = _make_params(braking_index=2.5, tau_sd=3000.0,
                         period=0.31686, p_dot=3.614e-13)
        tau_char_yr = p.period / (2 * p.p_dot) / (365.0 * 86400.0)
        expected = 2 * tau_char_yr / (p.braking_index - 1) - p.tau_sd
        assert p.true_age == pytest.approx(expected, rel=1e-10)

    def test_e0_37_positive(self):
        p = _make_params()
        assert p.e0_37 > 0

    def test_validation_negative_esn(self):
        with pytest.raises(ValueError, match="esn"):
            _make_params(esn=-1.0)

    def test_validation_negative_mej(self):
        with pytest.raises(ValueError, match="mej"):
            _make_params(mej=-1.0)

    def test_validation_eta_b_range(self):
        with pytest.raises(ValueError, match="eta_b"):
            _make_params(eta_b=1.5)

    def test_validation_e_break_above_e_max(self):
        with pytest.raises(ValueError, match="e_break"):
            _make_params(e_break=600000.0, e_max=500000.0)

    def test_validation_ic_length_mismatch(self):
        with pytest.raises(ValueError, match="ic_temps"):
            _make_params(n_ic=1, ic_temps=[], ic_norms=[1e-10])

    def test_from_theta(self):
        """Test the legacy 12-element theta vector constructor."""
        theta = [1.0, 6.0, 0.4, 1400.0, 2.5, 3000.0,
                 0.01, 3.0, 500000.0, 50.0, 0.8, 2.5]
        p = PWNScalarParams.from_theta(theta, period=0.31686, p_dot=3.614e-13)
        assert p.esn == 1.0
        assert p.distance == 1400.0
        assert p.p2 == 2.5

    def test_from_theta_distance_override(self):
        theta = [1.0, 6.0, 0.4, 9999.0, 2.5, 3000.0,
                 0.01, 3.0, 500000.0, 50.0, 0.8, 2.5]
        p = PWNScalarParams.from_theta(theta, period=0.31686, p_dot=3.614e-13,
                                      distance=1400.0)
        assert p.distance == 1400.0  # override, not 9999


# ===================================================================
# 6. Fitting infrastructure
# ===================================================================

def _make_simple_model():
    """Minimal PWNModel for testing fitting interface (no model eval)."""
    model = PWNModel()
    # bypass Quantity for speed
    model._parameters['period']._set_internal(0.31686)
    model._parameters['p_dot']._set_internal(3.614e-13)

    for name, log, lo, hi in [
        ('esn',     True,  0.1,   3.16),      # 10^-1 to 10^0.5
        ('mej',     True,  1.0,   100.0),
        ('n_ism',   True,  0.01,  10.0),
        ('braking_index', False, 1.5, 3.5),
        ('tau_sd',  True,  316.,  31623.),     # 10^2.5 to 10^4.5
        ('eta_b',   True,  1e-4,  0.5),
        ('e_min',   True,  0.1,   100.0),
        ('e_max',   True,  1e4,   1e7),
        ('e_break', True,  1.0,   1e4),
        ('p1',      False, 0.0,   2.0),
        ('p2',      False, 2.0,   3.5),
    ]:
        p = model._parameters[name]
        p.frozen = False
        p._bounds = (lo, hi)

    model._parameters['distance'].frozen = False
    model._parameters['distance']._bounds = (800, 2000)
    model._parameters['distance'].prior = (1400, 400)

    return model


def _make_simple_spec():
    """Build a _FitSpec from the simple model."""
    model = _make_simple_model()
    obs = [
        Observable.angular_size_snr(45.0, 10.0, 'arcmin'),
        Observable.angular_size_pwn(12.0, 6.0, 'arcmin'),
    ]
    return build_fit_spec(model, obs), model


class TestFitSpec:

    def test_ndim(self):
        spec, _ = _make_simple_spec()
        assert spec.ndim == 12

    def test_param_names(self):
        spec, _ = _make_simple_spec()
        assert 'esn' in spec.param_names
        assert 'p2' in spec.param_names

    def test_param_labels_log(self):
        spec, _ = _make_simple_spec()
        labels = spec.param_labels
        esn_idx = spec.param_names.index('esn')
        bi_idx = spec.param_names.index('braking_index')
        assert labels[esn_idx] == 'log(esn)'
        assert labels[bi_idx] == 'braking_index'

    def test_ic_field_in_spec(self):
        model = _make_simple_model()
        model.add_ic_field(temp=1000, norm=1e-10, frozen=False)
        model._parameters['ic_temp_1']._bounds = (10, 30000)
        model._parameters['ic_norm_1']._bounds = (1e-14, 1e-8)
        obs = [Observable.angular_size_snr(45.0, 10.0, 'arcmin')]
        spec = build_fit_spec(model, obs)
        assert spec.n_ic == 1
        assert 'ic_temp_1' in spec.param_names
        assert 'ic_norm_1' in spec.param_names


class TestThetaConversion:

    def test_roundtrip_linear(self):
        spec, model = _make_simple_spec()
        theta = model.parameters.free_fit_values
        kw = _theta_to_evolve_args(theta, spec)
        assert kw['esn_51'] == pytest.approx(model.esn.value, rel=1e-6)
        assert kw['p1'] == pytest.approx(model.p1.value, rel=1e-6)
        assert kw['p2'] == pytest.approx(model.p2.value, rel=1e-6)

    def test_roundtrip_log(self):
        spec, model = _make_simple_spec()
        theta = model.parameters.free_fit_values
        kw = _theta_to_evolve_args(theta, spec)
        assert kw['mej'] == pytest.approx(model.mej.value, rel=1e-6)
        assert kw['n_ism'] == pytest.approx(model.n_ism.value, rel=1e-6)
        assert kw['eta_b'] == pytest.approx(model.eta_b.value, rel=1e-6)

    def test_frozen_params_applied(self):
        spec, model = _make_simple_spec()
        theta = model.parameters.free_fit_values
        kw = _theta_to_evolve_args(theta, spec)
        assert kw['vel_psr'] == 0.0
        assert kw['eta_g'] == 0.0

    def test_ic_field_assembly(self):
        model = _make_simple_model()
        model.add_ic_field(temp=1000, norm=1e-10, frozen=False)
        model._parameters['ic_temp_1']._bounds = (10, 30000)
        model._parameters['ic_norm_1']._bounds = (1e-14, 1e-8)
        obs = [Observable.angular_size_snr(45.0, 10.0, 'arcmin')]
        spec = build_fit_spec(model, obs)
        theta = model.parameters.free_fit_values
        kw = _theta_to_evolve_args(theta, spec)
        assert kw['n_ic'] == 1
        assert len(kw['ic_temps']) == 1
        assert kw['ic_temps'][0] == pytest.approx(1000.0, rel=1e-6)
        assert kw['ic_norms'][0] == pytest.approx(1e-10, rel=1e-6)


class TestLogPrior:

    def test_within_bounds(self):
        spec, model = _make_simple_spec()
        theta = model.parameters.free_fit_values
        lp = log_prior(theta, spec)
        assert np.isfinite(lp)

    def test_outside_bounds_returns_neginf(self):
        spec, model = _make_simple_spec()
        theta = model.parameters.free_fit_values
        theta[0] = 10.0
        assert log_prior(theta, spec) == -np.inf

    def test_gaussian_prior_at_center(self):
        """Distance prior: Gaussian centered at 1400, sigma=400."""
        spec, model = _make_simple_spec()
        theta_center = model.parameters.free_fit_values.copy()
        theta_offset = theta_center.copy()
        dist_idx = spec.param_names.index('distance')
        theta_center[dist_idx] = 1400
        theta_offset[dist_idx] = 1800  # 1 sigma away

        lp_center = log_prior(theta_center, spec)
        lp_offset = log_prior(theta_offset, spec)
        assert lp_center > lp_offset
        assert (lp_center - lp_offset) == pytest.approx(0.5, abs=0.01)

    def test_boundary_exact(self):
        spec, model = _make_simple_spec()
        theta = np.array([fp.bounds[0] for fp in spec.free_params])
        lp = log_prior(theta, spec)
        assert np.isfinite(lp)


class TestInitWalkers:

    def test_shape(self):
        spec, model = _make_simple_spec()
        p0 = _init_walkers(spec, model, 24)
        assert p0.shape == (24, 12)

    def test_within_bounds(self):
        spec, model = _make_simple_spec()
        p0 = _init_walkers(spec, model, 100)
        for i, fp in enumerate(spec.free_params):
            lo, hi = fp.bounds
            assert np.all(p0[:, i] >= lo)
            assert np.all(p0[:, i] <= hi)


# ===================================================================
# 7. Chi² weight handling
# ===================================================================

class TestWeights:

    def test_weight_field_in_observable(self):
        obs = Observable.angular_size_snr(45.0, 10.0, 'arcmin')
        obs.weight = 4.0
        assert obs.weight == 4.0

    def test_default_weight_is_one(self):
        obs = Observable.flux_density(1.0, 'TeV', 1e-12, 1e-13, 'photons/cm2/s/TeV')
        assert obs.weight == 1.0


# ===================================================================
# 8. Constants modules
# ===================================================================

class TestConstants_Modules:
    """Verify constant source switching infrastructure."""

    def test_constants_source_is_set(self):
        assert isinstance(C.CONSTANTS_SOURCE, str)
        assert C.CONSTANTS_SOURCE in ('astropy', 'gsz2009')

    def test_gsz2009_and_astropy_export_same_names(self):
        from viper.constants import GSZ2009, ASTROPY
        gsz_names = set(vars(GSZ2009))
        ast_names = set(vars(ASTROPY))
        assert gsz_names == ast_names

    def test_c_light_differs_between_sources(self):
        """gsz2009 uses c = 3.0e10, astropy uses 2.998e10."""
        from viper.constants import GSZ2009, ASTROPY
        assert GSZ2009.c_light == 3.0e10
        assert ASTROPY.c_light == pytest.approx(2.998e10, rel=1e-3)
        assert GSZ2009.c_light != ASTROPY.c_light

    def test_pc_differs_between_sources(self):
        """gsz2009 uses pc = 3.09e18, astropy is more precise."""
        from viper.constants import GSZ2009, ASTROPY
        assert GSZ2009.pc == 3.09e18
        assert ASTROPY.pc == pytest.approx(3.0857e18, rel=1e-3)


# ===================================================================
# 9. Integration parameter validation
# ===================================================================

class TestIntegrationValidation:
    """Verify evolve_pwn rejects bad integration modes."""

    _DUMMY = dict(
        esn_51=1.0, mej=6.0, n_ism=0.4, braking_index=2.5,
        tau_sd=3000.0, age=10000.0, e0_37=1.0, vel_psr=0.0,
        eta_g=0.0, eta_b=0.01,
        e_min=3.0, e_max=5e5, e_break=50.0, p1=0.8, p2=2.5,
    )

    def test_invalid_integration_raises(self):
        from viper.evolution import evolve_pwn
        with pytest.raises(ValueError, match="integration must be"):
            evolve_pwn(integration='bogus', **self._DUMMY)

    def test_valid_modes_are_documented_set(self):
        from viper.evolution import evolve_pwn
        with pytest.raises(ValueError, match="integration must be") as exc_info:
            evolve_pwn(integration='bogus', **self._DUMMY)
        msg = str(exc_info.value)
        for mode in ('euler', 'pc_radius', 'pc_full'):
            assert mode in msg, f"{mode!r} not listed in error message"


# ===================================================================
# 10. Parameter unit enforcement
# ===================================================================

class TestParameterUnits:
    """Verify astropy unit enforcement on Parameter objects."""

    def test_physical_param_rejects_bare_float(self):
        model = PWNModel()
        with pytest.raises(ValueError, match="requires an astropy Quantity"):
            model.esn.value = 1.0

    def test_physical_param_accepts_quantity(self):
        model = PWNModel()
        model.esn.value = 1e51 * u.erg
        assert model.esn.value == pytest.approx(1.0, rel=1e-6)

    def test_dimensionless_param_rejects_quantity(self):
        model = PWNModel()
        with pytest.raises(ValueError, match="dimensionless"):
            model.p1.value = 0.8 * u.m

    def test_dimensionless_param_accepts_bare_float(self):
        model = PWNModel()
        model.p1.value = 1.2
        assert model.p1.value == 1.2

    def test_bounds_accept_quantities(self):
        model = PWNModel()
        model.esn.bounds = (1e49 * u.erg, 3e51 * u.erg)
        lo, hi = model.esn.bounds
        assert lo == pytest.approx(0.01, rel=1e-6)
        assert hi == pytest.approx(3.0, rel=1e-6)

    def test_unit_conversion_error_message(self):
        model = PWNModel()
        with pytest.raises(ValueError, match="Cannot convert"):
            model.esn.value = 5.0 * u.m


# ===================================================================
# 11. Parameter fit_value roundtrip
# ===================================================================

class TestParameterFitValue:
    """Verify log/linear fit-space conversions."""

    def test_log_param_fit_value(self):
        p = Parameter('test', default=100.0, log=True)
        assert p.fit_value == pytest.approx(2.0)

    def test_log_param_set_from_fit_value(self):
        p = Parameter('test', default=1.0, log=True)
        p.set_from_fit_value(3.0)
        assert p.value == pytest.approx(1000.0)

    def test_linear_param_fit_value(self):
        p = Parameter('test', default=2.5, log=False)
        assert p.fit_value == 2.5

    def test_fit_bounds_log(self):
        p = Parameter('test', default=100.0, log=True, bounds=(10.0, 1000.0))
        lo, hi = p.fit_bounds
        assert lo == pytest.approx(1.0)
        assert hi == pytest.approx(3.0)

    def test_fit_bounds_linear(self):
        p = Parameter('test', default=2.0, log=False, bounds=(1.0, 3.0))
        assert p.fit_bounds == (1.0, 3.0)

    def test_fit_label_log(self):
        p = Parameter('esn', default=1.0, log=True)
        assert p.fit_label == 'log(esn)'

    def test_fit_label_linear(self):
        p = Parameter('p1', default=0.8, log=False)
        assert p.fit_label == 'p1'


class TestParameterValidation:
    """Bounds/prior validation in constructor and property setters."""

    def test_constructor_rejects_reversed_bounds(self):
        with pytest.raises(ValueError, match='must be < upper bound'):
            Parameter('x', default=1.0, bounds=(3.0, 1.0))

    def test_constructor_rejects_nonpositive_bounds_for_log_param(self):
        with pytest.raises(ValueError, match='log-sampled'):
            Parameter('x', default=1.0, log=True, bounds=(-1.0, 3.0))
        with pytest.raises(ValueError, match='log-sampled'):
            Parameter('x', default=1.0, log=True, bounds=(0.0, 3.0))

    def test_setter_rejects_nan_bounds(self):
        p = Parameter('x', default=1.0)
        with pytest.raises(ValueError, match='must be finite'):
            p.bounds = (float('nan'), 3.0)
        with pytest.raises(ValueError, match='must be finite'):
            p.bounds = (1.0, float('nan'))

    def test_linear_param_negative_bounds_still_allowed(self):
        p = Parameter('esc_index', default=0.0, bounds=(-1.0, 2.0))
        assert p.fit_bounds == (-1.0, 2.0)

    def test_prior_rejects_bad_sigma(self):
        p = Parameter('mej', default=6.0)
        with pytest.raises(ValueError, match='sigma > 0'):
            p.prior = (10.0, 0.0)
        with pytest.raises(ValueError, match='sigma > 0'):
            p.prior = (10.0, -3.0)
        with pytest.raises(ValueError, match='sigma > 0'):
            p.prior = (float('nan'), 3.0)
        with pytest.raises(ValueError, match='sigma > 0'):
            Parameter('mej', default=6.0, prior=(10.0, 0.0))
        p.prior = (10.0, 3.0)
        assert p.prior == (10.0, 3.0)
        p.prior = None
        assert p.prior is None

    def test_fit_bounds_raises_on_late_log_flip(self):
        # used to return nan silently
        p = Parameter('x', default=1.0, bounds=(-5.0, 5.0))
        p.log = True
        with pytest.raises(ValueError, match='log-sampled'):
            p.fit_bounds


# ===================================================================
# 12. PWNModel construction and API
# ===================================================================

class TestPWNModelAPI:
    """Verify PWNModel construction and methods."""

    def test_default_construction(self):
        model = PWNModel()
        assert hasattr(model, 'esn')
        assert hasattr(model, 'p2')
        assert hasattr(model, 'parameters')

    def test_set_pulsar_by_jname(self):
        from viper import PulsarCatalog
        model = PWNModel()
        model.set_pulsar('J0534+2200')
        # Crab period ~ 0.0334 s
        assert 0.03 < model._parameters['period'].value < 0.04

    def test_add_ic_field(self):
        model = PWNModel()
        model.add_ic_field(temp=2753, norm=8.72e-11)
        assert 'ic_temp_1' in model._parameters
        assert 'ic_norm_1' in model._parameters
        assert model._parameters['ic_temp_1'].value == pytest.approx(2753.0)

    def test_parameters_returns_parameterset(self):
        model = PWNModel()
        ps = model.parameters
        assert isinstance(ps, ParameterSet)

    def test_add_multiple_ic_fields(self):
        model = PWNModel()
        model.add_ic_field(temp=2753, norm=8.72e-11)
        model.add_ic_field(temp=30, norm=1e-12)
        assert 'ic_temp_2' in model._parameters
        assert 'ic_norm_2' in model._parameters

    def test_add_ic_field_failed_conversion_leaves_model_unchanged(self):
        """A bad temp or norm Quantity must not half-add a field."""
        import astropy.units as u
        model = PWNModel()
        with pytest.raises(u.UnitConversionError):
            model.add_ic_field(temp=30.0, norm=5.0 * u.K)
        with pytest.raises(u.UnitConversionError):
            model.add_ic_field(temp=5.0 * u.eV / u.cm**3, norm=0.35)
        assert model.n_ic == 0
        assert 'ic_temp_1' not in model._parameters
        assert 'ic_norm_1' not in model._parameters
        assert model.ic_temps == [] and model.ic_norms == []
        # next add gets index 1
        assert model.add_ic_field(temp=30.0, norm=0.35) == 1
        assert model.ic_temps == [pytest.approx(30.0)]
        assert model.ic_norms == [pytest.approx(0.35)]


# ===================================================================
# 13. PulsarCatalog
# ===================================================================

class TestPulsarCatalog:
    """Verify ATNF catalog lookups."""

    def test_lookup_by_jname(self):
        from viper import PulsarCatalog
        cat = PulsarCatalog()
        entry = cat['J0534+2200']
        assert entry.period == pytest.approx(0.0334, rel=0.01)

    def test_lookup_by_bname(self):
        from viper import PulsarCatalog
        cat = PulsarCatalog()
        entry = cat['B0531+21']
        assert entry.period == pytest.approx(0.0334, rel=0.01)

    def test_unknown_name_raises(self):
        from viper import PulsarCatalog
        cat = PulsarCatalog()
        with pytest.raises(KeyError):
            cat['J9999+9999']


def test_range_observable_flat_top():
    """Test that range observables produce flat-top chi2."""
    from viper.observables import Observable
    obs = Observable.angular_size_pwn_range(10.2, 13.8, 1.8, 'arcmin')
    assert obs.range_lo == 10.2
    assert obs.range_hi == 13.8
    assert obs.error == 1.8
    assert obs.obs_type.value == 'angular_size_pwn'


# ===================================================================
# evolve_pwn replay_dt_sequence
# ===================================================================

@pytest.mark.slow
def test_replay_dt_sequence_self_consistent():
    """Live mode + replay of its own dt sequence must be bit-identical."""
    import os
    os.environ.setdefault("VIPER_CONSTANTS", "gsz2009")
    import importlib, viper
    from viper.evolution import evolve_pwn

    kw = dict(
        esn_51=1.0, mej=9.8, n_ism=0.15, braking_index=3.0, tau_sd=5000.0,
        age=10000.0, e0_37=1.0, vel_psr=0.0, eta_g=0.999, eta_b=0.06,
        e_min=1.1, e_max=263e3, e_break=10000.0, p1=2.5, p2=1.6,
        f_max=0.0, kT_max=0.0, n_ic=0, ic_temps=[], ic_norms=[],
        integration="pc_full",
    )
    r1 = evolve_pwn(**kw)
    dts = [d.delta_t for d in r1.dyninfo]
    r2 = evolve_pwn(**kw, replay_dt_sequence=dts)

    np.testing.assert_array_equal(r1.photon_spectrum, r2.photon_spectrum)
    np.testing.assert_array_equal(r1.electron_spectrum, r2.electron_spectrum)
    assert len(r1.dyninfo) == len(r2.dyninfo)


@pytest.mark.slow
def test_replay_dt_sequence_lookup_matches_step_count():
    """Lookup mode replaying live's dt sequence must take same n_steps."""
    import os
    os.environ.setdefault("VIPER_CONSTANTS", "gsz2009")
    from viper.evolution import evolve_pwn
    import viper.constants as C

    kw = dict(
        esn_51=1.0, mej=9.8, n_ism=0.15, braking_index=3.0, tau_sd=5000.0,
        age=10000.0, e0_37=1.0, vel_psr=0.0, eta_g=0.999, eta_b=0.06,
        e_min=1.1, e_max=263e3, e_break=10000.0, p1=2.5, p2=1.6,
        f_max=0.0, kT_max=0.0, n_ic=0, ic_temps=[], ic_norms=[],
        integration="pc_full",
    )
    r_live = evolve_pwn(use_cmb_lookup=False, **kw)
    dts = [d.delta_t for d in r_live.dyninfo]
    r_lookup = evolve_pwn(use_cmb_lookup=True,
                           n_elec_bins=C.ELEC_GRID_FIXED_NBINS,
                           replay_dt_sequence=dts, **kw)
    assert len(r_lookup.dyninfo) == len(r_live.dyninfo)



@pytest.mark.slow
def test_use_cmb_lookup_matches_live_within_tolerance():
    """Lookup mode matches live within 5% integrated and flux-weighted."""
    from viper.evolution import evolve_pwn
    import viper.constants as C
    kw = dict(
        esn_51=0.5, mej=6.0, n_ism=1.0,
        braking_index=2.7, tau_sd=4700.0,
        age=5800.0, e0_37=20.0,
        vel_psr=0.0, eta_g=0.999, eta_b=0.001,
        e_min=1.0, e_max=4e5, e_break=100.0,
        p1=1.8, p2=3.0, f_max=0.0, kT_max=0.0,
        n_ic=0, ic_temps=[], ic_norms=[],
        integration="pc_full",
    )
    r_live = evolve_pwn(use_cmb_lookup=False, **kw)
    times = np.array([d.time for d in r_live.dyninfo])
    dts = np.diff(np.concatenate([[0], times]))
    r_lookup = evolve_pwn(
        use_cmb_lookup=True, n_elec_bins=C.ELEC_GRID_FIXED_NBINS,
        replay_dt_sequence=dts.tolist(), **kw,
    )
    ph_l = r_lookup.photon_spectrum
    ph_v = r_live.photon_spectrum
    n = min(len(ph_l), len(ph_v))
    assert ph_l[:n].sum() > 0 and ph_v[:n].sum() > 0
    total_ratio = ph_l[:n].sum() / ph_v[:n].sum()
    assert 0.95 < total_ratio < 1.05, (
        f"integrated photon flux ratio {total_ratio:.4f} outside [0.95, 1.05]"
    )
    nz = (ph_l[:n] > 0) & (ph_v[:n] > 0)
    rel = np.abs(ph_l[:n][nz] - ph_v[:n][nz]) / np.abs(ph_v[:n][nz])
    fw_weight = (ph_l[:n][nz] + ph_v[:n][nz]) / 2
    fw = (rel * fw_weight).sum() / max(fw_weight.sum(), 1e-300)
    assert fw < 0.05, (
        f"flux-weighted photon disagreement {fw:.3%} exceeds 5%"
    )


def test_cmb_lookup_files_load_for_both_sources():
    """Both per-source lookup files must be present and have the expected shape."""
    import importlib.resources
    import viper.constants as C
    for src in ("gsz2009", "astropy"):
        ref = importlib.resources.files("viper.data").joinpath(f"cmb_pow_nu_ic_{src}.npz")
        with ref.open("rb") as f:
            npz = np.load(f, allow_pickle=False)
            assert "pow_nu_ic" in npz.files
            assert npz["pow_nu_ic"].shape == (C.ELEC_GRID_FIXED_NBINS + 2, C.N_PHOT_FREQ + 2)
            file_src = npz["constants_source"]
            if hasattr(file_src, "item"):
                file_src = file_src.item()
            assert str(file_src) == src


def test_fit_spec_stepping_options():
    """Stepper options pass from model to spec to evolve args; defaults otherwise."""
    import numpy as np
    from viper.fitting import build_fit_spec, _theta_to_evolve_args

    opts = ('timestep_events', 'snapshot_mode', 'event_dt_floor_yr',
             'timestep_cooling')

    m = PWNModel()
    m.parameters['p1'].frozen = False
    m.parameters['p1'].bounds = (1.0, 3.0)
    spec = build_fit_spec(m, [])
    assert spec.timestep_events is False
    assert spec.snapshot_mode == 'secant'
    assert spec.event_dt_floor_yr == 0.5

    theta = np.array([np.log10(spec.all_values[fp.name]) if fp.log
                      else spec.all_values[fp.name]
                      for fp in spec.free_params])
    kw = _theta_to_evolve_args(theta, spec)
    assert kw['timestep_events'] is False
    assert kw['snapshot_mode'] == 'secant'
    assert kw['event_dt_floor_yr'] == 0.5

    m2 = PWNModel()
    m2.parameters['p1'].frozen = False
    m2.parameters['p1'].bounds = (1.0, 3.0)
    m2.timestep_events = True
    m2.snapshot_mode = 'instantaneous'
    m2.event_dt_floor_yr = 0.25
    m2.timestep_cooling = 'total_loss10'
    spec2 = build_fit_spec(m2, [])
    assert spec.timestep_cooling == 'compat'
    kw2 = _theta_to_evolve_args(theta, spec2)
    assert kw2['timestep_events'] is True
    assert kw2['snapshot_mode'] == 'instantaneous'
    assert kw2['event_dt_floor_yr'] == 0.25
    assert kw2['timestep_cooling'] == 'total_loss10'

    # opts additive; other args unchanged
    a = {k: v for k, v in kw.items() if k not in opts}
    b = {k: v for k, v in kw2.items() if k not in opts}
    assert a.keys() == b.keys()
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, float) and np.isnan(va):
            assert isinstance(vb, float) and np.isnan(vb), k
        else:
            assert va == vb, k


def test_cooling_bound_hand_formula():
    """total_loss10 bound matches the hand formula on a synthetic grid."""
    import numpy as np
    from viper import constants as C
    from viper.evolution import _cooling_bound, _COOLING_PREFACTOR
    from viper.grids import make_log_grid
    from viper.radiation import csynch_calc_vec

    g = make_log_grid(1.0, 1.0e6, 102)
    g.number[:] = 0.0
    occ = [10, 50, 90]
    for i in occ:
        g.number[i] = 1.0e30
    bpwn, nelec, rad = 5e-6, 1.0e10, 3.0e18
    pie = np.zeros(g.n_bins)
    pie[50] = 1.0e-3                      # IC power on one bin

    got = _cooling_bound(g, bpwn, nelec, rad, pie, 0.5)

    log_avg = g.log_avg[occ]
    cs = csynch_calc_vec(log_avg, bpwn, nelec, np.log10(rad))
    e_erg = 10.0 ** (log_avg - C.log_erg)
    cic = np.array([0.0, pie[50] / e_erg[1] ** 2, 0.0])
    t10 = _COOLING_PREFACTOR / ((cs + cic) * e_erg) / C.yr
    assert np.isclose(got, max(t10.min(), 0.5), rtol=1e-12)

    # floor engages under fast cooling
    got_floor = _cooling_bound(g, 0.1, nelec, rad, pie, 0.5)
    assert got_floor == 0.5

    # empty grid -> no bound
    g.number[:] = 0.0
    assert _cooling_bound(g, bpwn, nelec, rad, pie, 0.5) > 1e29


# ======================================================================
# PWNModel attribute-assignment guard
# ======================================================================

class TestParameterAssignmentGuard:
    """model.<param> = value must raise, not silently shadow."""

    def test_bare_assignment_raises_with_hint(self):
        m = PWNModel()
        with pytest.raises(AttributeError, match=r"model\.esn\.value"):
            m.esn = 2.0

    def test_quantity_assignment_raises(self):
        m = PWNModel()
        with pytest.raises(AttributeError, match="Parameter"):
            m.distance = 1.4 * u.kpc

    def test_guard_covers_added_ic_fields(self):
        m = PWNModel()
        m.add_ic_field(temp=35.0, norm=0.35)
        with pytest.raises(AttributeError, match=r"ic_norm_1"):
            m.ic_norm_1 = 1.0

    def test_normal_paths_still_work(self):
        m = PWNModel()
        m.esn.value = 2e51 * u.erg          # Parameter.value setter
        assert m.esn.value == pytest.approx(2.0)
        m.esn.frozen = False                # Parameter attribute
        m.use_cmb_lookup = True             # plain model attribute
        m.n_elec_bins = 400
        m.injection_model = 'broken_powerlaw'
        m2 = m.copy()                       # deepcopy bypasses __setattr__
        assert m2.esn.value == pytest.approx(2.0)
        assert not m2.esn.frozen


# ======================================================================
# PWNModel.apply_frozen_params
# ======================================================================

class TestApplyFrozenParams:
    """frozen_params values are INTERNAL units; ._scale must not re-apply."""

    def test_scaled_param_not_rescaled(self):
        m = PWNModel()
        m.apply_frozen_params({'esn': 1.2})
        assert m.esn.value == pytest.approx(1.2)   # NOT 1.2e-51
        assert m.esn.frozen

    def test_unscaled_params(self):
        m = PWNModel()
        m.apply_frozen_params({'mej': 8.0, 'distance': 1400.0, 'p1': 1.1})
        assert m.mej.value == pytest.approx(8.0)
        assert m.distance.value == pytest.approx(1400.0)
        assert m.p1.value == pytest.approx(1.1)
        assert m.mej.frozen and m.distance.frozen and m.p1.frozen

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="unknown parameter"):
            PWNModel().apply_frozen_params({'esm': 1.0})


# ======================================================================
# Shared physics validation
# ======================================================================

def _timed_model():
    """PWNModel with timing set (validation otherwise fails on NaN)."""
    m = PWNModel()
    m.period._set_internal(0.1)
    m.p_dot._set_internal(1e-13)
    return m


class TestValidationParity:
    """PWNModel.validate and PWNScalarParams.validate share check_physical_inputs."""

    def _scalar_kwargs(self, **over):
        kw = dict(esn=1.0, mej=6.0, n_ism=0.4, braking_index=2.5,
                  tau_sd=3000.0, period=0.1, p_dot=1e-13, eta_b=0.01,
                  e_min=3.0, e_max=5e5, e_break=50.0, p1=0.8, p2=2.5)
        kw.update(over)
        return kw

    @pytest.mark.parametrize("name, bad, msg", [
        ('e_break', 1e6,  'e_break must be < e_max'),   # > e_max = 5e5
        ('e_max',   1.0,  'e_max must be >= e_min'),    # < e_min = 3
        ('f_max',   1.5,  'f_max must be in'),
        ('vel_psr', -1.0, 'vel_psr must be >= 0'),
        ('eta_g',   1.0,  'eta_g must be in'),
        ('kT_max',  -1.0, 'kT_max must be >= 0'),
        ('distance', 0.0, 'distance must be positive'),
        ('distance', float('nan'), 'distance must be positive and finite'),
    ])
    def test_both_apis_reject(self, name, bad, msg):
        with pytest.raises(ValueError, match=msg):
            PWNScalarParams(**self._scalar_kwargs(**{name: bad}))
        m = _timed_model()
        m.parameters[name]._set_internal(bad)
        with pytest.raises(ValueError, match=msg):
            m.validate()

    def test_defaults_pass_both(self):
        PWNScalarParams(**self._scalar_kwargs())
        _timed_model().validate()

    def test_e_break_zero_single_pl_still_allowed(self):
        PWNScalarParams(**self._scalar_kwargs(e_break=0.0))
        m = _timed_model()
        m.parameters['e_break']._set_internal(0.0)
        m.validate()

    def test_maxwellian_requires_positive_kt(self):
        """f_max > 0 with kT_max == 0 divided by zero in inject_maxwellian."""
        with pytest.raises(ValueError, match='kT_max must be > 0 when f_max'):
            PWNScalarParams(**self._scalar_kwargs(f_max=0.3))
        m = _timed_model()
        m.parameters['f_max']._set_internal(0.3)
        with pytest.raises(ValueError, match='kT_max must be > 0 when f_max'):
            m.validate()
        PWNScalarParams(**self._scalar_kwargs(f_max=0.3, kT_max=0.5))
        m.parameters['kT_max']._set_internal(0.5)
        m.validate()

    def test_ic_field_values_validated(self):
        """Negative IC temps and norms rejected; norm 0 stays valid."""
        with pytest.raises(ValueError, match=r'ic_temps\[0\] must be positive'):
            PWNScalarParams(**self._scalar_kwargs(),
                           n_ic=1, ic_temps=[-30.0], ic_norms=[0.35])
        with pytest.raises(ValueError, match=r'ic_norms\[0\] must be >= 0'):
            PWNScalarParams(**self._scalar_kwargs(),
                           n_ic=1, ic_temps=[30.0], ic_norms=[-0.35])
        m = _timed_model()
        m.add_ic_field(temp=-30.0, norm=0.35)
        with pytest.raises(ValueError, match=r'ic_temps\[0\] must be positive'):
            m.validate()
        PWNScalarParams(**self._scalar_kwargs(),
                       n_ic=1, ic_temps=[30.0], ic_norms=[0.0])


class TestLazyGammapyImport:
    """PEP 562 lazy export: `import viper` must not pull in gammapy."""

    def test_gammapy_not_imported_until_touched(self):
        code = (
            "import sys\n"
            "import viper\n"
            "assert 'viper.gammapy' not in sys.modules, "
            "'viper.gammapy imported eagerly'\n"
            "assert 'gammapy' not in sys.modules, "
            "'gammapy imported eagerly'\n"
            "cls = viper.PWNSpectralModel\n"
            "assert 'viper.gammapy' in sys.modules\n"
            "assert cls.__name__ == 'PWNSpectralModel'\n"
        )
        r = subprocess.run([sys.executable, '-c', code],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
