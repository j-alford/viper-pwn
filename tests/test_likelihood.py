"""Statistical-layer tests: log_likelihood, log_prior, log_probability, prior_transform."""

import logging

import numpy as np
import pytest
import astropy.units as u

import viper.fitting as F
from viper import PWNModel
from viper.evolution import PWNModelResult, DynState
from viper.observables import Observable, extract_all


# ===================================================================
# Synthetic-model scaffolding (no evolution runs)
# ===================================================================

RAD_PWN_PC = 5.0
RAD_SNR_PC = 15.0
DIST_PC = 1000.0   # PWNModel default distance


def _fake_spectrum():
    """Positive power-law L_nu over 1e8..1e28 Hz (units: 1e40 erg/s/Hz)."""
    freqs = np.logspace(8, 28, 300)
    spectrum = (freqs / 1.0e15) ** -0.6
    return freqs, spectrum


def _fake_result(freqs=None, spectrum=None):
    if freqs is None:
        freqs, spectrum = _fake_spectrum()
    res = PWNModelResult()
    res.dyninfo = [DynState(rad_pwn=RAD_PWN_PC, rad_snr=RAD_SNR_PC)]
    res.photon_freqs = freqs
    res.photon_spectrum = spectrum
    return res


def _make_spec(observables, tau_sd_yr=None):
    """Spec with one free bounded parameter (esn) and valid timing."""
    m = PWNModel()
    m.set_pulsar(period=0.3168 * u.s, p_dot=3.614e-13)
    if tau_sd_yr is not None:
        m.tau_sd.value = tau_sd_yr * u.yr
    m.esn.frozen = False
    m.esn.bounds = (1e50 * u.erg, 1e52 * u.erg)
    return F.build_fit_spec(m, observables), m


THETA = np.array([0.0])   # log10(esn=1.0), mid-bounds


def _predicted(observables):
    """Predicted values exactly as log_likelihood computes them."""
    freqs, spectrum = _fake_spectrum()
    return extract_all(observables, freqs, spectrum,
                       RAD_PWN_PC, RAD_SNR_PC, DIST_PC)


@pytest.fixture
def fake_evolve(monkeypatch):
    """Replace the expensive evolution with the canned synthetic result."""
    calls = []

    def _fake(**kw):
        calls.append(kw)
        return _fake_result()

    monkeypatch.setattr(F, 'evolve_pwn', _fake)
    return calls


# ===================================================================
# 1. Gaussian branch + weights
# ===================================================================

class TestGaussianBranch:

    def test_matches_hand_computation(self, fake_evolve):
        obs = [
            Observable.flux_density(1.4, 'GHz', 400.0, 30.0, 'Jy'),
            Observable.angular_size_pwn(17.0, 1.0, 'arcmin'),
            Observable.band_flux(1e15, 1e16, 'Hz', 1.0e-11, 1.0e-12),
        ]
        spec, _ = _make_spec(obs)
        pred = _predicted(obs)
        chi2 = sum(((p - o.value) / o.error) ** 2 for p, o in zip(pred, obs))
        ll = F.log_likelihood(THETA, spec)
        assert np.isfinite(ll)
        assert ll == pytest.approx(-0.5 * chi2, rel=1e-12)

    def test_weights_multiply(self, fake_evolve):
        obs_w1 = [Observable.flux_density(1.4, 'GHz', 400.0, 30.0, 'Jy')]
        obs_w3 = [Observable.flux_density(1.4, 'GHz', 400.0, 30.0, 'Jy')]
        obs_w3[0].weight = 3.0
        spec1, _ = _make_spec(obs_w1)
        spec3, _ = _make_spec(obs_w3)
        ll1 = F.log_likelihood(THETA, spec1)
        ll3 = F.log_likelihood(THETA, spec3)
        assert ll3 == pytest.approx(3.0 * ll1, rel=1e-12)


# ===================================================================
# 2. One-sided limits (upper / lower) + the 0.1*value sigma fallback
# ===================================================================

class TestLimits:

    def test_satisfied_upper_limit_contributes_zero(self, fake_evolve):
        base = [Observable.flux_density(1.4, 'GHz', 400.0, 30.0, 'Jy')]
        pred_ul = _predicted(
            [Observable.flux_density(10.0, 'GHz', 1.0, 1.0, 'Jy')])[0]
        with_ul = base + [Observable.flux_density(
            10.0, 'GHz', pred_ul * 2.0, 0.0, 'Jy', upper_limit=True)]
        spec_a, _ = _make_spec(base)
        spec_b, _ = _make_spec(with_ul)
        assert (F.log_likelihood(THETA, spec_b)
                == pytest.approx(F.log_likelihood(THETA, spec_a), rel=1e-12))

    def test_violated_upper_limit_uses_limit_sigma(self, fake_evolve):
        pred = _predicted(
            [Observable.flux_density(10.0, 'GHz', 1.0, 1.0, 'Jy')])[0]
        # error=0 -> 0.1*value sigma
        ul_val = pred / 2.0
        obs = [Observable.flux_density(10.0, 'GHz', ul_val, 0.0, 'Jy',
                                       upper_limit=True)]
        spec, _ = _make_spec(obs)
        expected = -0.5 * ((pred - ul_val) / (0.1 * ul_val)) ** 2
        assert F.log_likelihood(THETA, spec) == pytest.approx(expected,
                                                              rel=1e-12)
        # explicit error -> used directly
        obs[0].error = 0.3 * ul_val
        spec2, _ = _make_spec(obs)
        expected2 = -0.5 * ((pred - ul_val) / (0.3 * ul_val)) ** 2
        assert F.log_likelihood(THETA, spec2) == pytest.approx(expected2,
                                                               rel=1e-12)

    def test_lower_limit_branch(self, fake_evolve):
        pred = _predicted(
            [Observable.band_flux(1e15, 1e16, 'Hz', 1.0, 1.0)])[0]
        # violated LL; 0.1*value sigma
        ll_val = pred * 2.0
        obs = [Observable.band_flux(1e15, 1e16, 'Hz', ll_val, 0.0,
                                    lower_limit=True)]
        spec, _ = _make_spec(obs)
        expected = -0.5 * ((pred - ll_val) / (0.1 * ll_val)) ** 2
        assert F.log_likelihood(THETA, spec) == pytest.approx(expected,
                                                              rel=1e-12)
        # satisfied LL -> zero contribution
        obs_ok = [Observable.band_flux(1e15, 1e16, 'Hz', pred / 2.0, 0.0,
                                       lower_limit=True)]
        spec_ok, _ = _make_spec(obs_ok)
        assert F.log_likelihood(THETA, spec_ok) == pytest.approx(0.0,
                                                                 abs=1e-15)


# ===================================================================
# 3. Flat-top range branch
# ===================================================================

class TestFlatTopRange:

    def test_inside_range_is_free(self, fake_evolve):
        pred = _predicted(
            [Observable.angular_size_pwn(17.0, 1.0, 'arcmin')])[0]
        obs = [Observable.angular_size_pwn_range(pred - 1, pred + 1, 0.5)]
        spec, _ = _make_spec(obs)
        assert F.log_likelihood(THETA, spec) == pytest.approx(0.0, abs=1e-15)

    def test_outside_range_penalized_from_nearest_edge(self, fake_evolve):
        pred = _predicted(
            [Observable.angular_size_pwn(17.0, 1.0, 'arcmin')])[0]
        # penalty from range_hi
        obs = [Observable.angular_size_pwn_range(pred - 5, pred - 3, 0.5)]
        spec, _ = _make_spec(obs)
        expected = -0.5 * ((pred - (pred - 3)) / 0.5) ** 2
        assert F.log_likelihood(THETA, spec) == pytest.approx(expected,
                                                              rel=1e-12)
        # penalty from range_lo
        obs = [Observable.angular_size_pwn_range(pred + 3, pred + 5, 0.5)]
        spec, _ = _make_spec(obs)
        expected = -0.5 * ((pred - (pred + 3)) / 0.5) ** 2
        assert F.log_likelihood(THETA, spec) == pytest.approx(expected,
                                                              rel=1e-12)


# ===================================================================
# 4. Validity gates
# ===================================================================

class TestValidityGates:

    def test_zero_prediction_on_regular_obs_penalized(self, fake_evolve):
        # above spectrum -> predicted 0.0: a (value/error)^2 miss,
        # not an exclusion
        obs = [Observable.flux_density(1e35, 'Hz', 1.0, 0.1,
                                       'erg/s/cm2/Hz')]
        spec, _ = _make_spec(obs)
        assert _predicted(obs)[0] == 0.0
        expected = -0.5 * (1.0 / 0.1) ** 2
        assert F.log_likelihood(THETA, spec) == pytest.approx(expected)

    def test_nan_spectral_index_is_unconstrained(self, monkeypatch):
        # truncated band: extraction returns NaN
        freqs, spectrum = _fake_spectrum()
        spectrum = spectrum.copy()
        spectrum[freqs > 1e20] = 0.0
        monkeypatch.setattr(F, 'evolve_pwn',
                            lambda **kw: _fake_result(freqs, spectrum))

        radio = Observable.flux_density(1.4, 'GHz', 400.0, 30.0, 'Jy')
        idx = Observable.spectral_index(1e25, 1e26, 'Hz', 2.0, 0.1)
        spec_both, _ = _make_spec([radio, idx])
        spec_radio, _ = _make_spec([radio])

        ll_both = F.log_likelihood(THETA, spec_both)
        ll_radio = F.log_likelihood(THETA, spec_radio)
        # NaN term skipped = unconstrained
        assert np.isfinite(ll_both)
        assert ll_both == pytest.approx(ll_radio, rel=1e-12)

    def test_negative_true_age_is_minus_inf_without_running(self,
                                                            fake_evolve):
        obs = [Observable.flux_density(1.4, 'GHz', 400.0, 30.0, 'Jy')]
        spec, _ = _make_spec(obs, tau_sd_yr=1.0e6)
        assert F.log_likelihood(THETA, spec) == -np.inf
        assert len(fake_evolve) == 0   # rejected before evolve_pwn

    def test_radiative_remnant_is_rejected(self, fake_evolve):
        # trad = 2.9e4 E51^(4/17) n^(-9/17) yr: with E51 = 1 and
        # n_ism = 100 the remnant turns radiative at ~2.5 kyr.
        obs = [Observable.flux_density(1.4, 'GHz', 400.0, 30.0, 'Jy')]
        spec, _ = _make_spec(obs)
        spec.all_values['n_ism'] = 100.0
        age = F.age_from_theta(THETA, spec)
        from viper.snr import calc_trad
        assert age > calc_trad(spec.all_values['esn'], 100.0)
        assert F.log_prior(THETA, spec) == -np.inf
        assert F.log_likelihood(THETA, spec) == -np.inf
        assert len(fake_evolve) == 0
        spec.all_values['n_ism'] = 1e-3
        assert np.isfinite(F.log_prior(THETA, spec))


# ===================================================================
# 5. Exception swallowing is logged (once per distinct failure)
# ===================================================================

class TestExceptionLogging:

    def test_model_crash_is_minus_inf_and_warned_once(self, monkeypatch,
                                                      caplog):
        def _boom(**kw):
            raise RuntimeError("synthetic crash for test")

        monkeypatch.setattr(F, 'evolve_pwn', _boom)
        monkeypatch.setattr(F, '_FAILURE_WARNED', set())
        obs = [Observable.flux_density(1.4, 'GHz', 400.0, 30.0, 'Jy')]
        spec, _ = _make_spec(obs)

        with caplog.at_level(logging.DEBUG, logger='viper.fitting'):
            assert F.log_likelihood(THETA, spec) == -np.inf
            assert F.log_likelihood(THETA, spec) == -np.inf

        warnings = [r for r in caplog.records
                    if r.levelno == logging.WARNING]
        debugs = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert len(warnings) == 1          # deduped after first occurrence
        assert 'synthetic crash' in warnings[0].getMessage()
        assert len(debugs) == 1            # repeat went to DEBUG


# ===================================================================
# 6. log_prior branches
# ===================================================================

class TestLogPrior:

    def _spec_with_prior(self, prior=None, prior_log=False):
        m = PWNModel()
        m.set_pulsar(period=0.3168 * u.s, p_dot=3.614e-13)
        m.esn.frozen = False
        m.esn.bounds = (1e50 * u.erg, 1e52 * u.erg)   # fit space [-1, 1]
        if prior is not None:
            m.esn.prior = prior
            m.esn.prior_log = prior_log
        obs = [Observable.flux_density(1.4, 'GHz', 400.0, 30.0, 'Jy')]
        return F.build_fit_spec(m, obs)

    def test_outside_bounds_is_minus_inf(self):
        spec = self._spec_with_prior()
        assert F.log_prior(np.array([1.5]), spec) == -np.inf
        assert F.log_prior(np.array([-1.5]), spec) == -np.inf

    def test_flat_inside_bounds_is_zero(self):
        spec = self._spec_with_prior()
        assert F.log_prior(np.array([0.3]), spec) == 0.0

    def test_physical_gaussian_on_log_param(self):
        spec = self._spec_with_prior(prior=(2.0, 0.5), prior_log=False)
        theta = np.array([0.4])
        expected = -0.5 * ((10.0 ** 0.4 - 2.0) / 0.5) ** 2
        assert F.log_prior(theta, spec) == pytest.approx(expected, rel=1e-12)

    def test_fit_space_gaussian_with_prior_log(self):
        spec = self._spec_with_prior(prior=(0.2, 0.1), prior_log=True)
        theta = np.array([0.4])
        expected = -0.5 * ((0.4 - 0.2) / 0.1) ** 2
        assert F.log_prior(theta, spec) == pytest.approx(expected, rel=1e-12)

    def test_log_probability_combines(self, fake_evolve):
        spec = self._spec_with_prior(prior=(0.2, 0.1), prior_log=True)
        theta = np.array([0.4])
        lp = F.log_prior(theta, spec)
        ll = F.log_likelihood(theta, spec)
        assert F.log_probability(theta, spec) == pytest.approx(lp + ll,
                                                               rel=1e-12)


class TestBestFitReporting:
    """get_best_fit: chi2 is -2*log_likelihood, prior penalty removed."""

    class _StubSampler:
        def __init__(self, chain, lp):
            self._chain, self._lp = np.asarray(chain), np.asarray(lp)

        def get_chain(self, discard=0, flat=False):
            c = self._chain[discard:]
            return c.reshape(-1, c.shape[-1]) if flat else c

        def get_log_prob(self, discard=0, flat=False):
            lp = self._lp[discard:]
            return lp.reshape(-1) if flat else lp

    def _spec(self, prior=None):
        m = PWNModel()
        m.set_pulsar(period=0.3168 * u.s, p_dot=3.614e-13)
        m.esn.frozen = False
        m.esn.bounds = (1e50 * u.erg, 1e52 * u.erg)   # fit space [-1, 1]
        if prior is not None:
            m.esn.prior = prior
            m.esn.prior_log = True
        obs = [Observable.flux_density(1.4, 'GHz', 400.0, 30.0, 'Jy')]
        return F.build_fit_spec(m, obs)

    def test_chi2_excludes_prior_penalty(self):
        spec = self._spec(prior=(0.2, 0.1))
        # shape (3, 1, 1); argmax at 0.4
        chain = np.array([[[0.0]], [[0.4]], [[0.6]]])
        lp = np.array([[-9.0], [-3.0], [-7.0]])
        best = F.get_best_fit(self._StubSampler(chain, lp), spec, burn_in=0)
        prior = -0.5 * ((0.4 - 0.2) / 0.1) ** 2   # -2.0 in fit space
        assert best['log_probability'] == pytest.approx(-3.0)
        assert best['log_prior'] == pytest.approx(prior, rel=1e-12)
        assert best['log_likelihood'] == pytest.approx(-3.0 - prior,
                                                       rel=1e-12)
        assert best['chi2'] == pytest.approx(-2.0 * (-3.0 - prior),
                                             rel=1e-12)
        assert best['esn'] == pytest.approx(10.0 ** 0.4, rel=1e-12)

    def test_chi2_equals_minus2lp_without_prior(self):
        spec = self._spec()
        chain = np.array([[[0.3]]])
        lp = np.array([[-4.5]])
        best = F.get_best_fit(self._StubSampler(chain, lp), spec, burn_in=0)
        assert best['log_prior'] == 0.0
        assert best['log_likelihood'] == pytest.approx(-4.5)
        assert best['chi2'] == pytest.approx(9.0)


# ===================================================================
# 7. NS prior_transform consistency with log_prior
# ===================================================================

class TestPriorTransform:

    def _spec(self, prior=None, prior_log=False):
        m = PWNModel()
        m.set_pulsar(period=0.3168 * u.s, p_dot=3.614e-13)
        m.esn.frozen = False
        m.esn.bounds = (1e50 * u.erg, 1e52 * u.erg)
        if prior is not None:
            m.esn.prior = prior
            m.esn.prior_log = prior_log
        obs = [Observable.flux_density(1.4, 'GHz', 400.0, 30.0, 'Jy')]
        return F.build_fit_spec(m, obs)

    def test_no_prior_is_uniform_on_fit_bounds(self):
        from viper.nested import prior_transform
        spec = self._spec()
        for ui, expect in ((0.0, -1.0), (0.5, 0.0), (1.0, 1.0)):
            theta = prior_transform(np.array([ui]), spec)
            assert theta[0] == pytest.approx(expect, abs=1e-12)

    def test_prior_log_matches_truncnorm_in_fit_space(self):
        from scipy.stats import truncnorm
        from viper.nested import prior_transform
        mu, sigma = 0.2, 0.1
        spec = self._spec(prior=(mu, sigma), prior_log=True)
        a, b = (-1.0 - mu) / sigma, (1.0 - mu) / sigma
        for ui in (0.05, 0.5, 0.95):
            theta = prior_transform(np.array([ui]), spec)
            expect = truncnorm.ppf(ui, a, b, loc=mu, scale=sigma)
            assert theta[0] == pytest.approx(expect, rel=1e-10)

    def test_physical_prior_on_log_param_matches_truncnorm_in_physical(self):
        # deliberate NS/MCMC measure difference
        from scipy.stats import truncnorm
        from viper.nested import prior_transform
        mu, sigma = 2.0, 0.5
        spec = self._spec(prior=(mu, sigma), prior_log=False)
        lo_phys, hi_phys = 0.1, 10.0
        a, b = (lo_phys - mu) / sigma, (hi_phys - mu) / sigma
        for ui in (0.05, 0.5, 0.95):
            theta = prior_transform(np.array([ui]), spec)
            expect_phys = truncnorm.ppf(ui, a, b, loc=mu, scale=sigma)
            assert 10.0 ** theta[0] == pytest.approx(expect_phys, rel=1e-10)


# ===================================================================
# 8. End-to-end: one real (tiny) evolution
# ===================================================================

class TestRealEvolutionSmoke:

    def test_valid_theta_gives_finite_loglike(self):
        m = PWNModel()
        # young system, fast evolution
        m.set_pulsar(period=0.1 * u.s, p_dot=1e-11)
        m.tau_sd.value = 100.0 * u.yr
        m.n_elec_bins = 100
        m.n_phot_bins = 100
        m.n_ic_target = 50
        m.esn.frozen = False
        m.esn.bounds = (1e50 * u.erg, 1e52 * u.erg)
        obs = [Observable.flux_density(1.4, 'GHz', 1.0, 0.5, 'Jy'),
               Observable.angular_size_pwn(1.0, 0.5, 'arcmin')]
        spec = F.build_fit_spec(m, obs)
        ll = F.log_likelihood(np.array([0.0]), spec)
        assert np.isfinite(ll)
        assert ll <= 0.0
