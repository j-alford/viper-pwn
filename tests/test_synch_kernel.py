"""Synchrotron emission kernels."""

import os
os.environ.setdefault('VIPER_CONSTANTS', 'gsz2009')  # noqa: E402

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import kv

from viper import constants as C
from viper import radiation as R
from viper.evolution import evolve_pwn
from viper.grids import make_log_grid


def _fx(x):
    return x * quad(lambda t: kv(5.0 / 3.0, t), x, np.inf)[0]


def _g_quadrature(x):
    """pitch-angle average of F"""
    val, _ = quad(lambda a: 0.5 * np.sin(a) ** 2 * _fx(x / np.sin(a)),
                  1e-4, np.pi - 1e-4, limit=200, epsabs=0.0, epsrel=1e-8)
    return val


class TestGTable:

    def test_matches_pitch_angle_quadrature(self):
        tx, ty = R._get_gx_table()
        for x in (1e-5, 1e-3, 0.03, 0.23, 1.0, 3.0, 6.0):
            table = np.interp(np.log10(x), tx, ty)
            assert table == pytest.approx(_g_quadrature(x), rel=2e-3), x

    def test_peak_location_and_value(self):
        # AKP 2010: max G = 0.7126 at x = 0.2292
        tx, ty = R._get_gx_table()
        i = np.argmax(ty)
        assert 10.0 ** tx[i] == pytest.approx(0.2292, rel=0.01)
        assert ty[i] == pytest.approx(0.7126, rel=1e-3)

    def test_akp10_d7_approximation_agrees(self):
        # AKP 2010 Eq. D7, quoted at 0.2%
        tx, ty = R._get_gx_table()
        x = np.logspace(-5, 1.2, 60)
        cb = np.cbrt(x)
        d7 = (1.808 * cb / np.sqrt(1.0 + 3.4 * cb ** 2)
              * (1.0 + 2.21 * cb ** 2 + 0.347 * cb ** 4)
              / (1.0 + 1.353 * cb ** 2 + 0.217 * cb ** 4) * np.exp(-x))
        table = np.interp(np.log10(x), tx, ty)
        assert np.max(np.abs(d7 / table - 1.0)) < 3e-3

    def test_total_power_two_thirds_of_fixed_pitch(self):
        # ∫G dx = (2/3) ∫F dx = (2/3)(8π/(9√3))
        tx, ty = R._get_gx_table()
        x = 10.0 ** tx
        total = np.trapezoid(ty, x)
        assert total == pytest.approx(2.0 / 3.0 * 8.0 * np.pi / (9.0 * np.sqrt(3.0)),
                                      rel=2e-3)


class TestKernelRegistry:

    def test_names_and_default(self):
        assert set(R.SYNCH_KERNELS) == {'isotropic', 'rms_pitch', 'legacy'}
        assert R.DEFAULT_SYNCH_KERNEL == 'isotropic'
        with pytest.raises(ValueError, match="synch_kernel"):
            R.get_synch_kernel('F')

    def test_legacy_aliases(self):
        log_amp, log_nu, _, _ = R.get_synch_kernel('legacy')
        assert log_amp == pytest.approx(R._LOG_SYNCH_CONST_B0, abs=1e-12)
        assert log_nu == pytest.approx(R._LOG_NU_CRIT_OFFSET, abs=1e-12)

    def test_isotropic_is_standard_critical_frequency(self):
        _, log_nu, _, _ = R.get_synch_kernel('isotropic')
        # ν_c = 3 e B γ² / (4π m c) at B = 1 G, E = 1 erg
        nu_c = 3.0 * C.e_charge / (4.0 * np.pi * C.m_electron * C.c_light) \
            / (C.m_electron * C.c_light ** 2) ** 2
        assert 10.0 ** log_nu == pytest.approx(nu_c, rel=1e-12)

    @pytest.mark.parametrize('kernel, expected', [
        ('isotropic', 1.0), ('rms_pitch', 1.0), ('legacy', np.sqrt(2.0 / 3.0))])
    def test_photon_spectrum_power_vs_cooling(self, kernel, expected):
        """emitted power vs c_synch E²"""
        elec = make_log_grid(3.0e-5, 1.0e6, 200)
        phot = make_log_grid(1.0e2, 1.0e25, 600)
        ibin = 110
        elec.number[ibin] = 1.0
        elec.update_log_number()
        bpwn = 3.0e-5
        pnu = np.full((elec.n_bins, phot.n_bins), -9999.0)
        pie = np.zeros(elec.n_bins)
        R.calc_photon_spectrum(elec, phot, bpwn, 1.0e18, 1.0e18, 0.0, 0,
                               pnu, pie, 1.0, snapshot_mode='instantaneous',
                               synch_kernel=kernel)
        emitted = np.sum(phot.number * phot.bin_width) * 1.0e40
        cs = R.csynch_calc_vec(elec.log_avg[ibin:ibin + 1], bpwn, 1e-30,
                               np.log10(3e18))[0]
        e_erg = 10.0 ** (elec.log_avg[ibin] - C.log_erg)
        assert emitted / (cs * e_erg ** 2 * 1.0e40) == pytest.approx(expected,
                                                                    rel=5e-3)


SMALL_KW = dict(
    esn_51=1.0, mej=6.0, n_ism=0.4,
    braking_index=2.5, tau_sd=3000.0, age=2000.0, e0_37=28.0,
    vel_psr=0.0, eta_g=0.0, eta_b=0.03,
    e_min=3.0, e_max=5.0e5, e_break=50.0, p1=0.8, p2=2.5,
    n_elec_bins=150, n_phot_bins=150, n_ic_target=80,
    dyn_step=-1,
)


class TestEvolveIntegration:

    @pytest.fixture(scope='class')
    def iso(self):
        return evolve_pwn(**SMALL_KW)

    def test_default_equals_explicit_isotropic(self, iso):
        again = evolve_pwn(**SMALL_KW, synch_kernel='isotropic')
        np.testing.assert_array_equal(iso.photon_spectrum, again.photon_spectrum)

    def test_rejects_unknown_value(self):
        with pytest.raises(ValueError, match="synch_kernel"):
            evolve_pwn(**SMALL_KW, synch_kernel='bogus')

    def test_cpp_compat_forces_legacy(self):
        a = evolve_pwn(**SMALL_KW, cpp_compat=True, synch_kernel='isotropic')
        b = evolve_pwn(**SMALL_KW, cpp_compat=True, synch_kernel='legacy')
        np.testing.assert_array_equal(a.photon_spectrum, b.photon_spectrum)

    def test_kernel_changes_synchrotron_only(self, iso):
        leg = evolve_pwn(**SMALL_KW, synch_kernel='legacy')
        # emission only
        np.testing.assert_array_equal(iso.electron_spectrum, leg.electron_spectrum)
        assert iso.dyninfo[-1].bpwn == leg.dyninfo[-1].bpwn
        # radio band: measured 0.83-0.85
        f = iso.photon_freqs
        radio = (f > 1e9) & (f < 1e11) & (leg.photon_spectrum > 0)
        ratio = leg.photon_spectrum[radio] / iso.photon_spectrum[radio]
        assert np.all((ratio > 0.80) & (ratio < 0.87))
        # IC band: identical
        ic = (f > 1e25) & (iso.photon_spectrum > 0)
        np.testing.assert_allclose(leg.photon_spectrum[ic],
                                   iso.photon_spectrum[ic], rtol=1e-12)


class TestSourceAndGammapy:

    def test_source_option_default(self):
        from viper.source import OPTION_DEFAULTS
        assert OPTION_DEFAULTS['synch_kernel'] == 'isotropic'

    def test_gammapy_extra_attr(self):
        gammapy = pytest.importorskip('gammapy')  # noqa: F841
        from viper.gammapy import PWNSpectralModel
        m = PWNSpectralModel(synch_kernel='legacy')
        assert m.synch_kernel == 'legacy'
        assert 'synch_kernel' in m._EXTRA_ATTRS
