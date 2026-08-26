"""Synchrotron self-Compton tests."""

import os
os.environ.setdefault('VIPER_CONSTANTS', 'gsz2009')  # noqa: E402

import pickle

import numpy as np
import pytest

from viper import constants as C
from viper.evolution import evolve_pwn, _fill_target_field
from viper.grids import make_log_grid
from viper import radiation as R

# ---------------------------------------------------------------------------
# Small-grid kernel fixtures (fast: no evolutions)
# ---------------------------------------------------------------------------

NE, NP, NT = 152, 152, 102


def _kernel_grids():
    elec = make_log_grid(3.0e-5, 1.0e6, NE)
    phot = make_log_grid(C.PHOT_FREQ_MIN, C.PHOT_FREQ_MAX, NP)
    tgt = make_log_grid(C.TARGET_FREQ_MIN, C.TARGET_FREQ_MAX, NT)
    return elec, phot, tgt


class TestCoolingMatrix:
    """K2[i,k] = photon-axis-summed KN kernel."""

    def test_k2_matches_kernel_path(self):
        # summation-order rounding tolerance
        elec, phot, tgt = _kernel_grids()
        _fill_target_field(tgt, NT - 2, 2, [35.0, 3000.0], [0.35, 0.5])

        pie_kernel = R.calc_pow_ic(
            phot, R.calc_sigma_and_pow_nu_ic(elec, phot, tgt))
        k2 = R.ic_cooling_matrix(elec, phot, tgt)
        pie_k2 = k2 @ (tgt.number * tgt.bin_width)

        mask = (pie_kernel > 0) | (pie_k2 > 0)
        assert np.any(mask)
        np.testing.assert_allclose(
            pie_k2[mask], pie_kernel[mask], rtol=1e-8,
            err_msg="K2 cooling matrix disagrees with the validated "
                    "kernel path beyond summation-order rounding")

    @pytest.mark.skipif(not R.HAS_NUMBA, reason="numba not installed")
    def test_backends_agree(self):
        elec, phot, tgt = _kernel_grids()
        k2_nb = R.ic_cooling_matrix(elec, phot, tgt)
        k2_np = R._ic_cooling_matrix_numpy(elec, phot, tgt)
        mask = (k2_nb > 0) | (k2_np > 0)
        np.testing.assert_allclose(k2_nb[mask], k2_np[mask], rtol=1e-12)

    def test_thomson_limit(self):
        # P = (4/3) σ_T c γ² u
        elec, phot, _ = _kernel_grids()
        tgt = make_log_grid(1.0e9, 1.0e15, NT)  # resolve the 30 K peak
        t_seed = 30.0
        for k in range(NT):
            tgt.number[k] = R.blackbody_number_density(tgt.bin_avg[k],
                                                       t_seed)
        tgt.update_log_number()
        u_seed = np.sum(tgt.number * C.h_cgs * tgt.bin_avg * tgt.bin_width)

        k2 = R.ic_cooling_matrix(elec, phot, tgt)
        p_k2 = k2 @ (tgt.number * tgt.bin_width)

        for e_gev in (0.3, 1.0, 3.0):
            i = int(np.argmin(np.abs(elec.bin_avg - e_gev)))
            gamma = 10.0 ** (elec.log_avg[i] - C.log_erg) / (
                C.m_electron * C.c_light ** 2)
            p_thomson = (4.0 / 3.0) * C.sigma_T * C.c_light * gamma ** 2 \
                * u_seed
            assert p_k2[i] == pytest.approx(p_thomson, rel=0.05), (
                f"E={elec.bin_avg[i]:.3g} GeV (γ={gamma:.0f}): K2 power "
                f"{p_k2[i]:.4e} vs Thomson {p_thomson:.4e}")


class TestSynchSeed:
    """Instantaneous synchrotron seed."""

    def test_seed_normalization(self):
        # emission-convention √(2/3) identity
        wide = make_log_grid(1.0e2, 1.0e25, 400)
        sqrt23 = np.sqrt(2.0 / 3.0)
        for bpwn in (1.0e-5, 1.0e-4):
            for ibin in (60, 90, 120):
                elec = make_log_grid(3.0e-5, 1.0e6, NE)
                elec.number[ibin] = 1.0
                elec.update_log_number()
                l40 = R.synch_seed_lum(elec, bpwn, wide)
                p_table = np.sum(l40 * wide.bin_width)
                cs = R.csynch_calc_vec(
                    elec.log_avg[ibin:ibin + 1], bpwn, 1e-30,
                    np.log10(3e18))[0]
                e_erg = 10.0 ** (elec.log_avg[ibin] - C.log_erg)
                ratio = p_table / (cs * e_erg ** 2)
                assert ratio == pytest.approx(sqrt23, rel=0.01), (
                    f"B={bpwn:.1e} G, E={elec.bin_avg[ibin]:.3g} GeV: "
                    f"∫L_ν dν / (c_synch E²) = {ratio:.5f}, expected "
                    f"√(2/3) = {sqrt23:.5f}")

    @pytest.mark.skipif(not R.HAS_NUMBA, reason="numba not installed")
    def test_backends_agree(self):
        elec, _, tgt = _kernel_grids()
        elec.number[:] = np.random.RandomState(7).lognormal(0.0, 2.0, NE)
        elec.update_log_number()
        bpwn = 1.0e-4
        a = R.synch_seed_lum(elec, bpwn, tgt)
        b = 10.0 ** (R._LOG_SYNCH_CONST_B0 + np.log10(bpwn)) \
            * R._synch_seed_lum_numpy(
                elec.log_avg, elec.number, np.log10(bpwn), tgt.log_avg,
                R._LOG_NU_CRIT_OFFSET, C.log_erg)
        mask = (a > 0) | (b > 0)
        np.testing.assert_allclose(a[mask], b[mask], rtol=1e-9)

    def test_seed_density_geometry(self):
        elec, _, tgt = _kernel_grids()
        elec.number[:] = np.random.RandomState(3).lognormal(0.0, 2.0, NE)
        elec.update_log_number()
        bpwn, rad = 1.0e-4, 3.0e18

        n_nu = R.ssc_seed_density(elec, bpwn, rad, tgt)
        u_tot = np.sum(n_nu * C.h_cgs * tgt.bin_avg * tgt.bin_width)
        l_tot = np.sum(R.synch_seed_lum(elec, bpwn, tgt) * 1e40
                       * tgt.bin_width)
        expect = R.SSC_GEOM_FACTOR * l_tot / (
            4.0 * np.pi * rad ** 2 * C.c_light)
        assert u_tot == pytest.approx(expect, rel=1e-12)

        assert not np.any(R.ssc_seed_density(elec, 0.0, rad, tgt))
        assert not np.any(R.ssc_seed_density(elec, bpwn, 0.0, tgt))


# ---------------------------------------------------------------------------
# End-to-end evolutions (reduced grids to stay fast)
# ---------------------------------------------------------------------------

# Crab-like quick test
CRAB_KW = dict(
    esn_51=1.0, mej=9.0, n_ism=0.1,
    braking_index=2.51, tau_sd=700.0, age=970.0, e0_37=310.0,
    vel_psr=0.0, eta_g=0.0, eta_b=0.03,
    e_min=1.0, e_max=2.0e6, e_break=300.0, p1=1.5, p2=2.5,
    n_elec_bins=200, n_phot_bins=200, n_ic_target=100,
    dyn_step=-1,
)

# all-epoch-weak SSC, pre-RS-collision
FENCE_KW = dict(
    esn_51=1.0, mej=6.0, n_ism=0.4,
    braking_index=2.5, tau_sd=3000.0, age=2000.0, e0_37=28.0,
    vel_psr=0.0, eta_g=0.0, eta_b=0.9,
    e_min=3.0, e_max=5.0e5, e_break=50.0, p1=0.8, p2=2.5,
    n_elec_bins=150, n_phot_bins=150, n_ic_target=80,
)

# CTA1-like: SSC weak today only
WEAK_KW = dict(
    esn_51=1.0, mej=6.0, n_ism=0.4,
    braking_index=2.5, tau_sd=3000.0, age=15532.0, e0_37=28.0,
    vel_psr=0.0, eta_g=0.0, eta_b=0.01,
    e_min=3.0, e_max=5.0e5, e_break=50.0, p1=0.8, p2=2.5,
    n_elec_bins=150, n_phot_bins=150, n_ic_target=80,
)


def _band_lum(result, lo_tev, hi_tev):
    e_tev = result.photon_freqs * C.h_eV / 1e12
    m = (e_tev >= lo_tev) & (e_tev <= hi_tev)
    return float(np.sum(result.photon_spectrum[m]
                        * result.phot_grid.bin_width[m]))


class TestEvolveIntegration:

    @pytest.fixture(scope='class')
    def crab_off(self):
        return evolve_pwn(**CRAB_KW)

    @pytest.fixture(scope='class')
    def crab_on(self):
        return evolve_pwn(**CRAB_KW, ssc_model='uniform')

    def test_default_equals_explicit_off(self, crab_off):
        again = evolve_pwn(**CRAB_KW, ssc_model='off')
        assert np.array_equal(again.photon_spectrum,
                              crab_off.photon_spectrum)
        assert np.array_equal(again.electron_spectrum,
                              crab_off.electron_spectrum)

    def test_rejects_unknown_value(self):
        with pytest.raises(ValueError, match="ssc_model"):
            evolve_pwn(**CRAB_KW, ssc_model='bogus')

    def test_cpp_compat_forces_off(self):
        a = evolve_pwn(**CRAB_KW, cpp_compat=True, ssc_model='uniform')
        b = evolve_pwn(**CRAB_KW, cpp_compat=True, ssc_model='off')
        assert np.array_equal(a.photon_spectrum, b.photon_spectrum)
        assert np.array_equal(a.electron_spectrum, b.electron_spectrum)

    def test_crab_like_run(self, crab_off, crab_on):
        # dt sequences fork, loose bounds
        r_vhe = _band_lum(crab_on, 0.1, 10.0) / _band_lum(crab_off, 0.1, 10.0)
        assert 1.5 < r_vhe < 30.0, f"VHE on/off ratio {r_vhe:.2f}"
        r_x = (_band_lum(crab_on, 1e-9, 1e-5)
               / _band_lum(crab_off, 1e-9, 1e-5))
        assert r_x == pytest.approx(1.0, rel=0.05), (
            f"synchrotron band moved by {abs(r_x - 1):.3%}; SSC feedback "
            f"should be weak")

    def test_warns_on_compat_cooling_bound(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger='viper.evolution'):
            evolve_pwn(**dict(CRAB_KW, age=5.0), ssc_model='uniform')
        assert any("total_loss10" in r.message for r in caplog.records)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger='viper.evolution'):
            evolve_pwn(**dict(CRAB_KW, age=5.0), ssc_model='uniform',
                       timestep_cooling='total_loss10')
        assert not any("total_loss10" in r.message for r in caplog.records)

    @staticmethod
    def _replayed_pair(kw):
        # dt-matched comparison
        off = evolve_pwn(**kw, dyn_step=1)
        dts = [d.delta_t for d in off.dyninfo[:-1]]
        on = evolve_pwn(**kw, dyn_step=1, ssc_model='uniform',
                        replay_dt_sequence=dts)
        return off, on

    @staticmethod
    def _flux_weighted_shift(off, on):
        f_off, f_on = off.photon_spectrum, on.photon_spectrum
        mask = (f_off > 0) | (f_on > 0)
        w = f_off[mask] / f_off[mask].sum()
        rel = np.abs(f_on[mask] - f_off[mask]) / np.maximum(f_off[mask],
                                                            1e-300)
        return float(np.sum(w * rel))

    def test_weak_ssc_leaves_dynamics_unchanged(self):
        # null coupling beyond direct seed
        off, on = self._replayed_pair(FENCE_KW)
        assert self._flux_weighted_shift(off, on) < 1e-4
        b_off = off.dyninfo[-1].bpwn
        b_on = on.dyninfo[-1].bpwn
        assert b_on == pytest.approx(b_off, rel=1e-5), (
            "SSC-on perturbed the dynamics of an all-epoch-weak source; "
            "non-physical coupling (static-array mutation?)")

    def test_old_source_shift_is_bounded(self):
        # real percent-level shift, bounded
        off, on = self._replayed_pair(WEAK_KW)
        assert self._flux_weighted_shift(off, on) < 0.05


class TestWiring:
    """run_config → model → _FitSpec → evolve args."""

    def _model(self):
        from viper.model import PWNModel
        m = PWNModel()
        m.period._set_internal(0.1)
        m.p_dot._set_internal(1e-13)
        return m

    def test_model_default_and_validate(self):
        m = self._model()
        assert m.ssc_model == 'off'
        m.validate()
        m.ssc_model = 'uniform'
        m.validate()
        m.ssc_model = 'bogus'
        with pytest.raises(ValueError, match="ssc_model"):
            m.validate()

    def test_apply_run_config(self):
        m = self._model()
        m.apply_run_config({'ssc_model': 'uniform'})
        assert m.ssc_model == 'uniform'
        m.apply_run_config({})
        assert m.ssc_model == 'off'
        with pytest.raises(ValueError, match="ssc_model"):
            m.apply_run_config({'ssc_model': 'on'})

    def test_fit_spec_carries_and_pickles(self):
        import astropy.units as u
        from viper.fitting import _build_fit_spec, _theta_to_evolve_args
        m = self._model()
        m.ssc_model = 'uniform'
        m.esn.frozen = False
        m.esn.bounds = (1e50 * u.erg, 1e52 * u.erg)
        spec = _build_fit_spec(m, observables=[])
        assert spec.ssc_model == 'uniform'
        spec2 = pickle.loads(pickle.dumps(spec))
        assert spec2.ssc_model == 'uniform'
        theta = np.array([np.mean(spec.free_params[0].bounds)])
        args = _theta_to_evolve_args(theta, spec2)
        assert args['ssc_model'] == 'uniform'

    def test_run_kwarg_override(self):
        m = self._model()
        with pytest.raises(ValueError, match="ssc_model"):
            m.run(ssc_model='bogus')
        assert m.ssc_model == 'bogus'  # set before validate
        m.ssc_model = 'off'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
