"""In-loop radiation coefficients against closed forms and quadrature."""
import numpy as np
import pytest
from scipy.integrate import quad

import viper.constants as C
from viper import radiation as R
from viper.evolution import _fill_target_field, evolve_pwn
from viper.grids import make_log_grid
from viper.radiative import kernels as K

from _analytic import (ic_thomson_J, ic_thomson_blackbody_lnu, jones_loss_rate,
                       logslope, mc2, pitch_angle_factor, powerlaw_bin_counts,
                       synch_fx_quad, synch_nu_c, synch_powerlaw_integral,
                       synch_powerlaw_lnu, thomson_loss_rate)

SIG = C.sigma_T / K.SIGMA_T          # standalone module builds sigma_T from r0
NT = C.N_IC_TARGET
P = 2.5
K_NORM = 1.0
NOIC = -9999.0


def _powerlaw_grid(e_lo_gev, e_hi_gev, n_bins, p=P, k=K_NORM):
    g = make_log_grid(e_lo_gev / 3.0, e_hi_gev * 3.0, n_bins)
    g.number[:] = powerlaw_bin_counts(g, k, p, e_lo_gev / C.GeV_per_erg,
                                      e_hi_gev / C.GeV_per_erg)
    g.update_log_number()
    return g


def _target(n_target=NT, n_ic=0, temps=(), norms=()):
    t = make_log_grid(C.TARGET_FREQ_MIN, C.TARGET_FREQ_MAX, n_target + 2)
    _fill_target_field(t, n_target, n_ic, list(temps), list(norms))
    return t


def _blackbody_target(temp, n_target=NT):
    t = make_log_grid(C.TARGET_FREQ_MIN, C.TARGET_FREQ_MAX, n_target + 2)
    for i in range(n_target):
        t.number[i] = R.blackbody_number_density(t.bin_avg[i], temp)
    t.update_log_number()
    return t


def _ic_tables(elec, phot, target):
    pnu = R.calc_sigma_and_pow_nu_ic(elec, phot, target)
    return pnu, R.calc_pow_ic(phot, pnu)


def _emit(elec, phot, b_gauss, pnu, pie, kind='isotropic', r_cm=1e18):
    phot.reset_number()
    R.calc_photon_spectrum(elec, phot, b_gauss, r_cm, r_cm, 0.0, 0, pnu, pie,
                           1.0, snapshot_mode='instantaneous',
                           synch_kernel=kind)
    return phot.number * 1e40


def _synch_only(elec, phot, b_gauss, kind='isotropic'):
    pnu = np.full((elec.n_bins, phot.n_bins), NOIC)
    return _emit(elec, phot, b_gauss, pnu, np.zeros(elec.n_bins), kind)


def _band(phot, lo, hi, lnu):
    return (phot.bin_avg > lo) & (phot.bin_avg < hi) & (lnu > 0)


# --- synchrotron ---------------------------------------------------------

class TestSynchrotronPowerLaw:

    E_LO, E_HI, B = 1.0, 1.0e6, 1.0e-4

    @pytest.mark.parametrize('p', [1.5, 2.5, 3.0])
    def test_closed_forms_match_quadrature(self, p):
        """Gamma form of the F integral; isotropic pitch-angle average."""
        i_quad = quad(lambda x: x ** ((p - 3) / 2) * synch_fx_quad(x), 0, 40,
                      limit=200)[0]
        assert i_quad == pytest.approx(synch_powerlaw_integral(p), rel=1e-8)
        avg = 0.5 * quad(lambda a: np.sin(a) ** ((p + 3) / 2), 0, np.pi)[0]
        assert avg == pytest.approx(pitch_angle_factor(p, 'isotropic'),
                                    rel=1e-8)

    def _band(self, phot, lnu):
        lo = 30.0 * synch_nu_c(self.E_LO / C.GeV_per_erg, self.B)
        hi = 1e-3 * synch_nu_c(self.E_HI / C.GeV_per_erg, self.B)
        return _band(phot, lo, hi, lnu)

    @pytest.mark.parametrize('kind', ['isotropic', 'rms_pitch', 'legacy'])
    def test_amplitude_and_slope(self, kind):
        """L_nu of a power law vs RL 6.36 with the kernel's pitch-angle factor."""
        elec = _powerlaw_grid(self.E_LO, self.E_HI, 502)
        phot = make_log_grid(1e6, 1e29, 1002)
        lnu = _synch_only(elec, phot, self.B, kind)
        m = self._band(phot, lnu)
        assert m.sum() > 200
        ratio = lnu[m] / synch_powerlaw_lnu(phot.bin_avg[m], K_NORM, P,
                                            self.B, kind)
        assert abs(np.median(ratio) - 1.0) < 5e-4
        assert abs(ratio - 1.0).max() < 2e-3
        slope = logslope(phot.bin_avg, lnu, phot.bin_avg[m].min(),
                         phot.bin_avg[m].max())
        assert slope == pytest.approx(-(P - 1) / 2, abs=1e-3)

    def test_excess_is_second_order_in_bin_width(self):
        """Median excess shrinks faster than 1/n_bins."""
        ex = []
        for ne in (252, 1002):
            elec = _powerlaw_grid(self.E_LO, self.E_HI, ne)
            phot = make_log_grid(1e6, 1e29, 1002)
            lnu = _synch_only(elec, phot, self.B)
            m = self._band(phot, lnu)
            ex.append(np.median(lnu[m] / synch_powerlaw_lnu(
                phot.bin_avg[m], K_NORM, P, self.B)) - 1.0)
        assert 0 < ex[1] < ex[0] < 1e-3
        assert ex[0] > 4.0 * ex[1]

    @pytest.mark.parametrize('kind', ['rms_pitch', 'legacy'])
    def test_kernel_factor_relative_to_isotropic(self, kind):
        elec = _powerlaw_grid(self.E_LO, self.E_HI, 502)
        phot = make_log_grid(1e6, 1e29, 1002)
        iso = _synch_only(elec, phot, self.B, 'isotropic')
        other = _synch_only(elec, phot, self.B, kind)
        m = self._band(phot, iso)
        expect = pitch_angle_factor(P, kind) / pitch_angle_factor(P, 'isotropic')
        assert np.median(other[m] / iso[m]) == pytest.approx(expect, rel=2e-4)


# --- inverse Compton, Thomson regime ---------------------------------------

class TestThomsonICPowerLaw:

    E_LO, E_HI = 0.1, 1.0e3        # GeV; b = 4 gamma kT / mc^2 < 4e-3 on the CMB

    @pytest.mark.parametrize('p', [1.5, 2.5, 3.0])
    def test_J_matches_quadrature(self, p):
        f = lambda q: 2 * q * np.log(q) + 1 + q - 2 * q * q
        j = quad(lambda q: q ** ((p - 1) / 2) * f(q), 0, 1)[0]
        assert j == pytest.approx(ic_thomson_J(p), rel=1e-8)

    def test_amplitude_and_slope(self):
        """L_nu of a power law on the CMB vs the Planck-integrated Thomson form."""
        elec = _powerlaw_grid(self.E_LO, self.E_HI, 502)
        phot = make_log_grid(C.PHOT_FREQ_MIN, C.PHOT_FREQ_MAX, 1002)
        pnu, _ = _ic_tables(elec, phot, _target())
        lin = np.where(pnu > NOIC + 1, 10.0 ** pnu, 0.0)
        lnu = (elec.number[:, None] * lin).sum(0) * 1e40
        ev = phot.bin_avg * C.h_eV
        m = (ev > 3e4) & (ev < 3e7) & (lnu > 0)
        assert m.sum() > 100
        ratio = lnu[m] / ic_thomson_blackbody_lnu(phot.bin_avg[m], K_NORM, P,
                                                  C.T_CMB)
        assert abs(np.median(ratio) - 1.0) < 5e-3
        assert abs(ratio - 1.0).max() < 6e-3
        slope = logslope(phot.bin_avg, lnu, phot.bin_avg[m].min(),
                         phot.bin_avg[m].max())
        assert slope == pytest.approx(-(P - 1) / 2, abs=1e-3)

    def test_per_electron_power_vs_target_grid(self):
        """(4/3) sigma_T c gamma^2 u, excess falling with target resolution."""
        elec = make_log_grid(0.3, 100.0, 102)
        phot = make_log_grid(C.PHOT_FREQ_MIN, C.PHOT_FREQ_MAX, 502)
        e_erg = elec.bin_avg / C.GeV_per_erg
        u = C.radiation_const * C.T_CMB ** 4
        sel = (elec.bin_avg > 1.0) & (elec.bin_avg < 30.0)
        ex = []
        for nt in (100, 200, 400):
            _, pie = _ic_tables(elec, phot, _target(nt))
            ex.append(np.median(pie[sel] / thomson_loss_rate(e_erg[sel], u)) - 1)
        assert ex[0] > ex[1] > ex[2] > 0
        assert ex[2] < 5e-3


# --- inverse Compton, Klein-Nishina regime ----------------------------------

class TestKleinNishinaPower:

    @pytest.fixture(scope='class')
    def cmb(self):
        elec = make_log_grid(1e-3, 1e8, 502)
        phot = make_log_grid(C.PHOT_FREQ_MIN, C.PHOT_FREQ_MAX, 502)
        _, pie = _ic_tables(elec, phot, _target())
        return elec, pie

    @pytest.mark.parametrize('e_gev', [10.0, 1e3, 1e5, 3e5])
    def test_power_vs_jones_quadrature(self, cmb, e_gev):
        """Per-electron IC power vs direct quadrature of BG70 2.48."""
        elec, pie = cmb
        i = np.argmin(abs(elec.bin_avg - e_gev))
        exact = jones_loss_rate(elec.bin_avg[i] / C.GeV_per_erg, C.T_CMB)
        assert pie[i] == pytest.approx(exact, rel=1.2e-2)

    def test_power_vs_standalone_loss_rate(self, cmb):
        """Cross-module: in-loop kernel vs the closed-form blackbody loss rate."""
        elec, pie = cmb
        sel = (elec.bin_avg > 1.0) & (elec.bin_avg < 3e5)
        exact = K.ic_loss_rate_blackbody(elec.bin_avg[sel] / C.GeV_per_erg,
                                         C.T_CMB) * SIG
        assert abs(pie[sel] / exact - 1.0).max() < 1.5e-2

    def test_power_lost_above_photon_grid_ceiling(self, cmb):
        """Registered: the overflow photon bin corrupts IC power above h nu_max."""
        elec, pie = cmb
        e_erg = elec.bin_avg / C.GeV_per_erg
        exact = K.ic_loss_rate_blackbody(e_erg, C.T_CMB) * SIG
        ceiling = C.PHOT_FREQ_MAX * C.h_eV / 1e9
        assert 3e5 < ceiling < 6e5
        i6, i7 = (np.argmin(abs(elec.bin_avg - e)) for e in (1e6, 1e7))
        assert pie[i6] / exact[i6] > 10.0
        assert pie[i7] / exact[i7] < 0.5
        phot = make_log_grid(C.PHOT_FREQ_MIN, 1e32, 502)
        _, tall = _ic_tables(elec, phot, _target())
        assert tall[i6] == pytest.approx(exact[i6], rel=3e-2)

    def test_deep_kn_power_depends_on_photon_grid(self):
        """Registered: 3000 K target, b ~ 1e3, power follows the photon binning."""
        elec = make_log_grid(1e5, 9e5, 12)
        i = np.argmin(abs(elec.bin_avg - 3e5))
        exact = jones_loss_rate(elec.bin_avg[i] / C.GeV_per_erg, 3000.0)
        r = {}
        for nb in (502, 2002, 8002):
            phot = make_log_grid(C.PHOT_FREQ_MIN, 1e32, nb)
            _, pie = _ic_tables(elec, phot, _blackbody_target(3000.0))
            r[nb] = pie[i] / exact
        assert r[502] > 3.0
        assert r[2002] < 0.8
        assert r[8002] == pytest.approx(1.0, abs=6e-2)


# --- run level ---------------------------------------------------------------

RUN_KW = dict(
    esn_51=1.0, mej=6.0, n_ism=0.4, braking_index=2.5, tau_sd=3000.0,
    age=2000.0, e0_37=28.0, vel_psr=0.0, eta_g=0.0, eta_b=0.03,
    e_min=3.0, e_max=5.0e5, e_break=50.0, p1=0.8, p2=2.5,
    n_elec_bins=300, n_phot_bins=300, n_ic_target=NT, dyn_step=-1,
    snapshot_mode='instantaneous',
)

THREE_FIELD_KW = dict(
    esn_51=1.0, mej=20.0, n_ism=0.01, braking_index=2.0, tau_sd=3500.0,
    age=2241.468354, e0_37=5.20375847, vel_psr=0.0, eta_g=0.0, eta_b=0.003,
    e_min=10.0, e_max=5e5, e_break=3000.0, p1=2.3, p2=2.4,
    n_ic=2, ic_temps=[35.0, 3000.0], ic_norms=[0.35358428, 0.49622557],
    dyn_step=-1, snapshot_mode='instantaneous',
)


def _final(res):
    d = res.dyninfo[-1]
    n = res.electron_spectrum
    e_erg = res.electron_energies / C.GeV_per_erg
    return d.bpwn * 1e-6, d.rad_pwn * C.pc, n, e_erg


def _exact_ic(n, e_erg, fields):
    return sum((n * K.ic_loss_rate_blackbody(e_erg, t, u) * SIG).sum()
               for t, u in fields) * 1e40


@pytest.mark.slow
class TestRunLevel:

    @pytest.fixture(scope='class')
    def run(self):
        return evolve_pwn(**RUN_KW)

    def test_synchrotron_radiated_equals_cooled(self, run):
        """Emitted synchrotron vs summed c_synch E^2, second order in nu binning."""
        b, r_cm, n, e_erg = _final(run)
        eg = run.elec_grid
        cs = R.csynch_calc_vec(eg.log_avg, b, n.sum(), np.log10(r_cm))
        cooled = (n * cs * e_erg ** 2).sum() * 1e40
        g = eg.copy()
        g.number[:] = n
        g.update_log_number()
        ex = []
        for nb in (152, 302, 602, 1202):
            phot = make_log_grid(C.PHOT_FREQ_MIN, C.PHOT_FREQ_MAX, nb)
            lnu = _synch_only(g, phot, b)
            ex.append((lnu * phot.bin_width).sum() / cooled - 1.0)
        assert all(0 < ex[i + 1] < ex[i] for i in range(3))
        assert ex[0] > 10.0 * ex[-1]
        assert ex[-1] < 3e-4

    def test_ic_cooling_power_vs_exact(self, run):
        """Summed pow_ic_elec vs the exact CMB loss rate of the final spectrum."""
        b, r_cm, n, e_erg = _final(run)
        exact = _exact_ic(n, e_erg, [(C.T_CMB, None)])
        ex = []
        for nb in (152, 302, 1202):
            phot = make_log_grid(C.PHOT_FREQ_MIN, C.PHOT_FREQ_MAX, nb)
            _, pie = _ic_tables(run.elec_grid, phot, _target())
            ex.append((n * pie).sum() * 1e40 / exact - 1.0)
        assert ex[0] > ex[1] > ex[2] > 0
        assert ex[1] < 1e-2

    def test_final_photon_spectrum_equals_total_cooling(self, run):
        b, r_cm, n, e_erg = _final(run)
        cs = R.csynch_calc_vec(run.elec_grid.log_avg, b, n.sum(),
                               np.log10(r_cm))
        cooled = ((n * cs * e_erg ** 2).sum() * 1e40
                  + _exact_ic(n, e_erg, [(C.T_CMB, None)]))
        pg = run.phot_grid
        emitted = (run.photon_spectrum * pg.bin_width).sum() * 1e40
        assert emitted / cooled == pytest.approx(1.0, abs=3e-3)

    def test_three_field_target_and_luminosities(self):
        """Extra fields carry the requested u; synch and IC match exact losses."""
        temps, norms = THREE_FIELD_KW['ic_temps'], THREE_FIELD_KW['ic_norms']
        t = _target(NT, 2, temps, norms)
        u = (t.number * t.bin_avg * C.h_eV * t.bin_width).sum()
        u_cmb = C.radiation_const * C.T_CMB ** 4 * C.eV_per_erg
        assert u == pytest.approx(u_cmb + sum(norms), rel=1e-3)
        res = evolve_pwn(**THREE_FIELD_KW)
        b, r_cm, n, e_erg = _final(res)
        fields = [(C.T_CMB, None)] + [(tt, uu / C.eV_per_erg)
                                      for tt, uu in zip(temps, norms)]
        l_s = (n * K.synch_loss_rate(e_erg / mc2(), b) * SIG).sum() * 1e40
        l_ic = _exact_ic(n, e_erg, fields)
        pg = res.phot_grid
        lw = res.photon_spectrum * pg.bin_width * 1e40
        assert lw[pg.bin_avg < 1e20].sum() / l_s == pytest.approx(1.0, abs=1e-2)
        assert lw[pg.bin_avg > 1e22].sum() / l_ic == pytest.approx(1.0, abs=2e-2)
        assert l_ic / l_s == pytest.approx(0.0857, abs=5e-3)
