"""One-zone particle-escape sink: registry, end-to-end behavior, run-config handling."""

import numpy as np
import pytest
import astropy.units as u

from viper.model import PWNModel
from viper import constants as C
from viper.escape import ESCAPE_MODELS, escape_tau_seconds, MEC2_GEV
from viper.fitting import build_fit_spec, _theta_to_evolve_args


def _cta1_like_model():
    """A small CTA-1-ish model (matches tests/test_sed_invariants.py)."""
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


# ===================================================================
# Registry-level unit tests (fast; no evolution)
# ===================================================================

_TEST_E = np.array([1.0, 1.0e2, 1.0e3, 1.0e4, 1.0e6])   # GeV
_TEST_ARGS = dict(B_g=3.0e-6, R_cm=3.0e18, v_cms=1.0e8, t_yr=1.0e4)


def test_registry_off_is_inf():
    tau = escape_tau_seconds('off', _TEST_E, 3e-6, 3e18, 1e8, 1e4, 1.0, 1.0)
    assert np.all(np.isinf(tau))
    assert tau.shape == _TEST_E.shape


def test_registry_finite_forms_positive():
    for name in ESCAPE_MODELS:
        if name == 'off':
            continue
        tau = escape_tau_seconds(name, _TEST_E, 3e-6, 3e18, 1e8, 1e4, 1.0, 1.0)
        assert tau.shape == _TEST_E.shape, name
        assert np.all(np.isfinite(tau)), name
        assert np.all(tau > 0), name


def test_registry_unknown_raises():
    with pytest.raises(ValueError):
        escape_tau_seconds('nonsense', _TEST_E, 3e-6, 3e18, 1e8, 1e4, 1.0, 1.0)


def test_bohm_hand_formula():
    """`bohm` tau matches the analytic e B R^2 / (2 gamma m_e c^3) (TIDE)."""
    E_gev, B, R = 1.0e3, 3.0e-6, 3.0e18
    tau = escape_tau_seconds('bohm', np.array([E_gev]), B, R, 0.0, 0.0,
                             1.0, 0.0)[0]
    gamma = E_gev / MEC2_GEV
    expected = (C.e_charge * B * R**2) / (2.0 * gamma * C.m_electron
                                          * C.c_light**3)
    assert tau == pytest.approx(expected, rel=1e-10)


def test_powerlaw_decade_ratio():
    """`powerlaw` tau ~ E^-index: a decade in E scales tau by 10^-index."""
    E = np.array([1.0e3, 1.0e4])   # 1, 10 TeV
    for index in (0.0, 0.5, 1.0, 1.5):
        tau = escape_tau_seconds('powerlaw', E, 3e-6, 3e18, 1e8, 1e4,
                                 1.0, index)
        assert tau[1] / tau[0] == pytest.approx(10.0 ** (-index), rel=1e-10)


def test_norm_scales_tau_inversely_for_bohm():
    """bohm efficiency `norm`: larger norm -> shorter escape time."""
    E = np.array([1.0e4])
    t1 = escape_tau_seconds('bohm', E, 3e-6, 3e18, 1e8, 1e4, 1.0, 0.0)[0]
    t2 = escape_tau_seconds('bohm', E, 3e-6, 3e18, 1e8, 1e4, 2.0, 0.0)[0]
    assert t2 == pytest.approx(t1 / 2.0, rel=1e-12)


# ===================================================================
# End-to-end behavior tests
# ===================================================================

def test_escape_off_is_noop():
    """escape_model='off' ignores esc params -> bit-identical to baseline."""
    res_base = _cta1_like_model().run(dyn_step=-1)

    m_off = _cta1_like_model()
    m_off.escape_model = 'off'
    m_off.parameters['esc_norm'].value = 5.0     # nonsense; must be ignored
    m_off.parameters['esc_index'].value = 2.0
    res_off = m_off.run(dyn_step=-1)

    np.testing.assert_array_equal(res_base.photon_spectrum,
                                  res_off.photon_spectrum)
    np.testing.assert_array_equal(res_base.electron_spectrum,
                                  res_off.electron_spectrum)
    assert np.all(res_off.escaped_spectrum == 0.0)


def test_escape_on_depletes_energy_dependently():
    """`bohm` escape removes high-E electrons more than low-E ones."""
    res_base = _cta1_like_model().run(dyn_step=-1)

    m_esc = _cta1_like_model()
    m_esc.escape_model = 'bohm'
    m_esc.parameters['esc_norm'].value = 1.0
    res_esc = m_esc.run(dyn_step=-1)

    E = res_base.electron_energies
    base = res_base.electron_spectrum
    esc = res_esc.electron_spectrum

    hi = (E > 1.0e4) & (base > 0)              # > 10 TeV
    lo = (E > 1.0) & (E < 1.0e2) & (base > 0)  # 1-100 GeV
    assert hi.sum() > 3 and lo.sum() > 3

    ratio_hi = esc[hi].sum() / base[hi].sum()
    ratio_lo = esc[lo].sum() / base[lo].sum()

    assert ratio_hi < 1.0                       # something escaped at high E
    assert ratio_hi < ratio_lo                  # high E depleted more
    assert res_esc.escaped_spectrum.sum() > 0.0


def test_escape_index_monotonic():
    """Steeper esc_index -> stronger high-energy depletion (`powerlaw`)."""
    def top_count(index):
        m = _cta1_like_model()
        m.escape_model = 'powerlaw'
        m.parameters['esc_norm'].value = 0.1
        m.parameters['esc_index'].value = index
        res = m.run(dyn_step=-1)
        E = res.electron_energies
        return res.electron_spectrum[E > 1.0e4].sum()

    assert top_count(1.5) < top_count(0.5)


def test_escape_feeds_dynamics():
    """Escape removes particle energy and perturbs dynamics; radius sign unasserted (post-RS)."""
    res_base = _cta1_like_model().run(dyn_step=-1)
    d_base = res_base.dyninfo[-1]

    m_esc = _cta1_like_model()
    m_esc.escape_model = 'constant'
    m_esc.parameters['esc_norm'].value = 0.1   # tau ~ 1e3 yr << age
    res_esc = m_esc.run(dyn_step=-1)
    d_esc = res_esc.dyninfo[-1]

    # escaped energy left the budget
    assert d_esc.epwnp_50 < d_base.epwnp_50
    # dynamics responded
    assert d_esc.rad_pwn != d_base.rad_pwn
    # escaped reservoir holds positive energy
    E = res_esc.electron_energies
    esc_energy = float(np.sum(res_esc.escaped_spectrum * E) / 10.0 ** C.log_erg)
    assert esc_energy > 0.0


def test_escape_records_escinfo_snapshots():
    """escinfo: empty when off; parallel to elecinfo + cumulative when on."""
    # OFF: no escinfo
    res_off = _cta1_like_model().run(elec_step=40)
    assert res_off.escinfo == []
    assert len(res_off.elecinfo) > 0

    # ON: parallel to elecinfo
    m = _cta1_like_model()
    m.escape_model = 'bohm'
    m.parameters['esc_norm'].value = 1.0
    res_on = m.run(elec_step=40)

    assert len(res_on.escinfo) == len(res_on.elecinfo)
    assert [t for t, _ in res_on.escinfo] == [t for t, _ in res_on.elecinfo]
    totals = [float(np.asarray(s).sum()) for _, s in res_on.escinfo]
    assert all(totals[i] <= totals[i + 1] + 1e-30
               for i in range(len(totals) - 1))
    assert totals[-1] > 0.0
    np.testing.assert_array_equal(res_on.escinfo[-1][1],
                                  res_on.escaped_spectrum)


# ===================================================================
# Configuration tests
# ===================================================================

def test_escape_param_and_toggle_roundtrip():
    """esc_norm/esc_index + escape_model flow model -> spec -> evolve args."""
    m = _cta1_like_model()
    m.escape_model = 'diffusion'
    m.parameters['esc_norm'].frozen = False
    m.parameters['esc_norm'].bounds = (0.01, 100.0)     # linear; log-sampled
    m.parameters['esc_index'].frozen = False
    m.parameters['esc_index'].bounds = (-1.0, 2.0)
    m.parameters['esc_norm'].value = 3.0
    m.parameters['esc_index'].value = 1.0

    spec = build_fit_spec(m, [])
    assert spec.escape_model == 'diffusion'
    assert 'esc_norm' in spec.param_names and 'esc_index' in spec.param_names

    theta = m.parameters.free_fit_values
    kw = _theta_to_evolve_args(theta, spec)
    assert kw['escape_model'] == 'diffusion'
    assert kw['esc_norm'] == pytest.approx(3.0, rel=1e-6)
    assert kw['esc_index'] == pytest.approx(1.0, rel=1e-6)

    # esc_norm log; esc_index linear
    i_norm = spec.param_names.index('esc_norm')
    assert spec.free_params[i_norm].log is True
    assert theta[i_norm] == pytest.approx(np.log10(3.0), rel=1e-10)
    i_idx = spec.param_names.index('esc_index')
    assert spec.free_params[i_idx].log is False


def test_escape_params_frozen_by_default():
    """esc_norm/esc_index must NOT join the fit unless explicitly unfrozen."""
    m = _cta1_like_model()
    m.parameters['p1'].frozen = False
    m.parameters['p1'].bounds = (0.5, 2.5)
    spec = build_fit_spec(m, [])
    assert 'esc_norm' not in spec.param_names
    assert 'esc_index' not in spec.param_names


def test_cpp_compat_forces_escape_off():
    """cpp_compat=True forces escape off (not part of the original model)."""
    res_off = _cta1_like_model().run(dyn_step=-1, cpp_compat=True)

    m_esc = _cta1_like_model()
    m_esc.escape_model = 'bohm'
    m_esc.parameters['esc_norm'].value = 1.0
    res_esc = m_esc.run(dyn_step=-1, cpp_compat=True)

    np.testing.assert_array_equal(res_off.photon_spectrum,
                                  res_esc.photon_spectrum)
    assert np.all(res_esc.escaped_spectrum == 0.0)
