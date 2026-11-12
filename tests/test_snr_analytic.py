"""SNR dynamics against the Truelove & McKee (1999) closed forms."""
import numpy as np
import pytest

import viper.constants as C
from viper import snr
from viper.evolution import evolve_pwn

# TM99 Table 6, n = 9
N = 9
L_ED, PHI_ED = 1.19, 0.55
T_ST, R_ST = 0.523, 0.725
T_CORE, R_CORE, V_CORE, A_CORE = 0.249, 0.371, 0.497, 0.162
XI_0 = 2.026                                      # eq. 53

ESN, MEJ, N_ISM = 1.0, 8.0, 0.1


def _cn_amplitude(n=N):
    """eq. 75 prefactor."""
    return (27.0 / (4.0 * np.pi) / (n * (n - 3.0)) * L_ED ** (n - 2.0) / PHI_ED
            * (10.0 / 3.0 * (n - 5.0) / (n - 3.0)) ** ((n - 3.0) / 2.0)) ** (1.0 / n)


def rb_star(t):
    """eqs. 75 and 57."""
    t = np.asarray(t, float)
    a = _cn_amplitude()
    ed = a * t ** ((N - 3.0) / N)
    st = np.maximum(R_ST ** 2.5 + np.sqrt(XI_0) * (t - T_ST), 1e-300) ** 0.4
    return np.where(t < T_ST, ed, st)


def vb_star(t):
    """eqs. 76 and 58."""
    t = np.asarray(t, float)
    a = _cn_amplitude()
    ed = (N - 3.0) / N * a * t ** (-3.0 / N)
    st = 0.4 * np.sqrt(XI_0) * np.maximum(R_ST ** 2.5 + np.sqrt(XI_0) * (t - T_ST), 1e-300) ** (-0.6)
    return np.where(t < T_ST, ed, st)


def rr_star(t):
    """R_b / l_ED before t_core, eq. 83 after."""
    t = np.asarray(t, float)
    early = rb_star(t) / L_ED
    with np.errstate(divide='ignore', invalid='ignore'):
        late = t * (R_CORE / T_CORE - A_CORE * (t - T_CORE)
                    - (V_CORE - A_CORE * T_CORE) * np.log(t / T_CORE))
    return np.where(t < T_CORE, early, late)


def _scales(esn=ESN, mej=MEJ, n_ism=N_ISM):
    """eqs. 1-3 with the code's rho_0 convention."""
    rho0 = n_ism * C.m_proton * C.mu_ism
    m = mej * C.M_sun
    e = esn * 1.0e51
    r_ch = m ** (1.0 / 3.0) * rho0 ** (-1.0 / 3.0)
    t_ch = e ** -0.5 * m ** (5.0 / 6.0) * rho0 ** (-1.0 / 3.0)
    return r_ch, t_ch, r_ch / t_ch


def test_table6_constants_are_the_code_constants():
    assert (C.m_ejecta_index, C.l_ed, C.phi_ed) == (N, L_ED, PHI_ED)
    assert (C.tst_star, C.rst_star, C.psi_0) == (T_ST, R_ST, XI_0)
    assert (C.tcore_star, C.rrscore_star, C.vrscore_star, C.arscore_star) == (
        T_CORE, R_CORE, V_CORE, A_CORE)


def test_table6_transition_is_consistent_with_eq75():
    """R_ST = A t_ST^((n-3)/n), eq. 82."""
    assert abs(_cn_amplitude() * T_ST ** ((N - 3.0) / N) / R_ST - 1.0) < 1e-3


def test_characteristic_scales():
    r_ch, t_ch, v_ch = snr.calc_characteristic_scales(ESN, MEJ, N_ISM)
    ref = _scales()
    assert np.allclose((r_ch, t_ch, v_ch), ref, rtol=1e-12)


@pytest.mark.parametrize('field', ['rad', 'vel', 'rs'])
def test_snr_module_matches_tm99(field):
    r_ch, t_ch, v_ch = _scales()
    ts = np.logspace(-2, np.log10(2.5), 400)
    ts = np.sort(np.concatenate([ts, [T_CORE, T_ST]]))
    yrs = ts * t_ch / C.yr
    if field == 'rad':
        code = np.array([snr.calc_rad_snr(y, t_ch, r_ch) for y in yrs]) / r_ch
        ref = rb_star(ts)
    elif field == 'vel':
        code = np.array([snr.calc_vel_snr(y, t_ch, v_ch) for y in yrs]) / v_ch
        ref = vb_star(ts)
    else:
        rads = [snr.calc_rad_snr(y, t_ch, r_ch) for y in yrs]
        code = np.array([snr.calc_rad_rs(y, t_ch, r_ch, rb) for y, rb in zip(yrs, rads)]) / r_ch
        ref = rr_star(ts)
    assert np.allclose(code, ref, rtol=1e-12, atol=0)


def test_blast_wave_continuous_at_t_st():
    """Table 6 rounding limits this to ~2e-3."""
    eps = 1e-9
    assert abs(rb_star(T_ST - eps) / rb_star(T_ST + eps) - 1.0) < 1e-3
    assert abs(vb_star(T_ST - eps) / vb_star(T_ST + eps) - 1.0) < 3e-3


def test_reverse_shock_continuous_at_t_core():
    eps = 1e-9
    assert abs(rr_star(T_CORE - eps) / rr_star(T_CORE + eps) - 1.0) < 1e-3


# --- full model, pulsar power negligible --------------------------------

def _rows(result):
    rows = result.dyninfo
    t = np.array([x.time for x in rows])
    keep = np.ones(len(rows), bool)
    keep[1:] = np.diff(t) > 0
    return [x for x, k in zip(rows, keep) if k and x.time > 0]


@pytest.fixture(scope='module', params=['euler', 'pc_full'])
def quiet_pulsar(request):
    r = evolve_pwn(
        integration=request.param, esn_51=ESN, mej=MEJ, n_ism=N_ISM,
        braking_index=3.0, tau_sd=1.0e4, age=20000.0, e0_37=1.0e-20,
        vel_psr=0.0, eta_g=0.0, eta_b=1.0e-3, e_min=10.0, e_max=1.0e3,
        e_break=100.0, p1=1.5, p2=2.5, n_ic=0, ic_temps=[], ic_norms=[],
        dyn_step=1, elec_step=-1, phot_step=-1, snapshot_mode='instantaneous',
    )
    return _rows(r)


@pytest.mark.slow
def test_quiet_pulsar_snr_follows_tm99(quiet_pulsar):
    r_ch, t_ch, v_ch = _scales()
    t = np.array([x.time for x in quiet_pulsar]) * C.yr / t_ch
    rb = np.array([x.rad_snr for x in quiet_pulsar]) * C.pc / r_ch
    vb = np.array([x.vsnr for x in quiet_pulsar]) * 1.0e5 / v_ch
    rr = np.array([x.rad_rs for x in quiet_pulsar]) * C.pc / r_ch
    assert t.max() > 2.0 and (t < T_CORE).sum() > 3 and (t > T_ST).sum() > 3
    assert np.allclose(rb, rb_star(t), rtol=1e-10, atol=0)
    assert np.allclose(vb, vb_star(t), rtol=1e-10, atol=0)
    pos = rr_star(t) > 0
    assert np.allclose(rr[pos], rr_star(t)[pos], rtol=1e-10, atol=0)


@pytest.mark.slow
def test_quiet_pulsar_reverse_shock_runs_past_centre(quiet_pulsar):
    """Registered: eq. 83 is applied beyond R_r = 0 (lhs_060 fork)."""
    rr = np.array([x.rad_rs for x in quiet_pulsar])
    assert rr[-1] < 0
