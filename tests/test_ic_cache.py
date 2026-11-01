"""viper.ic_cache: per-process IC tables (K2 matrix, blackbody field kernels)."""

import numpy as np
import pytest

from viper import constants as C
from viper import ic_cache
from viper.grids import make_log_grid, LOG_EMPTY_CUT
from viper.evolution import (evolve_pwn, _fill_target_field,
                             _cmb_lookup_compatible, _load_cmb_lookup)
from viper.radiation import calc_sigma_and_pow_nu_ic, calc_pow_ic, ic_cooling_matrix

NE, NP, NT = C.ELEC_GRID_FIXED_NBINS + 2, C.N_PHOT_FREQ + 2, C.N_IC_TARGET + 2


def _grids():
    return (make_log_grid(C.ELEC_GRID_FIXED_MIN, C.ELEC_GRID_FIXED_MAX, NE),
            make_log_grid(C.PHOT_FREQ_MIN, C.PHOT_FREQ_MAX, NP),
            make_log_grid(C.TARGET_FREQ_MIN, C.TARGET_FREQ_MAX, NT))


def _lin(k):
    return np.where(k > LOG_EMPTY_CUT, 10.0 ** k, 0.0)


@pytest.fixture(autouse=True)
def _fresh_cache():
    ic_cache.clear()
    yield
    ic_cache.clear()


def test_k2_cached_once_and_equal():
    elec, phot, tgt = _grids()
    a = ic_cache.cached_k2(elec, phot, tgt)
    b = ic_cache.cached_k2(elec, phot, tgt)
    assert a is b and not a.flags.writeable
    np.testing.assert_array_equal(a, ic_cooling_matrix(elec, phot, tgt))
    # a different grid is a different entry
    elec2 = make_log_grid(1e-5, 1e7, NE)
    assert ic_cache.cached_k2(elec2, phot, tgt) is not a


def test_fill_blackbody_matches_fill_target_field():
    _, _, tgt = _grids()
    T, u = 60.0, 0.35
    ref = make_log_grid(C.TARGET_FREQ_MIN, C.TARGET_FREQ_MAX, NT)
    _fill_target_field(ref, NT - 2, 1, [T], [u])
    cmb = make_log_grid(C.TARGET_FREQ_MIN, C.TARGET_FREQ_MAX, NT)
    _fill_target_field(cmb, NT - 2, 0, [], [])
    ic_cache.fill_blackbody(tgt, T, ic_cache.norm_dim_from_u(T, u))
    np.testing.assert_allclose(tgt.number + cmb.number, ref.number, rtol=1e-12)


def test_field_sum_exact_at_nodes():
    """At node temperatures the cached sum equals one live kernel on the
    summed field (linearity), to rounding."""
    elec, phot, tgt = _grids()
    temps = [ic_cache.node_temperature(75), ic_cache.node_temperature(190)]  # 31.6 K, 6310 K
    norms = [0.5, 0.9]
    live_tgt = make_log_grid(C.TARGET_FREQ_MIN, C.TARGET_FREQ_MAX, NT)
    _fill_target_field(live_tgt, NT - 2, 2, temps, norms)
    live = calc_sigma_and_pow_nu_ic(elec, phot, live_tgt)
    cmb_tgt = make_log_grid(C.TARGET_FREQ_MIN, C.TARGET_FREQ_MAX, NT)
    _fill_target_field(cmb_tgt, NT - 2, 0, [], [])
    cmb = calc_sigma_and_pow_nu_ic(elec, phot, cmb_tgt)
    cached = ic_cache.external_field_kernel(cmb, elec, phot, tgt, temps, norms)
    pl, pc = _lin(live), _lin(cached)
    m = pl > 1e-30 * pl.max()
    np.testing.assert_allclose(pc[m], pl[m], rtol=1e-9)
    assert np.all(pc[~m] <= 1e-30 * pl.max())
    # second call hits the cache
    assert len(ic_cache._NODES) == 2
    ic_cache.external_field_kernel(cmb, elec, phot, tgt, temps, norms)
    assert len(ic_cache._NODES) == 2


def test_field_kernel_interpolation_accuracy():
    """Mid-node temperature: per-electron IC power to 1e-3, bins to 2 %."""
    elec, phot, tgt = _grids()
    for T, u in [(ic_cache.node_temperature(75) * 10 ** 0.01, 1.2),
                 (ic_cache.node_temperature(190) * 10 ** 0.01, 0.9)]:
        ref_tgt = make_log_grid(C.TARGET_FREQ_MIN, C.TARGET_FREQ_MAX, NT)
        ic_cache.fill_blackbody(ref_tgt, T, ic_cache.norm_dim_from_u(T, u))
        ref = calc_sigma_and_pow_nu_ic(elec, phot, ref_tgt)
        k = ic_cache.field_kernel(T, u, elec, phot, tgt)
        P_ref, P_k = calc_pow_ic(phot, ref), calc_pow_ic(phot, k)
        m = P_ref > 1e-3 * P_ref.max()
        assert np.abs(P_k[m] / P_ref[m] - 1).max() < 1e-3
        pr, pk = _lin(ref), _lin(k)
        row = pr.max(axis=1, keepdims=True)
        mm = pr > 1e-4 * np.where(row > 0, row, 1.0)
        assert np.abs(pk[mm] / pr[mm] - 1).max() < 0.02


def test_zero_norm_field_is_empty():
    elec, phot, tgt = _grids()
    k = ic_cache.field_kernel(60.0, 0.0, elec, phot, tgt)
    assert np.all(k <= LOG_EMPTY_CUT)


def test_lookup_gate():
    args = (True, False, 2, 0, C.ELEC_GRID_FIXED_NBINS, C.N_PHOT_FREQ, C.N_IC_TARGET)
    ok, why = _cmb_lookup_compatible(*args)
    assert not ok and 'use_field_lookup' in why
    if _load_cmb_lookup() is None:
        pytest.skip('no CMB lookup file for this constants source')
    ok, why = _cmb_lookup_compatible(*args, use_field_lookup=True)
    assert ok, why


def test_model_and_spec_plumbing():
    from viper.model import PWNModel
    from viper.fitting import build_fit_spec
    m = PWNModel()
    assert m.use_field_lookup is False
    m.apply_run_config({'use_field_lookup': True})
    assert m.use_field_lookup and m.use_cmb_lookup
    m.period._set_internal(0.0334); m.p_dot._set_internal(4.2e-13)
    m.esn.frozen = False
    import astropy.units as u
    m.esn.bounds = (0.1e51 * u.erg, 5e51 * u.erg)
    spec = build_fit_spec(m, [])
    assert spec.use_field_lookup is True


_CRAB_LIKE = dict(
    esn_51=1.0, mej=6.0, n_ism=0.05, braking_index=2.5, tau_sd=700.0,
    age=972.0, e0_37=300.0, vel_psr=0.0, eta_g=0.0, eta_b=0.05,
    e_min=0.5, e_max=3.0e6, e_break=1000.0, p1=1.6, p2=2.5, f_max=0.0,
    kT_max=0.0, n_ic=2, ic_temps=[30.17, 6556.0], ic_norms=[1.2, 0.9],
    dyn_step=-1,
)


def _power(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.max() <= 0.0:          # log10 convention with LOG_EMPTY holes
        return np.where(arr > LOG_EMPTY_CUT, 10.0 ** arr, 0.0)
    return np.where(np.isfinite(arr), arr, 0.0)


def _flux_weighted_diff(r1, r2):
    f1 = _power(r1.photon_spectrum)
    f2 = _power(r2.photon_spectrum)
    w = f1 / f1.sum()
    return float(np.sum(w * np.abs(f2 - f1) / np.maximum(f1, 1e-300)))


@pytest.mark.skipif(_load_cmb_lookup() is None, reason='no CMB lookup file')
def test_evolve_field_lookup_vs_live():
    """Field lookup (fixed grid + cached kernels) against the live path
    (dynamic grid): the known fixed-grid approximation, a few per cent."""
    live = evolve_pwn(use_cmb_lookup=False, **_CRAB_LIKE)
    fast = evolve_pwn(use_cmb_lookup=True, use_field_lookup=True, **_CRAB_LIKE)
    d_live, d_fast = live.dyninfo[-1], fast.dyninfo[-1]
    assert abs(d_fast.rad_pwn / d_live.rad_pwn - 1) < 0.01
    assert abs(d_fast.bpwn / d_live.bpwn - 1) < 0.02
    assert _flux_weighted_diff(live, fast) < 0.05
    # the field nodes are now cached for this grid
    assert len(ic_cache._NODES) >= 2


def test_ssc_tensor_matches_live_kernel():
    elec, phot, tgt = _grids()
    flags = (True, True, True)
    A = ic_cache.cached_ssc_tensor(elec, phot, tgt, flags)
    assert A.dtype == np.float32 and A.shape == (NE, NP, NT)
    assert ic_cache.cached_ssc_tensor(elec, phot, tgt, flags) is A
    # synchrotron-like seed: power law with a cutoff across the target grid
    nu = tgt.bin_avg
    seed = 1e-10 * (nu / 1e12) ** -1.3 * np.exp(-nu / 1e16)
    seed[0] = seed[-1] = 0.0
    tgt.number[:] = seed
    tgt.update_log_number()
    live = _lin(calc_sigma_and_pow_nu_ic(elec, phot, tgt))
    fast = ic_cache.ssc_power(A, tgt.number)
    m = live > 1e-12 * live.max()
    np.testing.assert_allclose(fast[m], live[m], rtol=2e-6)
    assert np.all(fast[~m] <= 1e-11 * live.max())
    logf = np.where(fast > 0, np.log10(np.maximum(fast, 1e-300)), -9999.0)
    logl = np.where(live > 0, np.log10(np.maximum(live, 1e-300)), -9999.0)
    np.testing.assert_allclose(calc_pow_ic(phot, logf), calc_pow_ic(phot, logl), rtol=1e-6)


@pytest.mark.skipif(_load_cmb_lookup() is None, reason='no CMB lookup file')
def test_evolve_ssc_tensor_vs_live_path():
    """Same fixed grid, SSC on: tensor path == live kernel path."""
    kw = dict(_CRAB_LIKE, n_ic=0, ic_temps=[], ic_norms=[], ssc_model='uniform',
              timestep_cooling='total_loss10', use_cmb_lookup=True)
    ic_cache.SSC_TENSOR_ENABLED = False
    try:
        live = evolve_pwn(**kw)
    finally:
        ic_cache.SSC_TENSOR_ENABLED = True
    fast = evolve_pwn(**kw)
    f1, f2 = _power(live.photon_spectrum), _power(fast.photon_spectrum)
    m = f1 > 1e-12 * f1.max()
    np.testing.assert_allclose(f2[m], f1[m], rtol=1e-5)
    assert fast.dyninfo[-1].bpwn == live.dyninfo[-1].bpwn   # cooling path untouched


def test_field_lookup_needs_cmb_lookup():
    from viper import PWNModel
    m = PWNModel()
    m.use_cmb_lookup = False
    m.use_field_lookup = True
    with pytest.raises(ValueError, match="use_field_lookup"):
        m.validate()
    with pytest.raises(ValueError, match="use_field_lookup"):
        PWNModel().apply_run_config({'use_cmb_lookup': False,
                                     'use_field_lookup': True})
    kw = dict(esn_51=1.0, mej=6.0, n_ism=0.4, braking_index=2.5, tau_sd=3000.0,
              age=2000.0, e0_37=28.0, vel_psr=0.0, eta_g=0.0, eta_b=0.03,
              e_min=3.0, e_max=5.0e5, e_break=50.0, p1=0.8, p2=2.5)
    with pytest.raises(ValueError, match="use_field_lookup"):
        evolve_pwn(use_cmb_lookup=False, use_field_lookup=True, **kw)
