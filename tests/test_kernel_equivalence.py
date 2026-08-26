"""Equivalence tests: intermediate scalar <-> vectorized <-> Numba."""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from viper import constants as C
from viper.grids import make_log_grid
from viper.radiation import (
    csynch_calc, csynch_calc_vec,
    new_energy_calc, new_energy_calc_vec,
    calc_fx, calc_fx_array,
    blackbody_number_density,
    _calc_sigma_and_pow_nu_ic_numpy,
    _build_fx_table,
)
from viper.particles import (
    place_in_bins, place_in_bins_batch,
    inject_powerlaw, inject_broken_powerlaw, inject_maxwellian,
    integrate_electron_energy,
)


def rng(seed=0):
    return np.random.default_rng(seed)


# =====================================================================
# 1. csynch_calc_vec  ==  [csynch_calc(E) for E]   (bit-identical)
# =====================================================================
def test_csynch_vec_matches_scalar():
    r = rng(1)
    log_energies = np.sort(r.uniform(-6.0, 2.0, 80))   # log10(E[erg])
    cases = [
        (1.0e-5, 1.0e-3, np.log10(3.0e18)),
        (3.2e-4, 5.0e-2, np.log10(1.0e19)),
        (0.0,    1.0e-2, np.log10(1.0e19)),   # bpwn == 0 -> all zero
        (1.0e-4, 0.0,    np.log10(1.0e19)),   # n_elec == 0
        (1.0e-3, 1.0e10, np.log10(1.0e16)),   # optically thick -> zero
    ]
    for bpwn, n_elec, log_rad in cases:
        vec = csynch_calc_vec(log_energies, bpwn, n_elec, log_rad)
        scalar = np.array([csynch_calc(le, bpwn, n_elec, log_rad)
                           for le in log_energies])
        assert np.array_equal(vec, scalar), (
            f"csynch mismatch at bpwn={bpwn}, n_elec={n_elec}: "
            f"max|diff|={np.max(np.abs(vec - scalar)):.3e}")


# =====================================================================
# 2. new_energy_calc_vec  vs  new_energy_calc  (per-bin)
# =====================================================================
def test_new_energy_vec_matches_scalar():
    r = rng(2)
    energies = 10.0 ** r.uniform(-6.0, 2.0, 64)        # erg-like
    log_c_rad = r.uniform(-40.0, -10.0, 64)
    log_rad = np.log10(3.0e18)
    log_old_rad = np.log10(2.9e18)
    ln_rad_rat = np.log(10.0 ** log_rad / 10.0 ** log_old_rad)

    # every branch covered
    cases = [
        (np.log10(5.0e7), False, ln_rad_rat),       # normal
        (np.log10(5.0e7), True,  ln_rad_rat),        # vel_neg
        (-999999.0,       False, ln_rad_rat),        # velocity == 0
        (np.log10(5.0e7), False, -ln_rad_rat),       # local_ln <= 0 path
        (-999999.0,       False, 1e-12),             # tiny denom -> abs_e_max
    ]
    for log_vel, vel_neg, lrr in cases:
        vec = new_energy_calc_vec(energies, log_c_rad, log_rad, log_old_rad,
                                  lrr, log_vel, vel_neg)
        scalar = np.array([
            new_energy_calc(e, lcr, log_rad, log_old_rad, lrr, log_vel, vel_neg)
            for e, lcr in zip(energies, log_c_rad)
        ])
        np.testing.assert_allclose(
            vec, scalar, rtol=1e-12, atol=0.0,
            err_msg=f"new_energy mismatch at log_vel={log_vel}, "
                    f"vel_neg={vel_neg}, ln_rad_rat={lrr}")

    # zero-cooling bins
    lcr0 = np.full(energies.shape, -np.inf)
    vec = new_energy_calc_vec(energies, lcr0, log_rad, log_old_rad,
                              ln_rad_rat, np.log10(5.0e7), False)
    scalar = np.array([
        new_energy_calc(e, -np.inf, log_rad, log_old_rad, ln_rad_rat,
                        np.log10(5.0e7), False)
        for e in energies
    ])
    np.testing.assert_allclose(vec, scalar, rtol=1e-12, atol=0.0)


# =====================================================================
# 3. place_in_bins_batch  vs  loop of place_in_bins
# =====================================================================
def _fresh_grid():
    return make_log_grid(1.0, 1.0e8, 200)   # GeV


def test_place_in_bins_batch_matches_loop():
    """Placements kept inside the +/-3-bin guess window."""
    g = _fresh_grid()
    abs_max = g.bin_max[g.n_bins - 1]
    bmin, bmax = g.bin_min, g.bin_max

    placements = []   # (new_min, new_max)
    rr = rng(3)
    # span 0: single bin
    for i in rr.integers(5, 190, 25):
        lo = bmin[i] + 0.3 * (bmax[i] - bmin[i])
        hi = bmin[i] + 0.6 * (bmax[i] - bmin[i])
        placements.append((lo, hi))
    # span 1: two adjacent bins
    for i in rr.integers(5, 188, 25):
        placements.append((0.5 * (bmin[i] + bmax[i]), 0.5 * (bmin[i + 1] + bmax[i + 1])))
    # span 2-3: inside guess window
    for i in rr.integers(5, 185, 20):
        placements.append((0.5 * (bmin[i] + bmax[i]), bmax[i + 2]))
    # swap branch: new_max < new_min
    for i in rr.integers(5, 185, 10):
        placements.append((bmax[i + 1], bmin[i]))
    # overflow clamp: new_max > abs_max
    placements.append((bmin[180], abs_max * 10.0))
    # zero-width
    for i in rr.integers(5, 190, 8):
        c = 0.5 * (bmin[i] + bmax[i])
        placements.append((c, c))

    values = rr.uniform(0.1, 10.0, len(placements))
    new_mins = np.array([p[0] for p in placements])
    new_maxs = np.array([p[1] for p in placements])

    g_batch = _fresh_grid()
    place_in_bins_batch(new_mins, new_maxs, values,
                        np.ones(len(placements), dtype=bool), g_batch)

    g_loop = _fresh_grid()
    for nmin, nmax, v in zip(new_mins, new_maxs, values):
        place_in_bins(float(nmin), float(nmax), float(v), g_loop)

    # FP accumulation order differs
    np.testing.assert_allclose(g_batch.number, g_loop.number,
                               rtol=1e-12, atol=1e-300)


# =====================================================================
# 3b. evolve_spectrum: fused numba kernel vs vectorised-numpy reference
# =====================================================================
def _es_grid(n_bins=102):
    return make_log_grid(1.0e-6, 1.0e8, n_bins)


def _es_run_both(old_grid, bpwn, old_radius, new_radius, velocity,
                 n_elec_40, min_ic_bin, pow_ic):
    """Run the numpy reference and the fused kernel on identical inputs."""
    from viper.evolution import evolve_spectrum, _evolve_spectrum_numpy

    n = old_grid.n_bins
    args = (bpwn, old_radius, new_radius, velocity, n_elec_40,
            int(min_ic_bin), pow_ic)

    g_np = _es_grid(n)
    s_np = tuple(np.full(n, -1.0) for _ in range(4))
    ref = _evolve_spectrum_numpy(old_grid, g_np, *args, scratch=s_np)

    g_kn = _es_grid(n)
    s_kn = tuple(np.full(n, -1.0) for _ in range(4))
    got = evolve_spectrum(old_grid, g_kn, *args, scratch=s_kn)

    return ref, g_np, s_np, got, g_kn, s_kn


def _es_scenarios():
    """(label, fill, ic, overrides): every branch of the fused kernel."""
    r_old, r_new = 3.0e18, 3.01e18
    base = dict(bpwn=1.0e-4, old_radius=r_old, new_radius=r_new,
                velocity=5.0e7, n_elec_40=1.0e-2, min_ic_bin=0)
    scen = []

    def add(label, fill='bpl', ic='none', **over):
        scen.append((label, fill, ic, {**base, **over}))

    add('typical_cooling', ic='band', min_ic_bin=30)
    add('bpwn_zero', ic='band', bpwn=0.0)
    add('optically_thick', ic='none', n_elec_40=1.0e10,
        old_radius=1.0e16, new_radius=1.001e16)
    add('n_elec_zero', ic='band', n_elec_40=0.0)
    add('velocity_zero_equal_radii', velocity=0.0, new_radius=r_old)
    add('velocity_zero_unequal_radii', velocity=0.0)
    add('vel_neg_expanding', velocity=-5.0e7)          # local_ln <= 0
    add('vel_neg_contracting', velocity=-5.0e7,        # full math, |v|
        old_radius=r_new, new_radius=r_old)
    add('contraction_pos_vel', old_radius=r_new, new_radius=r_old)
    add('pure_adiabatic', bpwn=0.0, ic='none')
    add('all_empty', fill='empty', ic='band')
    add('single_bin', fill='single', ic='band')
    add('unoccupied_with_cooling', fill='sparse', ic='dense')
    add('bin0_ic_cooled', ic='dense')                  # pow_ic[0] > 0
    add('min_ic_mid_gates_lower_bins', ic='dense', min_ic_bin=51)
    add('min_ic_last_bin', ic='dense', min_ic_bin=101)
    add('min_ic_past_end', ic='dense', min_ic_bin=109)
    add('nan_synthetic_inf_over_inf', ic='dense', velocity=1.0e305)
    add('ssc_shaped_dense', fill='bpl', ic='dense')
    add('deep_cooling_overflow', fill='bpl', ic='huge')
    return scen


@pytest.mark.parametrize(
    "label,fill,ic,kw",
    _es_scenarios(),
    ids=[s[0] for s in _es_scenarios()])
def test_evolve_spectrum_kernel_matches_numpy(label, fill, ic, kw):
    """Fused kernel must be bit-identical to the numpy path, never a tolerance."""
    pytest.importorskip("numba")

    old = _es_grid()
    n = old.n_bins
    rr = rng(11)

    if fill == 'bpl':
        inject_broken_powerlaw(old, 2.5, 10.0, 1.0e6, 3.0e3, 1.6, 2.5)
    elif fill == 'single':
        old.number[60] = 3.7
        old.update_log_number()
    elif fill == 'sparse':
        old.number[15] = 1.1
        old.number[85] = 0.3
        old.update_log_number()
    # 'empty': leave all zeros

    pow_ic = np.zeros(n)
    if ic == 'band':
        pow_ic[30:90] = 10.0 ** rr.uniform(-24.0, -14.0, 60)
    elif ic == 'dense':
        pow_ic[:] = 10.0 ** rr.uniform(-24.0, -12.0, n)
    elif ic == 'huge':
        # span>=2 cases pinned elsewhere
        pow_ic[:] = 10.0 ** rr.uniform(-8.0, -4.0, n)

    ref, g_np, s_np, got, g_kn, s_kn = _es_run_both(
        old, kw['bpwn'], kw['old_radius'], kw['new_radius'],
        kw['velocity'], kw['n_elec_40'], kw['min_ic_bin'], pow_ic)

    assert ref[0] == got[0], f"{label}: new_energy {ref[0]!r} != {got[0]!r}"
    assert ref[1] == got[1], f"{label}: rad_loss {ref[1]!r} != {got[1]!r}"
    assert np.array_equal(g_np.number, g_kn.number), (
        f"{label}: destination grid number diverges "
        f"(first at bin {np.argmax(g_np.number != g_kn.number)})")
    assert np.array_equal(g_np.log_number, g_kn.log_number), (
        f"{label}: destination grid log_number diverges")
    for k, name in enumerate(("new_e_min", "new_e_max",
                              "energy_terms", "rad_loss_terms")):
        assert np.array_equal(s_np[k], s_kn[k]), (
            f"{label}: {name} diverges "
            f"(first at bin {np.argmax(s_np[k] != s_kn[k])})")


def test_place_in_bins_kernel_matches_batch():
    """Fused deposit kernel must be bit-identical to place_in_bins_batch."""
    pytest.importorskip("numba")
    from viper.evolution import _place_in_bins_kernel

    g_ref = _fresh_grid()
    n_bins = g_ref.n_bins
    bmin, bmax = g_ref.bin_min, g_ref.bin_max
    abs_max = bmax[n_bins - 1]
    rr = rng(13)

    placements = []
    # span 0: inside one bin
    for i in rr.integers(5, 190, 20):
        lo = bmin[i] + 0.25 * (bmax[i] - bmin[i])
        hi = bmin[i] + 0.75 * (bmax[i] - bmin[i])
        placements.append((lo, hi))
    # span 1: two adjacent bins
    for i in rr.integers(5, 188, 20):
        placements.append((0.5 * (bmin[i] + bmax[i]),
                           0.5 * (bmin[i + 1] + bmax[i + 1])))
    # span 2-5
    for i in rr.integers(5, 180, 15):
        placements.append((0.5 * (bmin[i] + bmax[i]),
                           bmax[i + 1 + int(rr.integers(1, 5))]))
    # swapped edges (new_max < new_min)
    for i in rr.integers(5, 185, 8):
        placements.append((bmax[i + 1], bmin[i]))
    # overflow clamp beyond abs_max
    placements.append((bmin[195], abs_max * 5.0))
    placements.append((bmin[150], abs_max * 2.0))
    # span-0 frac fallback 1.0
    for i in rr.integers(5, 190, 5):
        c = 0.6 * bmin[i] + 0.4 * bmax[i]
        placements.append((c, c))
    # span-1 0.5/0.5 fallback
    for i in rr.integers(5, 190, 5):
        placements.append((bmax[i], bmax[i]))
    # frac == 0 skip branches
    for i in rr.integers(5, 180, 5):
        placements.append((bmin[i], bmax[i + 3]))
    # pins cross-phase accumulation order
    for i in rr.integers(20, 170, 6):
        placements.append((bmin[i] + 0.1 * (bmax[i] - bmin[i]),
                           bmin[i] + 0.2 * (bmax[i] - bmin[i])))
        placements.append((0.5 * (bmin[i - 1] + bmax[i - 1]),
                           0.5 * (bmin[i] + bmax[i])))
        placements.append((0.5 * (bmin[i - 2] + bmax[i - 2]), bmax[i]))

    new_mins = np.array([p[0] for p in placements])
    new_maxs = np.array([p[1] for p in placements])
    values = rr.uniform(0.1, 10.0, len(placements))
    mask = np.ones(len(placements), dtype=bool)
    mask[rr.integers(0, len(placements), 10)] = False  # skipped sources

    g_batch = _fresh_grid()
    place_in_bins_batch(new_mins, new_maxs, values, mask, g_batch)

    g_kern = _fresh_grid()
    _place_in_bins_kernel(new_mins, new_maxs, values, mask,
                          g_kern.bin_min, g_kern.bin_max, g_kern.number)

    assert np.array_equal(g_batch.number, g_kern.number), (
        "deposit kernel diverges from place_in_bins_batch (first at bin "
        f"{np.argmax(g_batch.number != g_kern.number)})")

    # all-masked-out no-op
    g_b0, g_k0 = _fresh_grid(), _fresh_grid()
    none = np.zeros(len(placements), dtype=bool)
    place_in_bins_batch(new_mins, new_maxs, values, none, g_b0)
    _place_in_bins_kernel(new_mins, new_maxs, values, none,
                          g_k0.bin_min, g_k0.bin_max, g_k0.number)
    assert np.array_equal(g_b0.number, g_k0.number)
    assert not g_k0.number.any()


def test_evolve_spectrum_kernel_production_shape():
    """Same A/B at the production grid size (502 bins)."""
    pytest.importorskip("numba")

    old = _es_grid(502)
    inject_broken_powerlaw(old, 2.5, 10.0, 1.0e6, 3.0e3, 1.6, 2.5)
    n = old.n_bins
    rr = rng(12)
    pow_ic = np.zeros(n)
    pow_ic[100:450] = 10.0 ** rr.uniform(-24.0, -12.0, 350)

    ref, g_np, s_np, got, g_kn, s_kn = _es_run_both(
        old, 1.0e-4, 3.0e18, 3.01e18, 5.0e7, 1.0e-2, 50, pow_ic)

    assert ref[0] == got[0] and ref[1] == got[1]
    assert np.array_equal(g_np.number, g_kn.number)
    assert np.array_equal(g_np.log_number, g_kn.log_number)
    for k in range(4):
        assert np.array_equal(s_np[k], s_kn[k])


# =====================================================================
# 4. _ic_kernel (Numba)  vs  _calc_sigma_and_pow_nu_ic_numpy
# =====================================================================
def _ic_test_grids():
    elec = make_log_grid(1.0, 1.0e8, 60)          # GeV
    phot = make_log_grid(1.0e8, 1.0e28, 60)       # Hz
    targ = make_log_grid(1.0e9, 1.0e13, 40)       # Hz (CMB band)
    for i in range(targ.n_bins):
        targ.number[i] = blackbody_number_density(targ.bin_avg[i], C.T_CMB)
    targ.update_log_number()
    return elec, phot, targ


def test_ic_kernel_matches_numpy():
    pytest.importorskip("numba")
    from viper.radiation import _calc_sigma_and_pow_nu_ic_numba

    elec, phot, targ = _ic_test_grids()
    ref = _calc_sigma_and_pow_nu_ic_numpy(elec, phot, targ,
                                          ic_skip_empty_bins=True)
    assert np.any(ref > -9998.0), "IC test grids produced no scattered power"
    ref_lin = np.where(ref > -9998.0, 10.0 ** ref, 0.0)

    for eb, jb, se in itertools.product([False, True], repeat=3):
        got = _calc_sigma_and_pow_nu_ic_numba(
            elec, phot, targ,
            ic_early_break=eb, ic_j_break=jb, ic_skip_empty_bins=se)
        got_lin = np.where(got > -9998.0, 10.0 ** got, 0.0)
        np.testing.assert_allclose(
            got_lin, ref_lin, rtol=1e-12, atol=1e-300,
            err_msg=f"IC kernel diverges at early_break={eb}, "
                    f"j_break={jb}, skip_empty={se}")

    # empty bins add 0
    ref_false = _calc_sigma_and_pow_nu_ic_numpy(elec, phot, targ,
                                                ic_skip_empty_bins=False)
    np.testing.assert_allclose(
        np.where(ref_false > -9998.0, 10.0 ** ref_false, 0.0),
        ref_lin, rtol=1e-12, atol=1e-300)


# =====================================================================
# 5. calc_fx_array  vs  calc_fx  (+ slow F(x) table-accuracy check)
# =====================================================================
def test_calc_fx_array_matches_scalar():
    r = rng(5)
    x = np.concatenate([
        10.0 ** r.uniform(-6.0, np.log10(C.max_synch_x) - 0.1, 60),
        np.array([0.0, -1.0, 1e-12, C.max_synch_x, C.max_synch_x * 2.0]),
    ])
    vec = calc_fx_array(x)
    scalar = np.array([calc_fx(float(xi)) for xi in x])
    assert np.array_equal(vec, scalar)


@pytest.mark.slow
def test_fx_table_accuracy_vs_quadrature():
    """Bound the F(x) table error vs scipy quadrature; 1% leaves margin for scipy variation."""
    from scipy.integrate import quad
    from scipy.special import kv

    xs = np.logspace(-3, np.log10(C.max_synch_x) - 0.5, 20)
    rel_errs = []
    for x in xs:
        truth = x * quad(lambda xi: kv(5.0 / 3.0, xi), x, np.inf,
                         limit=200)[0]
        table = calc_fx(float(x))
        if truth > 0:
            rel_errs.append(abs(table - truth) / truth)
    rel_errs = np.array(rel_errs)
    assert rel_errs.max() < 1e-2, (
        f"F(x) table max rel error {rel_errs.max():.3%} exceeds 1% "
        f"(median {np.median(rel_errs):.3%})")


# =====================================================================
# 6. injectors: vectorized vs scalar reference + energy conservation
# =====================================================================
def _scalar_inject_broken_powerlaw(grid, energy_40, e_min, e_max,
                                   e_break, p1, p2):
    """Literal scalar transliteration of the original per-bin injection loop."""
    if energy_40 <= 0:
        return
    erg = C.GeV_per_erg
    n1 = (np.log(e_break / e_min) if p1 == 2
          else (e_break ** (2.0 - p1) - e_min ** (2.0 - p1)) / (2.0 - p1))
    n2 = (np.log(e_max / e_break) if p2 == 2
          else (e_max ** (2.0 - p2) - e_break ** (2.0 - p2)) / (2.0 - p2))
    n_break = energy_40 * erg / (e_break ** p1 * n1 + e_break ** p2 * n2)
    for i in range(grid.n_bins):
        bmn, bmx = grid.bin_min[i], grid.bin_max[i]
        if bmx >= e_min and bmn < e_break:
            lo = max(bmn, e_min)
            hi = min(bmx, e_break)
            if p1 == 1:
                grid.number[i] += n_break * e_break * np.log(hi / lo)
            else:
                grid.number[i] += (n_break * e_break ** p1 / (1.0 - p1)
                                   * (hi ** (1.0 - p1) - lo ** (1.0 - p1)))
        if bmx >= e_break and bmn <= e_max:
            lo = max(bmn, e_break)
            hi = min(bmx, e_max)
            if p2 == 1:
                grid.number[i] += n_break * e_break * np.log(hi / lo)
            else:
                grid.number[i] += (n_break * e_break ** p2 / (1.0 - p2)
                                   * (hi ** (1.0 - p2) - lo ** (1.0 - p2)))
    grid.update_log_number()


@pytest.mark.parametrize("p1,p2", [(1.6, 2.5), (2.0, 3.0), (1.0, 2.2),
                                   (2.3, 2.0)])
def test_inject_broken_powerlaw_matches_scalar(p1, p2):
    e_min, e_max, e_break = 10.0, 1.0e6, 3.0e3
    g_vec = make_log_grid(1.0, 1.0e8, 200)
    g_ref = make_log_grid(1.0, 1.0e8, 200)
    inject_broken_powerlaw(g_vec, 1.0, e_min, e_max, e_break, p1, p2)
    _scalar_inject_broken_powerlaw(g_ref, 1.0, e_min, e_max, e_break, p1, p2)
    np.testing.assert_allclose(g_vec.number, g_ref.number,
                               rtol=1e-12, atol=1e-300)


def test_injector_energy_conservation():
    """Each injector deposits ~energy_40 within bin-discretization error."""
    energy_40 = 2.5
    e_min, e_max, e_break = 10.0, 1.0e6, 3.0e3

    g = make_log_grid(1.0, 1.0e8, 400)
    inject_powerlaw(g, energy_40, e_min, e_max, 2.2)
    assert abs(integrate_electron_energy(g) - energy_40) / energy_40 < 1e-2

    g = make_log_grid(1.0, 1.0e8, 400)
    inject_broken_powerlaw(g, energy_40, e_min, e_max, e_break, 1.6, 2.5)
    assert abs(integrate_electron_energy(g) - energy_40) / energy_40 < 1e-2

    # kT in GeV; tail-truncation tolerance
    g = make_log_grid(1.0e-2, 1.0e3, 400)
    inject_maxwellian(g, energy_40, kT=5.0)
    assert abs(integrate_electron_energy(g) - energy_40) / energy_40 < 5e-2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
