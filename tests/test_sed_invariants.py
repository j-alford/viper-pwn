"""Physical SED invariants: IC norm (eV/cm^3) semantics."""

import numpy as np
import astropy.units as u

from viper.model import PWNModel
from viper import constants as C


def _cta1_like_model():
    """A small CTA-1-ish model used by the IC norm invariant tests."""
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


def test_ic_norm_zero_is_noop():
    """``add_ic_field(temp=T, norm=0)`` must reproduce the CMB-only SED."""
    m_cmb_only = _cta1_like_model()
    m_zero     = _cta1_like_model()
    m_zero.add_ic_field(temp=2.728, norm=0.0)

    res_a = m_cmb_only.run(dyn_step=-1)
    res_b = m_zero.run(dyn_step=-1)

    np.testing.assert_array_equal(res_a.photon_spectrum, res_b.photon_spectrum)


def test_ic_norm_evcm3_doubles_cmb_in_ic_band():
    """An extra T=T_CMB field at u=u_CMB must approximately double the IC."""
    T = 2.728
    u_cmb_evcm3 = C.radiation_const * T**4 * C.eV_per_erg

    m_a = _cta1_like_model()
    m_b = _cta1_like_model()
    m_b.add_ic_field(temp=T, norm=u_cmb_evcm3)

    res_a = m_a.run(dyn_step=-1)
    res_b = m_b.run(dyn_step=-1)

    freq = res_a.photon_freqs
    spec_a = res_a.photon_spectrum
    spec_b = res_b.photon_spectrum

    # IC-only band above synchrotron cutoff
    h_eV = 4.135667696e-15
    is_ic = (freq * h_eV > 1.0e6) & (spec_a > 0)
    assert is_ic.sum() > 10, "Too few IC-band bins."

    # brightest IC bin
    nufnu_ic = freq[is_ic] * spec_a[is_ic]
    ic_peak_local = int(np.argmax(nufnu_ic))
    ic_peak = np.where(is_ic)[0][ic_peak_local]

    ratio = spec_b[ic_peak] / spec_a[ic_peak]
    photon_eV = freq[ic_peak] * h_eV
    assert 1.5 < ratio < 2.3, (
        f"IC peak ratio = {ratio:.3f} at photon E = {photon_eV:.2e} eV. "
        f"Expected ~2 for an extra field with u=u_CMB(={u_cmb_evcm3:.3f} "
        f"eV/cm^3) at T=T_CMB. If ~1.26, the eV/cm^3 conversion in "
        f"_fill_target_field was reverted to a dimensionless multiplier."
    )


def _crab_like_kwargs():
    """A young, strongly magnetised nebula: electrons at e_max cool in a
    fraction of the 0.5 yr timestep floor."""
    return dict(
        esn_51=1.0, mej=9.9, n_ism=0.1, braking_index=2.51, tau_sd=700.0,
        age=300.0, e0_37=335.0, vel_psr=0.0, eta_g=0.0, eta_b=0.027,
        e_min=0.05, e_max=2.4e6, e_break=670.0, p1=1.56, p2=2.41,
        f_max=0.0, kT_max=0.0, n_ic=0, ic_temps=[], ic_norms=[],
        elec_step=-1, phot_step=-1,
    )


def test_resolve_final_cooling_matches_refined_timesteps():
    """With resolve_final_cooling the final spectrum agrees with a run
    whose last steps are subdivided ten-fold; without it the synchrotron
    cutoff region is off by tens of percent."""
    from viper.evolution import evolve_pwn
    kw = _crab_like_kwargs()
    default = evolve_pwn(dyn_step=1, **kw)
    dts = [d.delta_t for d in default.dyninfo][:-1]
    seq = dts[:-100] + [d / 50 for d in dts[-100:] for _ in range(50)]
    reference = evolve_pwn(replay_dt_sequence=seq, **kw)
    resolved = evolve_pwn(resolve_final_cooling=True, **kw)
    e_eV = default.photon_freqs * C.h_eV
    nufnu = lambda r: r.photon_spectrum * r.photon_freqs
    band = (e_eV > 1e4) & (e_eV < 1.5e8)
    worst_default = np.max(np.abs(nufnu(default)[band] / nufnu(reference)[band] - 1))
    worst_resolved = np.max(np.abs(nufnu(resolved)[band] / nufnu(reference)[band] - 1))
    assert worst_default > 0.10
    assert worst_resolved < 0.06
