"""Multi-IC-field tests: kernel linearity + pinned end-to-end regression.

Regenerate: VIPER_CONSTANTS=gsz2009 python3 tests/test_multi_ic.py --regen
"""

import os
os.environ.setdefault('VIPER_CONSTANTS', 'gsz2009')  # noqa: E402

import numpy as np
import pytest

from viper import constants as C
from viper.evolution import evolve_pwn, _fill_target_field
from viper.grids import make_log_grid
from viper.radiation import calc_sigma_and_pow_nu_ic

REF_PATH = os.path.join(os.path.dirname(__file__), 'reference',
                        'multi_ic_reference.npz')

# reference IC fields, eV/cm^3
IC_TEMPS = [35.0, 3000.0]
IC_NORMS = [0.35, 0.5]

# literals: fixture must not drift
EVOLVE_KW = dict(
    esn_51=1.0, mej=6.0, n_ism=0.4,
    braking_index=2.5, tau_sd=3000.0, age=15532.0, e0_37=28.0,
    vel_psr=0.0, eta_g=0.0, eta_b=0.01,
    e_min=3.0, e_max=5.0e5, e_break=50.0, p1=0.8, p2=2.5,
    n_ic=2, ic_temps=IC_TEMPS, ic_norms=IC_NORMS,
    dyn_step=-1,
)

DYN_FIELDS = ('time', 'rad_pwn', 'rad_snr', 'rad_rs', 'bpwn',
              'epwn_50', 'epwnb_50', 'epwnp_50', 'vpwn', 'mswpwn')

RTOL = 1e-10


def _linear_power(log_table):
    """Convert the log10 pow_nu_ic table (-9999 sentinel) to linear."""
    out = np.zeros_like(log_table)
    mask = log_table > -9998.0
    out[mask] = 10.0 ** log_table[mask]
    return out


class TestKernelLinearity:
    """pow_nu_ic must be exactly linear in the target photon field."""

    def test_multi_field_equals_sum_of_parts(self):
        ne, np_, nt = 152, 152, 102
        elec = make_log_grid(3.0e-5, 1.0e6, ne)
        phot = make_log_grid(C.PHOT_FREQ_MIN, C.PHOT_FREQ_MAX, np_)

        t_cmb = make_log_grid(C.TARGET_FREQ_MIN, C.TARGET_FREQ_MAX, nt)
        t_all = make_log_grid(C.TARGET_FREQ_MIN, C.TARGET_FREQ_MAX, nt)
        t_extra = make_log_grid(C.TARGET_FREQ_MIN, C.TARGET_FREQ_MAX, nt)

        _fill_target_field(t_cmb, nt - 2, 0, [], [])
        _fill_target_field(t_all, nt - 2, 2, IC_TEMPS, IC_NORMS)
        t_extra.number[:] = np.maximum(t_all.number - t_cmb.number, 0.0)
        t_extra.update_log_number()

        p_cmb = _linear_power(calc_sigma_and_pow_nu_ic(elec, phot, t_cmb))
        p_all = _linear_power(calc_sigma_and_pow_nu_ic(elec, phot, t_all))
        p_extra = _linear_power(calc_sigma_and_pow_nu_ic(elec, phot,
                                                         t_extra))

        p_sum = p_cmb + p_extra
        floor = p_all.max() * 1e-30
        mask = (p_all > floor) | (p_sum > floor)
        assert np.any(mask)
        np.testing.assert_allclose(
            p_sum[mask], p_all[mask], rtol=1e-10,
            err_msg="IC kernel is not linear in the target field: the "
                    "multi-field path disagrees with the sum of its "
                    "single-field parts.")


def _run_production_shaped():
    return evolve_pwn(**EVOLVE_KW)


def _fixture_payload(result):
    dyn = result.dyninfo[-1]
    return dict(
        dyn=np.array([getattr(dyn, f) for f in DYN_FIELDS]),
        electron_spectrum=result.electron_spectrum,
        electron_energies=result.electron_energies,
        photon_spectrum=result.photon_spectrum,
        photon_freqs=result.photon_freqs,
        constants_source=np.array(C.CONSTANTS_SOURCE),
        ic_temps=np.array(IC_TEMPS), ic_norms=np.array(IC_NORMS),
    )


class TestProductionMultiICRegression:
    """Python-vs-Python pin of the production-shaped 2-extra-field run."""

    @pytest.fixture(scope='class')
    def reference(self):
        if not os.path.exists(REF_PATH):
            pytest.skip(f"fixture missing: {REF_PATH}; regenerate via "
                        f"VIPER_CONSTANTS=gsz2009 "
                        f"python3 tests/test_multi_ic.py --regen")
        ref = np.load(REF_PATH, allow_pickle=False)
        src = str(ref['constants_source'])
        if src != C.CONSTANTS_SOURCE:
            pytest.skip(f"fixture constants_source={src} != runtime "
                        f"{C.CONSTANTS_SOURCE}")
        return ref

    @pytest.fixture(scope='class')
    def fresh(self):
        return _fixture_payload(_run_production_shaped())

    def test_dynamics_pinned(self, reference, fresh):
        for i, name in enumerate(DYN_FIELDS):
            got, want = fresh['dyn'][i], reference['dyn'][i]
            assert got == pytest.approx(want, rel=RTOL), (
                f"{name}: {got!r} != pinned {want!r}")

    def test_electron_spectrum_pinned(self, reference, fresh):
        np.testing.assert_allclose(
            fresh['electron_spectrum'], reference['electron_spectrum'],
            rtol=RTOL, atol=0.0)

    def test_photon_spectrum_pinned(self, reference, fresh):
        np.testing.assert_allclose(
            fresh['photon_spectrum'], reference['photon_spectrum'],
            rtol=RTOL, atol=0.0)

    def test_grids_pinned(self, reference, fresh):
        np.testing.assert_allclose(fresh['electron_energies'],
                                   reference['electron_energies'],
                                   rtol=RTOL)
        np.testing.assert_allclose(fresh['photon_freqs'],
                                   reference['photon_freqs'], rtol=RTOL)


if __name__ == '__main__':
    import sys
    if '--regen' in sys.argv:
        payload = _fixture_payload(_run_production_shaped())
        os.makedirs(os.path.dirname(REF_PATH), exist_ok=True)
        np.savez_compressed(REF_PATH, **payload)
        print(f"wrote {REF_PATH} (constants_source="
              f"{C.CONSTANTS_SOURCE})")
        dyn = payload['dyn']
        for name, val in zip(DYN_FIELDS, dyn):
            print(f"  {name:>10s} = {val:.6g}")
    else:
        print(__doc__)
