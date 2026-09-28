"""snapshot_injection: snapshots without the current step's injection deposit.

Pins (a) the default ('included') is bit-identical to the pre-option code
path, (b) 'excluded' snapshots equal the 'included' ones minus a deposit
confined to the injection range, (c) that deposit carries the step's
spin-down energy, and (d) cpp_compat forces 'included'.
"""
import numpy as np
import pytest
import astropy.units as u

from viper import PWNModel
import viper.constants as C


def _model():
    m = PWNModel()
    m.set_pulsar('J1930+1852')
    m.distance.value = 6.0 * u.kpc
    m.esn.value = 1e51 * u.erg
    m.mej.value = 20.0 * u.M_sun
    m.n_ism.value = 0.01 / u.cm ** 3
    m.braking_index.value = 2.0
    m.tau_sd.value = 3500 * u.yr
    m.eta_b.value = 3e-3
    m.e_min.value = 10.0 * u.GeV
    m.e_max.value = 500 * u.TeV
    m.e_break.value = 3.0 * u.TeV
    m.p1.value = 2.3
    m.p2.value = 2.4
    return m


@pytest.fixture(scope='module')
def runs():
    inc = _model().run(dyn_step=1, elec_step=1, phot_step=-1)
    dflt = _model().run(dyn_step=1, elec_step=1, phot_step=-1,
                        snapshot_injection='included')
    exc = _model().run(dyn_step=1, elec_step=1, phot_step=-1,
                       snapshot_injection='excluded')
    return inc, dflt, exc


def test_default_is_included(runs):
    inc, dflt, _ = runs
    assert np.array_equal(inc.electron_spectrum, dflt.electron_spectrum)
    assert np.array_equal(inc.photon_spectrum, dflt.photon_spectrum)
    for (t1, s1), (t2, s2) in zip(inc.elecinfo, dflt.elecinfo):
        assert t1 == t2 and np.array_equal(s1, s2)


def test_injection_range_recorded(runs):
    inc, _, _ = runs
    assert inc.e_min_inj == pytest.approx(10.0)
    assert inc.e_max_inj == pytest.approx(5.0e5)


def test_excluded_is_included_minus_a_deposit_in_range(runs):
    inc, _, exc = runs
    g = inc.elec_grid
    assert len(inc.elecinfo) == len(exc.elecinfo)
    outside = (g.bin_max <= inc.e_min_inj) | (g.bin_min >= inc.e_max_inj)
    for (t1, s_inc), (t2, s_exc) in zip(inc.elecinfo[1:], exc.elecinfo[1:]):
        assert t1 == t2
        dep = s_inc - s_exc
        assert np.all(dep >= -1e-12 * s_inc.max())
        assert np.allclose(dep[outside], 0.0, atol=1e-12 * s_inc.max())
        assert dep[~outside].sum() > 0.0
    # same holds for the final state
    assert np.all(inc.electron_spectrum - exc.electron_spectrum >= -1e-12)


def test_deposit_carries_the_step_spindown_energy(runs):
    inc, _, exc = runs
    g = inc.elec_grid
    # last recorded intermediate step: deposit energy vs (1 - eta_b) * Edot * dt
    k = len(inc.dyninfo) - 2
    dyn = inc.dyninfo[k]
    dep = inc.elecinfo[k][1] - exc.elecinfo[k][1]
    e_dep_40 = (dep * g.bin_avg).sum() / C.GeV_per_erg
    e_sd_40 = dyn.edot_40 * dyn.delta_t * C.yr * (1.0 - 3e-3)
    assert e_dep_40 == pytest.approx(e_sd_40, rel=0.03)


def test_top_bins_lose_the_dt_lump(runs):
    inc, _, exc = runs
    g = inc.elec_grid
    i_top = int(np.searchsorted(g.bin_max, inc.e_max_inj))
    # the e_max bin holds essentially only the last step's deposit when included
    assert inc.electron_spectrum[i_top] > 0
    assert exc.electron_spectrum[i_top] < 0.5 * inc.electron_spectrum[i_top]


def test_photon_spectrum_changes_only_where_expected(runs):
    inc, _, exc = runs
    # removing the fresh e_max deposit can only lower the emission.  The
    # deposit is a sizeable share of the barely-cooled top of the spectrum, so
    # the synchrotron peak/cutoff can move by tens of percent; the radio (fully
    # resident electrons) moves by a few percent.
    ref = inc.photon_freqs * inc.photon_spectrum
    pos = inc.photon_spectrum > 0
    rel = np.zeros_like(ref)
    rel[pos] = ((exc.photon_spectrum[pos] - inc.photon_spectrum[pos])
                / inc.photon_spectrum[pos])
    assert np.all(rel[pos] <= 1e-9)
    bulk = ref > 0.1 * ref.max()
    assert np.median(np.abs(rel[bulk])) < 0.25
    radio = pos & (inc.photon_freqs < 1e10)
    assert np.abs(rel[radio]).max() < 0.05
    tail = pos & (ref > 1e-3 * ref.max())
    assert np.abs(rel[tail]).max() > 0.05


def test_cpp_compat_forces_included():
    m1 = _model(); m2 = _model()
    r1 = m1.run(dyn_step=-1, cpp_compat=True)
    r2 = m2.run(dyn_step=-1, cpp_compat=True, snapshot_injection='excluded')
    assert np.array_equal(r1.electron_spectrum, r2.electron_spectrum)
    assert np.array_equal(r1.photon_spectrum, r2.photon_spectrum)


def test_bad_value_rejected():
    with pytest.raises(ValueError):
        _model().run(dyn_step=-1, snapshot_injection='half')
