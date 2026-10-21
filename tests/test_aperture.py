import numpy as np

from viper.observables import Observable, extract_observable


def _flat_spectrum():
    freqs = np.geomspace(2.4e16, 2.4e19, 400)
    return freqs, np.ones_like(freqs)


def test_aperture_scaling_matches_power_law():
    freqs, spec = _flat_spectrum()
    d = 1000.0
    total = Observable.band_flux(0.5, 3.0, 'keV', 1e-12, 1e-13)
    ap = Observable.band_flux(0.5, 3.0, 'keV', 1e-12, 1e-13, aperture_arcmin=12.0)
    for rad_pc in (2.0, 3.49, 6.0):
        theta = np.degrees(rad_pc / d) * 60.0
        f_tot = extract_observable(total, freqs, spec, rad_pc, 10.0, d)
        f_ap = extract_observable(ap, freqs, spec, rad_pc, 10.0, d)
        assert np.isclose(f_ap, f_tot * (12.0 / theta) ** 1.5)


def test_aperture_index_two_is_area_scaling():
    freqs, spec = _flat_spectrum()
    ap = Observable.band_flux(0.5, 3.0, 'keV', 1e-12, 1e-13, aperture_arcmin=6.0, aperture_index=2.0)
    total = Observable.band_flux(0.5, 3.0, 'keV', 1e-12, 1e-13)
    d = 1000.0
    rad_pc = d * np.radians(12.0 / 60.0)
    assert np.isclose(extract_observable(ap, freqs, spec, rad_pc, 10.0, d),
                      0.25 * extract_observable(total, freqs, spec, rad_pc, 10.0, d))


def test_default_is_unchanged():
    freqs, spec = _flat_spectrum()
    obs = Observable.band_flux(0.5, 3.0, 'keV', 1e-12, 1e-13)
    assert obs.aperture_arcmin is None
    a = extract_observable(obs, freqs, spec, 1.0, 10.0, 1000.0)
    b = extract_observable(obs, freqs, spec, 9.0, 10.0, 1000.0)
    assert a == b


def test_aperture_must_be_positive():
    import pytest
    with pytest.raises(ValueError):
        Observable.band_flux(0.5, 3.0, 'keV', 1e-12, 1e-13, aperture_arcmin=0.0)
