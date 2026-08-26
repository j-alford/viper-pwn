"""Construction-time validation of Observable."""

import pytest

from viper.observables import Observable, ObsType


class TestValidConstructions:
    """The supported conventions must keep constructing cleanly."""

    def test_measured_point(self):
        Observable.flux_density(1.4, 'GHz', 433.0, 30.0, 'Jy')

    def test_upper_limit_with_zero_error_uses_fallback_sigma(self):
        ob = Observable.flux_density(100.0, 'GeV', 1e-12, 0.0,
                                     'erg/s/cm2', upper_limit=True)
        assert ob.limit_sigma == pytest.approx(1e-13)

    def test_flat_top_range(self):
        Observable.angular_size_pwn_range(10.2, 13.8, 1.0)

    def test_band_and_index(self):
        Observable.band_flux(2.0, 10.0, 'keV', 5.4e-12, 3.5e-14)
        Observable.spectral_index(2.0, 10.0, 'keV', 2.09, 0.01)


class TestRejectedConstructions:

    def test_zero_error_measured_point(self):
        with pytest.raises(ValueError, match='error must be > 0'):
            Observable.flux_density(1.4, 'GHz', 433.0, 0.0, 'Jy')

    def test_negative_error(self):
        with pytest.raises(ValueError, match='error must be finite and >= 0'):
            Observable.spectral_index(2, 10, 'keV', 2.09, -0.01)

    def test_nan_value(self):
        with pytest.raises(ValueError, match='value must be finite'):
            Observable.band_flux(2, 10, 'keV', float('nan'), 1e-13)

    def test_reversed_band(self):
        with pytest.raises(ValueError, match='band edges'):
            Observable.band_flux(10.0, 2.0, 'keV', 5.4e-12, 3.5e-14)

    def test_zero_width_band(self):
        with pytest.raises(ValueError, match='band edges'):
            Observable.spectral_index(2.0, 2.0, 'keV', 2.09, 0.01)

    def test_reversed_flat_top(self):
        with pytest.raises(ValueError, match='flat-top range'):
            Observable.angular_size_pwn_range(13.8, 10.2, 1.0)

    def test_zero_sigma_flat_top(self):
        with pytest.raises(ValueError, match='error must be > 0'):
            Observable.angular_size_pwn_range(10.2, 13.8, 0.0)

    def test_simultaneous_upper_and_lower_limit(self):
        with pytest.raises(ValueError, match='mutually exclusive'):
            Observable.band_flux(2, 10, 'keV', 5e-12, 1e-13,
                                 upper_limit=True, lower_limit=True)

    def test_zero_value_limit_with_zero_error(self):
        with pytest.raises(ValueError, match='nonzero value'):
            Observable.flux_density(1.4, 'GHz', 0.0, 0.0, 'Jy',
                                    upper_limit=True)

    def test_half_specified_flat_top(self):
        with pytest.raises(ValueError, match='together'):
            Observable(obs_type=ObsType.ANGULAR_SIZE_PWN, value=12.0,
                       error=1.0, range_lo=10.2)

    def test_message_names_the_label(self):
        with pytest.raises(ValueError, match='VERITAS'):
            Observable.flux_density(311, 'GeV', 1.1e-11, 0.0,
                                    'photons/cm2/s/TeV', label='VERITAS')
