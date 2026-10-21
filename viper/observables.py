"""
Observable extraction from PWN model results.

Observable types
----------------
1. Angular size:   SNR or PWN radius → angular size at distance d
2. Flux density:   model luminosity interpolated at one frequency/energy
3. Band flux:      integrated energy flux over a frequency/energy range
4. Spectral index: power-law photon index over a frequency/energy range
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from . import constants as C

logger = logging.getLogger(__name__)

# Unit conversion constants

# Energy → Hz conversion factors (E = hν)
_ENERGY_TO_HZ = {
    'Hz': 1.0,
    'hz': 1.0,
    'GHz': 1.0e9,
    'ghz': 1.0e9,
    'MHz': 1.0e6,
    'mhz': 1.0e6,
    'eV': 1.0 / C.h_eV,
    'ev': 1.0 / C.h_eV,
    'keV': 1.0e3 / C.h_eV,
    'kev': 1.0e3 / C.h_eV,
    'MeV': 1.0e6 / C.h_eV,
    'mev': 1.0e6 / C.h_eV,
    'GeV': 1.0e9 / C.h_eV,
    'gev': 1.0e9 / C.h_eV,
    'TeV': 1.0e12 / C.h_eV,
    'tev': 1.0e12 / C.h_eV,
}

# Flux density output conversion: from erg/s/cm²/Hz
_FLUX_DENSITY_FROM_CGS = {
    'Jy': 1.0e23,
    'jy': 1.0e23,
    'mJy': 1.0e26,
    'mjy': 1.0e26,
    'erg/s/cm2/Hz': 1.0,
}

# 1 erg = X of unit; from C.eV_per_erg, never hardcode
_ERG_PER_ENERGY_UNIT = {
    'eV': C.eV_per_erg,
    'keV': C.eV_per_erg * 1.0e-3,
    'MeV': C.eV_per_erg * 1.0e-6,
    'GeV': C.eV_per_erg * 1.0e-9,
    'TeV': C.eV_per_erg * 1.0e-12,
}

# Angular size: from radians
_ANGLE_FROM_RAD = {
    'rad': 1.0,
    'deg': 180.0 / np.pi,
    'arcmin': 180.0 / np.pi * 60.0,
    'arcsec': 180.0 / np.pi * 3600.0,
}


def _to_hz(value: float, unit: str) -> float:
    """Convert a frequency or energy value to Hz."""
    if unit not in _ENERGY_TO_HZ:
        raise ValueError(
            f"Unknown frequency/energy unit '{unit}'. "
            f"Supported: {list(_ENERGY_TO_HZ.keys())}"
        )
    return value * _ENERGY_TO_HZ[unit]


# Observable types

class ObsType(Enum):
    ANGULAR_SIZE_SNR = 'angular_size_snr'
    ANGULAR_SIZE_PWN = 'angular_size_pwn'
    FLUX_DENSITY = 'flux_density'
    BAND_FLUX = 'band_flux'
    SPECTRAL_INDEX = 'spectral_index'


@dataclass
class Observable:
    """A single observable quantity to compare against data.

    Use the class methods to create observables of each type.
    """
    obs_type: ObsType
    value: float          # in user's units
    error: float          # 1σ measurement uncertainty
    label: str = ''       # human-readable label for plots/tables
    upper_limit: bool = False  # value is an upper bound
    lower_limit: bool = False  # penalize only when model < value
    weight: float = 1.0       # chi2 weight (>1 upweights)
    range_lo: Optional[float] = None  # flat-top lower bound
    range_hi: Optional[float] = None  # flat-top upper bound
    aperture_arcmin: Optional[float] = None
    aperture_index: float = 1.5

    # Internal, set by class methods (Hz/CGS)
    _freq_hz: float = 0.0           # for flux_density
    _freq_lo_hz: float = 0.0        # for band_flux / spectral_index
    _freq_hi_hz: float = 0.0        # for band_flux / spectral_index
    _angle_unit: str = 'arcmin'     # for angular sizes
    _flux_unit: str = ''            # user's flux unit string
    _energy_unit: str = ''          # user's energy/freq unit string

    def __post_init__(self) -> None:
        """Reject malformed definitions at construction."""
        tag = f" ({self.label})" if self.label else ""
        if self.upper_limit and self.lower_limit:
            raise ValueError(
                f"Observable{tag}: upper_limit and lower_limit are "
                f"mutually exclusive")
        if not np.isfinite(self.value):
            raise ValueError(
                f"Observable{tag}: value must be finite, got {self.value}")
        if not np.isfinite(self.error) or self.error < 0:
            raise ValueError(
                f"Observable{tag}: error must be finite and >= 0, "
                f"got {self.error}")
        if self.error == 0:
            if not (self.upper_limit or self.lower_limit):
                raise ValueError(
                    f"Observable{tag}: error must be > 0 for a measured "
                    f"point or flat-top range (error 0 is allowed only "
                    f"for upper/lower limits, where limit_sigma falls "
                    f"back to 10% of the limit value)")
            if self.value == 0:
                raise ValueError(
                    f"Observable{tag}: a limit with error 0 needs a "
                    f"nonzero value for the 10% limit_sigma fallback")
        if (self.range_lo is None) != (self.range_hi is None):
            raise ValueError(
                f"Observable{tag}: range_lo and range_hi must be given "
                f"together, got ({self.range_lo}, {self.range_hi})")
        if self.range_lo is not None:
            if (not np.isfinite(self.range_lo)
                    or not np.isfinite(self.range_hi)
                    or not self.range_lo < self.range_hi):
                raise ValueError(
                    f"Observable{tag}: flat-top range must satisfy "
                    f"range_lo < range_hi (finite), got "
                    f"({self.range_lo}, {self.range_hi})")
        if self._freq_lo_hz != 0.0 or self._freq_hi_hz != 0.0:
            if not (0.0 < self._freq_lo_hz < self._freq_hi_hz):
                raise ValueError(
                    f"Observable{tag}: band edges must satisfy "
                    f"0 < lo < hi, got ({self._freq_lo_hz:.6g}, "
                    f"{self._freq_hi_hz:.6g}) Hz; a reversed or "
                    f"zero-width band integrates to nonsense")

    @property
    def limit_sigma(self) -> float:
        """Effective Gaussian σ for one-sided (upper/lower) limit penalties.

        Reported error when nonzero, else 10% of the limit value.
        """
        return self.value * 0.1 if self.error == 0 else self.error

    # --- Constructors ---

    @classmethod
    def angular_size_snr(cls, value: float, error: float,
                         unit: str = 'arcmin', label: str = '',
                         weight: float = 1.0) -> Observable:
        """SNR angular radius."""
        if unit not in _ANGLE_FROM_RAD:
            raise ValueError(f"Unknown angle unit '{unit}'. Use: {list(_ANGLE_FROM_RAD.keys())}")
        return cls(
            obs_type=ObsType.ANGULAR_SIZE_SNR,
            value=value, error=error, label=label or f'θ_snr [{unit}]',
            _angle_unit=unit,
            weight=weight,
        )

    @classmethod
    def angular_size_pwn(cls, value: float, error: float,
                         unit: str = 'arcmin', label: str = '',
                         weight: float = 1.0) -> Observable:
        """PWN angular radius."""
        if unit not in _ANGLE_FROM_RAD:
            raise ValueError(f"Unknown angle unit '{unit}'. Use: {list(_ANGLE_FROM_RAD.keys())}")
        return cls(
            obs_type=ObsType.ANGULAR_SIZE_PWN,
            value=value, error=error, label=label or f'θ_pwn [{unit}]',
            _angle_unit=unit,
            weight=weight,
        )


    @classmethod
    def angular_size_pwn_range(cls, lo: float, hi: float, sigma: float,
                               unit: str = 'arcmin', label: str = '',
                               weight: float = 1.0) -> Observable:
        """PWN angular radius with flat-top likelihood.

        Zero chi2 contribution when predicted is in [lo, hi].
        Outside that range, penalized with the given sigma.
        """
        if unit not in _ANGLE_FROM_RAD:
            raise ValueError(f"Unknown angle unit '{unit}'. Use: {list(_ANGLE_FROM_RAD.keys())}")
        mid = 0.5 * (lo + hi)
        return cls(
            obs_type=ObsType.ANGULAR_SIZE_PWN,
            value=mid, error=sigma,
            label=label or f'\u03b8_pwn [{unit}] ({lo}-{hi})',
            _angle_unit=unit,
            range_lo=lo, range_hi=hi,
            weight=weight,
        )

    @classmethod
    def angular_size_snr_range(cls, lo: float, hi: float, sigma: float,
                               unit: str = 'arcmin', label: str = '',
                               weight: float = 1.0) -> Observable:
        """SNR angular radius with flat-top likelihood."""
        if unit not in _ANGLE_FROM_RAD:
            raise ValueError(f"Unknown angle unit '{unit}'. Use: {list(_ANGLE_FROM_RAD.keys())}")
        mid = 0.5 * (lo + hi)
        return cls(
            obs_type=ObsType.ANGULAR_SIZE_SNR,
            value=mid, error=sigma,
            label=label or f'\u03b8_snr [{unit}] ({lo}-{hi})',
            _angle_unit=unit,
            range_lo=lo, range_hi=hi,
            weight=weight,
        )

    @classmethod
    def flux_density(cls, at: float, at_unit: str,
                     value: float, error: float, flux_unit: str,
                     label: str = '', upper_limit: bool = False,
                     weight: float = 1.0) -> Observable:
        """Flux density at a single frequency or energy.

        Parameters
        ----------
        at : float
            Frequency or energy where the measurement was made.
        at_unit : str
            Unit of ``at`` (e.g. 'GHz', 'keV', 'GeV', 'TeV').
        value, error : float
            Measured flux density and 1σ error; for upper limits,
            error is ignored.
        flux_unit : str
            Unit of the measured value: 'Jy', 'mJy', 'erg/s/cm2/Hz',
            'photons/cm2/s/TeV', 'photons/cm2/s/GeV', 'erg/s/cm2' (νFν).
        upper_limit : bool
            If True, value is an upper bound (model must be ≤ value).
        weight : float
            Multiplies this point's chi2 term (>1 upweights).
        """
        freq_hz = _to_hz(at, at_unit)
        ul_tag = ' [UL]' if upper_limit else ''
        auto_label = label or f'F({at} {at_unit}) [{flux_unit}]{ul_tag}'
        return cls(
            obs_type=ObsType.FLUX_DENSITY,
            value=value, error=error, label=auto_label,
            upper_limit=upper_limit,
            weight=weight,
            _freq_hz=freq_hz, _flux_unit=flux_unit, _energy_unit=at_unit,
        )

    @classmethod
    def band_flux(cls, lo: float, hi: float, energy_unit: str,
                  value: float, error: float,
                  flux_unit: str = 'erg/s/cm2',
                  label: str = '',
                  lower_limit: bool = False,
                  upper_limit: bool = False,
                  weight: float = 1.0,
                  aperture_arcmin: Optional[float] = None,
                  aperture_index: float = 1.5) -> Observable:
        """Integrated energy flux over a band.

        Parameters
        ----------
        lo, hi : float
            Lower and upper bounds of the energy/frequency band.
        energy_unit : str
            Unit of lo and hi.
        value, error : float
            Measured integrated flux and 1σ error.
        flux_unit : str
            Currently only 'erg/s/cm2' supported.
        weight : float
            Multiplies this point's chi2 term (>1 upweights).
        aperture_arcmin : float, optional
            Angular radius the flux was measured inside.  The model's
            total band flux is scaled by
            (aperture_arcmin / theta_pwn)**aperture_index before the
            comparison, where theta_pwn is the model's PWN angular
            radius at that parameter point.  Leave unset to compare
            the total flux.
        aperture_index : float
            Exponent of the enclosed-flux law F(<theta) ∝ theta**k;
            1.5 for surface brightness falling as r^-0.5.
        """
        if flux_unit != 'erg/s/cm2':
            raise ValueError(f"band_flux only supports 'erg/s/cm2', got '{flux_unit}'")
        if aperture_arcmin is not None and aperture_arcmin <= 0:
            raise ValueError(f"band_flux: aperture_arcmin must be > 0, got {aperture_arcmin}")
        auto_label = label or f'F({lo}-{hi} {energy_unit}) [{flux_unit}]'
        if aperture_arcmin is not None and not label:
            auto_label += f" <{aperture_arcmin:g}'"
        return cls(
            obs_type=ObsType.BAND_FLUX,
            value=value, error=error, label=auto_label,
            upper_limit=upper_limit,
            lower_limit=lower_limit,
            weight=weight,
            aperture_arcmin=aperture_arcmin,
            aperture_index=aperture_index,
            _freq_lo_hz=_to_hz(lo, energy_unit),
            _freq_hi_hz=_to_hz(hi, energy_unit),
            _flux_unit=flux_unit, _energy_unit=energy_unit,
        )

    @classmethod
    def spectral_index(cls, lo: float, hi: float, energy_unit: str,
                       value: float, error: float,
                       label: str = '',
                       weight: float = 1.0) -> Observable:
        """Power-law photon index Γ (dN/dE ∝ E^{-Γ}) over an energy band.

        Parameters
        ----------
        lo, hi : float
            Lower and upper bounds of the energy band for the fit.
        energy_unit : str
            Unit of lo and hi.
        value, error : float
            Measured photon index and 1σ error.
        weight : float
            Multiplies this point's chi2 term (>1 upweights).
        """
        auto_label = label or f'Γ({lo}-{hi} {energy_unit})'
        return cls(
            obs_type=ObsType.SPECTRAL_INDEX,
            value=value, error=error, label=auto_label,
            weight=weight,
            _freq_lo_hz=_to_hz(lo, energy_unit),
            _freq_hi_hz=_to_hz(hi, energy_unit),
            _energy_unit=energy_unit,
        )


# Extraction: model result → predicted observable value

def extract_observable(obs: Observable, photon_freqs: np.ndarray,
                       photon_spectrum: np.ndarray,
                       rad_pwn_pc: float, rad_snr_pc: float,
                       distance_pc: float) -> float:
    """Compute the model-predicted value of an observable.

    Parameters
    ----------
    obs : Observable
        The observable definition (type, units, etc.)
    photon_freqs : ndarray
        Model output frequencies [Hz].
    photon_spectrum : ndarray
        Model output luminosity density [10^40 erg/s/Hz].
    rad_pwn_pc, rad_snr_pc : float
        PWN and SNR radii [pc].
    distance_pc : float
        Distance to the source [pc].

    Returns
    -------
    float
        Predicted value in the same units as obs.value.
    """
    if obs.obs_type == ObsType.ANGULAR_SIZE_SNR:
        return _extract_angular_size(rad_snr_pc, distance_pc, obs._angle_unit)

    elif obs.obs_type == ObsType.ANGULAR_SIZE_PWN:
        return _extract_angular_size(rad_pwn_pc, distance_pc, obs._angle_unit)

    elif obs.obs_type == ObsType.FLUX_DENSITY:
        return _extract_flux_density(
            obs._freq_hz, photon_freqs, photon_spectrum,
            distance_pc, obs._flux_unit,
        )

    elif obs.obs_type == ObsType.BAND_FLUX:
        flux = _extract_band_flux(
            obs._freq_lo_hz, obs._freq_hi_hz,
            photon_freqs, photon_spectrum, distance_pc,
        )
        if obs.aperture_arcmin is not None:
            theta_pwn = _extract_angular_size(rad_pwn_pc, distance_pc, 'arcmin')
            flux *= (obs.aperture_arcmin / theta_pwn) ** obs.aperture_index
        return flux

    elif obs.obs_type == ObsType.SPECTRAL_INDEX:
        return _extract_spectral_index(
            obs._freq_lo_hz, obs._freq_hi_hz,
            photon_freqs, photon_spectrum,
        )

    else:
        raise ValueError(f"Unknown observable type: {obs.obs_type}")


def extract_all(observables: list[Observable],
                photon_freqs: np.ndarray, photon_spectrum: np.ndarray,
                rad_pwn_pc: float, rad_snr_pc: float,
                distance_pc: float) -> np.ndarray:
    """Extract all observables at once.  Returns array of predicted values."""
    return np.array([
        extract_observable(obs, photon_freqs, photon_spectrum,
                           rad_pwn_pc, rad_snr_pc, distance_pc)
        for obs in observables
    ])


# Internal extraction functions

def _extract_angular_size(radius_pc: float, distance_pc: float,
                          unit: str) -> float:
    """Convert physical radius to angular size at a given distance."""
    theta_rad = radius_pc / distance_pc  # small-angle approximation
    return theta_rad * _ANGLE_FROM_RAD[unit]


def _model_flux_density_cgs(freq_hz: float, photon_freqs: np.ndarray,
                             photon_spectrum: np.ndarray,
                             distance_pc: float) -> float:
    """Interpolate model spectrum and convert to F_ν [erg/s/cm²/Hz].

    The model stores L_ν in units of 10^40 erg/s/Hz.
    F_ν = L_ν / (4π d²).
    """
    mask = photon_spectrum > 0
    if not np.any(mask):
        return 0.0

    log_freq = np.log10(photon_freqs[mask])
    log_lum = np.log10(photon_spectrum[mask])

    log_freq_target = np.log10(freq_hz)

    # Outside model range: zero flux (normal when e_max is low)
    if log_freq_target < log_freq[0] or log_freq_target > log_freq[-1]:
        return 0.0

    log_lum_interp = np.interp(log_freq_target, log_freq, log_lum)
    lum_40 = 10.0 ** log_lum_interp  # 10^40 erg/s/Hz

    d_cm = distance_pc * C.pc  # cm
    f_nu = lum_40 * 1.0e40 / (4.0 * np.pi * d_cm**2)  # erg/s/cm²/Hz
    return f_nu


def _extract_flux_density(freq_hz: float, photon_freqs: np.ndarray,
                          photon_spectrum: np.ndarray,
                          distance_pc: float, flux_unit: str) -> float:
    """Extract flux density at a single frequency in the requested unit."""
    f_nu = _model_flux_density_cgs(freq_hz, photon_freqs, photon_spectrum,
                                    distance_pc)

    if flux_unit in _FLUX_DENSITY_FROM_CGS:
        return f_nu * _FLUX_DENSITY_FROM_CGS[flux_unit]

    # νFν = E²dN/dE [erg/s/cm²] at a single frequency
    if flux_unit in ('erg/s/cm2', 'nufnu'):
        return f_nu * freq_hz

    # F_ν = h²ν × dN/dE (E = hν); then per-erg → per energy unit
    for eu in ('TeV', 'GeV', 'MeV', 'keV', 'eV'):
        tag = f'photons/cm2/s/{eu}'
        if flux_unit == tag:
            h = C.h_cgs  # erg·s
            if freq_hz <= 0:
                return 0.0
            # dN/dE [phot/cm²/s/erg]
            dnde_per_erg = f_nu / (h * h * freq_hz)
            erg_per_unit = 1.0 / _ERG_PER_ENERGY_UNIT[eu]
            return dnde_per_erg * erg_per_unit

    raise ValueError(
        f"Unknown flux unit '{flux_unit}'. Supported: "
        f"{list(_FLUX_DENSITY_FROM_CGS.keys())} + 'erg/s/cm2' (νFν) + "
        f"photons/cm2/s/{{TeV,GeV,MeV,keV,eV}}"
    )


def flux_density_to_nufnu(value: float, freq_hz: float,
                          flux_unit: str) -> float:
    """Convert a single-frequency flux measurement to νFν [erg/s/cm²].

    Inverse of the unit handling in ``_extract_flux_density``; used to
    place data points of any flux unit on an SED plot.
    """
    if flux_unit in ('erg/s/cm2', 'nufnu'):
        return value
    if flux_unit in _FLUX_DENSITY_FROM_CGS:
        return value / _FLUX_DENSITY_FROM_CGS[flux_unit] * freq_hz
    for eu in ('TeV', 'GeV', 'MeV', 'keV', 'eV'):
        if flux_unit == f'photons/cm2/s/{eu}':
            dnde_per_erg = value * _ERG_PER_ENERGY_UNIT[eu]
            f_nu = dnde_per_erg * C.h_cgs * C.h_cgs * freq_hz
            return f_nu * freq_hz
    raise ValueError(f"Unknown flux unit '{flux_unit}'")


def _extract_band_flux(freq_lo: float, freq_hi: float,
                       photon_freqs: np.ndarray,
                       photon_spectrum: np.ndarray,
                       distance_pc: float) -> float:
    """Integrate F_ν over a frequency band → erg/s/cm².

    Uses the trapezoidal rule on the model frequency grid within
    the requested band.
    """
    d_cm = distance_pc * C.pc
    mask = (photon_freqs >= freq_lo) & (photon_freqs <= freq_hi) & (photon_spectrum > 0)

    if mask.sum() < 2:
        # Fallback: interpolate at band edges
        f_lo = _model_flux_density_cgs(freq_lo, photon_freqs, photon_spectrum, distance_pc)
        f_hi = _model_flux_density_cgs(freq_hi, photon_freqs, photon_spectrum, distance_pc)
        return 0.5 * (f_lo + f_hi) * (freq_hi - freq_lo)

    freqs = photon_freqs[mask]
    lum_40 = photon_spectrum[mask]
    f_nu = lum_40 * 1.0e40 / (4.0 * np.pi * d_cm**2)

    # Integrate partial edge bins via interpolated band-edge points
    if freqs[0] > freq_lo:
        f_lo = _model_flux_density_cgs(freq_lo, photon_freqs,
                                       photon_spectrum, distance_pc)
        freqs = np.concatenate(([freq_lo], freqs))
        f_nu = np.concatenate(([f_lo], f_nu))
    if freqs[-1] < freq_hi:
        f_hi = _model_flux_density_cgs(freq_hi, photon_freqs,
                                       photon_spectrum, distance_pc)
        freqs = np.concatenate((freqs, [freq_hi]))
        f_nu = np.concatenate((f_nu, [f_hi]))

    # numpy 2.0 renamed trapz → trapezoid
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    return _trapz(f_nu, freqs)


def _extract_spectral_index(freq_lo: float, freq_hi: float,
                            photon_freqs: np.ndarray,
                            photon_spectrum: np.ndarray) -> float:
    """Fit power-law photon index Γ (dN/dE ∝ E^{-Γ}) over an energy band.

    Linear least-squares fit in log-log space over model bins in the band.
    """
    mask = ((photon_freqs >= freq_lo) & (photon_freqs <= freq_hi)
            & (photon_spectrum > 0))

    if mask.sum() < 2:
        return np.nan

    log_nu = np.log10(photon_freqs[mask])
    log_lnu = np.log10(photon_spectrum[mask])

    # If L_ν ∝ ν^s then F_ν ∝ ν^s, and dN/dE ∝ E^{-(1-s)} = E^{-Γ},
    # so Γ = 1 - s.
    coeffs = np.polyfit(log_nu, log_lnu, 1)
    spectral_slope = coeffs[0]  # s in L_ν ∝ ν^s
    gamma = 1.0 - spectral_slope
    return gamma
