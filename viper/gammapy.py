"""
Using VIPER in gammapy

PWNSpectralModel is a gammapy SpectralModel class, so it works
similarly to an analytic spectral model within gammapy.

Usage
-----
>>> from viper.gammapy import PWNSpectralModel
>>> model = PWNSpectralModel.for_cta1()
>>> print(model)
>>> model.plot(energy_bounds=[1e-7, 100] * u.TeV)

Can also be used standalone (without gammapy) via ``evaluate_dnde``.

CMB is always included as an IC target; up to 2 additional thermal
fields via ``ic_temp_N`` / ``ic_norm_N``.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

import numpy as np

from .model import PWNModel
from . import constants as C

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unit conversions use the package constants; never mix constant sets.
# ---------------------------------------------------------------------------
_H_CGS = C.h_cgs                       # Planck constant [erg s]
_ERG_PER_TEV = 1.0e12 / C.eV_per_erg   # erg per TeV
_PC_CM = C.pc                          # parsec in cm


def _run_and_convert(
    esn, mej, n_ism, braking_index, tau_sd, eta_b,
    e_min, e_max, e_break, p1, p2,
    period, p_dot, distance_pc,
    ic_temp_1=0.0, ic_norm_1=0.0,
    ic_temp_2=0.0, ic_norm_2=0.0,
    injection_model='broken_powerlaw',
    ssc_model='off',
):
    """Run the PWN model and return (log10_energy_TeV, log10_dnde) arrays.

    Returns the photon spectrum as dN/dE [cm^-2 s^-1 TeV^-1] on the
    model's native frequency grid, converted to energy in TeV.
    """
    model = PWNModel()
    model.period._set_internal(period)
    model.p_dot._set_internal(p_dot)
    model.esn._set_internal(esn)
    model.mej._set_internal(mej)
    model.n_ism._set_internal(n_ism)
    model.braking_index._set_internal(braking_index)
    model.tau_sd._set_internal(tau_sd)
    model.eta_b._set_internal(eta_b)
    model.e_min._set_internal(e_min)
    model.e_max._set_internal(e_max)
    model.e_break._set_internal(e_break)
    model.p1._set_internal(p1)
    model.p2._set_internal(p2)
    model.distance._set_internal(distance_pc)
    model.injection_model = injection_model
    model.ssc_model = ssc_model

    if ic_temp_1 > 0 and ic_norm_1 > 0:
        model.add_ic_field(temp=ic_temp_1, norm=ic_norm_1)
    if ic_temp_2 > 0 and ic_norm_2 > 0:
        model.add_ic_field(temp=ic_temp_2, norm=ic_norm_2)

    result = model.run(dyn_step=-1)

    freq = result.photon_freqs       # Hz
    spec = result.photon_spectrum    # 10^40 erg/s/Hz

    d_cm = distance_pc * _PC_CM
    # dN/dE = spec * 1e40 * erg_per_TeV / (4π d² h² ν)
    dnde = spec * 1e40 * _ERG_PER_TEV / (
        4.0 * np.pi * d_cm**2 * _H_CGS**2 * freq
    )

    energy_tev = _H_CGS * freq / _ERG_PER_TEV

    mask = (dnde > 0) & (energy_tev > 0)
    if not np.any(mask):
        return None, None

    log_e = np.log10(energy_tev[mask])
    log_f = np.log10(dnde[mask])

    return log_e, log_f


def evaluate_dnde(energy_tev, log_e_grid, log_f_grid):
    """Interpolate dN/dE onto arbitrary energy array.

    Parameters
    ----------
    energy_tev : array-like
        Energies in TeV.
    log_e_grid, log_f_grid : arrays
        Log10 energy and log10 dN/dE from ``_run_and_convert``.

    Returns
    -------
    dnde : ndarray
        Differential photon flux [cm^-2 s^-1 TeV^-1].
    """
    energy_tev = np.atleast_1d(np.asarray(energy_tev, dtype=float))
    result = np.zeros_like(energy_tev)

    valid = energy_tev > 0
    if not np.any(valid) or log_e_grid is None:
        return result

    log_e = np.log10(energy_tev[valid])

    # Zero outside grid bounds
    in_range = (log_e >= log_e_grid[0]) & (log_e <= log_e_grid[-1])
    if np.any(in_range):
        interp_vals = np.interp(
            log_e[in_range], log_e_grid, log_f_grid,
        )
        tmp = np.zeros_like(log_e)
        tmp[in_range] = 10.0 ** interp_vals
        result[valid] = tmp

    return result


# ---------------------------------------------------------------------------
# Gammapy SpectralModel subclass (optional import)
# ---------------------------------------------------------------------------

try:
    import astropy.units as u
    from gammapy.modeling import Parameter
    from gammapy.modeling.models import SpectralModel

    _HAS_GAMMAPY = True
except ImportError:
    _HAS_GAMMAPY = False

if _HAS_GAMMAPY:

    class PWNSpectralModel(SpectralModel):
        r"""Gammapy spectral model wrapping the Gelfand+2009 PWN evolution.

        Runs the full dynamical + radiative model for each unique set of
        parameters and interpolates dN/dE onto the requested energy grid.
        Results are cached so repeated evaluations with the same parameters
        (e.g., during gammapy's numerical integration) are instant.

        Parameters
        ----------
        period : float
            Pulsar spin period [s]; plain attribute, not a fit Parameter.
        p_dot : float
            Period derivative [s s^-1]; plain attribute.
        injection_model : str
            Injection-spectrum shape (``viper.injection`` registry name);
            default ``'broken_powerlaw'``.
        ssc_model : str
            Synchrotron self-Compton seed, ``'off'`` (default) or
            ``'uniform'``.
        esn : float
            Supernova kinetic energy [10^51 erg].
        mej : float
            Ejecta mass [Msun].
        n_ism : float
            ISM number density [cm^-3].
        distance : float
            Distance to source [pc].
        braking_index : float
            Pulsar braking index.
        tau_sd : float
            Spin-down timescale [yr].
        eta_b : float
            Magnetic fraction of pulsar wind.
        e_min : float
            Minimum injection energy [GeV].
        e_max : float
            Maximum injection energy [GeV].
        e_break : float
            Break energy [GeV].
        p1 : float
            Low-energy spectral index.
        p2 : float
            High-energy spectral index.
        ic_temp_1, ic_norm_1 : float
            Temperature [K] and energy density [eV cm^-3] of extra IC
            field 1; 0 = disabled.
        ic_temp_2, ic_norm_2 : float
            Temperature [K] and energy density [eV cm^-3] of extra IC
            field 2; 0 = disabled.

        Notes
        -----
        All physics Parameters start frozen, matching ``PWNModel``;
        unfreeze the subset you fit.
        """

        tag = ["PWNSpectralModel", "pwn"]

        # --- Physics parameters (all frozen by default) ---
        # Defaults and interp="log" choices must match PWNModel._PARAM_DEFS.
        esn = Parameter("esn", 1.0, min=0.01, max=10.0,
                        frozen=True)  # [10^51 erg]
        mej = Parameter("mej", 6.0, min=1.0, max=50.0, unit="solMass",
                        frozen=True)
        n_ism = Parameter("n_ism", 0.4, min=1e-4, max=10.0,
                          unit="1 / cm3", interp="log", frozen=True)
        distance = Parameter("distance", 1000.0, min=10.0, frozen=True,
                             unit="pc")
        braking_index = Parameter("braking_index", 2.5, min=1.01, max=3.99,
                                  frozen=True)
        tau_sd = Parameter("tau_sd", 3000.0, min=100.0, max=1e6, unit="yr",
                           interp="log", frozen=True)
        eta_b = Parameter("eta_b", 0.01, min=1e-5, max=0.5,
                          interp="log", frozen=True)
        e_min = Parameter("e_min", 3.0, min=0.01, max=100.0, unit="GeV",
                          interp="log", frozen=True)
        e_max = Parameter("e_max", 5e5, min=1e3, max=1e8, unit="GeV",
                          interp="log", frozen=True)
        e_break = Parameter("e_break", 50.0, min=1.0, max=1e6, unit="GeV",
                            interp="log", frozen=True)
        p1 = Parameter("p1", 0.8, min=0.1, max=3.0, frozen=True)
        p2 = Parameter("p2", 2.5, min=1.0, max=4.0, frozen=True)

        # --- Optional IC photon fields (frozen/zero = disabled) ---
        ic_temp_1 = Parameter("ic_temp_1", 0.0, min=0.0, frozen=True, unit="K")
        ic_norm_1 = Parameter("ic_norm_1", 0.0, min=0.0, frozen=True, unit="eV / cm3")
        ic_temp_2 = Parameter("ic_temp_2", 0.0, min=0.0, frozen=True, unit="K")
        ic_norm_2 = Parameter("ic_norm_2", 0.0, min=0.0, frozen=True, unit="eV / cm3")

        _unit_map = {
            "mej": u.solMass,
            "n_ism": u.Unit("1 / cm3"),
            "distance": u.pc,
            "tau_sd": u.yr,
            "e_min": u.GeV,
            "e_max": u.GeV,
            "e_break": u.GeV,
            "ic_temp_1": u.K,
            "ic_temp_2": u.K,
            "ic_norm_1": u.eV / u.cm**3,
            "ic_norm_2": u.eV / u.cm**3,
        }

        def __init__(self, period=0.1, p_dot=1e-13,
                     injection_model='broken_powerlaw',
                     ssc_model='off', **kwargs):
            self.period = period
            self.p_dot = p_dot
            self.injection_model = injection_model
            self.ssc_model = ssc_model
            self._cache_hash = None
            self._cache_log_e = None
            self._cache_log_f = None
            # Auto-attach units to bare numeric values
            for name, unit in self._unit_map.items():
                if name in kwargs and not hasattr(kwargs[name], "unit"):
                    kwargs[name] = kwargs[name] * unit
            super().__init__(**kwargs)

        # Gammapy auto-converts Quantity parameters to display units
        # before evaluate(); convert back before stripping magnitudes.
        _INTERNAL_UNITS = {
            "mej": "solMass",
            "n_ism": "1 / cm3",
            "distance": "pc",
            "tau_sd": "yr",
            "e_min": "GeV",
            "e_max": "GeV",
            "e_break": "GeV",
            "ic_temp_1": "K",
            "ic_temp_2": "K",
            "ic_norm_1": "eV / cm3",
            "ic_norm_2": "eV / cm3",
        }

        def _to_internal(self, name, val):
            """Convert ``val`` to the model's internal unit for ``name``."""
            target = self._INTERNAL_UNITS.get(name)
            if target is not None and hasattr(val, "to"):
                return float(val.to(target).value)
            if hasattr(val, "value"):
                return float(val.value)
            return float(val)

        @staticmethod
        def _strip(val):
            """Strip units; kept for dimensionless inputs and back-compat."""
            if hasattr(val, "value"):
                return float(val.value)
            return float(val)

        def _param_hash(self, esn, mej, n_ism, distance, braking_index,
                        tau_sd, eta_b, e_min, e_max, e_break, p1, p2,
                        ic_temp_1, ic_norm_1, ic_temp_2, ic_norm_2):
            """Hash of all parameter values for cache invalidation.

            Uses canonical internal-unit values so two Quantity inputs
            in different but compatible units hash to the same key.
            """
            t = self._to_internal
            key = (t('esn', esn), t('mej', mej), t('n_ism', n_ism),
                   t('distance', distance), t('braking_index', braking_index),
                   t('tau_sd', tau_sd), t('eta_b', eta_b),
                   t('e_min', e_min), t('e_max', e_max), t('e_break', e_break),
                   t('p1', p1), t('p2', p2),
                   t('ic_temp_1', ic_temp_1), t('ic_norm_1', ic_norm_1),
                   t('ic_temp_2', ic_temp_2), t('ic_norm_2', ic_norm_2),
                   self.period, self.p_dot, self.injection_model,
                   self.ssc_model)
            return hashlib.md5(str(key).encode()).hexdigest()

        def evaluate(self, energy, esn, mej, n_ism, distance,
                     braking_index, tau_sd, eta_b, e_min, e_max,
                     e_break, p1, p2,
                     ic_temp_1, ic_norm_1, ic_temp_2, ic_norm_2):
            """Evaluate dN/dE at the given energies.

            Parameters
            ----------
            energy : `~astropy.units.Quantity`
                Energies (with units).
            esn, mej, n_ism, distance, braking_index, tau_sd, eta_b : \
float or `~astropy.units.Quantity`
                Physics parameters in gammapy display units; converted
                back to internal units.
            e_min, e_max, e_break, p1, p2 : float or `~astropy.units.Quantity`
                Injection-spectrum parameters, same handling.
            ic_temp_1, ic_norm_1, ic_temp_2, ic_norm_2 : \
float or `~astropy.units.Quantity`
                Additional IC-field parameters (0 = field disabled).

            Returns
            -------
            dnde : `~astropy.units.Quantity`
                Differential photon flux [cm^-2 s^-1 TeV^-1].
            """
            if hasattr(energy, "to"):
                e_tev = energy.to("TeV").value
            else:
                e_tev = np.asarray(energy, dtype=float)

            h = self._param_hash(
                esn, mej, n_ism, distance, braking_index, tau_sd,
                eta_b, e_min, e_max, e_break, p1, p2,
                ic_temp_1, ic_norm_1, ic_temp_2, ic_norm_2,
            )
            if h != self._cache_hash:
                logger.info("PWNSpectralModel: running model (params changed)")
                t = self._to_internal
                log_e, log_f = _run_and_convert(
                    esn=t('esn', esn),
                    mej=t('mej', mej),
                    n_ism=t('n_ism', n_ism),
                    braking_index=t('braking_index', braking_index),
                    tau_sd=t('tau_sd', tau_sd),
                    eta_b=t('eta_b', eta_b),
                    e_min=t('e_min', e_min),
                    e_max=t('e_max', e_max),
                    e_break=t('e_break', e_break),
                    p1=t('p1', p1), p2=t('p2', p2),
                    period=self.period, p_dot=self.p_dot,
                    distance_pc=t('distance', distance),
                    ic_temp_1=t('ic_temp_1', ic_temp_1),
                    ic_norm_1=t('ic_norm_1', ic_norm_1),
                    ic_temp_2=t('ic_temp_2', ic_temp_2),
                    ic_norm_2=t('ic_norm_2', ic_norm_2),
                    injection_model=self.injection_model,
                    ssc_model=self.ssc_model,
                )
                self._cache_hash = h
                self._cache_log_e = log_e
                self._cache_log_f = log_f

            dnde = evaluate_dnde(e_tev, self._cache_log_e, self._cache_log_f)
            return dnde * u.Unit("cm-2 s-1 TeV-1")

        # ------------------------------------------------------------------
        # Serialization: the default path drops non-Parameter attributes.
        # ------------------------------------------------------------------

        _EXTRA_ATTRS = ("period", "p_dot", "injection_model", "ssc_model")

        def to_dict(self, full_output=False):
            """Serialize, including the non-Parameter physics attributes."""
            data = super().to_dict(full_output=full_output)
            inner = data.get("spectral", data)
            for attr in self._EXTRA_ATTRS:
                inner[attr] = getattr(self, attr)
            return data

        @classmethod
        def from_dict(cls, data, **kwargs):
            """Reconstruct, restoring the non-Parameter physics attributes."""
            key0 = next(iter(data))
            wrapped = key0 in ("spatial", "temporal", "spectral")
            inner = dict(data[key0] if wrapped else data)
            for attr in cls._EXTRA_ATTRS:
                if attr in inner:
                    kwargs.setdefault(attr, inner.pop(attr))
            outer = {key0: inner} if wrapped else inner
            return super().from_dict(outer, **kwargs)

        @classmethod
        def for_cta1(cls):
            """Representative parameters for CTA 1 (PSR J0007+7303): the
            package defaults with the J0007+7303 timing."""
            return cls(
                period=0.31686, p_dot=3.614e-13,
                esn=1.0,
                mej=6.0 * u.solMass,
                n_ism=0.4 * u.Unit("1 / cm3"),
                distance=1400.0 * u.pc,
                braking_index=2.5,
                tau_sd=3000.0 * u.yr,
                eta_b=0.01,
                e_min=3.0 * u.GeV,
                e_max=5e5 * u.GeV,
                e_break=50.0 * u.GeV,
                p1=0.8, p2=2.5,
            )

        @classmethod
        def for_g54(cls, theta=None):
            """Pre-configured model for G54.1+0.3 (PSR J1930+1852).

            Parameters
            ----------
            theta : array-like of 14 floats, optional
                Full parameter vector. If None, uses the best-fit values
                from Alford, Zhang & Gelfand (2025).
            """
            if theta is None:
                theta = [
                    9.49577657e-01, 2.08950062e+01, 8.49315308e-03,
                    6.22334375e+03, 1.90038240e+00, 3.57365910e+03,
                    2.39626337e-03, 9.98260706e+00, 4.00865090e+05,
                    3.23166390e+03, 2.29260528e+00, 2.42195225e+00,
                    # ic_temp_1 [K]; ic_norm_1 [eV/cm^3]
                    2.75305883e+03, 2.3613683661006988e+01,
                ]
            return cls(
                period=0.136, p_dot=7.5112e-13,
                esn=theta[0],
                mej=theta[1] * u.solMass,
                n_ism=theta[2] * u.Unit("1 / cm3"),
                distance=theta[3] * u.pc,
                braking_index=theta[4],
                tau_sd=theta[5] * u.yr,
                eta_b=theta[6],
                e_min=theta[7] * u.GeV,
                e_max=theta[8] * u.GeV,
                e_break=theta[9] * u.GeV,
                p1=theta[10], p2=theta[11],
                ic_temp_1=theta[12] * u.K,
                ic_norm_1=theta[13],
            )

    # Registration lets Models.read resolve the model tag.
    from gammapy.modeling.models import SPECTRAL_MODEL_REGISTRY

    if not any(m is PWNSpectralModel for m in SPECTRAL_MODEL_REGISTRY):
        SPECTRAL_MODEL_REGISTRY.append(PWNSpectralModel)

else:
    class PWNSpectralModel:
        """Placeholder when gammapy is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "gammapy is required for PWNSpectralModel. "
                "Install it with: pip install gammapy\n"
                "The standalone helper evaluate_dnde() works without "
                "gammapy."
            )
