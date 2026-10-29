"""Astropy-units radiative models."""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import astropy.units as u

from . import kernels as K
from .quadrature import QuadratureInfo, integrate_population, trapz_loglog
from .seeds import Blackbody, TableSeed, as_seed, ssc_seed

__all__ = ["Synchrotron", "InverseCompton", "SSC", "TableDistribution",
           "PowerLaw", "ExponentialCutoffPowerLaw", "BrokenPowerLaw"]

_DEFAULT_EMIN = 1.0 * u.GeV
_DEFAULT_EMAX = (1e9 * K.MEC2) * u.erg          # gamma = 1e9
_FLUX_UNIT = 1 / (u.s * u.cm**2 * u.eV)
_LUM_UNIT = 1 / (u.s * u.eV)


# distributions
class TableDistribution:
    """Tabulated particle distribution."""

    def __init__(self, energy, dNdE) -> None:
        self.energy = u.Quantity(energy).to(u.erg)
        self.dNdE = u.Quantity(dNdE).to(1 / u.erg)
        if np.any(np.diff(self.energy.value) <= 0):
            raise ValueError("energy must be strictly increasing")
        self.Emin = self.energy[0]
        self.Emax = self.energy[-1]
        self._lnE = np.log(self.energy.value)
        v = self.dNdE.value
        self._lnN = np.log(np.where(v > 0, v, 1e-300))

    def __call__(self, E) -> u.Quantity:
        e = u.Quantity(E).to_value(u.erg)
        lne = np.log(e)
        out = np.exp(np.interp(lne, self._lnE, self._lnN))
        out = np.where((lne >= self._lnE[0]) & (lne <= self._lnE[-1]), out, 0.0)
        out = np.where(out < 1e-299, 0.0, out)
        return out / u.erg


class PowerLaw:
    """Power law."""

    def __init__(self, amplitude, e_0, alpha) -> None:
        self.amplitude = u.Quantity(amplitude)
        self.e_0 = u.Quantity(e_0)
        self.alpha = float(alpha)

    def __call__(self, E) -> u.Quantity:
        x = (u.Quantity(E) / self.e_0).to_value(u.one)
        return self.amplitude * x ** -self.alpha


class ExponentialCutoffPowerLaw:
    """Exponential-cutoff power law."""

    def __init__(self, amplitude, e_0, alpha, e_cutoff, beta: float = 1.0) -> None:
        self.amplitude = u.Quantity(amplitude)
        self.e_0 = u.Quantity(e_0)
        self.alpha = float(alpha)
        self.e_cutoff = u.Quantity(e_cutoff)
        self.beta = float(beta)

    def __call__(self, E) -> u.Quantity:
        E = u.Quantity(E)
        x = (E / self.e_0).to_value(u.one)
        xc = (E / self.e_cutoff).to_value(u.one)
        return self.amplitude * x ** -self.alpha * np.exp(-xc**self.beta)


class BrokenPowerLaw:
    """Broken power law."""

    def __init__(self, amplitude, e_0, e_break, alpha_1, alpha_2) -> None:
        self.amplitude = u.Quantity(amplitude)
        self.e_0 = u.Quantity(e_0)
        self.e_break = u.Quantity(e_break)
        self.alpha_1 = float(alpha_1)
        self.alpha_2 = float(alpha_2)

    def __call__(self, E) -> u.Quantity:
        E = u.Quantity(E)
        x = (E / self.e_0).to_value(u.one)
        xb = (self.e_break / self.e_0).to_value(u.one)
        lo = x ** -self.alpha_1
        hi = xb ** (self.alpha_2 - self.alpha_1) * x ** -self.alpha_2
        return self.amplitude * np.where(x < xb, lo, hi)


# base classes
def _photon_energy_erg(photon_energy) -> tuple[np.ndarray, u.Quantity, tuple]:
    q = u.Quantity(photon_energy)
    e = q.to(u.erg, equivalencies=u.spectral())
    return np.atleast_1d(e.value).ravel(), e, q.shape


def _is_zero_distance(distance) -> bool:
    if isinstance(distance, u.Quantity):
        return bool(distance.value == 0)
    return distance == 0


class _Composable:
    """flux, sed, luminosity and sum."""

    def _rate(self, eps: np.ndarray) -> np.ndarray:  # photons/(s erg)
        raise NotImplementedError

    def flux(self, photon_energy, distance=1 * u.kpc) -> u.Quantity:
        """Photon flux; distance=0 is intrinsic."""
        eps, _, shape = _photon_energy_erg(photon_energy)
        rate = self._rate(eps).reshape(shape) / u.erg / u.s
        if _is_zero_distance(distance):
            return rate.to(_LUM_UNIT)
        d = u.Quantity(distance).to(u.cm)
        return (rate / (4.0 * np.pi * d**2)).to(_FLUX_UNIT)

    def sed(self, photon_energy, distance=1 * u.kpc) -> u.Quantity:
        """E^2 dN/dE."""
        _, e, shape = _photon_energy_erg(photon_energy)
        f = self.flux(photon_energy, distance)
        out = (f * e.reshape(shape) ** 2)
        return out.to(u.erg / u.s if _is_zero_distance(distance)
                      else u.erg / (u.cm**2 * u.s))

    def luminosity(self, photon_energy) -> u.Quantity:
        """Intrinsic nu L_nu, erg/s."""
        return self.sed(photon_energy, distance=0)

    def __add__(self, other: "_Composable") -> "SumModel":
        return SumModel([self, other])


class SumModel(_Composable):
    """Sum of models."""

    def __init__(self, models: Iterable[_Composable]) -> None:
        flat = []
        for m in models:
            flat.extend(m.models if isinstance(m, SumModel) else [m])
        self.models = flat

    def _rate(self, eps: np.ndarray) -> np.ndarray:
        return np.sum([m._rate(eps) for m in self.models], axis=0)


class _ElectronModel(_Composable):
    """Electron-population base class."""

    def __init__(self, particle_distribution: Callable, Eemin=None, Eemax=None,
                 rtol: float = 1e-4, floor: float = 1e-6,
                 max_per_decade: float = 4096.0) -> None:
        self.particle_distribution = particle_distribution
        if Eemin is None:
            Eemin = getattr(particle_distribution, "Emin", _DEFAULT_EMIN)
        if Eemax is None:
            Eemax = getattr(particle_distribution, "Emax", _DEFAULT_EMAX)
        self.Eemin = u.Quantity(Eemin).to(u.erg)
        self.Eemax = u.Quantity(Eemax).to(u.erg)
        if not self.Eemax > self.Eemin:
            raise ValueError("Eemax must exceed Eemin")
        self.rtol = float(rtol)
        self.floor = float(floor)
        self.max_per_decade = float(max_per_decade)
        self.quadrature: QuadratureInfo | None = None
        # check units
        test = u.Quantity(particle_distribution(np.array([1.0, 10.0]) * u.TeV))
        if not test.unit.is_equivalent(1 / u.erg):
            raise u.UnitsError("particle_distribution must return 1/energy")

    def _density(self, E: np.ndarray) -> np.ndarray:
        return np.asarray(u.Quantity(self.particle_distribution(E * u.erg))
                          .to_value(1 / u.erg), dtype=float)

    def _kernel(self, E: np.ndarray, eps: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _rate(self, eps: np.ndarray) -> np.ndarray:
        res, info = integrate_population(
            self._kernel, self._density,
            self.Eemin.value, self.Eemax.value, eps, rtol=self.rtol,
            floor=self.floor, max_per_decade=self.max_per_decade)
        self.quadrature = info
        return res

    @property
    def We(self) -> u.Quantity:
        """Total particle energy."""
        E = np.geomspace(self.Eemin.value, self.Eemax.value, 4097)
        return trapz_loglog(E * self._density(E), E) * u.erg


# synchrotron
class Synchrotron(_ElectronModel):
    """Synchrotron emission."""

    def __init__(self, particle_distribution, B=3.24e-6 * u.G,
                 kernel: str = "isotropic", sin_alpha: float = K.SQRT_2_3,
                 **kwargs) -> None:
        super().__init__(particle_distribution, **kwargs)
        self.B = u.Quantity(B).to(u.G)
        if kernel not in ("isotropic", "fixed", "turbulent"):
            raise ValueError(f"unknown synchrotron kernel {kernel!r}")
        self.kernel = kernel
        self.sin_alpha = float(sin_alpha)

    def _kernel(self, E: np.ndarray, eps: np.ndarray) -> np.ndarray:
        return K.synch_photon_rate(eps, E / K.MEC2, float(self.B.value),
                                   self.kernel, self.sin_alpha)

    def energy_loss_rate(self, E) -> u.Quantity:
        """Synchrotron loss rate."""
        e = u.Quantity(E).to_value(u.erg)
        return K.synch_loss_rate(e / K.MEC2, float(self.B.value)) * u.erg / u.s


# inverse Compton
class InverseCompton(_ElectronModel):
    """Inverse Compton emission."""

    def __init__(self, particle_distribution, seeds=("CMB",), **kwargs) -> None:
        super().__init__(particle_distribution, **kwargs)
        if isinstance(seeds, (str, Blackbody, TableSeed)):
            seeds = [seeds]
        self.seeds = [as_seed(s) for s in seeds]
        self._select: list | None = None

    @property
    def seed_names(self) -> list[str]:
        return [s.name for s in self.seeds]

    def _pick(self, seed) -> list:
        if seed is None:
            return self.seeds
        if isinstance(seed, int):
            return [self.seeds[seed]]
        for s in self.seeds:
            if s.name == seed:
                return [s]
        raise ValueError(f"no seed named {seed!r}; have {self.seed_names}")

    def _kernel(self, E: np.ndarray, eps: np.ndarray) -> np.ndarray:
        g = E / K.MEC2
        seeds = self._select if self._select is not None else self.seeds
        out = seeds[0]._kernel(eps, g, 0.1 * self.rtol)
        for s in seeds[1:]:
            out = out + s._kernel(eps, g, 0.1 * self.rtol)
        return out

    def _rate(self, eps: np.ndarray) -> np.ndarray:
        # ln(E - eps) grid
        seeds = self._select if self._select is not None else self.seeds
        e0max = max(s._eps_max() for s in seeds)
        Emax = self.Eemax.value
        dmin = 0.5 * eps * K.MEC2**2 / (4.0 * Emax * e0max)
        res, info = integrate_population(
            self._kernel, self._density, self.Eemin.value, Emax, eps,
            rtol=self.rtol, floor=self.floor,
            max_per_decade=self.max_per_decade, threshold=eps, delta_min=dmin)
        self.quadrature = info
        return res

    def flux(self, photon_energy, distance=1 * u.kpc, seed=None) -> u.Quantity:
        """Photon flux, optionally one seed."""
        self._select = None if seed is None else self._pick(seed)
        try:
            return super().flux(photon_energy, distance)
        finally:
            self._select = None

    def sed(self, photon_energy, distance=1 * u.kpc, seed=None) -> u.Quantity:
        """E^2 dN/dE, optionally one seed."""
        _, e, shape = _photon_energy_erg(photon_energy)
        f = self.flux(photon_energy, distance, seed=seed)
        return (f * e.reshape(shape) ** 2).to(
            u.erg / u.s if _is_zero_distance(distance) else u.erg / (u.cm**2 * u.s))

    def energy_loss_rate(self, E, seed=None) -> u.Quantity:
        """IC loss rate."""
        e = np.asarray(u.Quantity(E).to_value(u.erg), dtype=float)
        tot = sum(s._loss(e) for s in self._pick(seed))
        return tot * u.erg / u.s


class SSC(InverseCompton):
    """Uniform-sphere SSC."""

    def __init__(self, particle_distribution, B, R, kernel: str = "isotropic",
                 extra_seeds=(), geometry: str = "uniform_sphere", **kwargs) -> None:
        self.synchrotron = Synchrotron(particle_distribution, B=B, kernel=kernel,
                                       **kwargs)
        seed = ssc_seed(self.synchrotron, R, geometry=geometry)
        super().__init__(particle_distribution, seeds=[seed, *extra_seeds],
                         **kwargs)
