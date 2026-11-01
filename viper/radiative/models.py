"""Astropy-units radiative models."""

from __future__ import annotations

import warnings
from typing import Callable, Iterable

import numpy as np
import astropy.units as u
from scipy.interpolate import CubicSpline, PchipInterpolator

from .. import constants as C
from . import hadronic as H
from . import kernels as K
from .quadrature import (QuadratureInfo, QuadratureWarning,
                         integrate_piecewise, trapz_loglog)
from .seeds import Seed, as_seed, ssc_seed

__all__ = ["Synchrotron", "InverseCompton", "SSC", "SumModel",
           "Bremsstrahlung", "PionDecay", "TableDistribution",
           "PowerLaw", "ExponentialCutoffPowerLaw", "BrokenPowerLaw"]

_DEFAULT_EMIN = 1.0 * u.GeV
_DEFAULT_EMAX = (1e9 * K.MEC2) * u.erg          # gamma = 1e9
_FLUX_UNIT = 1 / (u.s * u.cm**2 * u.eV)
_LUM_UNIT = 1 / (u.s * u.eV)
_ON_UNCONVERGED = ("warn", "raise", "ignore")


# distributions
class TableDistribution:
    """Tabulated particle distribution, log-log."""

    def __init__(self, energy, dNdE, interpolation: str = "pchip") -> None:
        self.energy = u.Quantity(energy).to(u.erg)
        self.dNdE = u.Quantity(dNdE).to(1 / u.erg)
        e = np.atleast_1d(self.energy.value)
        v = np.atleast_1d(self.dNdE.value)
        if e.ndim != 1 or v.shape != e.shape:
            raise ValueError("energy and dNdE must be 1-D of equal length")
        if e.size < 2 or np.any(np.diff(e) <= 0):
            raise ValueError("energy must be strictly increasing")
        pos = v > 0
        if pos.sum() < 2:
            raise ValueError("need at least two positive points")
        self._lnE = np.log(e)
        self._pos = pos
        lne, lnv = self._lnE[pos], np.log(v[pos])
        if interpolation == "pchip":
            self._f = PchipInterpolator(lne, lnv, extrapolate=False)
        elif interpolation == "cubic":
            self._f = CubicSpline(lne, lnv, extrapolate=False)
        elif interpolation == "linear":
            self._f = lambda x: np.interp(x, lne, lnv)
        else:
            raise ValueError("interpolation must be 'pchip', 'cubic' or 'linear'")
        self.interpolation = interpolation
        self.Emin = self.energy[pos][0]
        self.Emax = self.energy[pos][-1]

    def __call__(self, E) -> u.Quantity:
        e = np.asarray(u.Quantity(E).to_value(u.erg), dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            lne = np.log(e)
        lnE, pos = self._lnE, self._pos
        i = np.clip(np.searchsorted(lnE, lne, side="right") - 1, 0, lnE.size - 2)
        node = (pos[i] & (lne == lnE[i])) | (pos[i + 1] & (lne == lnE[i + 1]))
        inside = (lne >= lnE[0]) & (lne <= lnE[-1]) & ((pos[i] & pos[i + 1]) | node)
        x = np.where(inside, lne, lnE[pos][0])
        out = np.where(inside, np.exp(self._f(x)), 0.0)
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


def _distance_cm(distance) -> float | None:
    """cm, or None when intrinsic."""
    if distance is None:
        return None
    d = u.Quantity(distance)
    if d.value == 0:
        return None
    d_cm = float(d.to_value(u.cm))
    if d_cm < 0:
        raise ValueError("distance must be non-negative")
    return d_cm


class _Composable:
    """flux, sed, luminosity and sum."""

    def _rate(self, eps: np.ndarray, **kw) -> np.ndarray:  # photons/(s erg)
        raise NotImplementedError

    def flux(self, photon_energy, distance=1 * u.kpc, **kw) -> u.Quantity:
        """Photon flux; distance None or 0 is intrinsic."""
        eps, _, shape = _photon_energy_erg(photon_energy)
        rate = self._rate(eps, **kw).reshape(shape) / u.erg / u.s
        d = _distance_cm(distance)
        if d is None:
            return rate.to(_LUM_UNIT)
        return (rate / (4.0 * np.pi * d**2 * u.cm**2)).to(_FLUX_UNIT)

    def sed(self, photon_energy, distance=1 * u.kpc, **kw) -> u.Quantity:
        """E^2 dN/dE."""
        _, e, shape = _photon_energy_erg(photon_energy)
        f = self.flux(photon_energy, distance, **kw)
        out = f * e.reshape(shape) ** 2
        return out.to(u.erg / u.s if _distance_cm(distance) is None
                      else u.erg / (u.cm**2 * u.s))

    def luminosity(self, photon_energy, **kw) -> u.Quantity:
        """Intrinsic nu L_nu, erg/s."""
        return self.sed(photon_energy, distance=None, **kw)

    def __add__(self, other: "_Composable") -> "SumModel":
        return SumModel([self, other])


class SumModel(_Composable):
    """Sum of models."""

    def __init__(self, models: Iterable[_Composable]) -> None:
        flat = []
        for m in models:
            flat.extend(m.models if isinstance(m, SumModel) else [m])
        self.models = flat

    def _rate(self, eps: np.ndarray, **kw) -> np.ndarray:
        return np.sum([m._rate(eps, **kw) for m in self.models], axis=0)

    @property
    def quadrature(self) -> QuadratureInfo | None:
        """Worst part."""
        infos = [m.quadrature for m in self.models
                 if getattr(m, "quadrature", None) is not None]
        if not infos:
            return None
        return max(infos, key=lambda i: (not i.converged, i.rel_error))


class _ParticleModel(_Composable):
    """Particle-population base class."""

    _default_Emin: u.Quantity = _DEFAULT_EMIN
    _default_Emax: u.Quantity = _DEFAULT_EMAX

    def __init__(self, particle_distribution: Callable, Emin=None, Emax=None,
                 rtol: float = 1e-4, floor: float = 1e-6,
                 max_per_decade: float = 4096.0,
                 on_unconverged: str = "warn") -> None:
        self.particle_distribution = particle_distribution
        if Emin is None:
            Emin = getattr(particle_distribution, "Emin", self._default_Emin)
        if Emax is None:
            Emax = getattr(particle_distribution, "Emax", self._default_Emax)
        self.Emin = u.Quantity(Emin).to(u.erg)
        self.Emax = u.Quantity(Emax).to(u.erg)
        if not self.Emax > self.Emin:
            raise ValueError("Emax must exceed Emin")
        self.rtol = float(rtol)
        self.floor = float(floor)
        self.max_per_decade = float(max_per_decade)
        if on_unconverged not in _ON_UNCONVERGED:
            raise ValueError(f"on_unconverged must be one of {_ON_UNCONVERGED}")
        self.on_unconverged = on_unconverged
        self.quadrature: QuadratureInfo | None = None
        test = u.Quantity(particle_distribution(np.array([1.0, 10.0]) * u.TeV))
        if not test.unit.is_equivalent(1 / u.erg):
            raise u.UnitsError("particle_distribution must return 1/energy")

    def _density(self, E: np.ndarray) -> np.ndarray:
        return np.asarray(u.Quantity(self.particle_distribution(E * u.erg))
                          .to_value(1 / u.erg), dtype=float)

    def _integrate(self, kernel: Callable, edges, eps: np.ndarray,
                   threshold=None, delta_min=None
                   ) -> tuple[np.ndarray, QuadratureInfo]:
        opts = dict(rtol=self.rtol, floor=self.floor,
                    max_per_decade=self.max_per_decade)
        if self.on_unconverged == "warn":
            return integrate_piecewise(kernel, self._density, edges, eps,
                                       threshold, delta_min, **opts)
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore" if self.on_unconverged == "ignore" else "error",
                QuadratureWarning)
            return integrate_piecewise(kernel, self._density, edges, eps,
                                       threshold, delta_min, **opts)

    def _finish(self, *infos: QuadratureInfo) -> QuadratureInfo:
        info = max(infos, key=lambda i: i.rel_error) if infos \
            else QuadratureInfo(0.0, 0.0, 0, True)
        self.quadrature = info
        if not info.converged and self.on_unconverged != "ignore":
            msg = (f"{type(self).__name__}: rel_error {info.rel_error:.2e} > "
                   f"rtol {self.rtol:.0e} at {info.per_decade:.0f} points per "
                   f"decade (max_per_decade={self.max_per_decade:g})")
            if self.on_unconverged == "raise":
                raise QuadratureWarning(msg)
            warnings.warn(msg, QuadratureWarning, stacklevel=4)
        return info

    @property
    def W(self) -> u.Quantity:
        """Total particle energy."""
        E = np.geomspace(self.Emin.value, self.Emax.value, 4097)
        return trapz_loglog(E * self._density(E), E) * u.erg


def _alias(attr: str) -> property:
    """Property aliasing a base attribute."""
    return property(lambda self: getattr(self, attr),
                    lambda self, v: setattr(self, attr, u.Quantity(v).to(u.erg)))


class _ElectronModel(_ParticleModel):
    """Electron-population base class."""

    def __init__(self, particle_distribution: Callable, Eemin=None, Eemax=None,
                 **kwargs) -> None:
        super().__init__(particle_distribution, Eemin, Eemax, **kwargs)

    Eemin = _alias("Emin")
    Eemax = _alias("Emax")
    We = property(lambda self: self.W)


class _ProtonModel(_ParticleModel):
    """Proton-population base class."""

    _default_Emin = (H.MPC2 + H.TP_TH) * u.erg
    _default_Emax = 10 * u.PeV

    def __init__(self, particle_distribution: Callable, Epmin=None, Epmax=None,
                 **kwargs) -> None:
        super().__init__(particle_distribution, Epmin, Epmax, **kwargs)

    Epmin = _alias("Emin")
    Epmax = _alias("Emax")
    Wp = property(lambda self: self.W)


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

    def _photon_rate(self, E: np.ndarray, eps: np.ndarray, k: int) -> np.ndarray:
        return K.synch_photon_rate(eps, E / K.MEC2, self.B.to_value(u.G),
                                   self.kernel, self.sin_alpha)

    def _rate(self, eps: np.ndarray) -> np.ndarray:
        res, info = self._integrate(self._photon_rate,
                                    [self.Emin.value, self.Emax.value], eps)
        self._finish(info)
        return res

    def energy_loss_rate(self, E) -> u.Quantity:
        """Synchrotron loss rate."""
        e = u.Quantity(E).to_value(u.erg)
        return K.synch_loss_rate(e / K.MEC2, self.B.to_value(u.G)) * u.erg / u.s


# inverse Compton
class InverseCompton(_ElectronModel):
    """Inverse Compton emission."""

    def __init__(self, particle_distribution, seeds=("CMB",), **kwargs) -> None:
        super().__init__(particle_distribution, **kwargs)
        if isinstance(seeds, (str, Seed)):
            seeds = [seeds]
        self.seeds = [as_seed(s) for s in seeds]

    @property
    def seed_names(self) -> list[str]:
        return [s.name for s in self.seeds]

    def _pick(self, seed) -> list[Seed]:
        if seed is None:
            return self.seeds
        if isinstance(seed, int):
            return [self.seeds[seed]]
        for s in self.seeds:
            if s.name == seed:
                return [s]
        raise ValueError(f"no seed named {seed!r}; have {self.seed_names}")

    def _photon_rate(self, E: np.ndarray, eps: np.ndarray,
                     seeds: list[Seed]) -> np.ndarray:
        g = E / K.MEC2
        out = seeds[0].photon_rate(eps, g, 0.1 * self.rtol)
        for s in seeds[1:]:
            out = out + s.photon_rate(eps, g, 0.1 * self.rtol)
        return out

    def _rate(self, eps: np.ndarray, seed=None) -> np.ndarray:
        seeds = self._pick(seed)
        e0max = max(s.eps_max for s in seeds)
        Emax = self.Emax.value
        # ln(E - eps) grid
        dmin = 0.5 * eps * K.MEC2**2 / (4.0 * Emax * e0max)
        res, info = self._integrate(
            lambda E, e, k: self._photon_rate(E, e, seeds),
            [self.Emin.value, Emax], eps, eps, dmin)
        self._finish(info)
        return res

    def energy_loss_rate(self, E, seed=None) -> u.Quantity:
        """IC loss rate."""
        e = np.asarray(u.Quantity(E).to_value(u.erg), dtype=float)
        tot = sum(s.loss_rate(e) for s in self._pick(seed))
        return tot * u.erg / u.s


class SSC(InverseCompton):
    """Uniform-sphere SSC."""

    def __init__(self, particle_distribution, B, R, kernel: str = "isotropic",
                 extra_seeds=(), geometry: str = "uniform_sphere", **kwargs) -> None:
        self.synchrotron = Synchrotron(particle_distribution, B=B, kernel=kernel,
                                       **kwargs)
        self.R = u.Quantity(R).to(u.cm)
        self.geometry = geometry
        self._seed_key = None
        super().__init__(particle_distribution,
                         seeds=[self._ssc_seed(), *extra_seeds], **kwargs)

    @property
    def B(self) -> u.Quantity:
        return self.synchrotron.B

    @B.setter
    def B(self, value) -> None:
        self.synchrotron.B = u.Quantity(value).to(u.G)

    def _key(self) -> tuple:
        syn = self.synchrotron
        return (syn.B.to_value(u.G), self.R.to_value(u.cm),
                id(syn.particle_distribution), syn.kernel, syn.sin_alpha)

    def _ssc_seed(self):
        self._seed_key = self._key()
        return ssc_seed(self.synchrotron, self.R, geometry=self.geometry)

    def _rate(self, eps: np.ndarray, seed=None) -> np.ndarray:
        if self._key() != self._seed_key:
            self.seeds[0] = self._ssc_seed()
        return super()._rate(eps, seed)


# bremsstrahlung
class Bremsstrahlung(_ElectronModel):
    """Relativistic bremsstrahlung, Baring99."""

    def __init__(self, particle_distribution, n0=1 / u.cm**3,
                 weight_ee: float = 1.088, weight_ep: float = 1.263,
                 elwert: bool = True, **kwargs) -> None:
        super().__init__(particle_distribution, **kwargs)
        self.n0 = u.Quantity(n0).to(u.cm**-3)
        self.weight_ee = float(weight_ee)
        self.weight_ep = float(weight_ep)
        self.elwert = bool(elwert)

    def _rate(self, eps: np.ndarray) -> np.ndarray:
        Emin, Emax = self.Emin.value, self.Emax.value
        mec2 = K.MEC2
        w = eps / mec2
        total = np.zeros(eps.size)
        infos = []
        if self.weight_ep != 0.0:
            thr = eps + mec2

            def kep(E, e, k):
                g = E / mec2
                return (C.c_light * np.sqrt(1.0 - 1.0 / g**2)
                        * H.brems_ep_sigma(e, g, 1.0, self.elwert))

            r, info = self._integrate(kep, [Emin, Emax], eps, thr, 1e-10 * thr)
            total += self.weight_ep * r
            infos.append(info)
        if self.weight_ee != 0.0:
            Esw = mec2 * (1.0 + H._T_SWITCH)
            edges = [Emin, Emax] if not Emin < Esw < Emax else [Emin, Esw, Emax]
            regimes = ["rel" if Emin >= Esw else "nr"] if len(edges) == 2 \
                else ["nr", "rel"]
            thr = mec2 * np.maximum(1.0 + H._ee_T_min(w), np.sqrt(1.0 + 4.0 * w))

            def kee(E, e, k):
                g = E / mec2
                return (C.c_light * np.sqrt(1.0 - 1.0 / g**2)
                        * H.brems_ee_sigma(e, g, regimes[k]))

            r, info = self._integrate(kee, edges, eps, thr, 1e-10 * thr)
            total += self.weight_ee * r
            infos.append(info)
        self._finish(*infos)
        return self.n0.to_value(u.cm**-3) * total


# pion decay
class PionDecay(_ProtonModel):
    """pp pi0 decay, KATV14; dN/dE in total energy."""

    def __init__(self, particle_distribution, nh=1 / u.cm**3,
                 nuclear_enhancement=True, model: str = "Pythia8",
                 **kwargs) -> None:
        super().__init__(particle_distribution, **kwargs)
        self.nh = u.Quantity(nh).to(u.cm**-3)
        self.model = H._check_model(model)
        if not isinstance(nuclear_enhancement, bool):
            nuclear_enhancement = float(nuclear_enhancement)
        self.nuclear_enhancement = nuclear_enhancement

    def _edges(self) -> tuple[np.ndarray, list]:
        """Piece edges and reference Tp."""
        lo, hi = self.Emin.value, self.Emax.value
        brk = [b for b in H._TP_BREAKS if b != 50.0 or self.model == "Pythia8"]
        brk = [H._TTH] + brk
        inner = [H.MPC2 + b * H._GEV for b in brk if lo < H.MPC2 + b * H._GEV < hi]
        edges = np.array([lo] + inner + [hi])
        tp = (edges - H.MPC2) / H._GEV
        ref = [0.5 * (tp[i] + tp[i + 1]) for i in range(len(tp) - 1)]
        return edges, ref

    def _rate(self, eps: np.ndarray) -> np.ndarray:
        edges, ref = self._edges()
        Eg = eps / H._GEV
        thr = H.MPC2 + H._tp_min_gev(Eg) * H._GEV
        nuc = self.nuclear_enhancement
        model = self.model

        def kernel(E, e, k):
            Tp = (E - H.MPC2) / H._GEV
            beta = np.sqrt(Tp * (Tp + 2.0 * H._MP)) / (Tp + H._MP)
            ds = H._dsigma_gev(e / H._GEV, Tp, model, ref[k], nuc)
            return (C.c_light * H._MB / H._GEV) * beta * ds

        res, info = self._integrate(kernel, edges, eps, thr,
                                    1e-10 * (thr - H.MPC2))
        self._finish(info)
        return self.nh.to_value(u.cm**-3) * res
