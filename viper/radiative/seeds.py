"""IC seed photon fields."""

from __future__ import annotations

import numpy as np
import astropy.units as u_

from . import kernels as K

__all__ = ["Blackbody", "TableSeed", "ssc_seed", "as_seed"]

_PRESETS = {
    # T [K], u [eV/cm^3]
    "CMB": (K.T_CMB, None),
    "FIR": (30.0, 0.5),
    "NIR": (3000.0, 1.0),
}


class Blackbody:
    """Blackbody seed field."""

    def __init__(self, T, u=None, theta=None, name: str | None = None) -> None:
        if isinstance(T, str):
            key = T.upper()
            if key not in _PRESETS:
                raise ValueError(f"unknown blackbody preset {T!r}")
            name = name or key
            T_K, u_eV = _PRESETS[key]
            T = T_K * u_.K
            if u is None and u_eV is not None:
                u = u_eV * u_.eV / u_.cm**3
        self.T = u_.Quantity(T).to(u_.K, equivalencies=u_.temperature_energy())
        if self.T.value <= 0:
            raise ValueError("temperature must be positive")
        self.u = None if u is None else u_.Quantity(u).to(u_.erg / u_.cm**3)
        self.theta = None if theta is None else u_.Quantity(theta).to(u_.rad)
        self.name = name or f"BB{self.T.value:g}K"

    @property
    def energy_density(self) -> u_.Quantity:
        """Field energy density."""
        if self.u is not None:
            return self.u
        return (K.A_RAD * self.T.value**4) * u_.erg / u_.cm**3

    def _eps_max(self) -> float:
        """Upper seed energy, 60 kT."""
        return 60.0 * K.C.k_B * float(self.T.value)

    def _u_cgs(self) -> float | None:
        return None if self.u is None else float(self.u.value)

    def _kernel(self, eps: np.ndarray, gamma: np.ndarray, rtol: float) -> np.ndarray:
        theta = None if self.theta is None else float(self.theta.value)
        return K.ic_blackbody_kernel(eps, gamma, float(self.T.value),
                                     self._u_cgs(), theta)

    def _loss(self, E: np.ndarray) -> np.ndarray:
        if self.theta is not None:
            raise NotImplementedError(
                "loss rate implemented for isotropic seeds only")
        return K.ic_loss_rate_blackbody(E, float(self.T.value), self._u_cgs())

    def __repr__(self) -> str:
        extra = "" if self.u is None else f", u={self.u:.4g}"
        ang = "" if self.theta is None else f", theta={self.theta:.4g}"
        return f"Blackbody({self.T:.6g}{extra}{ang}, name={self.name!r})"



class TableSeed:
    """Tabulated isotropic seed field."""

    def __init__(self, energy, density, interpolation: str = "cubic",
                 name: str | None = None) -> None:
        E = u_.Quantity(energy).to(u_.erg, equivalencies=u_.spectral())
        dens = u_.Quantity(density)
        if dens.unit.is_equivalent(1 / (u_.cm**3 * u_.erg)):
            dnde = dens.to_value(1 / (u_.cm**3 * u_.erg))
        elif dens.unit.is_equivalent(u_.erg / u_.cm**3):
            dnde = dens.to_value(u_.erg / u_.cm**3) / E.value**2
        else:
            raise u_.UnitsError("density must be dn/dE (1/(cm3 eV)) or "
                                "E^2 dn/dE (eV/cm3)")
        self.table = K.SeedTable(E.value, dnde, interpolation)
        self.name = name or "table"

    @property
    def energy_density(self) -> u_.Quantity:
        """Table energy density."""
        return self.table.energy_density() * u_.erg / u_.cm**3

    def _eps_max(self) -> float:
        return float(self.table.eps[-1])

    def _kernel(self, eps: np.ndarray, gamma: np.ndarray, rtol: float) -> np.ndarray:
        return K.ic_seed_kernel(eps, gamma, self.table, rtol=rtol)

    def _loss(self, E: np.ndarray) -> np.ndarray:
        return K.ic_loss_rate_seed(E, self.table)

    def __repr__(self) -> str:
        return f"TableSeed({self.table.eps.size} points, name={self.name!r})"


def as_seed(seed) -> Blackbody | TableSeed:
    """Coerce to a seed object."""
    if isinstance(seed, (Blackbody, TableSeed)):
        return seed
    if isinstance(seed, str):
        return Blackbody(seed)
    raise TypeError(f"cannot interpret {seed!r} as a seed photon field")


def ssc_seed(synchrotron, R, geometry: str = "uniform_sphere",
             energy=None, factor: float | None = None,
             name: str = "SSC") -> TableSeed:
    """Synchrotron seed, AA96 factor 2.24."""
    if factor is None:
        if geometry != "uniform_sphere":
            raise ValueError("only geometry='uniform_sphere' is implemented")
        factor = 2.24
    R_cm = u_.Quantity(R).to_value(u_.cm)
    if energy is None:
        gmin = synchrotron.Eemin.to_value(u_.erg) / K.MEC2
        gmax = synchrotron.Eemax.to_value(u_.erg) / K.MEC2
        B = synchrotron.B.to_value(u_.G)
        e_lo = 1e-6 * K.C.h_cgs * K.synch_nu_c(gmin, B)
        e_hi = 50.0 * K.C.h_cgs * K.synch_nu_c(gmax, B)
        n = int(np.ceil(20 * np.log10(e_hi / e_lo))) + 1
        energy = np.geomspace(e_lo, e_hi, n) * u_.erg
    energy = u_.Quantity(energy).to(u_.erg, equivalencies=u_.spectral())
    q = synchrotron.flux(energy, distance=0).to_value(1 / (u_.s * u_.erg))
    dnde = factor * q / (4.0 * np.pi * R_cm**2 * K.C.c_light)
    return TableSeed(energy, dnde / (u_.cm**3 * u_.erg), name=name)
