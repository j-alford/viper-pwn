"""
VIPER is a python implementation of the Gelfand et al. (2009) pulsar wind nebula (PWN) model.

It is a semi-analytic model of the evolution of a PWN inside a non-radiative supernova remnant (SNR).

Reference: Gelfand, Slane & Zhang (2009), ApJ 703, 2051.

Package structure
-----------------
    viper/
    ├── parameter.py      Parameter class with unit enforcement
    ├── catalog.py        ATNF pulsar catalogue
    ├── model.py          PWNModel: Parameter-based top-level interface
    ├── constants.py      Physical constants (cgs units)
    ├── grids.py          SpectralGrid: log-spaced energy/frequency bins
    ├── pulsar.py         E_dot, initial energy, spin-down calculations
    ├── snr.py            SNR dynamics: Sedov, reverse shock, interior
    ├── particles.py      Electron injection + bin redistribution
    ├── injection.py      Continuous injection-spectrum source  (injection_model)
    ├── escape.py         One-zone particle-escape sink (escape_model)
    ├── radiation.py      Synchrotron + IC emission, F(x) table, Numba kernel
    ├── evolution.py      Main time loop + CMB IC lookup
    ├── observables.py    Extract observables (angular sizes, fluxes, indices)
    ├── fitting.py        MCMC fitting with emcee (model + observables)
    ├── nested.py         Nested-sampling with UltraNest
    ├── plotting/         SED, particle-spectrum, movie, and corner plots
    ├── gammapy.py        PWNSpectralModel for gammapy fitting
    ├── scalar_params.py  Scalar-dataclass API (PWNScalarParams) for one-off runs
    └── data/
        ├── psrcat.db                       ATNF pulsar catalogue
        ├── psrcat_ref                      Catalogue reference codes
        ├── cmb_pow_nu_ic_gsz2009.npz       CMB IC lookup table (gsz2009 constants)
        └── cmb_pow_nu_ic_astropy.npz       CMB IC lookup table (astropy constants)

"""

__version__ = "0.3.0"

from .parameter import Parameter, ParameterSet
from .evolution import evolve_pwn, PWNModelResult, DynState
from .grids import SpectralGrid, make_log_grid
from .model import PWNModel
from .catalog import PulsarCatalog
from .observables import Observable
from .fitting import (
    run_mcmc, build_fit_spec, log_probability, log_likelihood,
    print_residuals, decode_theta,
)
from .nested import run_nested, prior_transform
from .plotting import plot_sed

# Legacy import (deprecated)
from .scalar_params import PWNScalarParams

# Gammapy integration (optional); resolved lazily via PEP 562.
def __getattr__(name):
    if name == "PWNSpectralModel":
        from .gammapy import PWNSpectralModel
        return PWNSpectralModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | {"PWNSpectralModel"})

__all__ = [
    "PWNModel",
    "Parameter",
    "ParameterSet",
    "PulsarCatalog",
    "Observable",
    "evolve_pwn",
    "PWNModelResult",
    "DynState",
    "SpectralGrid",
    "make_log_grid",
    "PWNScalarParams",
    "PWNSpectralModel",
    "run_mcmc",
    "build_fit_spec",
    "log_probability",
    "log_likelihood",
    "print_residuals",
    "decode_theta",
    "run_nested",
    "prior_transform",
    "plot_sed",
]
