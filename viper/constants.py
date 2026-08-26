"""
Physical and model constants used throughout VIPER (cgs units unless otherwise noted)

Physical constants come in two complete sets:

- ``GSZ2009``: values matching the original C++ implementation of
  Gelfand, Slane & Zhang (2009).
- ``ASTROPY``: current best values from astropy (CODATA 2018 /
  IAU 2015).  Default.

To switch sets, set the VIPER_CONSTANTS environment variable to
either 'gsz2009' or 'astropy' before importing viper.
"""

import os
from types import SimpleNamespace

import numpy as np

# ---------------------------------------------------------------------------
# Source selection
# ---------------------------------------------------------------------------
_env = os.environ.get('VIPER_CONSTANTS')
if _env is None and 'PWNMODEL_CONSTANTS' in os.environ:
    import warnings
    warnings.warn("PWNMODEL_CONSTANTS is deprecated; use VIPER_CONSTANTS",
                  DeprecationWarning)
    _env = os.environ['PWNMODEL_CONSTANTS']
CONSTANTS_SOURCE: str = _env if _env is not None else 'astropy'
del _env

# Values matching the original C++ implementation; retained for exact
# reproducibility of published results.
GSZ2009 = SimpleNamespace(
    yr=31556926.0,              # seconds per year
    pc=3.09e18,                 # cm per parsec
    M_sun=1.99e33,              # grams
    m_proton=1.67262158e-24,    # grams
    c_light=3.0e10,             # cm/s
    m_electron=9.10938188e-28,  # grams
    e_charge=4.8032068e-10,     # esu
    GeV_per_erg=624.150974,     # GeV per erg
    h_cgs=6.626068e-27,         # erg s
    h_eV=4.1356668e-15,         # eV s
    k_B=1.380658e-16,           # erg/K
    mec2_keV=510.998910,        # keV
    sigma_T=6.6524586e-25,      # cm^2
)


def _astropy_set() -> SimpleNamespace:
    from astropy import constants as _ac
    from astropy import units as _u
    return SimpleNamespace(
        yr=_u.yr.to(_u.s),
        pc=_ac.pc.cgs.value,
        M_sun=_ac.M_sun.cgs.value,
        m_proton=_ac.m_p.cgs.value,
        c_light=_ac.c.cgs.value,
        m_electron=_ac.m_e.cgs.value,
        e_charge=_ac.e.gauss.value,
        GeV_per_erg=(1 * _u.erg).to(_u.GeV).value,
        h_cgs=_ac.h.cgs.value,
        h_eV=_ac.h.to(_u.eV * _u.s).value,
        k_B=_ac.k_B.cgs.value,
        mec2_keV=(_ac.m_e * _ac.c**2).to(_u.keV).value,
        sigma_T=_ac.sigma_T.cgs.value,
    )


# Current best values (CODATA 2018 / IAU 2015).
ASTROPY = _astropy_set()

_active = ASTROPY if CONSTANTS_SOURCE == 'astropy' else GSZ2009
yr: float = _active.yr
pc: float = _active.pc
M_sun: float = _active.M_sun
m_proton: float = _active.m_proton
c_light: float = _active.c_light
m_electron: float = _active.m_electron
e_charge: float = _active.e_charge
GeV_per_erg: float = _active.GeV_per_erg
h_cgs: float = _active.h_cgs
h_eV: float = _active.h_eV
k_B: float = _active.k_B
mec2_keV: float = _active.mec2_keV
sigma_T: float = _active.sigma_T
del _active

# ---------------------------------------------------------------------------
# Constants that are the same regardless of source
# ---------------------------------------------------------------------------
pi: float = np.pi
mu_ism: float = 1.0 / 1.64       # ISM mean molecular weight
I_ns: float = 1.0e45             # NS moment of inertia (g cm^2)

# ---------------------------------------------------------------------------
# SNR model constants (Truelove & McKee 1999)
# ---------------------------------------------------------------------------
m_ejecta_index: int = 9
l_ed: float = 1.19
phi_ed: float = 0.55
tst_star: float = 0.523
rst_star: float = 0.725
psi_0: float = 2.026

# Reverse shock parameters (TM99 Equations 83, 84 and Table 6)
rrscore_star: float = 0.371
tcore_star: float = 0.249
vrscore_star: float = 0.497
arscore_star: float = 0.162

# Adiabatic index of SNR ejecta
gamma_snr: float = 5.0 / 3.0

# ---------------------------------------------------------------------------
# Synchrotron calculation parameters
# ---------------------------------------------------------------------------
max_synch_x: float = 20.0

# ---------------------------------------------------------------------------
# Simulation grid defaults
# ---------------------------------------------------------------------------
N_ELEC_BINS: int = 500
N_IC_TARGET: int = 200
N_PHOT_FREQ: int = 500

TARGET_FREQ_MIN: float = 1.0e8            # Hz
TARGET_FREQ_MAX: float = 1.0e3 / h_eV     # Hz
PHOT_FREQ_MIN: float = 1.0e6              # Hz
PHOT_FREQ_MAX: float = 5.0e14 / h_eV      # Hz

# Fixed electron grid for the CMB IC lookup; must span prior ranges.
ELEC_GRID_FIXED_MIN: float = 1.0e-6      # GeV
ELEC_GRID_FIXED_MAX: float = 1.0e8       # GeV
ELEC_GRID_FIXED_NBINS: int = 500         # ~36 bins/decade
T_CMB: float = 2.728                     # K

# ---------------------------------------------------------------------------
# Pre-computed logarithmic constants
# ---------------------------------------------------------------------------
log_two: float = np.log10(2.0)
log_3: float = np.log10(3.0)
log_c: float = np.log10(c_light)
log_e_charge: float = np.log10(e_charge)
log_m_electron: float = np.log10(m_electron)
log_erg: float = np.log10(GeV_per_erg)
log_yr: float = np.log10(yr)
log_h_cgs: float = np.log10(h_cgs)

# Stefan-Boltzmann radiation constant a = 4 sigma/c [erg cm^-3 K^-4]
radiation_const: float = (
    8.0 * np.pi**5 * k_B**4 / (15.0 * h_cgs**3 * c_light**3)
)
eV_per_erg: float = GeV_per_erg * 1.0e9
