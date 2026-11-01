"""Shared radiative test inputs."""

import numpy as np
import astropy.units as u

from viper.radiative import ExponentialCutoffPowerLaw

NE = ExponentialCutoffPowerLaw(1e36 / u.eV, 1 * u.TeV, 2.3, 50 * u.TeV)
RNG = dict(Eemin=1 * u.GeV, Eemax=1 * u.PeV)
RNG_P = dict(Epmin=1 * u.GeV, Epmax=1 * u.PeV)
E_SYN = np.geomspace(1e-7, 1e7, 100) * u.eV
E_IC = np.geomspace(1e6, 1e15, 100) * u.eV
