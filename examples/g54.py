"""
G54.1+0.3 (PSR J1930+1852): an example source file.

Copy this file, replace the values with your own, and run

    python3 tools/run_mcmc.py my_source.py

Each run gets its own folder under results/ holding the chain, the best
fit, and a copy of this file as it was when the run started.

The observables and starting values are those of the README fitting
example (see Gelfand, Slane & Temim 2015, ApJ 807, 30).  The bounds and
the prior are illustrative.
"""
import astropy.units as u
from viper import Observable

# Spin period, period derivative and distance come from the ATNF
# catalogue.  For a pulsar that is not in it, set PERIOD and P_DOT
# instead, e.g. PERIOD = 0.137 * u.s and P_DOT = 7.5e-13.
PULSAR = 'J1930+1852'

# What the model is compared against.  Add weight=... to any entry to
# make it count more (or less) in the fit, and upper_limit=True for an
# upper limit.
OBSERVABLES = [
    Observable.angular_size_snr(6.6, 0.4, 'arcmin'),
    Observable.angular_size_pwn(1.14, 0.04, 'arcmin'),
    Observable.flux_density(1.4, 'GHz', 433, 30, 'mJy'),
    Observable.flux_density(4.7, 'GHz', 327, 25, 'mJy'),
    Observable.band_flux(2, 10, 'keV', 5.43e-12, 0.035e-12),
    Observable.spectral_index(2, 10, 'keV', 2.09, 0.01),
    Observable.flux_density(311, 'GeV', 1.10e-11, 0.56e-11,
                            'photons/cm2/s/TeV'),
]

# Parameters to fit: (starting value, lower bound, upper bound).
PARAMETERS = {
    'mej':      (21 * u.M_sun,     1 * u.M_sun,     100 * u.M_sun),
    'n_ism':    (0.008 / u.cm**3,  1e-4 / u.cm**3,  1 / u.cm**3),
    'tau_sd':   (3574 * u.yr,      100 * u.yr,      1e5 * u.yr),
    'eta_b':    (0.0024,           1e-5,            0.5),
    'e_break':  (3232 * u.GeV,     10 * u.GeV,      1e5 * u.GeV),
    'distance': (6.223 * u.kpc,    4 * u.kpc,       10 * u.kpc),
}

# Parameters held fixed.  Move an entry up to PARAMETERS (and give it
# bounds) to fit it.
FROZEN = {
    'esn':           0.95e51 * u.erg,
    'braking_index': 1.9,
    'e_min':         10 * u.GeV,
    'e_max':         400 * u.TeV,
    'p1':            2.29,
    'p2':            2.42,
}

# Gaussian priors: (center, sigma).
PRIORS = {
    'distance': (6.2 * u.kpc, 1.0 * u.kpc),
}

# Photon fields for inverse-Compton scattering, on top of the CMB (which
# is always included).  To fit a field, give it bounds, e.g.
# 'temp_bounds': (1000 * u.K, 5000 * u.K).
IC_FIELDS = [
    {'temp': 2753 * u.K, 'norm': 23.6137 * u.eV / u.cm**3},
]

# MCMC settings.  nwalkers must be at least twice the number of
# parameters being fit.
FIT = {
    'nwalkers': 32,
    'nsteps': 2000,
}
